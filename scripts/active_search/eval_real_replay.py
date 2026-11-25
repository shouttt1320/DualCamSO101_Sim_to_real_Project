#!/usr/bin/env python3
"""
[Real-World Offline Replay & Policy Diagnostic]
================================================
실물 IK로 수집한 HDF5 에피소드(real_ik)의 실제 양안 카메라 프레임(cam_left, cam_right)과
관절 상태(qpos)를 시뮬레이션 학습 모델(DINOv2-ACT)에 입력해,
실물 로봇을 움직이지 않고도 모델의 예측 궤적을 검증하고 오차를 시각화합니다.

[실행 예시]
  bash scripts/active_search/eval_real_replay.sh
  bash scripts/active_search/eval_real_replay.sh --hdf5 Robot_Arm_Project/data/so101_active_search/real_ik/episode_0000.hdf5
  bash scripts/active_search/eval_real_replay.sh --ckpt Robot_Arm_Project/ckpt/so101_active_search/domain_rand/policy_best3.pth
"""

import os
import sys
import glob
import pickle
import argparse
import numpy as np
import h5py
import torch
from torchvision import transforms
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

from so101_dual_dinov2_act.models import build_act_model
from so101_dual_dinov2_act.config import (
    CAMERA_NAMES, CHUNK_SIZE, IMAGE_SIZE, LATENT_DIM,
    STATE_DIM, BACKBONE, HIDDEN_DIM, DIM_FEEDFORWARD,
    ENC_LAYERS, DEC_LAYERS, NHEADS
)

JOINT_LABELS = ["Pan (deg)", "Lift (deg)", "Elbow (deg)", "W_Flex (deg)", "W_Roll (deg)", "Gripper (deg)"]


def estimate_phase_bounds(actions_rad: np.ndarray):
    """에피소드 그리퍼 각도로 Search / Grasp / Transport / Place 경계 추정"""
    T = len(actions_rad)
    grip_deg = np.degrees(actions_rad[:, 5])
    idx1 = np.where(grip_deg > 50.0)[0]
    t_search_end = int(idx1[0]) if len(idx1) > 0 else 0
    idx2 = np.where(grip_deg[t_search_end:] < 20.0)[0]
    t_grasp_done = int(t_search_end + idx2[0]) if len(idx2) > 0 else int(T * 0.4)
    idx3 = np.where(grip_deg[t_grasp_done:] > 50.0)[0]
    t_place_start = int(t_grasp_done + idx3[0]) if len(idx3) > 0 else int(T * 0.8)
    return t_search_end, t_grasp_done, t_place_start


def resolve_hdf5_path(hdf5_arg: str, data_dir: str) -> str:
    if hdf5_arg:
        if not os.path.exists(hdf5_arg):
            raise FileNotFoundError(f"HDF5 file not found: {hdf5_arg}")
        return hdf5_arg
    files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not files:
        raise FileNotFoundError(
            f"No real_ik episodes found in {data_dir}.\n"
            f"먼저 실물 IK 데이터 수집기(bash scripts/active_search/real_ik_collect.sh)로 1개 이상 수집해주세요."
        )
    return files[-1]


def load_policy(ckpt_path: str, stats_path: str, device: torch.device):
    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    policy = build_act_model(
        state_dim=STATE_DIM,
        num_queries=CHUNK_SIZE,
        camera_names=CAMERA_NAMES,
        backbone_name=BACKBONE,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, norm_stats


def main():
    parser = argparse.ArgumentParser(description="Offline Replay Evaluation on Real-World IK HDF5 Episode")
    parser.add_argument("--hdf5", type=str, default=None, help="Path to real HDF5 episode (default: latest in real_ik)")
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "data", "so101_active_search", "real_robot"))
    parser.add_argument("--ckpt", type=str, default=os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", "domain_rand", "policy_best.pth"))
    parser.add_argument("--stats", type=str, default=None, help="Path to dataset_stats.pkl (default: next to ckpt)")
    parser.add_argument("--stride", type=int, default=5, help="Frame evaluation stride")
    parser.add_argument("--lookahead", type=int, default=20, help="Chunk lookahead step k to compare trend")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path")
    args = parser.parse_args()

    hdf5_path = resolve_hdf5_path(args.hdf5, args.data_dir)
    stats_path = args.stats if args.stats else os.path.join(os.path.dirname(args.ckpt), "dataset_stats.pkl")
    out_png = args.out if args.out else os.path.join(os.path.dirname(hdf5_path), "real_replay_eval.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(">>> [REAL-WORLD OFFLINE REPLAY]")
    print(f"  • Episode:    {hdf5_path}")
    print(f"  • Checkpoint: {args.ckpt}")
    print(f"  • Device:     {device}")
    print("=" * 80)

    policy, norm_stats = load_policy(args.ckpt, stats_path, device)
    q_mean = torch.tensor(norm_stats["qpos_mean"], dtype=torch.float32, device=device)
    q_std = torch.tensor(norm_stats["qpos_std"], dtype=torch.float32, device=device)
    a_mean = norm_stats["action_mean"]
    a_std = norm_stats["action_std"]

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    with h5py.File(hdf5_path, "r") as f:
        qpos_all = f["observations/qpos"][()]
        act_all = f["action"][()]
        imgs_l = f["observations/images/cam_left"][()]
        imgs_r = f["observations/images/cam_right"][()]

    T = len(act_all)
    t_se, t_gd, t_ps = estimate_phase_bounds(act_all)
    steps = list(range(0, T, max(1, args.stride)))

    pred_k0 = np.zeros((len(steps), 6), dtype=np.float32)
    pred_kk = np.zeros((len(steps), 6), dtype=np.float32)
    k_idx = min(args.lookahead, CHUNK_SIZE - 1)

    with torch.no_grad():
        for i, ts in enumerate(steps):
            img_l = torch.from_numpy(imgs_l[ts]).permute(2, 0, 1).float() / 255.0
            img_r = torch.from_numpy(imgs_r[ts]).permute(2, 0, 1).float() / 255.0
            img_l = transform(img_l)
            img_r = transform(img_r)
            imgs = torch.stack([img_l, img_r], dim=0).unsqueeze(0).to(device)
            q_norm = ((torch.from_numpy(qpos_all[ts]).float().to(device) - q_mean) / q_std).unsqueeze(0)

            pred_norm, _, _ = policy(qpos=q_norm, image=imgs)
            pred_rad = pred_norm[0].cpu().numpy() * a_std + a_mean
            pred_k0[i] = pred_rad[0]
            pred_kk[i] = pred_rad[k_idx]

    gt_deg = np.degrees(act_all[steps])
    p0_deg = np.degrees(pred_k0)
    pk_deg = np.degrees(pred_kk)
    err_deg = np.abs(p0_deg - gt_deg)

    print("\n[1] 전체 관절별 예측 오차 MAE (deg):")
    for j, name in enumerate(JOINT_LABELS):
        print(f"  • {name:14s}: {err_deg[:, j].mean():.2f}°  (Max: {err_deg[:, j].max():.2f}°)")

    print("\n[2] 작업 단계별 평균 오차 MAE (deg):")
    steps_arr = np.array(steps)
    phase_masks = [
        ("Search", steps_arr < t_se),
        ("Grasp", (steps_arr >= t_se) & (steps_arr < t_gd)),
        ("Transport", (steps_arr >= t_gd) & (steps_arr < t_ps)),
        ("Place", steps_arr >= t_ps),
    ]
    for pname, mask in phase_masks:
        if np.any(mask):
            print(f"  • {pname:10s} ({int(mask.sum()):3d} pts): {err_deg[mask].mean():.2f}°")

    # t=0 초기 반응 진단 (바닥 쳐박기 vs 탐색)
    gt_t0_next = np.degrees(act_all[min(k_idx, T - 1)] - qpos_all[0])
    pred_t0_next = pk_deg[0] - np.degrees(qpos_all[0])
    print(f"\n[3] t=0 초기 동작 방향 진단 (현재 qpos 대비 +{k_idx} step 변화량):")
    print(f"  • GT Real IK : Pan={gt_t0_next[0]:+.1f}°, Lift={gt_t0_next[1]:+.1f}°, Elbow={gt_t0_next[2]:+.1f}°, Grip={np.degrees(act_all[min(k_idx, T-1), 5]):.1f}°")
    print(f"  • Policy Pred: Pan={pred_t0_next[0]:+.1f}°, Lift={pred_t0_next[1]:+.1f}°, Elbow={pred_t0_next[2]:+.1f}°, Grip={pk_deg[0, 5]:.1f}°")

    # 시각화
    fig, axes = plt.subplots(3, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()
    for j in range(6):
        ax = axes[j]
        ax.plot(steps, gt_deg[:, j], label="GT Real IK", color="#1f77b4", linewidth=2.0)
        ax.plot(steps, p0_deg[:, j], label="Pred (k=0)", color="#d62728", linestyle="--", linewidth=1.6)
        ax.plot(steps, pk_deg[:, j], label=f"Pred (k={k_idx})", color="#ff7f0e", linestyle=":", linewidth=1.5)
        for bx in (t_se, t_gd, t_ps):
            if 0 < bx < T:
                ax.axvline(bx, color="gray", linestyle="-.", alpha=0.5)
        ax.set_ylabel(JOINT_LABELS[j])
        ax.set_title(f"{JOINT_LABELS[j]} | MAE={err_deg[:, j].mean():.2f}°")
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(loc="best", fontsize=8)
    axes[4].set_xlabel("Timestep")
    axes[5].set_xlabel("Timestep")
    fig.suptitle(f"Real-World Offline Replay: {os.path.basename(hdf5_path)} vs {os.path.basename(args.ckpt)}", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    print(f"\n📊 [Saved] 궤적 비교 그래프 저장 완료: {out_png}")


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""
[Checkpoint Diagnostic Tool]
==============================
저장된 여러 에포크 체크포인트(.pth)들을 로드하여:
  1) 작업 구간별 L1 손실 (Search / Grasp / Transport / Place)
  2) 관절별 L1 손실 (Pan / Lift / Elbow / Wrist_Flex / Wrist_Roll / Gripper)
  3) 큐브 스폰 각도 구간별 L1 손실 (-65° ~ +65°)
을 정밀하게 역산·비교하고 시각화 그래프(checkpoint_diagnosis.png)를 생성합니다.
"""

import os
import re
import glob
import pickle
import argparse
import numpy as np
import h5py
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
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


def get_episode_metadata(hdf5_path):
    """에피소드 HDF5에서 단계 경계 및 큐브 스폰 각도(theta_deg) 추출"""
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][()]
        T = len(actions)
        grip_deg = np.degrees(actions[:, 5])

        idx1 = np.where(grip_deg > 50.0)[0]
        t_search_end = int(idx1[0]) if len(idx1) > 0 else 0

        idx2 = np.where(grip_deg[t_search_end:] < 20.0)[0]
        t_grasp_done = int(t_search_end + idx2[0]) if len(idx2) > 0 else int(T * 0.4)

        idx3 = np.where(grip_deg[t_grasp_done:] > 50.0)[0]
        t_place_start = int(t_grasp_done + idx3[0]) if len(idx3) > 0 else int(T * 0.8)

        xy = f.attrs.get("cube_spawn_xy", [0.24, 0.0])
        theta_deg = float(np.degrees(np.arctan2(xy[1], xy[0])))

    return {
        "T": T,
        "bounds": (t_search_end, t_grasp_done, t_place_start),
        "theta_deg": theta_deg,
    }


def build_deterministic_eval_samples(data_dir, norm_stats, stride=25):
    """모든 에피소드에서 일정 간격(stride)으로 결정론적(Deterministic) 평가 샘플 사전 추출"""
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    samples = []
    print(f">>> [Data] {len(hdf5_files)}개 에피소드에서 결정론적 평가 샘플 추출 중 (stride={stride})...")
    for fpath in hdf5_files:
        meta = get_episode_metadata(fpath)
        T = meta["T"]
        t_se, t_gd, t_ps = meta["bounds"]
        theta_deg = meta["theta_deg"]

        with h5py.File(fpath, "r") as root:
            actions = root["action"][()]
            qpos_all = root["observations/qpos"][()]

            # 각 단계에서 골고루 샘플링
            for ts in range(0, max(1, T - 10), stride):
                if ts < t_se:
                    phase = 0  # Search
                elif ts < t_gd:
                    phase = 1  # Grasp
                elif ts < t_ps:
                    phase = 2  # Transport
                else:
                    phase = 3  # Place

                # 양안 이미지 로드
                cam_imgs = []
                for cname in CAMERA_NAMES:
                    key = f"observations/images/{cname}"
                    if key not in root:
                        key = f"observations/images/{cname.replace('cam_', '')}"
                    img = root[key][ts]
                    t_img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                    cam_imgs.append(transform(t_img))
                image_tensor = torch.stack(cam_imgs, dim=0)

                qpos_norm = (qpos_all[ts] - norm_stats["qpos_mean"]) / norm_stats["qpos_std"]
                qpos_tensor = torch.from_numpy(qpos_norm).float()

                # Action slice
                act_slice = actions[ts:ts + CHUNK_SIZE]
                actual_len = act_slice.shape[0]
                padded = np.zeros((CHUNK_SIZE, 6), dtype=np.float32)
                padded[:actual_len] = act_slice
                is_pad = np.zeros(CHUNK_SIZE, dtype=bool)
                if actual_len < CHUNK_SIZE:
                    padded[actual_len:] = act_slice[-1]
                    is_pad[actual_len:] = True

                act_norm = (padded - norm_stats["action_mean"]) / norm_stats["action_std"]

                samples.append({
                    "image": image_tensor,
                    "qpos": qpos_tensor,
                    "action": torch.from_numpy(act_norm).float(),
                    "is_pad": torch.from_numpy(is_pad),
                    "phase": phase,
                    "theta_deg": theta_deg,
                })

    print(f"  ✅ 총 {len(samples)}개 평가 샘플 준비 완료.")
    return samples


def evaluate_checkpoint(model, ckpt_path, samples, device, batch_size=32):
    """단일 체크포인트에 대한 전체/구간별/관절별/각도별 L1 오차 측정"""
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model.eval()

    phase_errors = {0: [], 1: [], 2: [], 3: []}
    joint_errors = []
    # 각도 구간: [-65,-35), [-35,-10), [-10,10], (10,35], (35,65]
    angle_bins = [
        ("L_Ext(-65~-35)", -90, -35),
        ("L_Mid(-35~-10)", -35, -10),
        ("Center(-10~+10)", -10, 10),
        ("R_Mid(+10~+35)", 10, 35),
        ("R_Ext(+35~+65)", 35, 90),
    ]
    angle_errors = {b[0]: [] for b in angle_bins}

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            imgs = torch.stack([s["image"] for s in batch]).to(device)
            qpos = torch.stack([s["qpos"] for s in batch]).to(device)
            acts = torch.stack([s["action"] for s in batch]).to(device)
            is_pad = torch.stack([s["is_pad"] for s in batch]).to(device)

            # Inference mode (no CVAE ground truth actions passed)
            a_hat, _, _ = model(qpos=qpos, image=imgs)

            raw_l1 = F.l1_loss(acts, a_hat, reduction="none")  # (B, chunk_size, 6)
            valid = (~is_pad).unsqueeze(-1).float()

            # 샘플별 평균 L1 (B,)
            s_l1 = (raw_l1 * valid).sum(dim=(1, 2)) / (valid.sum(dim=(1, 2)) * 6.0).clamp(min=1.0)
            s_l1_np = s_l1.cpu().numpy()

            # 관절별 L1 (B, 6)
            j_l1 = (raw_l1 * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
            joint_errors.append(j_l1.cpu().numpy())

            for idx, s in enumerate(batch):
                err = float(s_l1_np[idx])
                phase_errors[s["phase"]].append(err)
                th = s["theta_deg"]
                for b_name, lo, hi in angle_bins:
                    if lo <= th <= hi:
                        angle_errors[b_name].append(err)
                        break

    all_joint = np.concatenate(joint_errors, axis=0).mean(axis=0)  # (6,)
    all_errs = [e for lst in phase_errors.values() for e in lst]

    return {
        "total": float(np.mean(all_errs)),
        "phase": {
            "Search": float(np.mean(phase_errors[0])) if phase_errors[0] else np.nan,
            "Grasp": float(np.mean(phase_errors[1])) if phase_errors[1] else np.nan,
            "Transport": float(np.mean(phase_errors[2])) if phase_errors[2] else np.nan,
            "Place": float(np.mean(phase_errors[3])) if phase_errors[3] else np.nan,
        },
        "joint": {
            "Pan": float(all_joint[0]),
            "Lift": float(all_joint[1]),
            "Elbow": float(all_joint[2]),
            "W_Flex": float(all_joint[3]),
            "W_Roll": float(all_joint[4]),
            "Gripper": float(all_joint[5]),
        },
        "angle": {k: (float(np.mean(v)) if v else np.nan) for k, v in angle_errors.items()},
    }


def plot_diagnosis(results_list, save_path):
    """체크포인트별 구간/관절/각도별 L1 오차 비교 그래프 생성"""
    labels = [r["label"] for r in results_list]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # 1. 작업 단계별 (Phase-wise)
    ax = axes[0, 0]
    for key, col in [("Search", "#1f77b4"), ("Grasp", "#ff7f0e"), ("Transport", "#2ca02c"), ("Place", "#9467bd")]:
        vals = [r["metrics"]["phase"][key] for r in results_list]
        ax.plot(x, vals, marker="o", linewidth=2, label=key, color=col)
    ax.plot(x, [r["metrics"]["total"] for r in results_list], marker="s", linestyle="--", color="black", label="Total Mean")
    ax.set_title("1. Phase-wise L1 Loss across Checkpoints", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("L1 Error (Normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 큐브 스폰 각도별 (Angle-wise)
    ax = axes[0, 1]
    angle_keys = list(results_list[0]["metrics"]["angle"].keys())
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#17becf", "#1f77b4"]
    for key, col in zip(angle_keys, colors):
        vals = [r["metrics"]["angle"][key] for r in results_list]
        ax.plot(x, vals, marker="o", linewidth=2, label=key, color=col)
    ax.set_title("2. Cube Spawn Angle-wise L1 Loss (-65° ~ +65°)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("L1 Error (Normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 관절별 (Joint-wise)
    ax = axes[1, 0]
    joint_keys = ["Pan", "Lift", "Elbow", "W_Flex", "W_Roll", "Gripper"]
    j_cols = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e", "#8c564b", "#e377c2"]
    for key, col in zip(joint_keys, j_cols):
        vals = [r["metrics"]["joint"][key] for r in results_list]
        ax.plot(x, vals, marker="^", linewidth=1.8, label=key, color=col)
    ax.set_title("3. Joint-wise L1 Loss across Checkpoints", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25)
    ax.set_ylabel("L1 Error (Normalized)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 최신(Best) vs 초기(Epoch 50) 각도별 바 차트 비교
    ax = axes[1, 1]
    bx = np.arange(len(angle_keys))
    width = 0.35
    first_res = results_list[0]
    best_res = min(results_list, key=lambda r: r["metrics"]["total"])
    vals_first = [first_res["metrics"]["angle"][k] for k in angle_keys]
    vals_best = [best_res["metrics"]["angle"][k] for k in angle_keys]
    ax.bar(bx - width/2, vals_first, width, label=f"Early ({first_res['label']})", color="#aec7e8")
    ax.bar(bx + width/2, vals_best, width, label=f"Best ({best_res['label']})", color="#1f77b4")
    ax.set_title("4. Angle Error Comparison: Early vs Best Checkpoint", fontsize=13, fontweight="bold")
    ax.set_xticks(bx)
    ax.set_xticklabels(angle_keys, rotation=15)
    ax.set_ylabel("L1 Error (Normalized)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"\n📊 진단 그래프 저장 완료: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="default_clean")
    parser.add_argument("--stride", type=int, default=30)
    args = parser.parse_args()

    data_dir = os.path.join(PROJECT_ROOT, "data", "so101_active_search", args.env)
    ckpt_dir = os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", args.env)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

    with open(stats_path, "rb") as f:
        norm_stats = pickle.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_act_model(
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
        train_backbone=False,
    ).to(device)

    # 체크포인트 파일 목록 정렬 (epoch 숫자 기준)
    epoch_ckpts = []
    for p in glob.glob(os.path.join(ckpt_dir, "policy_best_epoch_*.pth")):
        m = re.search(r"epoch_(\d+)\.pth$", p)
        if m:
            epoch_ckpts.append((int(m.group(1)), f"Ep{m.group(1)}", p))
    epoch_ckpts.sort(key=lambda t: t[0])

    best_p = os.path.join(ckpt_dir, "policy_best.pth")
    if os.path.exists(best_p):
        epoch_ckpts.append((99999, "Best", best_p))

    samples = build_deterministic_eval_samples(data_dir, norm_stats, stride=args.stride)

    print("\n" + "=" * 95)
    print(f"{'Checkpoint':<10} | {'Total':>6} | {'Search':>6} | {'Grasp':>6} | {'Trans':>6} | {'Place':>6} || {'Pan':>5} | {'Arm':>5} | {'Grip':>5}")
    print("-" * 95)

    results = []
    for _, label, path in epoch_ckpts:
        m = evaluate_checkpoint(model, path, samples, device)
        results.append({"label": label, "path": path, "metrics": m})
        arm_avg = np.mean([m["joint"]["Lift"], m["joint"]["Elbow"], m["joint"]["W_Flex"]])
        print(
            f"{label:<10} | {m['total']:.4f} | {m['phase']['Search']:.4f} | {m['phase']['Grasp']:.4f} | "
            f"{m['phase']['Transport']:.4f} | {m['phase']['Place']:.4f} || "
            f"{m['joint']['Pan']:.4f} | {arm_avg:.4f} | {m['joint']['Gripper']:.4f}"
        )
    print("=" * 95)

    out_png = os.path.join(ckpt_dir, "checkpoint_diagnosis.png")
    plot_diagnosis(results, out_png)


if __name__ == "__main__":
    main()

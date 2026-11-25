"""양안(Dual Eyes) 카메라 DINOv2 ACT 모델 학습 스크립트."""

import argparse
import csv
import json
import os
import pickle
import time
from copy import deepcopy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from so101_dual_dinov2_act.config import (
    BACKBONE,
    BASE_DIR,
    BATCH_SIZE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DATA_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    IMAGE_SIZE,
    KL_WEIGHT,
    LATENT_DIM,
    LR,
    LR_BACKBONE,
    NHEADS,
    NUM_EPOCHS,
    SAVE_EVERY,
    SEED,
    STATE_DIM,
    TRAIN_BACKBONE,
    UNFREEZE_BACKBONE_BLOCKS,
    WEIGHT_DECAY,
    STEPS_PER_EPISODE_PER_EPOCH,
    QPOS_NOISE_STD,
    QPOS_DROPOUT_PROB,
    PATCH_DROPOUT_PROB,
    WEIGHTED_SAMPLING,
    START_WEIGHT_MULTIPLIER,
    GRASP_WEIGHT_MULTIPLIER,
)
from so101_dual_dinov2_act.dataset import load_dual_eyes_dataloaders
from so101_dual_dinov2_act.models import build_act_model, kl_divergence


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_loss(
    model,
    data,
    kl_weight: float,
    device: torch.device,
    gripper_loss_weight: float = 1.0,
    aux_loss_weight: float = 0.2,
):
    """L1 모방 오차, CVAE KL 발산, 15x20 보조 히트맵/보임 손실, 단계별/관절별 세부 손실 계산"""
    gt_heatmap = None
    gt_vis = None
    if len(data) == 8:
        image_data, qpos_data, action_data, is_pad_data, phase_id, patch_mask, gt_heatmap, gt_vis = data
        phase_id = phase_id.to(device)
        patch_mask = patch_mask.to(device) if model.training else None
        gt_heatmap = gt_heatmap.to(device)
        gt_vis = gt_vis.to(device)
    elif len(data) == 6:
        image_data, qpos_data, action_data, is_pad_data, phase_id, patch_mask = data
        phase_id = phase_id.to(device)
        patch_mask = patch_mask.to(device) if model.training else None
    elif len(data) == 5:
        image_data, qpos_data, action_data, is_pad_data, phase_id = data
        phase_id = phase_id.to(device)
        patch_mask = None
    else:
        image_data, qpos_data, action_data, is_pad_data = data
        phase_id = None
        patch_mask = None

    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad_data = is_pad_data.to(device)

    # Forward Pass
    a_hat, is_pad_hat, (mu, logvar) = model(
        qpos=qpos_data,
        image=image_data,
        actions=action_data,
        is_pad=is_pad_data,
        patch_mask=patch_mask,
    )

    # 1. L1 액션 모방 손실 (원본 오차 및 가중치 적용 오차)
    raw_l1 = F.l1_loss(action_data, a_hat, reduction="none")  # (B, chunk_size, 6)
    valid_mask = (~is_pad_data).unsqueeze(-1).float()       # (B, chunk_size, 1)

    if gripper_loss_weight > 1.0:
        joint_weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, gripper_loss_weight], device=device)
        weighted_l1 = raw_l1 * joint_weights
    else:
        weighted_l1 = raw_l1

    l1_loss = (weighted_l1 * valid_mask).mean()

    # 2. CVAE KL 발산 손실
    total_kld, _, _ = kl_divergence(mu, logvar)
    kl_loss = total_kld[0]

    # 2.5 보조 손실 (15x20 패치 히트맵 BCE + MaxPool 큐브 보임 확률 BCE)
    aux_loss = torch.tensor(0.0, device=device)
    vis_acc = None
    if aux_loss_weight > 0.0 and gt_heatmap is not None and getattr(model, "last_aux", None) is not None:
        hm_logits = model.last_aux["heatmap_logits"]
        vis_logit = model.last_aux["vis_logit"]
        pos_w = torch.tensor([15.0], device=device)
        hm_bce = F.binary_cross_entropy_with_logits(hm_logits, gt_heatmap, pos_weight=pos_w)
        vis_bce = F.binary_cross_entropy_with_logits(vis_logit, gt_vis)
        aux_loss = 0.5 * (hm_bce + vis_bce)
        with torch.no_grad():
            pred_vis = (torch.sigmoid(vis_logit) >= 0.5).float()
            vis_acc = float((pred_vis == gt_vis).float().mean().item()) * 100.0

    # 3. 종합 손실
    total_loss = l1_loss + kl_weight * kl_loss + aux_loss_weight * aux_loss

    # 4. 관절별 L1 오차 분해 (6-DOF: Pan, Lift, Elbow, Wrist_Flex, Wrist_Roll, Gripper)
    with torch.no_grad():
        joint_l1 = (raw_l1 * valid_mask).sum(dim=(0, 1)) / valid_mask.sum(dim=(0, 1)).clamp(min=1.0)
        joint_l1_list = joint_l1.cpu().tolist()

        # 5. 작업 단계별 L1 오차 분해 (0: Search, 1: Grasp, 2: Transport, 3: Place)
        phase_l1 = {}
        if phase_id is not None:
            # 샘플별 평균 L1 (B,)
            sample_l1 = (raw_l1 * valid_mask).sum(dim=(1, 2)) / (valid_mask.sum(dim=(1, 2)) * 6.0).clamp(min=1.0)
            for p_idx, p_name in enumerate(["search", "grasp", "transport", "place"]):
                p_mask = (phase_id == p_idx)
                phase_l1[p_name] = float(sample_l1[p_mask].mean().item()) if p_mask.any() else None

    return {
        "loss": total_loss,
        "l1": l1_loss,
        "kl": kl_loss,
        "aux": aux_loss,
        "vis_acc": vis_acc,
        "joint_l1": joint_l1_list,
        "phase_l1": phase_l1,
    }


def plot_training_curves(history: list, save_path: str):
    """학습/검증 전체 손실 및 작업 단계별/관절별 손실 곡선 그래프 저장"""
    if not history:
        return

    epochs = [r["epoch"] for r in history]
    train_loss = [r["train_loss"] for r in history]
    val_loss = [r["val_loss"] for r in history]
    train_l1 = [r["train_l1"] for r in history]
    val_l1 = [r["val_l1"] for r in history]
    train_kl = [r["train_kl"] for r in history]
    val_kl = [r["val_kl"] for r in history]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # [0, 0] Total Loss
    axes[0, 0].plot(epochs, train_loss, label="Train Loss", color="royalblue")
    axes[0, 0].plot(epochs, val_loss, label="Val Loss", color="crimson", linestyle="--")
    axes[0, 0].set_title("Total Loss (L1 + KL)")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # [0, 1] L1 Loss
    axes[0, 1].plot(epochs, train_l1, label="Train L1", color="royalblue")
    axes[0, 1].plot(epochs, val_l1, label="Val L1", color="crimson", linestyle="--")
    axes[0, 1].set_title("L1 Action Loss")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("L1 Error")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # [0, 2] KL Divergence
    axes[0, 2].plot(epochs, train_kl, label="Train KL", color="royalblue")
    axes[0, 2].plot(epochs, val_kl, label="Val KL", color="crimson", linestyle="--")
    axes[0, 2].set_title("KL Divergence")
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("KL")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # [1, 0] 작업 단계별 Val L1 추이
    phase_styles = [
        ("val_search", "Search", "#1f77b4"),
        ("val_grasp", "Grasp", "#ff7f0e"),
        ("val_transport", "Transport", "#2ca02c"),
        ("val_place", "Place", "#9467bd"),
    ]
    for key, label, color in phase_styles:
        vals = [r.get(key, np.nan) for r in history]
        axes[1, 0].plot(epochs, vals, label=label, color=color, linewidth=1.8)
    axes[1, 0].set_title("Val L1 by Task Phase")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("L1 Error")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # [1, 1] 주요 팔 관절 Val L1 추이
    arm_styles = [
        ("val_pan", "Shoulder Pan", "#d62728"),
        ("val_lift", "Shoulder Lift", "#1f77b4"),
        ("val_elbow", "Elbow Flex", "#2ca02c"),
        ("val_wrist_flex", "Wrist Flex", "#ff7f0e"),
    ]
    for key, label, color in arm_styles:
        vals = [r.get(key, np.nan) for r in history]
        axes[1, 1].plot(epochs, vals, label=label, color=color, linewidth=1.6)
    axes[1, 1].set_title("Val L1 by Arm Joint")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("L1 Error")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # [1, 2] 말단(손목 회전 & 그리퍼) Val L1 추이
    ee_styles = [
        ("val_wrist_roll", "Wrist Roll", "#8c564b"),
        ("val_gripper", "Gripper", "#e377c2"),
    ]
    for key, label, color in ee_styles:
        vals = [r.get(key, np.nan) for r in history]
        axes[1, 2].plot(epochs, vals, label=label, color=color, linewidth=1.8)
    axes[1, 2].set_title("Val L1 by End-Effector Joint")
    axes[1, 2].set_xlabel("Epoch")
    axes[1, 2].set_ylabel("L1 Error")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def train_dual_eyes_act(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== SO-101 양안 DINOv2 ACT 모델 학습 시작 (Device: {device}) ===")
    print(f"  백본 모델: {args.backbone} (Train Backbone: {args.train_backbone})")
    print(f"  카메라 목록: {args.camera_names}")
    print(f"  입력 해상도: {args.image_size} (14의 배수)")
    print(f"  예측 청크(CHUNK_SIZE): {args.chunk_size} steps")
    print(f"  배치 크기: {args.batch_size}, 에포크: {args.num_epochs}")
    print(f"  에피소드당 샘플링 수: {args.steps_per_episode} steps/ep (총 {29 * args.steps_per_episode} samples/epoch)")
    print(f"  비전 의존성 강화 설정:")
    print(f"    - 가중 샘플링: {args.weighted_sampling} (출발 x{args.start_weight_mult}, 파지 x{args.grasp_weight_mult})")
    print(f"    - 관절 노이즈 (QPOS Noise Std): {args.qpos_noise_std} rad")
    print(f"    - 관절 드롭아웃 (QPOS Dropout Prob): {args.qpos_dropout_prob * 100:.0f}% (qpos 마스킹)")
    print(f"    - 배경 패치 드롭아웃 (Red-Protected Patch Dropout): {args.patch_dropout_prob * 100:.0f}%")
    print(f"    - 상자 위 그리퍼 파인튜닝: {args.fine_tune_gripper} (Gripper Loss Weight: x{args.gripper_loss_weight})")
    print(f"    - 가중치 이어받기 (Resume): {args.resume}")

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # 1. 데이터셋 로드
    train_loader, val_loader, norm_stats = load_dual_eyes_dataloaders(
        data_dir=args.data_dir,
        camera_names=args.camera_names,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
        image_size=args.image_size,
        steps_per_episode=args.steps_per_episode,
        weighted_sampling=args.weighted_sampling,
        start_weight_mult=args.start_weight_mult,
        grasp_weight_mult=args.grasp_weight_mult,
        fine_tune_gripper=args.fine_tune_gripper,
        qpos_noise_std=args.qpos_noise_std,
        qpos_dropout_prob=args.qpos_dropout_prob,
        patch_dropout_prob=args.patch_dropout_prob,
    )

    # 정규화 통계치 저장
    stats_path = os.path.join(args.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(norm_stats, f)
    print(f"[Stats] 정규화 통계치 저장 완료: {stats_path}")

    # 2. 모델 인스턴스 생성
    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=args.chunk_size,
        camera_names=args.camera_names,
        backbone_name=args.backbone,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
        train_backbone=args.train_backbone,
        unfreeze_blocks=args.unfreeze_blocks,
    ).to(device)

    # 기존 가중치 이어받기 (Resume / Fine-tuning)
    if args.resume:
        resume_path = os.path.join(args.ckpt_dir, "policy_best.pth")
        if os.path.exists(resume_path):
            try:
                ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            except TypeError:
                ckpt = torch.load(resume_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            print(f"🔄 [RESUME] 기존 최고 가중치 로드 완료: {resume_path}")
        else:
            print(f"⚠️ [WARN] Resume할 가중치를 찾을 수 없어 새로 시작합니다: {resume_path}")

    # 파라미터 그룹 분리 (백본 미세조정 여부에 따른 LR 분리)
    param_dicts = [
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    history = []

    # 기존 학습 로그·그래프 덮어쓰기 방지 (train_log.csv / training_curves.png 버전 자동 증가)
    log_csv_path = os.path.join(args.ckpt_dir, "train_log.csv")
    curves_png_path = os.path.join(args.ckpt_dir, "training_curves.png")
    if os.path.exists(log_csv_path) or os.path.exists(curves_png_path):
        ver = 2
        while (
            os.path.exists(os.path.join(args.ckpt_dir, f"train_log_{ver}.csv"))
            or os.path.exists(os.path.join(args.ckpt_dir, f"training_curves_{ver}.png"))
        ):
            ver += 1
        log_csv_path = os.path.join(args.ckpt_dir, f"train_log_{ver}.csv")
        curves_png_path = os.path.join(args.ckpt_dir, f"training_curves_{ver}.png")
    print(f"📝 [Log] 학습 로그 저장 경로: {log_csv_path}")

    with open(log_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch", "train_loss", "train_l1", "train_kl",
            "val_loss", "val_l1", "val_kl", "val_aux", "val_vis_acc",
            "val_search", "val_grasp", "val_transport", "val_place",
            "val_pan", "val_lift", "val_elbow", "val_wrist_flex", "val_wrist_roll", "val_gripper",
            "time_s"
        ])

    start_time = time.time()

    # 3. 학습 루프
    for epoch in range(1, args.num_epochs + 1):
        epoch_start = time.time()
        model.train()
        train_losses = []
        train_l1s = []
        train_kls = []

        for data in train_loader:
            optimizer.zero_grad()
            loss_dict = compute_loss(
                model, data, args.kl_weight, device,
                gripper_loss_weight=args.gripper_loss_weight,
                aux_loss_weight=args.aux_loss_weight,
            )
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss_dict["loss"].item())
            train_l1s.append(loss_dict["l1"].item())
            train_kls.append(loss_dict["kl"].item())

        # 검증 루프 (단계별/관절별 세부 통계 동시 집계)
        model.eval()
        val_losses = []
        val_l1s = []
        val_kls = []
        val_auxs = []
        val_vis_accs = []
        val_joint_l1_list = []
        val_phase_dict = {"search": [], "grasp": [], "transport": [], "place": []}

        with torch.no_grad():
            for data in val_loader:
                loss_dict = compute_loss(
                    model, data, args.kl_weight, device,
                    gripper_loss_weight=args.gripper_loss_weight,
                    aux_loss_weight=args.aux_loss_weight,
                )
                val_losses.append(loss_dict["loss"].item())
                val_l1s.append(loss_dict["l1"].item())
                val_kls.append(loss_dict["kl"].item())
                val_auxs.append(loss_dict["aux"].item())
                if loss_dict["vis_acc"] is not None:
                    val_vis_accs.append(loss_dict["vis_acc"])
                val_joint_l1_list.append(loss_dict["joint_l1"])
                for p_name, p_val in loss_dict["phase_l1"].items():
                    if p_val is not None:
                        val_phase_dict[p_name].append(p_val)

        t_loss = float(np.mean(train_losses))
        t_l1 = float(np.mean(train_l1s))
        t_kl = float(np.mean(train_kls))
        v_loss = float(np.mean(val_losses))
        v_l1 = float(np.mean(val_l1s))
        v_kl = float(np.mean(val_kls))
        v_aux = float(np.mean(val_auxs)) if val_auxs else 0.0
        v_vis_acc = float(np.mean(val_vis_accs)) if val_vis_accs else float("nan")

        # 단계별 평균 (샘플 없는 단계는 NaN)
        v_search = float(np.mean(val_phase_dict["search"])) if val_phase_dict["search"] else float("nan")
        v_grasp = float(np.mean(val_phase_dict["grasp"])) if val_phase_dict["grasp"] else float("nan")
        v_transport = float(np.mean(val_phase_dict["transport"])) if val_phase_dict["transport"] else float("nan")
        v_place = float(np.mean(val_phase_dict["place"])) if val_phase_dict["place"] else float("nan")

        # 관절별 평균 (6,)
        v_joints = np.mean(val_joint_l1_list, axis=0)
        v_pan, v_lift, v_elbow, v_wflex, v_wroll, v_grip = [float(x) for x in v_joints]

        epoch_elapsed = time.time() - epoch_start

        epoch_record = {
            "epoch": epoch,
            "train_loss": t_loss,
            "train_l1": t_l1,
            "train_kl": t_kl,
            "val_loss": v_loss,
            "val_l1": v_l1,
            "val_kl": v_kl,
            "val_search": v_search,
            "val_grasp": v_grasp,
            "val_transport": v_transport,
            "val_place": v_place,
            "val_pan": v_pan,
            "val_lift": v_lift,
            "val_elbow": v_elbow,
            "val_wrist_flex": v_wflex,
            "val_wrist_roll": v_wroll,
            "val_gripper": v_grip,
        }
        history.append(epoch_record)

        with open(log_csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, t_loss, t_l1, t_kl, v_loss, v_l1, v_kl, v_aux, v_vis_acc,
                v_search, v_grasp, v_transport, v_place,
                v_pan, v_lift, v_elbow, v_wflex, v_wroll, v_grip,
                f"{epoch_elapsed:.2f}"
            ])

        if epoch % 10 == 0 or epoch == 1:
            print(f"[Epoch {epoch:4d}/{args.num_epochs}] "
                  f"Train: {t_loss:.4f} (L1: {t_l1:.4f}, KL: {t_kl:.4f}) | "
                  f"Val: {v_loss:.4f} (L1: {v_l1:.4f}, Aux: {v_aux:.4f}, VisAcc: {v_vis_acc:.1f}%) | "
                  f"{epoch_elapsed:.1f}s")
            print(f"  ├─ [작업 단계별 Val L1] Search: {v_search:.4f} | Grasp: {v_grasp:.4f} | Transport: {v_transport:.4f} | Place: {v_place:.4f}")
            print(f"  └─ [관절별 Val L1]      Pan: {v_pan:.4f} | Lift: {v_lift:.4f} | Elb: {v_elbow:.4f} | W_Flex: {v_wflex:.4f} | W_Roll: {v_wroll:.4f} | Grip: {v_grip:.4f}")

        ckpt_metrics = {
            "val_phase_l1": {"search": v_search, "grasp": v_grasp, "transport": v_transport, "place": v_place},
            "val_joint_l1": {"pan": v_pan, "lift": v_lift, "elbow": v_elbow, "wrist_flex": v_wflex, "wrist_roll": v_wroll, "gripper": v_grip},
            "val_vis_acc": v_vis_acc,
        }

        # 최고 성능 모델 체크포인트 저장
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            save_best_path = os.path.join(args.ckpt_dir, args.save_name)
            torch.save(
                {"epoch": epoch, "backbone_name": args.backbone, "model_state_dict": model.state_dict(), "val_loss": v_loss, "metrics": ckpt_metrics},
                save_best_path,
            )

        # 주기별 체크포인트 및 시각화 저장
        if epoch % args.save_every == 0 or epoch == args.num_epochs:
            prefix = os.path.splitext(args.save_name)[0]
            torch.save(
                {"epoch": epoch, "backbone_name": args.backbone, "model_state_dict": model.state_dict(), "val_loss": v_loss, "metrics": ckpt_metrics},
                os.path.join(args.ckpt_dir, f"{prefix}_epoch_{epoch}.pth"),
            )
            plot_training_curves(history, curves_png_path)

    total_time = time.time() - start_time
    print(f"=== 학습 완료 (총 소요 시간: {total_time/60:.1f}분) ===")
    print(f"최고 Val Loss: {best_val_loss:.4f} -> 저장 위치: {os.path.join(args.ckpt_dir, args.save_name)}")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="양안 DINOv2 ACT 모델 학습")
    parser.add_argument("--data_dir", type=str, default=DATA_DIR)
    parser.add_argument("--ckpt_dir", type=str, default=CKPT_DIR)
    parser.add_argument("--backbone", type=str, default=BACKBONE)
    parser.add_argument("--train_backbone", action="store_true", default=TRAIN_BACKBONE)
    parser.add_argument("--unfreeze_blocks", type=int, default=UNFREEZE_BACKBONE_BLOCKS, help="DINOv2 마지막 N개 블록만 부분 미세조정")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr_backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--kl_weight", type=float, default=KL_WEIGHT)
    parser.add_argument("--chunk_size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--save_every", type=int, default=SAVE_EVERY)
    parser.add_argument("--save_name", type=str, default="policy_best.pth", help="저장할 최고 성능 체크포인트 파일명 (기본: policy_best.pth)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--camera_names", nargs="+", default=CAMERA_NAMES)
    parser.add_argument("--image_size", nargs=2, type=int, default=IMAGE_SIZE)
    parser.add_argument("--steps_per_episode", type=int, default=STEPS_PER_EPISODE_PER_EPOCH)
    parser.add_argument("--qpos_noise_std", type=float, default=QPOS_NOISE_STD)
    parser.add_argument("--qpos_dropout_prob", type=float, default=QPOS_DROPOUT_PROB)
    parser.add_argument("--patch_dropout_prob", type=float, default=PATCH_DROPOUT_PROB, help="빨간 물체 보호 후 배경 패치 드롭아웃 비율 (기법 C)")
    parser.add_argument("--weighted_sampling", type=str2bool, default=WEIGHTED_SAMPLING, help="가중 샘플링 활성화 여부 (True/False)")
    parser.add_argument("--no_weighted_sampling", dest="weighted_sampling", action="store_false", help="가중 샘플링 끄기 (균일 샘플링 강제)")
    parser.add_argument("--start_weight_mult", type=float, default=START_WEIGHT_MULTIPLIER)
    parser.add_argument("--grasp_weight_mult", type=float, default=GRASP_WEIGHT_MULTIPLIER)
    parser.add_argument("--resume", action="store_true", default=False, help="기존 최고 가중치(policy_best.pth) 이어받아 학습")
    parser.add_argument("--fine_tune_gripper", action="store_true", default=False, help="상자 위 그리퍼 개방 집중 파인튜닝 모드")
    parser.add_argument("--gripper_loss_weight", type=float, default=1.0, help="그리퍼 관절 Loss 가중치 배수 (기본 1.0, 파인튜닝 시 3.0 권장)")
    parser.add_argument("--aux_loss_weight", type=float, default=0.2, help="15x20 큐브 히트맵 및 보임 확률 보조 손실 가중치 (기본 0.2)")

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    train_dual_eyes_act(args)

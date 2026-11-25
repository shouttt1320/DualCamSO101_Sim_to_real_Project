#!/usr/bin/env python3
"""
[Stage 0 Diagnosis]
1) Frozen DINOv2 (vits14 vs vits14_reg) Linear Probe: Sim Train -> Sim Val & Real Test
   - Compares Global Average Pooling vs Patch-wise 15x20 Heatmap + Max Pooling
   - Saves visual contact sheets of exact Sim & Real images used (with GT label & pred prob)
2) Real Camera & ACT Inference Loop Timing Benchmark (p50 / p95 ms, actual camera FPS)
"""

import glob
import os
import pickle
import sys
import time
import cv2
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project")

for p in [SCRIPT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from so101_dual_dinov2_act.config import (
    BACKBONE, CAMERA_NAMES, CHUNK_SIZE, DEC_LAYERS, DIM_FEEDFORWARD,
    ENC_LAYERS, HIDDEN_DIM, IMAGE_SIZE, LATENT_DIM, NHEADS, STATE_DIM,
)
from so101_dual_dinov2_act.models import build_act_model

OUT_DIR = os.path.join(PROJECT_ROOT, "data", "so101_active_search", "stage0_diagnosis")
os.makedirs(OUT_DIR, exist_ok=True)


def count_red_pixels(rgb_u8: np.ndarray) -> int:
    hsv = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 75, 50]), np.array([12, 255, 255]))
    m2 = cv2.inRange(hsv, np.array([165, 75, 50]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(m1, m2)
    # Ignore bottom 15% gripper/mount area if needed? Keep full frame to match policy
    return int(np.count_nonzero(mask))


def collect_frames_from_hdf5(ep_paths, max_pos_per_ep=3, max_neg_per_ep=3, pos_thr=120, neg_thr=15, stride=15):
    samples = []  # (rgb_u8, label, meta_str)
    for ep_path in ep_paths:
        ep_name = os.path.basename(ep_path).replace(".hdf5", "")
        with h5py.File(ep_path, "r") as f:
            n_steps = f["action"].shape[0]
            pos_cnt, neg_cnt = 0, 0
            # Sample across early/mid steps (up to grasp/approach)
            idxs = list(range(0, min(n_steps, 420), stride))
            for t in idxs:
                if pos_cnt >= max_pos_per_ep and neg_cnt >= max_neg_per_ep:
                    break
                for cam in ("cam_left", "cam_right"):
                    if pos_cnt >= max_pos_per_ep and neg_cnt >= max_neg_per_ep:
                        break
                    img = f[f"observations/images/{cam}"][t]
                    rpx = count_red_pixels(img)
                    if rpx >= pos_thr and pos_cnt < max_pos_per_ep:
                        samples.append((img.copy(), 1, f"{ep_name} t={t} {cam[4]} px={rpx}"))
                        pos_cnt += 1
                    elif rpx <= neg_thr and neg_cnt < max_neg_per_ep:
                        samples.append((img.copy(), 0, f"{ep_name} t={t} {cam[4]} px={rpx}"))
                        neg_cnt += 1
    return samples


def save_montage(samples, probs, title, out_path, max_n=24):
    # Pick balanced subset (up to max_n//2 pos, max_n//2 neg)
    pos_i = [i for i, s in enumerate(samples) if s[1] == 1]
    neg_i = [i for i, s in enumerate(samples) if s[1] == 0]
    half = max_n // 2
    sel = pos_i[:half] + neg_i[:half]
    if not sel:
        return
    cols = 6
    rows = int(np.ceil(len(sel) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.8, rows * 2.4))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for k, idx in enumerate(sel):
        img, lbl, meta = samples[idx]
        p = probs[idx] if probs is not None else -1.0
        pred = 1 if p >= 0.5 else 0
        ok = (pred == lbl)
        color = "green" if ok else "red"
        axes[k].imshow(img)
        axes[k].set_title(f"GT={lbl} P={p:.2f} {'OK' if ok else 'ERR'}\n{meta}", fontsize=7.5, color=color)
    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close(fig)


@torch.no_grad()
def extract_patch_features(backbone, samples, tf, device, batch_size=32):
    feats = []
    labels = []
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        imgs = torch.stack([tf(torch.from_numpy(s[0]).permute(2, 0, 1).float() / 255.0) for s in batch]).to(device)
        out = backbone.forward_features(imgs)["x_norm_patchtokens"]  # (B, 300, 384)
        feats.append(out.cpu())
        labels.extend([s[1] for s in batch])
    return torch.cat(feats, dim=0), torch.tensor(labels, dtype=torch.float32)


def train_and_eval_probe(f_tr, y_tr, f_va, y_va, f_re, y_re, mode="max_patch", epochs=400, lr=5e-2, wd=1e-2):
    # f_*: (N, P, D)
    D = f_tr.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_tr = f_tr.to(device)
    Y_tr = y_tr.to(device)
    X_va = f_va.to(device)
    X_re = f_re.to(device)

    if mode == "avg":
        X_tr = X_tr.mean(dim=1)
        X_va = X_va.mean(dim=1)
        X_re = X_re.mean(dim=1)

    head = nn.Linear(D, 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    pos_cnt = float((Y_tr == 1).sum().item())
    neg_cnt = float((Y_tr == 0).sum().item())
    pos_w = torch.tensor([neg_cnt / max(pos_cnt, 1.0)], device=device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)

    def forward_logits(X):
        if mode == "avg":
            return head(X).squeeze(-1)
        else:
            # Patch-wise linear -> spatial max pooling over 15x20 patches
            return head(X).squeeze(-1).max(dim=1).values

    for _ in range(epochs):
        opt.zero_grad()
        loss = crit(forward_logits(X_tr), Y_tr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        p_va = torch.sigmoid(forward_logits(X_va)).cpu().numpy()
        p_re = torch.sigmoid(forward_logits(X_re)).cpu().numpy()

    def metrics(probs, y_true):
        yt = y_true.numpy()
        pr = (probs >= 0.5).astype(np.float32)
        acc = float((pr == yt).mean()) * 100.0
        pos_m = yt == 1
        neg_m = yt == 0
        tpr = float((pr[pos_m] == 1).mean()) * 100.0 if pos_m.any() else 0.0
        tnr = float((pr[neg_m] == 0).mean()) * 100.0 if neg_m.any() else 0.0
        return acc, tpr, tnr

    return metrics(p_va, y_va), metrics(p_re, y_re), p_va, p_re


def run_timing_benchmark(device, tf):
    print("\n" + "=" * 76)
    print(">>> [Stage 0 - Part 2] Real Camera & ACT Inference Loop Timing (p50 / p95)")
    print("=" * 76)
    ckpt_dir = os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", "real_wall_twin")
    ck = torch.load(os.path.join(ckpt_dir, "policy_best.pth"), map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    ql = sd["query_embed.weight"].shape[0] if "query_embed.weight" in sd else CHUNK_SIZE
    model = build_act_model(
        state_dim=STATE_DIM, num_queries=ql, camera_names=CAMERA_NAMES,
        backbone_name=BACKBONE, hidden_dim=HIDDEN_DIM, dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, nheads=NHEADS, latent_dim=LATENT_DIM,
        train_backbone=False,
    ).to(device).eval()
    model.load_state_dict(sd)

    # Warmup
    with torch.inference_mode():
        dq = torch.zeros(1, STATE_DIM, device=device)
        di = torch.zeros(1, 2, 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=device)
        for _ in range(5):
            model(qpos=dq, image=di)
        torch.cuda.synchronize()

    # Check real camera capture
    from hardware.camera_interface import MultiCameraCapture
    cap = None
    cam_ok = False
    try:
        cap = MultiCameraCapture(
            camera_dict={"cam_left": 2, "cam_right": 0},
            width=640, height=480, fps=30,
            disable_autofocus=True, focus_dict={"cam_left": 25, "cam_right": 25},
        )
        cap.start()
        time.sleep(0.2)
        cam_ok = True
    except Exception as e:
        print(f"  [Warn] Camera open failed ({e}); benchmarking with synthetic frames.")

    t_cam, t_prep, t_inf, t_tot = [], [], [], []
    # Also measure actual new-frame interval on /dev/video2
    unique_frames = 0
    prev_hash = None
    t_fps_start = time.perf_counter()

    for _ in range(60):
        t0 = time.perf_counter()
        if cam_ok:
            fr = cap.read_all()
            bl, br = fr.get("cam_left"), fr.get("cam_right")
            if bl is None:
                bl = np.zeros((480, 640, 3), dtype=np.uint8)
            if br is None:
                br = np.zeros((480, 640, 3), dtype=np.uint8)
            hval = int(bl[240, 320, 0]) + int(bl[100, 100, 1]) * 256
            if hval != prev_hash:
                unique_frames += 1
                prev_hash = hval
        else:
            bl = np.zeros((480, 640, 3), dtype=np.uint8)
            br = np.zeros((480, 640, 3), dtype=np.uint8)
        t1 = time.perf_counter()

        rl = cv2.cvtColor(bl, cv2.COLOR_BGR2RGB)
        rr = cv2.cvtColor(br, cv2.COLOR_BGR2RGB)
        im = torch.stack([
            tf(torch.from_numpy(rl).permute(2, 0, 1).float() / 255.0),
            tf(torch.from_numpy(rr).permute(2, 0, 1).float() / 255.0),
        ], dim=0).unsqueeze(0).to(device)
        t2 = time.perf_counter()

        with torch.inference_mode():
            model(qpos=dq, image=im)
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        t_cam.append((t1 - t0) * 1000.0)
        t_prep.append((t2 - t1) * 1000.0)
        t_inf.append((t3 - t2) * 1000.0)
        t_tot.append((t3 - t0) * 1000.0)

    fps_elapsed = time.perf_counter() - t_fps_start
    if cap is not None:
        cap.stop()

    def stat(arr):
        a = np.array(arr)
        return np.median(a), np.percentile(a, 95)

    for name, arr in [("cam.read_all", t_cam), ("preprocess", t_prep), ("GPU infer", t_inf), ("Total (no motor/GUI)", t_tot)]:
        p50, p95 = stat(arr)
        print(f"  {name:<22}: p50 = {p50:6.2f} ms | p95 = {p95:6.2f} ms")
    mean_tot = float(np.mean(t_tot))
    print(f"  Projected Max Hz (no motor/GUI): {1000.0 / mean_tot:.1f} Hz")
    if cam_ok:
        print(f"  Observed unique camera frames  : {unique_frames} in {fps_elapsed:.2f}s (~{unique_frames/max(fps_elapsed,1e-3):.1f} FPS)")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    sim_eps = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "so101_active_search", "real_wall_twin", "episode_*.hdf5")))
    real_eps = sorted(glob.glob(os.path.join(PROJECT_ROOT, "data", "so101_active_search", "real_robot", "episode_*.hdf5")))

    print(f"[Data] Sim episodes: {len(sim_eps)}, Real episodes: {len(real_eps)}")
    sim_tr = collect_frames_from_hdf5(sim_eps[:55], max_pos_per_ep=3, max_neg_per_ep=3, pos_thr=120, neg_thr=10, stride=20)
    sim_va = collect_frames_from_hdf5(sim_eps[55:], max_pos_per_ep=3, max_neg_per_ep=3, pos_thr=120, neg_thr=10, stride=20)
    real_te = collect_frames_from_hdf5(real_eps, max_pos_per_ep=15, max_neg_per_ep=15, pos_thr=150, neg_thr=20, stride=10)

    # Also add the 2 user-uploaded empty-table real screenshots (cropped into left/right views)
    up_dir = "/home/jw/.gemini/antigravity/brain/3dd3fd7f-fd74-4c14-9f8a-521095585341/.user_uploaded"
    for fname, tag in [
        ("media_1795389015431_840d8ce9.jpg", "real_empty_close"),
        ("media_1795389015433_7b7599f6.jpg", "real_empty_edge"),
    ]:
        fpath = os.path.join(up_dir, fname)
        if os.path.exists(fpath):
            bgr = cv2.imread(fpath)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                top_h = int(h * 0.78)  # exclude bottom HUD bar
                half_w = w // 2
                left_c = cv2.resize(rgb[:top_h, :half_w], (640, 480))
                right_c = cv2.resize(rgb[:top_h, half_w:], (640, 480))
                real_te.append((left_c, 0, f"{tag}_L"))
                real_te.append((right_c, 0, f"{tag}_R"))

    def cnt(s):
        return sum(1 for x in s if x[1] == 1), sum(1 for x in s if x[1] == 0)

    print(f"  Sim Train: {len(sim_tr)} (pos={cnt(sim_tr)[0]}, neg={cnt(sim_tr)[1]})")
    print(f"  Sim Val  : {len(sim_va)} (pos={cnt(sim_va)[0]}, neg={cnt(sim_va)[1]})")
    print(f"  Real Test: {len(real_te)} (pos={cnt(real_te)[0]}, neg={cnt(real_te)[1]})")

    print("\n" + "=" * 76)
    print(">>> [Stage 0 - Part 1] Frozen DINOv2 Linear Probe (Sim Train -> Sim Val & Real Test)")
    print("=" * 76)
    print(f"{'Backbone':<18} | {'Head':<9} | {'Sim Val Acc (TPR/TNR)':<24} | {'Real Test Acc (TPR/TNR)':<24}")
    print("-" * 86)

    best_real_probs = None
    best_sim_probs = None
    best_label = ""

    for bname in ("dinov2_vits14", "dinov2_vits14_reg"):
        bb = torch.hub.load("facebookresearch/dinov2", bname, trust_repo=True).to(device).eval()
        f_tr, y_tr = extract_patch_features(bb, sim_tr, tf, device)
        f_va, y_va = extract_patch_features(bb, sim_va, tf, device)
        f_re, y_re = extract_patch_features(bb, real_te, tf, device)

        for mode in ("avg", "max_patch"):
            (va_acc, va_tpr, va_tnr), (re_acc, re_tpr, re_tnr), p_va, p_re = train_and_eval_probe(
                f_tr, y_tr, f_va, y_va, f_re, y_re, mode=mode
            )
            print(f"{bname:<18} | {mode:<9} | {va_acc:5.1f}% ({va_tpr:5.1f}/{va_tnr:5.1f})   | {re_acc:5.1f}% ({re_tpr:5.1f}/{re_tnr:5.1f})")
            if bname == "dinov2_vits14_reg" and mode == "max_patch":
                best_sim_probs = p_va
                best_real_probs = p_re
                best_label = f"{bname} ({mode})"
        del bb
        torch.cuda.empty_cache()

    # Save visual montages of exact Sim Val & Real Test images used!
    art_dir = "/home/jw/.gemini/antigravity/brain/3dd3fd7f-fd74-4c14-9f8a-521095585341"
    for d in (OUT_DIR, art_dir):
        save_montage(sim_va, best_sim_probs, f"Stage 0 Sim Val Samples - {best_label}", os.path.join(d, "stage0_sim_val_samples.png"), max_n=24)
        save_montage(real_te, best_real_probs, f"Stage 0 Real Test Samples - {best_label}", os.path.join(d, "stage0_real_test_samples.png"), max_n=24)
    print(f"\n[Saved Sample Montages] {OUT_DIR}/stage0_sim_val_samples.png & stage0_real_test_samples.png")

    run_timing_benchmark(device, tf)


if __name__ == "__main__":
    main()


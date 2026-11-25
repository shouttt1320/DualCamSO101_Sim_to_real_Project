#!/usr/bin/env python3
"""
[Active Search] SO-101 능동 탐색 & 도메인 랜덤화 데이터셋 종합 검수 스크립트
=============================================================================
수집된 HDF5 데이터셋에 대해 다음 항목을 자동 점검하고 시각화 리포트를 생성합니다:
  1. 파일 무결성 & 결측치(NaN) & 흑백/과노출 프레임 검사
  2. 큐브 위치(X, Y) 및 각도(Theta), 거리(R) 분포 편향 검사 (좌/우 균형, Y < -0.20m 공백 확인)
  3. 능동 탐색 발동 비율, 스테레오 오차(mm), 에피소드 길이 분포
  4. 그리퍼 파지·개방 물리 사이클 검증 (닫힘 < 25° -> 말단 개방 > 45°)
  5. 도메인 랜덤화(바닥·큐브·조명·파란 팔·방해물) 다양성 썸네일 몽타주 생성

[실행 예시]
  python3 scripts/active_search/inspect_search_dataset.py --env domain_rand
  python3 scripts/active_search/inspect_search_dataset.py --env default_clean
  python3 scripts/active_search/inspect_search_dataset.py --dir /path/to/hdf5_dir
"""

import os
import glob
import argparse
import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))


def inspect_dataset(data_dir: str, out_dir: str, max_montage: int = 20):
    hdf5_files = sorted(glob.glob(os.path.join(data_dir, "episode_*.hdf5")))
    if not hdf5_files:
        print(f"❌ [ERROR] No episode_*.hdf5 files found in: {data_dir}")
        return

    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "=" * 80)
    print(f">>> [INSPECT SEARCH DATASET] {data_dir} ({len(hdf5_files)} episodes)")
    print("=" * 80)

    records = []
    warnings = []

    for fpath in hdf5_files:
        fname = os.path.basename(fpath)
        try:
            with h5py.File(fpath, "r") as root:
                required = [
                    "action",
                    "observations/qpos",
                    "observations/qvel",
                    "observations/images/cam_left",
                    "observations/images/cam_right",
                ]
                missing = [k for k in required if k not in root]
                if missing:
                    warnings.append(f"{fname}: missing keys {missing}")
                    continue

                qpos = root["observations/qpos"][()]
                act = root["action"][()]
                qvel = root["observations/qvel"][()]
                img_l = root["observations/images/cam_left"]
                T = len(qpos)

                if np.isnan(qpos).any() or np.isnan(act).any() or np.isnan(qvel).any():
                    warnings.append(f"{fname}: NaN detected in qpos/action/qvel")

                # 메타데이터
                spawn_xy = root.attrs.get("cube_spawn_xy", [np.nan, np.nan])
                x_m, y_m = float(spawn_xy[0]), float(spawn_xy[1])
                r_m = float(np.hypot(x_m, y_m))
                theta_deg = float(np.degrees(np.arctan2(y_m, x_m))) if np.isfinite(x_m) else np.nan
                search_trig = bool(root.attrs.get("search_triggered", False))
                stereo_err = float(root.attrs.get("stereo_error_mm", np.nan))

                # 관측 프레임 (t=25 부근이 OBSERVATION_POSE로 테이블/큐브/방해물이 잘 보임)
                obs_idx = min(25, T - 1)
                obs_frame = img_l[obs_idx]
                bright_mean = float(np.mean(obs_frame))
                if bright_mean < 15.0 or bright_mean > 245.0:
                    warnings.append(f"{fname}: abnormal brightness ({bright_mean:.1f})")

                # 그리퍼 물리 사이클 검사 (deg): 후반부(투하 구간)에 실제로 열렸는지 확인
                grip_deg = np.degrees(qpos[:, 5])
                min_grip = float(np.min(grip_deg))
                max_grip_late = float(np.max(grip_deg[len(grip_deg) // 2 :]))
                if min_grip >= 25.0:
                    warnings.append(f"{fname}: gripper never closed firmly (min={min_grip:.1f}°)")
                if max_grip_late <= 45.0:
                    warnings.append(f"{fname}: gripper never opened in place phase (max_late={max_grip_late:.1f}°)")

                if np.isfinite(stereo_err) and stereo_err > 20.0:
                    warnings.append(f"{fname}: high stereo error ({stereo_err:.1f}mm)")

                records.append(
                    {
                        "fname": fname,
                        "fpath": fpath,
                        "steps": T,
                        "x": x_m,
                        "y": y_m,
                        "r": r_m,
                        "theta": theta_deg,
                        "search": search_trig,
                        "stereo_err": stereo_err,
                        "bright_mean": bright_mean,
                        "obs_idx": obs_idx,
                    }
                )
        except Exception as e:
            warnings.append(f"{fname}: failed to read ({e})")

    if not records:
        print("❌ [ERROR] No valid episodes could be parsed.")
        return

    xs = np.array([r["x"] for r in records])
    ys = np.array([r["y"] for r in records])
    rs = np.array([r["r"] for r in records])
    thetas = np.array([r["theta"] for r in records])
    steps = np.array([r["steps"] for r in records])
    searches = np.array([r["search"] for r in records])
    errs = np.array([r["stereo_err"] for r in records])
    brights = np.array([r["bright_mean"] for r in records])

    # 1. 분포 편향 리포트
    valid_xy = np.isfinite(thetas)
    n_valid = int(np.sum(valid_xy))
    n_left = int(np.sum(thetas[valid_xy] < 0))
    n_right = int(np.sum(thetas[valid_xy] >= 0))
    n_y_neg20 = int(np.sum(ys[valid_xy] < -0.20))
    n_y_pos20 = int(np.sum(ys[valid_xy] > 0.20))

    print(f"  • Valid Episodes:    {len(records)} / {len(hdf5_files)}")
    print(f"  • Episode Steps:     min={steps.min()}, mean={steps.mean():.1f}, max={steps.max()}")
    print(f"  • Frame Brightness:  min={brights.min():.1f}, mean={brights.mean():.1f}, max={brights.max():.1f}")
    print(f"  • Active Search:     {int(searches.sum())} / {len(records)} ({searches.mean()*100:.1f}%)")
    if np.isfinite(errs).any():
        print(f"  • Stereo Error (mm): mean={np.nanmean(errs):.2f}mm, max={np.nanmax(errs):.2f}mm")

    if n_valid > 0:
        print("\n[Spatial & Angular Distribution]")
        print(f"  • Left (Theta < 0°): {n_left} ({n_left/n_valid*100:.1f}%) | Right (Theta >= 0°): {n_right} ({n_right/n_valid*100:.1f}%)")
        print(f"  • Extreme Edges:     Y < -0.20m: {n_y_neg20}개 | Y > +0.20m: {n_y_pos20}개")
        print(f"  • Radius Range (m):  [{rs[valid_xy].min():.3f}, {rs[valid_xy].max():.3f}] (mean={rs[valid_xy].mean():.3f})")
        print(f"  • Angle Range (deg): [{thetas[valid_xy].min():+.1f}°, {thetas[valid_xy].max():+.1f}°]")

        bins = [(-60, -35), (-35, -15), (-15, 15), (15, 35), (35, 60)]
        bin_strs = []
        for lo, hi in bins:
            cnt = int(np.sum((thetas[valid_xy] >= lo) & (thetas[valid_xy] < hi)))
            bin_strs.append(f"[{lo:+d}°,{hi:+d}°): {cnt}")
        print("  • Angle Bins:        " + " | ".join(bin_strs))

    print(f"\n[Warnings ({len(warnings)})]")
    if warnings:
        for w in warnings[:20]:
            print(f"  ⚠️  {w}")
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more warnings")
    else:
        print("  ✅ No integrity, brightness, or gripper warnings found.")

    # 2. 분포 시각화 PNG (2x2)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"Dataset Distribution & Quality: {os.path.basename(data_dir)} (N={len(records)})", fontsize=14, fontweight="bold")

    # (1) 2D Workspace Scatter
    ax = axes[0, 0]
    if n_valid > 0:
        ax.scatter(xs[~searches & valid_xy], ys[~searches & valid_xy], c="tab:blue", label="Direct Pick", alpha=0.8, s=45)
        ax.scatter(xs[searches & valid_xy], ys[searches & valid_xy], c="tab:orange", label="Active Search", alpha=0.8, s=45)
        ax.axhline(0.20, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.axhline(-0.20, color="red", linestyle="--", linewidth=0.9, alpha=0.7, label="Y = -0.20m threshold")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_title("Cube Spawn Workspace (X vs Y)")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m] (+Left / -Right in World)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # (2) Angle Histogram
    ax = axes[0, 1]
    if n_valid > 0:
        ax.hist(thetas[valid_xy], bins=np.linspace(-60, 60, 13), color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(0.0, color="red", linestyle="--", linewidth=1.2)
    ax.set_title(f"Angle Distribution (Left={n_left}, Right={n_right})")
    ax.set_xlabel("Theta [deg]")
    ax.set_ylabel("Episode Count")
    ax.grid(True, alpha=0.3)

    # (3) Radius Histogram
    ax = axes[1, 0]
    if n_valid > 0:
        ax.hist(rs[valid_xy], bins=10, color="seagreen", edgecolor="black", alpha=0.85)
    ax.set_title("Radius Distribution (R [m])")
    ax.set_xlabel("R [m]")
    ax.set_ylabel("Episode Count")
    ax.grid(True, alpha=0.3)

    # (4) Episode Steps vs Stereo Error
    ax = axes[1, 1]
    if np.isfinite(errs).any():
        ax.scatter(steps, errs, c="purple", alpha=0.75, s=40)
        ax.axhline(10.0, color="orange", linestyle="--", linewidth=0.9, label="10mm")
        ax.set_ylabel("Stereo Error [mm]")
        ax.legend(fontsize=8)
    else:
        ax.hist(steps, bins=12, color="purple", edgecolor="black", alpha=0.8)
        ax.set_ylabel("Count")
    ax.set_title("Episode Length vs Stereo Error")
    ax.set_xlabel("Total Timesteps (50Hz)")
    ax.grid(True, alpha=0.3)

    def _next_unique_path(path):
        if not os.path.exists(path):
            return path
        stem, ext = os.path.splitext(path)
        ver = 2
        while os.path.exists(f"{stem}_{ver}{ext}"):
            ver += 1
        return f"{stem}_{ver}{ext}"

    plt.tight_layout()
    dist_path = _next_unique_path(os.path.join(out_dir, "dataset_distribution.png"))
    plt.savefig(dist_path, dpi=140)
    plt.close(fig)

    # 3. 도메인 랜덤화 썸네일 몽타주
    sample_indices = np.linspace(0, len(records) - 1, min(max_montage, len(records)), dtype=int)
    n_show = len(sample_indices)
    cols = 4
    rows = int(np.ceil(n_show / cols))
    fig_m, axes_m = plt.subplots(rows, cols, figsize=(16, 2.6 * rows))
    axes_m = np.atleast_2d(axes_m)
    fig_m.suptitle(f"Domain Randomization Montage (cam_left @ obs frame, {n_show} samples)", fontsize=14, fontweight="bold")

    for plot_i, rec_idx in enumerate(sample_indices):
        r = plot_i // cols
        c = plot_i % cols
        rec = records[rec_idx]
        with h5py.File(rec["fpath"], "r") as root:
            frame = root["observations/images/cam_left"][rec["obs_idx"]]
        axes_m[r, c].imshow(frame)
        axes_m[r, c].set_title(
            f"{rec['fname'].replace('.hdf5', '')} | th={rec['theta']:+.0f}° | S={rec['search']}",
            fontsize=8.5,
        )
        axes_m[r, c].axis("off")

    for empty_i in range(n_show, rows * cols):
        axes_m[empty_i // cols, empty_i % cols].axis("off")

    plt.tight_layout()
    montage_path = _next_unique_path(os.path.join(out_dir, "domain_randomization_montage.png"))
    plt.savefig(montage_path, dpi=130)
    plt.close(fig_m)

    print("\n[Saved Visual Reports]")
    print(f"  • Distribution Graph:  {dist_path}")
    print(f"  • Randomization Grid:  {montage_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect SO-101 Active Search HDF5 Dataset")
    parser.add_argument("--env", type=str, default="domain_rand", help="Dataset environment preset name")
    parser.add_argument("--dir", type=str, default=None, help="Optional explicit HDF5 dataset directory")
    parser.add_argument("--out_dir", type=str, default=None, help="Output directory for PNG reports")
    parser.add_argument("--max_montage", type=int, default=20, help="Max episodes in visual montage")
    args = parser.parse_args()

    target_dir = args.dir if args.dir else os.path.join(WORKSPACE_DIR, "Robot_Arm_Project/data/so101_active_search", args.env)
    report_dir = args.out_dir if args.out_dir else os.path.join(target_dir, "inspect_report")
    inspect_dataset(target_dir, report_dir, max_montage=args.max_montage)

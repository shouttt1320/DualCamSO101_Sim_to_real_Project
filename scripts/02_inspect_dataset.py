#!/usr/bin/env python3
"""
[Step 02] SO-101 HDF5 Dataset Inspector & Camera/Joint Visualizer
================================================================
수집된 HDF5 데모 데이터셋(Robot_Arm_Project/data/so101_dual_eyes/episode_*.hdf5)의
무결성을 검사하고, 양안 카메라(cam_left, cam_right) 녹화 화면과 6개 모터 관절값(qpos, action)을
시각화 대시보드(PNG)로 생성하여 데이터 품질을 검증합니다.

[실행 방법]
  1) 전체 요약 검사 및 최신 에피소드 대시보드 생성:
     python3 scripts/02_inspect_dataset.py
     # 또는 Isaac Sim 번들 파이썬: ~/isaacsim/python.sh scripts/02_inspect_dataset.py

  2) 특정 에피소드(예: episode_0005.hdf5) 지정 확인:
     python3 scripts/02_inspect_dataset.py --episode 5

  3) 모든 에피소드 시각화 PNG 일괄 생성:
     python3 scripts/02_inspect_dataset.py --all --no-show

[CLI 주요 옵션]
  --dir       : HDF5 데이터셋 폴더 경로 (기본값: Robot_Arm_Project/data/so101_dual_eyes)
  --episode   : 시각화할 에피소드 인덱스 번호 (기본값: -1, 최신 에피소드)
  --all       : 모든 에피소드 대시보드 PNG 일괄 생성 (기본값: False)
  --no-show   : GUI 창을 띄우지 않고 파일만 저장 (기본값: False)
"""

import os
import sys
import glob
import argparse
import h5py
import numpy as np
import matplotlib

# Isaac Sim 번들 파이썬에는 tkinter가 없으므로 파일 저장은 항상 안정적인 Agg 백엔드 사용
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project/data/so101_dual_eyes")

JOINT_NAMES = [
    "1: shoulder_pan",
    "2: shoulder_lift",
    "3: elbow_flex",
    "4: wrist_flex",
    "5: wrist_roll",
    "6: gripper",
]


def inspect_single_hdf5(filepath: str, idx: int, total: int) -> bool:
    """단일 HDF5 파일 구조, 해상도, 결측치 및 관절 범위 검증"""
    filename = os.path.basename(filepath)
    try:
        with h5py.File(filepath, "r") as root:
            required = ["action", "observations/qpos", "observations/qvel",
                        "observations/images/cam_left", "observations/images/cam_right"]
            missing = [k for k in required if k not in root]
            if missing:
                print(f"[ERROR] {filename}: 누락된 키 발견 -> {missing}")
                return False

            act = root["action"][()]
            qpos = root["observations/qpos"][()]
            qvel = root["observations/qvel"][()]
            img_l = root["observations/images/cam_left"]
            img_r = root["observations/images/cam_right"]

            T = len(qpos)
            l_shape = img_l.shape
            r_shape = img_r.shape
            spawn_xy = root.attrs.get("cube_spawn_xy", [np.nan, np.nan])

            # 샘플 프레임 밝기 확인 (처음/중간/끝)
            sample_indices = [0, T // 2, T - 1]
            l_mean = float(np.mean([img_l[i] for i in sample_indices]))
            r_mean = float(np.mean([img_r[i] for i in sample_indices]))

            has_nan = bool(np.isnan(qpos).any() or np.isnan(act).any() or np.isnan(qvel).any())
            is_black = (l_mean < 1.0) or (r_mean < 1.0)

            q_deg_min = np.degrees(np.min(qpos, axis=0))
            q_deg_max = np.degrees(np.max(qpos, axis=0))

            status = "OK"
            if has_nan:
                status = "WARN(NaN)"
            elif is_black:
                status = "WARN(Black Frame)"

            print(
                f"[{idx+1:02d}/{total:02d}] {filename} | T={T:3d} steps | "
                f"Spawn=({spawn_xy[0]:.3f}, {spawn_xy[1]:.3f}) | "
                f"Cam={l_shape[2]}x{l_shape[1]} (L:{l_mean:.1f}, R:{r_mean:.1f}) | "
                f"Grip=[{q_deg_min[5]:.1f}°, {q_deg_max[5]:.1f}°] | [{status}]"
            )
            return status == "OK"

    except Exception as e:
        print(f"[ERROR] {filename} 읽기 실패: {e}")
        return False


def visualize_episode(filepath: str, show_window: bool = True):
    """단일 에피소드의 양안 카메라 프레임 타임라인 + 6축 모터 관절값(qpos vs action) 그래프 생성"""
    filename = os.path.basename(filepath)
    ep_stem = os.path.splitext(filename)[0]

    with h5py.File(filepath, "r") as root:
        act_rad = root["action"][()]                    # (T, 6)
        qpos_rad = root["observations/qpos"][()]        # (T, 6)
        img_l_ds = root["observations/images/cam_left"] # (T, H, W, 3)
        img_r_ds = root["observations/images/cam_right"]
        T, H, W, _ = img_l_ds.shape
        spawn_xy = root.attrs.get("cube_spawn_xy", [0.0, 0.0])

        # t=0 및 주요 진행 시점 6개 프레임 추출 (관측 -> 접근 -> 파지 -> 리프트 -> 상자 상공 -> 투하)
        img_l0 = np.array(img_l_ds[0], copy=True)
        img_r0 = np.array(img_r_ds[0], copy=True)

        # 주요 진행 시점 6개 프레임 추출 (t=0 순수 관측/홈포즈 -> 접근 -> 파지 -> 리프트 -> 상자 상공 -> 투하)
        phase_ratios = [0.0, 0.22, 0.42, 0.62, 0.82, 0.96]
        phase_labels = ["1. Observe (Home)", "2. Pre-Grasp", "3. Clamp", "4. Lift", "5. Box Hover", "6. Release"]
        frame_indices = [min(max(int(T * r), 0), T - 1) for r in phase_ratios]

        frames_l = [np.array(img_l_ds[i], copy=True) for i in frame_indices]
        frames_r = [np.array(img_r_ds[i], copy=True) for i in frame_indices]

    qpos_deg = np.degrees(qpos_rad)
    act_deg = np.degrees(act_rad)
    timesteps = np.arange(T)

    fig = plt.figure(figsize=(20, 12))
    fig.suptitle(
        f"SO-101 Dataset Inspection: {filename}  |  Steps={T}  |  Cam={W}x{H}  |  "
        f"Cube Spawn XY=({spawn_xy[0]:.3f}m, {spawn_xy[1]:.3f}m)",
        fontsize=14,
        fontweight="bold",
    )

    # 상단 2행 × 6열: 좌/우 카메라 주요 시점 화면
    gs = fig.add_gridspec(4, 6, height_ratios=[1.0, 1.0, 1.25, 1.25], hspace=0.38, wspace=0.22)

    for col, (f_idx, p_label) in enumerate(zip(frame_indices, phase_labels)):
        ax_l = fig.add_subplot(gs[0, col])
        ax_l.imshow(frames_l[col])
        ax_l.set_title(f"{p_label} (t={f_idx})\n[cam_left] Grip={qpos_deg[f_idx, 5]:.1f}°", fontsize=9.5)
        ax_l.axis("off")

        ax_r = fig.add_subplot(gs[1, col])
        ax_r.imshow(frames_r[col])
        ax_r.set_title(f"[cam_right] Pan={qpos_deg[f_idx, 0]:.1f}°, Sh={qpos_deg[f_idx, 1]:.1f}°", fontsize=9.5)
        ax_r.axis("off")

    # 하단 2행 × 3열: 6개 모터 관절 각도 (실제 qpos vs 명령 action)
    for j in range(6):
        row = 2 + (j // 3)
        col_slice = slice((j % 3) * 2, (j % 3) * 2 + 2)
        ax_j = fig.add_subplot(gs[row, col_slice])

        ax_j.plot(timesteps, qpos_deg[:, j], label="Measured qpos (deg)", color="#1f77b4", linewidth=2.0)
        ax_j.plot(timesteps, act_deg[:, j], label="Commanded action (deg)", color="#d62728", linestyle="--", linewidth=1.5)
        for f_idx in frame_indices:
            ax_j.axvline(f_idx, color="gray", linestyle=":", alpha=0.4)

        ax_j.set_title(f"{JOINT_NAMES[j]}  [min={np.min(qpos_deg[:, j]):.1f}°, max={np.max(qpos_deg[:, j]):.1f}°]", fontsize=10.5, fontweight="bold")
        ax_j.set_xlabel("Time step (30 Hz)")
        ax_j.set_ylabel("Angle (deg)")
        ax_j.grid(True, alpha=0.3)
        if j == 0:
            ax_j.legend(loc="upper right", fontsize=8.5)

    out_png = os.path.join(DATASET_DIR, f"inspect_{ep_stem}.png")
    plt.savefig(out_png, dpi=140, bbox_inches="tight")
    print(f"  -> [VISUAL SAVED] {out_png}")

    if show_window and os.environ.get("DISPLAY"):
        print("  -> [WINDOW] 시각화 창을 표시합니다. (창을 닫으면 종료됩니다)")
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="SO-101 HDF5 Dataset Inspector & Visualizer")
    parser.add_argument("--dir", type=str, default=DATASET_DIR, help="HDF5 데이터셋 폴더 경로")
    parser.add_argument("--episode", type=int, default=-1, help="시각화할 에피소드 번호 (기본값: 최신 에피소드)")
    parser.add_argument("--all", action="store_true", help="모든 에피소드의 시각화 PNG 생성")
    parser.add_argument("--no-show", action="store_true", help="GUI 창을 띄우지 않고 PNG 파일만 저장")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "episode_*.hdf5")))
    print("=" * 95)
    print(f"[SO-101 HDF5 DATASET INSPECTOR] Directory: {args.dir}")
    print(f"Total HDF5 episodes found: {len(files)}")
    print("=" * 95)

    if not files:
        print("[INFO] 저장된 에피소드 파일(episode_*.hdf5)이 아직 없습니다.")
        print("       먼저 Isaac Sim Script Editor에서 scripts/collect_demonstrations.py 를 실행해 주세요.")
        return

    all_ok = True
    for idx, fpath in enumerate(files):
        ok = inspect_single_hdf5(fpath, idx, len(files))
        all_ok = all_ok and ok

    print("=" * 95)
    print(f"Integrity Status: {'[ALL EPISODES VALID]' if all_ok else '[WARNINGS DETECTED]'}")
    print("=" * 95)

    show_win = (not args.no_show) and (not args.all)
    if args.all:
        for fpath in files:
            visualize_episode(fpath, show_window=False)
    else:
        if args.episode >= 0:
            target_file = os.path.join(args.dir, f"episode_{args.episode:04d}.hdf5")
            if not os.path.exists(target_file):
                print(f"[ERROR] {target_file} 파일을 찾을 수 없습니다.")
                return
        else:
            target_file = files[-1]
        visualize_episode(target_file, show_window=show_win)


if __name__ == "__main__":
    main()

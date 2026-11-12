#!/usr/bin/env python3
"""
[Step 03] SO-101 HDF5 Episode Video Renderer & HUD Player
=========================================================
저장된 HDF5 데모 데이터셋(Robot_Arm_Project/data/so101_dual_eyes/episode_*.hdf5)의
양안 카메라(cam_left, cam_right) 영상과 실시간 관절 각도(deg) 텔레메트리를
하나의 Side-by-Side MP4(또는 GIF) 동영상으로 변환하여 저장합니다.

[실행 방법]
  1) 최신 에피소드 MP4 비디오 렌더링:
     ~/isaacsim/python.sh scripts/03_render_episode_video.py

  2) 특정 에피소드(예: 0번) MP4 비디오 렌더링:
     ~/isaacsim/python.sh scripts/03_render_episode_video.py --episode 0

  3) 모든 에피소드 비디오 일괄 변환:
     ~/isaacsim/python.sh scripts/03_render_episode_video.py --all

  4) 가벼운 미리보기용 GIF 애니메이션으로 렌더링:
     ~/isaacsim/python.sh scripts/03_render_episode_video.py --episode 0 --gif

[CLI 주요 옵션]
  --dir       : HDF5 데이터셋 폴더 경로 (기본값: Robot_Arm_Project/data/so101_dual_eyes)
  --episode   : 변환할 에피소드 번호 (기본값: -1, 최신 에피소드)
  --all       : 모든 에피소드를 일괄 렌더링 (기본값: False)
  --gif       : MP4 대신 웹 임베드용 GIF 형식으로 생성 (기본값: False)
  --speed     : 비디오 재생 배속 (기본값: 1.0)
"""

import os
import sys
import glob
import argparse
import h5py
import numpy as np
import cv2
import imageio

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project/data/so101_dual_eyes")
OUTPUT_VIDEO_DIR = os.path.join(DATASET_DIR, "videos")
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)


def render_single_episode_video(hdf5_path: str, is_gif: bool = False, speed_multiplier: float = 1.0) -> str:
    filename = os.path.basename(hdf5_path)
    ep_stem = os.path.splitext(filename)[0]
    ext = "gif" if is_gif else "mp4"
    out_path = os.path.join(OUTPUT_VIDEO_DIR, f"{ep_stem}.{ext}")

    print(f"\n[RENDERING {ext.upper()}] {filename} -> {out_path}")

    with h5py.File(hdf5_path, "r") as root:
        img_l_ds = root["observations/images/cam_left"]
        img_r_ds = root["observations/images/cam_right"]
        qpos_arr = np.degrees(root["observations/qpos"][()])  # (T, 6) in deg
        action_arr = np.degrees(root["action"][()])          # (T, 6) in deg
        T, H, W, _ = img_l_ds.shape
        fps = int(root.attrs.get("fps", 50) * speed_multiplier)
        spawn_xy = root.attrs.get("cube_spawn_xy", [0.0, 0.0])

        hud_h = 60
        out_w = W * 2
        out_h = H + hud_h

        # GIF일 경우 용량 최적화를 위해 2스텝 건너뛰기
        step_stride = 2 if is_gif else 1
        target_fps = max(int(fps / step_stride), 1)

        frames_to_write = []
        writer = None
        if not is_gif:
            writer = imageio.get_writer(
                out_path,
                fps=target_fps,
                codec="libx264",
                pixelformat="yuv420p",
                quality=8,
                macro_block_size=1,
            )

        for t in range(0, T, step_stride):
            # 1) 좌/우 RGB 프레임 복사
            frame_l = np.array(img_l_ds[t], copy=True)
            frame_r = np.array(img_r_ds[t], copy=True)

            # 카메라 레이블 오버레이 (RGB 공간에서 렌더링)
            cv2.putText(frame_l, "Left Eye (cam_left)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame_r, "Right Eye (cam_right)", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2, cv2.LINE_AA)

            cams_combined = np.hstack([frame_l, frame_r])

            # 2) 하단 텔레메트리 HUD 생성 (RGB)
            hud = np.zeros((hud_h, out_w, 3), dtype=np.uint8)
            hud[:] = (35, 35, 35)

            t_sec = t / 50.0
            total_sec = T / 50.0
            info_text1 = (
                f"[{ep_stem}] Step: {t:03d}/{T} ({t_sec:.1f}s/{total_sec:.1f}s) | "
                f"Cube Spawn: ({spawn_xy[0]:.3f}m, {spawn_xy[1]:.3f}m) | FPS: {fps}"
            )
            cv2.putText(hud, info_text1, (15, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

            q = qpos_arr[t]
            act = action_arr[t]
            info_text2 = (
                f"Pan:{q[0]:5.1f} | ShLift:{q[1]:5.1f} | Elbow:{q[2]:5.1f} | "
                f"WristPitch:{q[3]:5.1f} | Roll:{q[4]:5.1f} | "
                f"Grip:{q[5]:4.1f} deg (cmd:{act[5]:4.1f})"
            )
            cv2.putText(hud, info_text2, (15, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 255, 80), 1, cv2.LINE_AA)

            full_frame = np.vstack([cams_combined, hud])

            if is_gif:
                # GIF 용량을 줄이기 위해 절반 해상도로 리사이즈
                small = cv2.resize(full_frame, (out_w // 2, out_h // 2), interpolation=cv2.INTER_AREA)
                frames_to_write.append(small)
            else:
                writer.append_data(full_frame)

        if is_gif:
            imageio.mimsave(out_path, frames_to_write, fps=target_fps, loop=0)
        else:
            writer.close()

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"  -> [COMPLETE] Saved: {out_path} ({size_mb:.1f} MB, {out_w}x{out_h})")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="SO-101 HDF5 Episode Video Renderer")
    parser.add_argument("--dir", type=str, default=DATASET_DIR, help="HDF5 데이터셋 경로")
    parser.add_argument("--episode", type=int, default=-1, help="변환할 에피소드 번호 (기본값: 최신 에피소드)")
    parser.add_argument("--all", action="store_true", help="모든 에피소드를 비디오로 일괄 변환")
    parser.add_argument("--gif", action="store_true", help="MP4 대신 GIF 형식으로 생성")
    parser.add_argument("--speed", type=float, default=1.0, help="비디오 재생 속도 배율 (기본값: 1.0)")
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "episode_*.hdf5")))
    if not files:
        print(f"[ERROR] No HDF5 episodes found in {args.dir}")
        return

    print("=" * 78)
    print(f"[SO-101 VIDEO RENDERER] Total Episodes: {len(files)} | Output: {OUTPUT_VIDEO_DIR}")
    print("=" * 78)

    if args.all:
        for fpath in files:
            render_single_episode_video(fpath, is_gif=args.gif, speed_multiplier=args.speed)
    else:
        if args.episode >= 0:
            target = os.path.join(args.dir, f"episode_{args.episode:04d}.hdf5")
            if not os.path.exists(target):
                print(f"[ERROR] Episode file not found: {target}")
                return
        else:
            target = files[-1]
        render_single_episode_video(target, is_gif=args.gif, speed_multiplier=args.speed)

    print("\n>>> All video rendering completed successfully!\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
[Step 01] SO-101 Sim-to-Real: 양안(Dual-Cam) 데모 데이터셋 자동 수집기
===================================================================
[핵심 기능]
1. 작업 영역 전역 큐브 배치 다양화:
   - 연속 무작위 모드 (--mode random): R in [0.275, 0.345]m, Theta in [-22.5°, +22.5°]
   - 조밀 격자 모드 (--mode dense): 7개 반경 x 9개 각도 = 63개 균등 그리드
   - 큐브 회전 다양화 (--random_yaw): 큐브 Z축 회전각 +-12° 무작위 적용
2. 검증된 `execute_vision_pick.py` 단일 파이프라인 호출:
   - Pick & Place 물리/궤적과 100% 동일한 파지 성공률 보장
3. 양안 원본 해상도(640x320) HDF5 저장:
   - observations/images/cam_left: (T, 320, 640, 3) uint8
   - observations/images/cam_right: (T, 320, 640, 3) uint8
   - observations/qpos: (T, 6) float32 (rad)
   - observations/qvel: (T, 6) float32 (rad/s)
   - action: (T, 6) float32 (rad)

[실행 방법]
  1) 터미널 Headless 초고속 백그라운드 수집 (권장):
     bash scripts/01_collect_demonstrations.sh --headless --num_episodes 50 --mode random
     또는: ~/isaacsim/python.sh scripts/01_collect_demonstrations.py --headless --num_episodes 50

  2) GUI 창을 보면서 수집:
     bash scripts/01_collect_demonstrations.sh --num_episodes 20 --mode random
     또는 Isaac Sim [Window] -> [Script Editor]에 붙여넣고 [Run (Ctrl+Enter)]

[주요 옵션]
  --headless             : 화면 렌더링 없이 초고속 백그라운드 수집 (약 10~15분 소요)
  --num_episodes <int>   : 수집할 목표 에피소드 수 (기본: 60)
  --mode <random|dense>  : 큐브 배치 방식 (random: 연속 무작위, dense: 7x9 균등 격자)
  --random_yaw           : 큐브 Z축 회전각(+-12도) 미세 무작위 회전 적용 (기본: True)
  --usd <path>           : 로드할 시뮬레이션 USD 스테이지 경로
"""

import os
import sys
import glob
import asyncio
import importlib
import traceback
import argparse
import h5py
import numpy as np

# -----------------------------------------------------------------------------
# 0. Standalone (Headless / Terminal) 모드 지원
# -----------------------------------------------------------------------------
is_kit_gui = "omni.kit.app" in sys.modules
simulation_app = None

if not is_kit_gui:
    from isaacsim.simulation_app import SimulationApp
    parser = argparse.ArgumentParser(description="SO-101 Headless Data Collector")
    parser.add_argument("--headless", action="store_true", default=True, help="Run headless without GUI")
    parser.add_argument("--usd", type=str, default="/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal/dual_cam_so101_redcube.usd")
    parser.add_argument("--num_episodes", type=int, default=60, help="Number of episodes to collect")
    parser.add_argument("--mode", type=str, default="random", choices=["random", "dense"], help="Sampling mode: random or dense")
    parser.add_argument("--random_yaw", action="store_true", default=True, help="Add random yaw rotation to cube")
    args, _ = parser.parse_known_args()

    simulation_app = SimulationApp({"headless": args.headless})
    import omni.usd
    omni.usd.get_context().open_stage(args.usd)

import omni.timeline
import omni.usd
import omni.kit.app
import omni.replicator.core as rep

WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

import execute_vision_pick
importlib.reload(execute_vision_pick)

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

# -----------------------------------------------------------------------------
# 1. 이전 실행 태스크 정리 및 타임라인 정지 리스너 (GUI 전용)
# -----------------------------------------------------------------------------
if is_kit_gui:
    global _active_collect_task, _collect_stop_sub
    for t_name in ["_active_collect_task", "_active_vision_pick_task", "_active_vision_pick_pipeline_task"]:
        if t_name in globals() and globals()[t_name] and not globals()[t_name].done():
            globals()[t_name].cancel()
            print(f"[CLEANUP] Cancelled previous task ({t_name}).")

    def _on_timeline_stop(e):
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            global _active_collect_task
            if "_active_collect_task" in globals() and _active_collect_task and not _active_collect_task.done():
                _active_collect_task.cancel()
                print("[TIMELINE] Stop pressed: Data collection halted immediately.")

    _collect_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
        _on_timeline_stop, name="collect_stop_listener"
    )

# -----------------------------------------------------------------------------
# 2. 데이터셋 저장 설정 & 25개 최적 균등 격자점 (5x5)
# -----------------------------------------------------------------------------
DATASET_DIR = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project/data/so101_dual_eyes")
os.makedirs(DATASET_DIR, exist_ok=True)

FPS = execute_vision_pick.FPS
IMG_W, IMG_H = 640, 320  # 640x320 원본 비율

# -----------------------------------------------------------------------------
# 2-1. 큐브 스폰 좌표 생성 (Random vs Dense Grid)
# -----------------------------------------------------------------------------
mode = args.mode if not is_kit_gui else "random"
num_target_episodes = args.num_episodes if not is_kit_gui else 60
use_random_yaw = args.random_yaw if not is_kit_gui else True

SPAWN_CANDIDATES = []  # [(x, y, yaw_deg), ...]

if mode == "dense":
    # 7개 반경 x 9개 각도 = 63개 균등 조밀 격자
    dense_radii = np.linspace(0.275, 0.345, 7)
    dense_angles = np.linspace(-22.0, 22.0, 9)
    for r_val in dense_radii:
        for deg_val in dense_angles:
            rad_val = np.radians(deg_val)
            x_val = float(r_val * np.cos(rad_val))
            y_val = float(r_val * np.sin(rad_val))
            yaw_val = float(np.random.uniform(-10.0, 10.0)) if use_random_yaw else 0.0
            SPAWN_CANDIDATES.append((x_val, y_val, yaw_val))
else:
    # 완전 무작위 균등 샘플링 (연속적인 보간 일반화에 최적)
    np.random.seed(42)  # 재현 가능한 시드
    for _ in range(num_target_episodes):
        r_val = np.random.uniform(0.275, 0.345)
        deg_val = np.random.uniform(-22.5, 22.5)
        rad_val = np.radians(deg_val)
        x_val = float(r_val * np.cos(rad_val))
        y_val = float(r_val * np.sin(rad_val))
        yaw_val = float(np.random.uniform(-12.0, 12.0)) if use_random_yaw else 0.0
        SPAWN_CANDIDATES.append((x_val, y_val, yaw_val))

print(f"[DATASET CONFIG] Mode: {mode.upper()}, Total Candidates: {len(SPAWN_CANDIDATES)}, Random Yaw: {use_random_yaw}")

# -----------------------------------------------------------------------------
# 3. 양안 카메라 Replicator Annotator (640x320)
# -----------------------------------------------------------------------------
left_cam_path = str(execute_vision_pick.left_cam_prim.GetPath())
right_cam_path = str(execute_vision_pick.right_cam_prim.GetPath())

rp_left = rep.create.render_product(left_cam_path, (IMG_W, IMG_H))
rp_right = rep.create.render_product(right_cam_path, (IMG_W, IMG_H))

annot_rgb_l = rep.AnnotatorRegistry.get_annotator("rgb")
annot_rgb_r = rep.AnnotatorRegistry.get_annotator("rgb")
annot_rgb_l.attach(rp_left)
annot_rgb_r.attach(rp_right)

def get_rgb_frame(annotator):
    data = annotator.get_data()
    if data is None or len(data) == 0:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8).copy()
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

def save_episode_hdf5(ep_idx, ep_buf, cube_xy):
    save_path = os.path.join(DATASET_DIR, f"episode_{ep_idx:04d}.hdf5")
    qpos_arr = np.array(ep_buf["qpos"], dtype=np.float32)
    action_arr = np.array(ep_buf["action"], dtype=np.float32)
    qvel_arr = np.zeros_like(qpos_arr)
    if len(qpos_arr) > 1:
        qvel_arr[1:] = (qpos_arr[1:] - qpos_arr[:-1]) * FPS

    total_steps = len(qpos_arr)
    with h5py.File(save_path, "w", rdcc_nbytes=1024**2 * 4) as root:
        root.attrs["sim"] = True
        root.attrs["total_timesteps"] = total_steps
        root.attrs["fps"] = FPS
        root.attrs["image_width"] = IMG_W
        root.attrs["image_height"] = IMG_H
        root.attrs["cube_spawn_xy"] = np.array(cube_xy, dtype=np.float32)

        root.create_dataset("action", data=action_arr)
        obs = root.create_group("observations")
        obs.create_dataset("qpos", data=qpos_arr)
        obs.create_dataset("qvel", data=qvel_arr)

        img_grp = obs.create_group("images")
        for cam_key in ["cam_left", "cam_right"]:
            img_data = np.array(ep_buf[cam_key], dtype=np.uint8)
            img_grp.create_dataset(
                cam_key,
                data=img_data,
                chunks=(1, IMG_H, IMG_W, 3),
                compression="gzip",
                compression_opts=2,
            )

    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"  [SAVED] {save_path} ({total_steps} steps, {IMG_W}x{IMG_H}, {size_mb:.1f} MB)")

def get_next_episode_index():
    existing = sorted(glob.glob(os.path.join(DATASET_DIR, "episode_*.hdf5")))
    return len(existing)

# -----------------------------------------------------------------------------
# 4. 메인 데이터 수집 루프 (25개 격자 순회)
# -----------------------------------------------------------------------------
async def run_data_collection():
    try:
        if not timeline.is_playing():
            timeline.play()

        app = omni.kit.app.get_app()
        for _ in range(20):
            await app.next_update_async()

        execute_vision_pick.ensure_initialized()
        saved_count = get_next_episode_index()

        mode_str = "HEADLESS" if (simulation_app and not is_kit_gui) else "GUI"
        print("\n" + "=" * 78)
        print(f">>> [SO-101 DATA COLLECTION ({mode_str})] 25 Grids (5x5) | Cam: {IMG_W}x{IMG_H} | 50Hz")
        print("=" * 78)

        for ep_i, (spawn_x, spawn_y, spawn_yaw) in enumerate(SPAWN_CANDIDATES):
            if not timeline.is_playing():
                break

            print(f"\n--- [Episode Candidate {ep_i+1}/{len(SPAWN_CANDIDATES)}] Cube Target XY=({spawn_x:.3f}, {spawn_y:.3f}), Yaw={spawn_yaw:.1f}° ---")

            # 1) 큐브를 이번 위치 및 각도로 이동
            execute_vision_pick.reset_cube_pose(spawn_x, spawn_y, 0.015, yaw_deg=spawn_yaw)
            for _ in range(10):
                await app.next_update_async()

            # 2) 녹화 버퍼 및 스텝 콜백 정의
            ep_buf = {"qpos": [], "action": [], "cam_left": [], "cam_right": []}

            def record_step(qpos_6, action_6):
                ep_buf["qpos"].append(qpos_6)
                ep_buf["action"].append(action_6)
                ep_buf["cam_left"].append(get_rgb_frame(annot_rgb_l))
                ep_buf["cam_right"].append(get_rgb_frame(annot_rgb_r))

            # 3) execute_vision_pick.py의 단일 Pick & Place 함수 호출 및 동기 녹화
            in_box = await execute_vision_pick.execute_single_pick_and_place(
                step_callback=record_step,
                use_camera_noise=False,
                auto_reset_after=False,
            )

            # 4) 상자 안착 성공 시 HDF5 파일로 저장
            if in_box and len(ep_buf["qpos"]) > 0:
                save_episode_hdf5(saved_count, ep_buf, (spawn_x, spawn_y))
                saved_count += 1
            else:
                print(f"  [DISCARD] Episode {ep_i+1} did not land in TargetBox (not saved).")

            # 5) 성공/실패 여부와 무관하게 즉시 다음 후보 위치로 큐브 이동
            if (ep_i + 1) < len(SPAWN_CANDIDATES):
                next_x, next_y, next_yaw = SPAWN_CANDIDATES[ep_i + 1]
            else:
                next_x, next_y, next_yaw = (0.310, 0.0, 0.0)
            execute_vision_pick.reset_cube_pose(next_x, next_y, 0.015, yaw_deg=next_yaw)
            for _ in range(10):
                await app.next_update_async()

        # 수집 완료 후 관측 자세로 복귀
        obs_pose = np.array([0.0, np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0], dtype=np.float32)
        await execute_vision_pick.move_joints_smoothly(obs_pose, duration_sec=1.4, grip_deg=0.0, step_callback=None)

        print("\n" + "=" * 78)
        print(f">>> [COLLECTION COMPLETE] Total saved episodes: {saved_count} in {DATASET_DIR}")
        print("=" * 78 + "\n")

    except asyncio.CancelledError:
        print("[CLEANUP] Data collection cancelled gracefully.")
    except Exception as e:
        print(f"\n[EXCEPTION ERROR in run_data_collection]: {e}")
        traceback.print_exc()


# GUI 모드와 Standalone Headless 모드 분기 실행
if is_kit_gui:
    _active_collect_task = asyncio.ensure_future(run_data_collection())
else:
    async def _run_standalone():
        await run_data_collection()
        simulation_app.close()

    asyncio.ensure_future(_run_standalone())
    while simulation_app.is_running():
        simulation_app.update()

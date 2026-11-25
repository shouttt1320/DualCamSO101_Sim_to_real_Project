#!/usr/bin/env python3
"""
[Step 02] SO-101 Sim-to-Real: Active Search & Pick Multi-Environment Dataset Collector
======================================================================================
환경 설정 모듈(env_manager), 행동 정책 모듈(search_and_pick_policy), 데이터 저장 모듈(dataset_recorder)을
순서대로 오케스트레이션하여 고품질 능동 시각 탐색 시연 데이터를 자동 수집합니다.

[실행 방법]
  1) 터미널 무화면(Headless) 초고속 수집 (기본 환경 50개):
     bash scripts/active_search/02_collect_search_demos.sh --num_episodes 50 --headless

  2) 다른 환경 지정 수집 (추후 확장):
     bash scripts/active_search/02_collect_search_demos.sh --env default_clean --num_episodes 50

  3) GUI 화면을 보면서 실시간 수집:
     bash scripts/active_search/02_collect_search_demos.sh --num_episodes 10
     또는 Isaac Sim Script Editor에서 실행
"""

import os
import sys
import argparse
import asyncio
import traceback
import numpy as np

# -----------------------------------------------------------------------------
# 0. 경로 및 환경 초기화
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from env_configs import ENV_REGISTRY
from dataset_recorder import HDF5DatasetRecorder

# Standalone (Headless / Terminal) 모드 지원
is_kit_gui = "omni.kit.app" in sys.modules
simulation_app = None

parser = argparse.ArgumentParser(description="SO-101 Multi-Environment Active Search Data Collector")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless without GUI")
parser.add_argument("--env", type=str, default="domain_rand", choices=list(ENV_REGISTRY.keys()), help="Environment preset name")
parser.add_argument("--usd", type=str, default=None, help="Optional custom USD stage path")
parser.add_argument("--num_episodes", type=int, default=50, help="Number of target episodes to collect")
parser.add_argument("--mode", type=str, default="random", choices=["random", "dense"], help="Sampling mode: random or dense")
parser.add_argument("--random_yaw", action="store_true", default=True, help="Randomize cube yaw rotation (+-15 deg)")
parser.add_argument("--no_cube_ratio", type=float, default=0.20, help="Ratio of No-Cube negative sweep demos (0.0~1.0, default: 0.20)")
parser.add_argument("--output_dir", type=str, default=os.path.join(WORKSPACE_DIR, "Robot_Arm_Project/data/so101_active_search"), help="Root output directory")

if not is_kit_gui:
    args, _ = parser.parse_known_args()
    from isaacsim.simulation_app import SimulationApp

    usd_to_open = args.usd if args.usd else ENV_REGISTRY[args.env]["usd_path"]
    simulation_app = SimulationApp({"headless": args.headless})

    import omni.usd
    omni.usd.get_context().open_stage(usd_to_open)
else:
    # GUI Script Editor 실행 시 기본 인자 사용
    args = parser.parse_args([])

# SimulationApp 구동 후 Omniverse 의존 모듈 import
from env_manager import EnvironmentManager
from search_and_pick_policy import ActiveSearchPolicy
from domain_randomizer import DomainRandomizer

import omni.timeline
import omni.usd
import omni.kit.app

try:
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
except ImportError:
    from omni.isaac.motion_generation import LulaKinematicsSolver

URDF_PATH = os.path.join(WORKSPACE_DIR, "so101_dual_cam.urdf")
DESC_YAML = os.path.join(WORKSPACE_DIR, "motion_policy_configs/so101/so101_robot_description.yaml")

timeline = omni.timeline.get_timeline_interface()

# GUI 타임라인 Stop 리스너
if is_kit_gui:
    global _active_collect_task, _collect_stop_sub
    for t_name in ["_active_collect_task", "_active_scan_pick_task"]:
        if t_name in globals() and globals()[t_name] and not globals()[t_name].done():
            globals()[t_name].cancel()

    def _on_timeline_stop(e):
        if e.type == int(omni.timeline.TimelineEventType.STOP):
            global _active_collect_task
            if "_active_collect_task" in globals() and _active_collect_task and not _active_collect_task.done():
                _active_collect_task.cancel()
                print("[TIMELINE] Stop pressed: Data collection halted immediately.")

    _collect_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
        _on_timeline_stop, name="collect_stop_listener"
    )


def generate_spawn_candidates(num_episodes, mode="random", use_random_yaw=True, env_cfg=None):
    """
    사각지대와 정면을 고르게 포함하는 광범위 작업 영역 큐브 좌표 생성:
    - 기본: R in [0.26, 0.35]m, Theta in [-55.0°, +55.0°]
    - real_wall_twin 등 테이블/상자 경계가 지정된 경우: 테이블 내부 + 상자 회피 필터 적용
    """
    safe_x = env_cfg.get("table_safe_x") if env_cfg else None
    safe_y = env_cfg.get("table_safe_y") if env_cfg else None
    box_pos = env_cfg.get("box_pos") if env_cfg else None
    box_excl = float(env_cfg.get("box_exclusion_dist", 0.115)) if env_cfg else 0.115

    def _is_valid_xy(x_v, y_v):
        if safe_x and not (safe_x[0] <= x_v <= safe_x[1]):
            return False
        if safe_y and not (safe_y[0] <= y_v <= safe_y[1]):
            return False
        if box_pos is not None:
            if np.hypot(x_v - float(box_pos[0]), y_v - float(box_pos[1])) < box_excl:
                return False
        return True

    r_min, r_max = (0.210, 0.340) if safe_x else (0.260, 0.350)
    th_lim = 58.0 if safe_x else 55.0

    candidates = []
    if mode == "dense":
        radii = np.linspace(r_min, r_max, 5)
        angles = np.linspace(-th_lim, th_lim, 11)
        for r_val in radii:
            for deg_val in angles:
                rad_val = np.radians(deg_val)
                x_val = float(r_val * np.cos(rad_val))
                y_val = float(r_val * np.sin(rad_val))
                if not _is_valid_xy(x_val, y_val):
                    continue
                yaw_val = float(np.random.uniform(-15.0, 15.0)) if use_random_yaw else 0.0
                candidates.append((x_val, y_val, yaw_val, r_val, deg_val))
    else:
        np.random.seed(42)
        while len(candidates) < num_episodes:
            r_val = float(np.random.uniform(r_min, r_max))
            deg_val = float(np.random.uniform(-th_lim, th_lim))
            # 테이블 환경(safe_x)에서는 50% 확률로 사각지대(|Theta|>=22° 또는 근거리 R<=0.255m) 비중 강화
            if safe_x and (len(candidates) % 2 == 1):
                if abs(deg_val) < 22.0 and r_val > 0.255:
                    continue
            rad_val = np.radians(deg_val)
            x_val = float(r_val * np.cos(rad_val))
            y_val = float(r_val * np.sin(rad_val))
            if not _is_valid_xy(x_val, y_val):
                continue
            yaw_val = float(np.random.uniform(-15.0, 15.0)) if use_random_yaw else 0.0
            candidates.append((x_val, y_val, yaw_val, r_val, deg_val))
    return candidates


async def run_data_collection_pipeline():
    """모듈식 데이터 수집 메인 파이프라인"""
    try:
        app = omni.kit.app.get_app()

        if not timeline.is_playing():
            timeline.play()

        for _ in range(20):
            await app.next_update_async()

        # ---------------------------------------------------------------------
        # 1. 3개 모듈 초기화 (Environment, Policy, Recorder)
        # ---------------------------------------------------------------------
        print("\n" + "=" * 82)
        print(f">>> [DATA COLLECTION] Initializing Modular Active Search Pipeline...")
        print(f"  • Environment:  {args.env} ({ENV_REGISTRY[args.env]['description']})")
        print(f"  • Target Count: {args.num_episodes} episodes (Mode: {args.mode.upper()})")
        print(f"  • Output Dir:   {os.path.join(args.output_dir, args.env)}")
        print("=" * 82)

        # 1) 환경 관리자 & 도메인 랜덤화
        env = EnvironmentManager(env_name=args.env, custom_usd_path=args.usd, img_w=640, img_h=480)
        env.ensure_initialized()

        use_randomizer = bool(ENV_REGISTRY[args.env].get("randomize", False)) and (env.room_builder is None)
        randomizer = DomainRandomizer(stage=env.stage) if use_randomizer else None
        if env.room_builder is not None:
            print(f"  • Randomizer:   REAL_WALL_TWIN (Folding Table + Backwall + RealRed Cloth + Clutter On/Off + Soft RectLight)")
        elif randomizer is not None:
            box_tr = omni.usd.get_world_transform_matrix(env.box_prim).ExtractTranslation()
            box_xy = (float(box_tr[0]), float(box_tr[1]))
            print(f"  • Randomizer:   ENABLED (Tabletop + Cube PBR + Lighting + Distractors + Cam Jitter)")

        # 2) 기구학 솔버 & 행동 정책
        kinematics_solver = LulaKinematicsSolver(robot_description_path=DESC_YAML, urdf_path=URDF_PATH)
        policy = ActiveSearchPolicy(kinematics_solver=kinematics_solver)

        # 3) 데이터셋 레코더 (환경별 하위 폴더 분리)
        env_output_dir = os.path.join(args.output_dir, args.env)
        recorder = HDF5DatasetRecorder(output_dir=env_output_dir)

        # 큐브 스폰 후보군 생성
        candidates = generate_spawn_candidates(args.num_episodes, mode=args.mode, use_random_yaw=args.random_yaw, env_cfg=env.cfg)
        total_targets = min(args.num_episodes, len(candidates))

        # 기존 에피소드 덮어쓰기 방지: 가장 큰 episode 번호 다음부터 이어서 저장
        existing_eps = [
            f for f in (os.listdir(env_output_dir) if os.path.exists(env_output_dir) else [])
            if f.startswith("episode_") and f.endswith(".hdf5")
        ]
        max_existing_idx = -1
        for fname in existing_eps:
            try:
                idx_val = int(fname[len("episode_") : -len(".hdf5")])
                if idx_val > max_existing_idx:
                    max_existing_idx = idx_val
            except ValueError:
                pass
        start_ep_idx = max_existing_idx + 1
        if start_ep_idx > 0:
            print(f"[APPEND MODE] 기존 데이터({len(existing_eps)}개) 보존 -> episode_{start_ep_idx:04d}.hdf5 부터 이어서 저장합니다.")

        success_count = 0
        search_count = 0
        no_cube_count = 0
        n_no_cube = int(round(total_targets * float(np.clip(args.no_cube_ratio, 0.0, 1.0))))
        no_cube_indices = set(np.linspace(0, total_targets - 1, n_no_cube, dtype=int).tolist()) if n_no_cube > 0 else set()

        # ---------------------------------------------------------------------
        # 2. 에피소드 수집 메인 루프
        # ---------------------------------------------------------------------
        for ep_idx in range(total_targets):
            if not timeline.is_playing():
                print("[COLLECTION HALTED] Timeline was stopped.")
                break

            is_no_cube_ep = (ep_idx in no_cube_indices)
            x_sp, y_sp, yaw_sp, r_sp, theta_sp = candidates[ep_idx]
            if is_no_cube_ep:
                print(f"\n>>> [Episode {ep_idx + 1:03d}/{total_targets:03d}] NO-CUBE Negative Demo (Sweep Only, No Grasp)")
                env.reset_cube_pose(-0.50, -0.50, z_val=-0.20, yaw_deg=0.0)
            else:
                print(f"\n>>> [Episode {ep_idx + 1:03d}/{total_targets:03d}] Spawning Cube: R={r_sp:.3f}m, Theta={theta_sp:+.1f}°, Yaw={yaw_sp:+.1f}°")
                env.reset_cube_pose(x_sp, y_sp, z_val=0.015, yaw_deg=yaw_sp)
            if env.room_builder is not None:
                env.room_builder.randomize_episode(cube_prim_path=env.cfg["cube_prim"])
            elif randomizer is not None:
                randomizer.randomize_lighting(intensity_range=(1000.0, 2400.0), temp_range=(3800.0, 6500.0), rotate=True)
                randomizer.randomize_table_texture(mode="white_dominant")
                randomizer.randomize_cube_material(cube_prim_path=env.cfg["cube_prim"])
                randomizer.spawn_distractors(
                    num_objects=int(np.random.randint(1, 4)),
                    cube_pos=(x_sp, y_sp),
                    box_pos=box_xy,
                    min_dist=0.07,
                )
                randomizer.spawn_background_props(
                    num_props=int(np.random.randint(4, 8)),
                    safe_radius=0.40,
                )
                # 고정 손목 카메라의 기하 일관성 유지를 위해 extrinsics 지터 비활성화
                # randomizer.jitter_camera_pose(env.left_cam_prim, rot_deg_range=0.5, trans_mm_range=1.0)
                # randomizer.jitter_camera_pose(env.right_cam_prim, rot_deg_range=0.5, trans_mm_range=1.0)
                randomizer.set_robot_arm_3d_printed_color(color_rgb=(0.06, 0.25, 0.78), jitter=0.03)
            recorder.reset_buffer()

            for _ in range(15):
                await app.next_update_async()

            # [Step 2: 행동 정책 모듈 실행] 능동 탐색 -> 정렬 -> 파지 -> 안착 (녹화 콜백 연동)
            success, search_trig, est_xy, err_mm = await policy.execute_search_and_pick(
                env=env,
                true_cube_x=x_sp,
                true_cube_y=y_sp,
                step_callback=recorder.record_step,
                no_cube_mode=is_no_cube_ep,
            )

            if search_trig:
                search_count += 1

            # [Step 3: 데이터 저장 모듈 실행] 성공 에피소드 HDF5 저장
            if success:
                success_count += 1
                if is_no_cube_ep:
                    no_cube_count += 1
                metadata = {
                    "env_name": args.env,
                    "cube_spawn_xy": np.array([-0.50, -0.50] if is_no_cube_ep else [x_sp, y_sp], dtype=np.float32),
                    "cube_yaw_deg": float(0.0 if is_no_cube_ep else yaw_sp),
                    "search_triggered": bool(search_trig),
                    "no_cube": bool(is_no_cube_ep),
                    "stereo_error_mm": float(err_mm),
                    "success": True,
                }
                save_idx = start_ep_idx + ep_idx
                save_res = recorder.save_to_hdf5(save_idx, metadata=metadata)
                if save_res:
                    save_path, steps, mb = save_res
                    tag = "NO-CUBE" if is_no_cube_ep else f"Err: {err_mm:.1f}mm"
                    print(f"  └─ [EPISODE SAVED 🏆] {os.path.basename(save_path)} | Steps: {steps} ({steps/50.0:.1f}s) | Size: {mb:.1f}MB | Search: {search_trig} | {tag}")
            else:
                recorder.reset_buffer()
                print(f"  └─ [EPISODE FAILED ❌] Missed target or placement error. Buffer discarded.")

        # ---------------------------------------------------------------------
        # 3. 최종 수집 통계 리포트
        # ---------------------------------------------------------------------
        print("\n" + "=" * 82)
        print(">>> [DATA COLLECTION COMPLETE]")
        print(f"  • Environment:       {args.env}")
        print(f"  • Total Attempted:   {total_targets} episodes")
        print(f"  • Success Recorded:  {success_count} / {total_targets} ({success_count / max(total_targets, 1) * 100:.1f}%)")
        print(f"  • Active Search Used:{search_count} episodes (No-Cube: {no_cube_count} episodes)")
        print(f"  • Output Directory:  {env_output_dir}")
        print("=" * 82 + "\n")

    except Exception as e:
        print(f"[ERROR in run_data_collection_pipeline]: {e}")
        traceback.print_exc()


# -----------------------------------------------------------------------------
# 엔트리포인트 (Standalone vs GUI)
# -----------------------------------------------------------------------------
if simulation_app is not None:
    task = asyncio.ensure_future(run_data_collection_pipeline())
    while not task.done():
        simulation_app.update()
    simulation_app.close()
else:
    _active_collect_task = asyncio.ensure_future(run_data_collection_pipeline())

#!/usr/bin/env python3
"""
[Step 05] SO-101 능동 탐색 (Active Search) 광각 사각지대 벤치마크 평가 스크립트
=============================================================================
사각지대(-50° ~ +50°) 및 다양한 거리(0.28m ~ 0.34m) 전 영역에 걸쳐
DINOv2-ACT 신경망 정책의 자율 탐색 및 픽앤플레이스 성공률을 자동 측정하고
단계별 마일스톤(Scan, Approach, Grasp, Transport, Place, Return)과
종합 CSV 보고서를 생성합니다.

[실행 방법 2가지]
  1) Isaac Sim GUI Script Editor:
     - Isaac Sim 상단 메뉴 [Window] -> [Script Editor] 열기
     - 본 파일 전체를 복사하여 붙여넣고 [Run (Ctrl+Enter)] 클릭

  2) 터미널 실행 (Standalone CLI):
     bash scripts/active_search/05_benchmark_active_search.sh
"""

import os
import sys
import argparse
import asyncio
import time
import csv
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# [GUI 스크립트 에디터 전용 기본 설정]
# -----------------------------------------------------------------------------
GUI_ENV = "default_clean"
GUI_R_LIST = [0.28, 0.30, 0.32]
GUI_THETA_LIST = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]
GUI_MAX_TIMESTEPS = 1150
GUI_INFERENCE_INTERVAL = 2

# -----------------------------------------------------------------------------
# 1. 실행 환경 감지 및 경로 설정
# -----------------------------------------------------------------------------
is_kit_gui = "omni.kit.app" in sys.modules

if "__file__" in globals() and "/tmp" not in os.path.abspath(__file__):
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
    WORKSPACE_DIR = os.path.dirname(SCRIPTS_DIR)
else:
    WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
    SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")
    SCRIPT_DIR = os.path.join(SCRIPTS_DIR, "active_search")

PROJECT_ROOT = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project")
NV_DIR = "/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR = "/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

for p in [
    SCRIPT_DIR,
    SCRIPTS_DIR,
    WORKSPACE_DIR,
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "models"),
    ML_DIR,
    "/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle",
]:
    if p not in sys.path:
        sys.path.insert(0, p)

if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = f"{NV_DIR}/cuda_runtime/lib:{NV_DIR}/cublas/lib:{NV_DIR}/cudnn/lib:{NV_DIR}/cusolver/lib:{os.environ['LD_LIBRARY_PATH']}"

from env_configs import ENV_REGISTRY

# -----------------------------------------------------------------------------
# 2. 인자 파싱 (CLI) 또는 GUI 설정 로드
# -----------------------------------------------------------------------------
sim_app = None

if not is_kit_gui:
    parser = argparse.ArgumentParser(description="SO-101 Active Search Wide-Angle Benchmark")
    parser.add_argument("--headless", action="store_true", default=True, help="화면 없이 백그라운드 고속 평가 (기본: True)")
    parser.add_argument("--gui", dest="headless", action="store_false", help="GUI 화면을 띄워 시각적으로 확인하며 평가")
    parser.add_argument("--env", type=str, default="default_clean", help="평가 대상 환경 프리셋")
    parser.add_argument("--usd", type=str, default=None, help="커스텀 USD 경로")
    parser.add_argument("--ckpt", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--stats", type=str, default=None, help="정규화 통계 경로")
    parser.add_argument("--max_timesteps", type=int, default=1150, help="에피소드당 최대 스텝 (기본: 1150)")
    parser.add_argument("--inference_interval", type=int, default=2, help="추론 간격 (기본: 2)")
    parser.add_argument("--r_list", nargs="+", type=float, default=GUI_R_LIST, help="테스트 거리 R 목록 [m]")
    parser.add_argument("--theta_list", nargs="+", type=float, default=GUI_THETA_LIST, help="테스트 각도 Theta 목록 [deg]")
    parser.add_argument("--random_yaw", action="store_true", default=False, help="큐브 Z축 미세 회전(-10~+10도) 추가 여부")

    args, _ = parser.parse_known_args()

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": args.headless})
    import omni.usd
    target_usd = args.usd if args.usd else ENV_REGISTRY[args.env]["usd_path"]
    print(f"[STAGE] Loading USD: {target_usd}")
    omni.usd.get_context().open_stage(target_usd)
else:
    class Args:
        pass
    args = Args()
    args.env = GUI_ENV
    args.usd = None
    args.ckpt = None
    args.stats = None
    args.max_timesteps = GUI_MAX_TIMESTEPS
    args.inference_interval = GUI_INFERENCE_INTERVAL
    args.r_list = GUI_R_LIST
    args.theta_list = GUI_THETA_LIST
    args.random_yaw = False

ENV_NAME = args.env
CKPT_ENV = ENV_REGISTRY.get(ENV_NAME, {}).get("ckpt_env", ENV_NAME)
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", CKPT_ENV)
CKPT_PATH = args.ckpt if args.ckpt else os.path.join(DEFAULT_CKPT_DIR, "policy_best.pth")
STATS_PATH = args.stats if args.stats else os.path.join(DEFAULT_CKPT_DIR, "dataset_stats.pkl")
CSV_REPORT_PATH = os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", ENV_NAME, "benchmark_report.csv")

# -----------------------------------------------------------------------------
# 3. GUI Script Editor 중복 실행 방지 및 정지 이벤트 리스너
# -----------------------------------------------------------------------------
import omni.timeline
import omni.kit.app

timeline = omni.timeline.get_timeline_interface()

global _active_bench_task, _bench_stop_sub
if "_active_bench_task" in globals() and _active_bench_task and not _active_bench_task.done():
    _active_bench_task.cancel()
    print("[CLEANUP] 이전 실행 중이던 벤치마크 태스크를 취소했습니다.")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        if "_active_bench_task" in globals() and _active_bench_task and not _active_bench_task.done():
            _active_bench_task.cancel()
            print("[CLEANUP] 시뮬레이션 정지 버튼 감지: 벤치마크 태스크를 중단했습니다.")

_bench_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(_on_timeline_stop)

from env_manager import EnvironmentManager
from models.so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    HIDDEN_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
    TEMPORAL_AGG_M,
)
from models.so101_dual_dinov2_act.models import build_act_model


def load_active_search_policy(ckpt_path, stats_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[POLICY] Loading Active Search Policy from: {ckpt_path}")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float().to(device)
    qpos_std = torch.from_numpy(stats["qpos_std"]).float().to(device)
    action_mean = stats["action_mean"]
    action_std = stats["action_std"]

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    actual_chunk_size = state_dict["query_embed.weight"].shape[0] if "query_embed.weight" in state_dict else CHUNK_SIZE
    ckpt_backbone = ckpt.get("backbone_name", "dinov2_vits14") if isinstance(ckpt, dict) else "dinov2_vits14"

    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=actual_chunk_size,
        camera_names=CAMERA_NAMES,
        backbone_name=ckpt_backbone,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
        train_backbone=False,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, transform, qpos_mean, qpos_std, action_mean, action_std, device, actual_chunk_size


async def move_joints_smoothly(env, target_rad, duration_sec=1.2, grip_deg=0.0):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    steps = max(1, int(duration_sec * 50))
    q_curr = env.robot_articulation.get_joint_positions()
    q_start = np.array(q_curr[:5], dtype=np.float32) if q_curr is not None and len(q_curr) >= 5 else np.zeros(5, dtype=np.float32)
    t_rad = np.array(target_rad[:5], dtype=np.float32)
    g_rad = np.radians(grip_deg)

    for i in range(1, steps + 1):
        if not timeline.is_playing():
            return False
        alpha = 0.5 * (1.0 - np.cos(np.pi * (i / steps)))
        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_start + alpha * (t_rad - q_start)
        q_full[5] = g_rad
        env.apply_full_joints(q_full)
        await app.next_update_async()
    return True


async def run_benchmark():
    try:
        timeline = omni.timeline.get_timeline_interface()
        if not timeline.is_playing():
            timeline.play()

        app = omni.kit.app.get_app()
        for _ in range(25):
            await app.next_update_async()

        env = EnvironmentManager(env_name=ENV_NAME, custom_usd_path=args.usd)
        env.ensure_initialized()

        model, transform, qpos_mean, qpos_std, action_mean, action_std, device, chunk_size = load_active_search_policy(CKPT_PATH, STATS_PATH)

        OBSERVATION_POSE = np.array([0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0], dtype=np.float32)

        test_points = []
        for r_val in args.r_list:
            for th_val in args.theta_list:
                test_points.append((float(r_val), float(th_val)))

        total_trials = len(test_points)
        print("\n" + "=" * 84)
        print(f">>> [SO-101 BENCHMARK] Evaluating Active Search Policy across {total_trials} Grid Points")
        print(f"  • Distance R:   {args.r_list} m")
        print(f"  • Angles Theta: {args.theta_list} deg")
        print(f"  • Max Steps:    {args.max_timesteps} steps (~{args.max_timesteps/50:.1f}s)")
        print(f"  • Report Path:  {CSV_REPORT_PATH}")
        print("=" * 84 + "\n")

        benchmark_records = []
        success_count = 0
        blind_total = 0
        blind_success = 0
        front_total = 0
        front_success = 0

        scan_count = 0
        grasp_count = 0
        trans_count = 0
        return_count = 0

        t_bench_start = time.time()

        for idx, (r_val, theta_deg) in enumerate(test_points):
            if not timeline.is_playing():
                break

            th_rad = np.radians(theta_deg)
            tx = float(r_val * np.cos(th_rad))
            ty = float(r_val * np.sin(th_rad))
            yaw_deg = float(np.random.uniform(-10.0, +10.0)) if args.random_yaw else 0.0
            is_blind = abs(theta_deg) >= 25.0

            if is_blind:
                blind_total += 1
            else:
                front_total += 1

            # 1) 홈 자세 복귀
            await move_joints_smoothly(env, OBSERVATION_POSE, duration_sec=1.0, grip_deg=0.0)

            # 2) 큐브 재배치
            env.reset_cube_pose(tx, ty, z_val=0.015, yaw_deg=yaw_deg)
            for _ in range(15):
                await app.next_update_async()

            # 3) 폐루프 추론 실행
            curr_pred_chunk = None
            pred_base_t = 0
            max_lift_height = 0.0
            trial_scan = False
            trial_grasp = False
            trial_trans = False

            # 실시간 Temporal Ensembling 버퍼 초기화
            all_time_actions = np.zeros((args.max_timesteps, args.max_timesteps + chunk_size, STATE_DIM), dtype=np.float32)
            action_populated = np.zeros(args.max_timesteps, dtype=bool)

            with torch.no_grad():
                for t in range(args.max_timesteps):
                    if not timeline.is_playing():
                        break

                    if t % args.inference_interval == 0 or curr_pred_chunk is None:
                        q_curr = env.robot_articulation.get_joint_positions()
                        q_curr_6 = np.array(q_curr[:6], dtype=np.float32) if q_curr is not None and len(q_curr) >= 6 else np.zeros(6, dtype=np.float32)

                        img_l, img_r = env.get_rgb_frames()

                        q_tensor = torch.from_numpy(q_curr_6).float().to(device)
                        q_norm = ((q_tensor - qpos_mean) / qpos_std).unsqueeze(0)

                        cam_tensors = []
                        for img_raw in [img_l, img_r]:
                            t_img = torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0
                            t_img = transform(t_img)
                            cam_tensors.append(t_img)
                        image_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(device)

                        a_hat, _, _ = model(qpos=q_norm, image=image_input)
                        curr_pred_chunk = a_hat[0].cpu().numpy() * action_std + action_mean
                        pred_base_t = t

                        all_time_actions[t, t : t + chunk_size] = curr_pred_chunk
                        action_populated[t] = True

                    # Temporal Ensembling 가중합 기반 액션 결정
                    history_start = max(0, t - chunk_size + 1)
                    actions_for_t = []
                    weights_for_t = []
                    for i in range(history_start, t + 1):
                        if action_populated[i]:
                            actions_for_t.append(all_time_actions[i, t])
                            age = t - i
                            weights_for_t.append(np.exp(-TEMPORAL_AGG_M * age))

                    if len(actions_for_t) > 0:
                        actions_arr = np.array(actions_for_t, dtype=np.float32)
                        w_arr = np.array(weights_for_t, dtype=np.float32)
                        w_arr = w_arr / w_arr.sum()
                        cmd_action = np.sum(actions_arr * w_arr[:, None], axis=0)
                    else:
                        offset = t - pred_base_t
                        cmd_action = curr_pred_chunk[offset] if offset < chunk_size else curr_pred_chunk[-1]

                    env.apply_full_joints(cmd_action)
                    await app.next_update_async()

                    # 마일스톤 추적
                    c_mat = omni.usd.get_world_transform_matrix(env.cube_prim)
                    cube_pos = np.array(c_mat.ExtractTranslation())
                    if cube_pos[2] > max_lift_height:
                        max_lift_height = float(cube_pos[2])

                    if abs(cmd_action[0]) > np.radians(25.0) and t < 250:
                        trial_scan = True
                    if max_lift_height > 0.040:
                        trial_grasp = True
                    if trial_grasp and cube_pos[1] < -0.10:
                        trial_trans = True

            # 최종 판정
            in_box, final_cube_pos, box_pos, dx, dy = env.check_cube_in_box()
            q_final = env.robot_articulation.get_joint_positions()
            trial_return = False
            if q_final is not None and len(q_final) >= 4:
                pose_err = np.linalg.norm(q_final[1:4] - OBSERVATION_POSE[1:4])
                if pose_err < 0.35:
                    trial_return = True

            if trial_scan or (not is_blind):
                scan_count += 1
            if trial_grasp:
                grasp_count += 1
            if trial_trans:
                trans_count += 1
            if in_box:
                success_count += 1
                if is_blind:
                    blind_success += 1
                else:
                    front_success += 1
            if trial_return:
                return_count += 1

            status_str = "SUCCESS 🏆" if in_box else "FAILED ❌"
            zone_str = "BLIND" if is_blind else "FRONT"
            print(f"  [{idx+1:02d}/{total_trials:02d}] R={r_val:.2f}m, Theta={theta_deg:+5.1f}° ({zone_str:5s}) -> {status_str} | Lift: {max_lift_height*100:4.1f}cm | Box dX={dx*100:4.1f}cm, dY={dy*100:4.1f}cm | Return={trial_return}")

            benchmark_records.append({
                "trial": idx + 1,
                "r_m": r_val,
                "theta_deg": theta_deg,
                "x_m": tx,
                "y_m": ty,
                "is_blind_spot": is_blind,
                "is_scan": trial_scan,
                "is_grasp": trial_grasp,
                "is_transport": trial_trans,
                "is_place": in_box,
                "is_return_home": trial_return,
                "dx_cm": dx * 100.0,
                "dy_cm": dy * 100.0,
                "max_lift_cm": max_lift_height * 100.0,
                "status": "SUCCESS" if in_box else "FAILED",
            })

        os.makedirs(os.path.dirname(CSV_REPORT_PATH), exist_ok=True)
        final_csv_path = CSV_REPORT_PATH
        if os.path.exists(final_csv_path):
            stem, ext = os.path.splitext(final_csv_path)
            ver = 2
            while os.path.exists(f"{stem}_{ver}{ext}"):
                ver += 1
            final_csv_path = f"{stem}_{ver}{ext}"
        with open(final_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "trial", "r_m", "theta_deg", "x_m", "y_m", "is_blind_spot",
                "is_scan", "is_grasp", "is_transport", "is_place", "is_return_home",
                "dx_cm", "dy_cm", "max_lift_cm", "status"
            ])
            writer.writeheader()
            writer.writerows(benchmark_records)

        total_time = time.time() - t_bench_start
        overall_rate = (success_count / total_trials) * 100.0 if total_trials > 0 else 0.0
        blind_rate = (blind_success / blind_total) * 100.0 if blind_total > 0 else 0.0
        front_rate = (front_success / front_total) * 100.0 if front_total > 0 else 0.0

        print("\n" + "=" * 84)
        print(">>> [SO-101 ACTIVE SEARCH BENCHMARK REPORT SUMMARY]")
        print(f"  • Total Evaluated:     {total_trials} trials ({total_time/60:.1f} minutes)")
        print(f"  • Overall Success:     {success_count}/{total_trials} ({overall_rate:.1f}%) 🏆")
        print(f"  • Blind-Spot (|θ|>=25°):{blind_success}/{blind_total} ({blind_rate:.1f}%)")
        print(f"  • Front-View (|θ|<25°): {front_success}/{front_total} ({front_rate:.1f}%)")
        print("-" * 84)
        print(f"  [Milestone Funnel]")
        print(f"    1) Active Scan Initiated:   {scan_count:2d}/{total_trials} ({(scan_count/total_trials)*100:.1f}%)")
        print(f"    2) Grasp & Lift (>4cm):     {grasp_count:2d}/{total_trials} ({(grasp_count/total_trials)*100:.1f}%)")
        print(f"    3) Box Transport Reached:   {trans_count:2d}/{total_trials} ({(trans_count/total_trials)*100:.1f}%)")
        print(f"    4) TargetBox In-Box Placed: {success_count:2d}/{total_trials} ({overall_rate:.1f}%)")
        print(f"    5) Safe Return to Home:     {return_count:2d}/{total_trials} ({(return_count/total_trials)*100:.1f}%)")
        print(f"  • CSV Report Saved: {CSV_REPORT_PATH}")
        print("=" * 84 + "\n")

    except asyncio.CancelledError:
        print("[CLEANUP] Benchmark task was cancelled.")
    except Exception as e:
        print(f"[ERROR in run_benchmark]: {e}")
        import traceback
        traceback.print_exc()


def main():
    global _active_bench_task
    if is_kit_gui:
        _active_bench_task = asyncio.ensure_future(run_benchmark())
    else:
        task = asyncio.ensure_future(run_benchmark())
        while not task.done() and sim_app.is_running():
            sim_app.update()
        sim_app.close()


if __name__ == "__main__" or is_kit_gui:
    main()

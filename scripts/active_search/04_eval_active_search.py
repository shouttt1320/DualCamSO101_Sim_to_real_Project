#!/usr/bin/env python3
"""
[Step 04] SO-101 능동 탐색 (Active Search) DINOv2-ACT 단일 폐루프 추론 스크립트
=============================================================================
사각지대(Blind Spot) 또는 작업대 임의의 위치에 놓인 큐브를
학습된 DINOv2-ACT 신경망 정책만으로 자율 탐색(Scan) -> 정렬(Gaze) -> 파지(Grasp) -> 투하(Place) -> 복귀(Return)
전 과정을 실시간 폐루프(Closed-Loop)로 시뮬레이션 평가합니다.

[실행 방법 2가지]
  1) Isaac Sim GUI Script Editor:
     - Isaac Sim 상단 메뉴 [Window] -> [Script Editor] 열기
     - 본 파일 전체를 복사하여 붙여넣고 [Run (Ctrl+Enter)] 클릭
     - 코드 상단의 GUI 설정값(GUI_CUBE_THETA, GUI_CUBE_R 등)을 바꿔가며 자유롭게 테스트 가능

  2) 터미널 실행 (Standalone CLI):
     bash scripts/active_search/04_eval_active_search.sh --cube_theta -45.0 --cube_r 0.31
"""

import os
import sys
import argparse
import asyncio
import time
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms

# -----------------------------------------------------------------------------
# [GUI 스크립트 에디터 전용 설정] 에디터에서 실행할 때 여기서 큐브 위치 조절 가능
# -----------------------------------------------------------------------------
GUI_ENV = "domain_rand"  # 환경 프리셋 (default_clean 등)
GUI_CUBE_R = 0.310         # 큐브 원점 거리 R [m] (0.28m ~ 0.34m)
GUI_CUBE_THETA = -4.0     # 큐브 각도 Theta [deg] (-60° ~ +60°, 사각지대 테스트 권장)
GUI_CUBE_YAW = 0.0         # 큐브 Z축 회전각 [deg]
GUI_MAX_TIMESTEPS = 1150   # 최대 추론 스텝 수 (~23초, 충분한 스캔/복귀 보장)
GUI_INFERENCE_INTERVAL = 2 # 비전 신경망 순전파 주기 (2스텝 = 0.04초)

# -----------------------------------------------------------------------------
# 1. 실행 환경 감지 (GUI Script Editor vs Standalone CLI) 및 경로 설정
# -----------------------------------------------------------------------------
is_kit_gui = "omni.kit.app" in sys.modules

# GUI 스크립트 에디터에서 임시 파일(/tmp/...)로 실행되는 경우를 대비한 절대 경로 폴백
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
    parser = argparse.ArgumentParser(description="SO-101 Active Search Policy Evaluation")
    parser.add_argument("--headless", action="store_true", default=False, help="GUI 창 없이 백그라운드 고속 실행")
    parser.add_argument("--env", type=str, default="default_clean", help=f"환경 프리셋 (선택: {list(ENV_REGISTRY.keys())})")
    parser.add_argument("--usd", type=str, default=None, help="커스텀 USD 경로 (미지정시 --env 프리셋 자동 로드)")
    parser.add_argument("--cube_r", type=float, default=0.300, help="큐브 원점 거리 R [m] (0.28 ~ 0.34)")
    parser.add_argument("--cube_theta", type=float, default=-45.0, help="큐브 각도 Theta [deg] (-60 ~ +60, 사각지대 테스트 권장)")
    parser.add_argument("--cube_yaw", type=float, default=0.0, help="큐브 Z축 회전 각도 [deg]")
    parser.add_argument("--cube_x", type=float, default=None, help="큐브 X 좌표 직접 지정 [m] (지정시 r, theta 무시)")
    parser.add_argument("--cube_y", type=float, default=None, help="큐브 Y 좌표 직접 지정 [m] (지정시 r, theta 무시)")
    parser.add_argument("--ckpt", type=str, default=None, help="체크포인트 경로")
    parser.add_argument("--stats", type=str, default=None, help="정규화 통계 경로")
    parser.add_argument("--max_timesteps", type=int, default=1150, help="최대 추론 타임스텝 수 (기본: 1150, 약 23초)")
    parser.add_argument("--inference_interval", type=int, default=2, help="신경망 추론 주기 (스텝 단위, 기본: 2)")

    args, _ = parser.parse_known_args()

    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": args.headless})
    import omni.usd
    target_usd = args.usd if args.usd else ENV_REGISTRY[args.env]["usd_path"]
    print(f"[STAGE] Loading USD: {target_usd}")
    omni.usd.get_context().open_stage(target_usd)
else:
    # GUI Script Editor 실행 시 기본 설정값 적용
    class Args:
        pass
    args = Args()
    args.env = GUI_ENV
    args.usd = None
    args.cube_r = GUI_CUBE_R
    args.cube_theta = GUI_CUBE_THETA
    args.cube_yaw = GUI_CUBE_YAW
    args.cube_x = None
    args.cube_y = None
    args.ckpt = None
    args.stats = None
    args.max_timesteps = GUI_MAX_TIMESTEPS
    args.inference_interval = GUI_INFERENCE_INTERVAL

# 좌표 및 경로 계산
if args.cube_x is not None and args.cube_y is not None:
    test_x = args.cube_x
    test_y = args.cube_y
    test_r = float(np.hypot(test_x, test_y))
    test_theta = float(np.degrees(np.arctan2(test_y, test_x)))
else:
    test_r = args.cube_r
    test_theta = args.cube_theta
    th_rad = np.radians(test_theta)
    test_x = float(test_r * np.cos(th_rad))
    test_y = float(test_r * np.sin(th_rad))

ENV_NAME = args.env
CKPT_ENV = ENV_REGISTRY.get(ENV_NAME, {}).get("ckpt_env", ENV_NAME)
DEFAULT_CKPT_DIR = os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", CKPT_ENV)
CKPT_PATH = args.ckpt if args.ckpt else os.path.join(DEFAULT_CKPT_DIR, "policy_best.pth")
STATS_PATH = args.stats if args.stats else os.path.join(DEFAULT_CKPT_DIR, "dataset_stats.pkl")

# -----------------------------------------------------------------------------
# 3. GUI Script Editor 중복 실행 방지 및 정지 이벤트 리스너
# -----------------------------------------------------------------------------
import omni.timeline
import omni.kit.app

timeline = omni.timeline.get_timeline_interface()

global _active_eval_task, _eval_stop_sub
if "_active_eval_task" in globals() and _active_eval_task and not _active_eval_task.done():
    _active_eval_task.cancel()
    print("[CLEANUP] 이전 실행 중이던 비전 추론 태스크를 취소했습니다.")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        if "_active_eval_task" in globals() and _active_eval_task and not _active_eval_task.done():
            _active_eval_task.cancel()
            print("[CLEANUP] 시뮬레이션 정지 버튼 감지: 추론 태스크를 중단했습니다.")

_eval_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(_on_timeline_stop)

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

# -----------------------------------------------------------------------------
# 4. 모델 로드 함수
# -----------------------------------------------------------------------------
def load_active_search_policy(ckpt_path, stats_path):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] Policy checkpoint not found: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"[ERROR] Dataset stats not found: {stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[POLICY] Loading Active Search DINOv2-ACT Policy on {device}...")
    print(f"  • Weights: {ckpt_path}")
    print(f"  • Stats:   {stats_path}")

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
    print(f"  • Checkpoint Chunk Size: {actual_chunk_size} | Backbone: {ckpt_backbone}")

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


async def move_joints_smoothly(env, target_deg_or_rad, duration_sec=1.2, grip_deg=0.0):
    app = omni.kit.app.get_app()
    timeline = omni.timeline.get_timeline_interface()
    steps = max(1, int(duration_sec * 50))
    q_curr = env.robot_articulation.get_joint_positions()
    if q_curr is None or len(q_curr) < 5:
        q_start = np.zeros(5, dtype=np.float32)
    else:
        q_start = np.array(q_curr[:5], dtype=np.float32)

    target_rad = np.array(target_deg_or_rad[:5], dtype=np.float32)
    grip_rad = np.radians(grip_deg)

    for i in range(1, steps + 1):
        if not timeline.is_playing():
            return False
        alpha = 0.5 * (1.0 - np.cos(np.pi * (i / steps)))
        q_interp = q_start + alpha * (target_rad - q_start)
        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_interp
        q_full[5] = grip_rad
        env.apply_full_joints(q_full)
        await app.next_update_async()
    return True


# -----------------------------------------------------------------------------
# 5. 폐루프 정책 추론 메인 루프
# -----------------------------------------------------------------------------
async def run_eval():
    try:
        if not timeline.is_playing():
            timeline.play()

        app = omni.kit.app.get_app()
        for _ in range(25):
            await app.next_update_async()

        env = EnvironmentManager(env_name=ENV_NAME, custom_usd_path=args.usd)
        env.ensure_initialized()

        model, transform, qpos_mean, qpos_std, action_mean, action_std, device, chunk_size = load_active_search_policy(CKPT_PATH, STATS_PATH)

        # 1. 홈 관측 자세 초기화
        OBSERVATION_POSE = np.array([0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0], dtype=np.float32)
        print("\n[INIT] Moving to Initial Home Observation Pose...")
        if not await move_joints_smoothly(env, OBSERVATION_POSE, duration_sec=1.2, grip_deg=0.0):
            return

        # 2. 큐브 배치
        print(f"[SPAWN] Placing RedCube at Target: R={test_r:.3f}m, Theta={test_theta:+.1f}°, Yaw={args.cube_yaw:+.1f}° (X={test_x:.3f}, Y={test_y:.3f})")
        env.reset_cube_pose(test_x, test_y, z_val=0.015, yaw_deg=args.cube_yaw)
        for _ in range(20):
            await app.next_update_async()

        print("\n" + "=" * 82)
        print(">>> [SO-101 CLOSED-LOOP EVALUATION] Autonomous Active Search Policy Running")
        print(f"  • Max Timesteps: {args.max_timesteps} steps (~{args.max_timesteps/50:.1f}s)")
        print(f"  • Inference Interval: Every {args.inference_interval} steps")
        print("=" * 82)

        curr_pred_chunk = None
        pred_base_t = 0
        max_lift_height = 0.0
        is_approach = False
        is_grasp = False
        is_transport = False
        is_scan = False

        # 실시간 Temporal Ensembling 버퍼 초기화
        all_time_actions = np.zeros((args.max_timesteps, args.max_timesteps + chunk_size, STATE_DIM), dtype=np.float32)
        action_populated = np.zeros(args.max_timesteps, dtype=bool)

        t_start = time.time()

        with torch.no_grad():
            for t in range(args.max_timesteps):
                if not timeline.is_playing():
                    print("[INFO] Simulation stopped by user.")
                    break

                # 1) INFERENCE_INTERVAL 마다 DINOv2-ACT 신경망 순전파
                if t % args.inference_interval == 0 or curr_pred_chunk is None:
                    q_curr = env.robot_articulation.get_joint_positions()
                    if q_curr is None or len(q_curr) < 6:
                        q_curr_6 = np.zeros(6, dtype=np.float32)
                    else:
                        q_curr_6 = np.array(q_curr[:6], dtype=np.float32)

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

                # 2) Temporal Ensembling 가중합 기반 액션 결정 (지수 감쇠 스무딩)
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

                # 3) 실시간 마일스톤 추적
                c_mat = omni.usd.get_world_transform_matrix(env.cube_prim)
                cube_pos = np.array(c_mat.ExtractTranslation())

                if cube_pos[2] > max_lift_height:
                    max_lift_height = float(cube_pos[2])

                if abs(cmd_action[0]) > np.radians(25.0) and t < 250:
                    is_scan = True
                if max_lift_height > 0.040:
                    is_grasp = True
                if is_grasp and cube_pos[1] < -0.10:
                    is_transport = True

                if t > 0 and t % 100 == 0:
                    grip_deg = np.degrees(cmd_action[5])
                    print(f"  [Step {t:4d}/{args.max_timesteps}] Pan: {np.degrees(cmd_action[0]):+5.1f}° | Grip: {grip_deg:4.1f}° | Cube Z: {cube_pos[2]*100:4.1f}cm")

        elapsed = time.time() - t_start
        in_box, final_cube_pos, box_pos, dx, dy = env.check_cube_in_box()

        print("\n" + "=" * 82)
        print(">>> [SO-101 ACTIVE SEARCH EVALUATION REPORT]")
        print(f"  • Spawned Cube:     R={test_r:.3f}m, Theta={test_theta:+.1f}° -> ({test_x:.3f}, {test_y:.3f})")
        print(f"  • Final Cube Pos:   ({final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f})")
        print(f"  • Box Distance:     dX={dx*100:.1f}cm, dY={dy*100:.1f}cm")
        print(f"  • Max Lift Height:  {max_lift_height*100:.1f}cm")
        print(f"  • Milestones:       Scan={is_scan} | Grasp={is_grasp} | Transport={is_transport} | Placed={in_box}")
        print(f"  • Total Time:       {elapsed:.1f}s ({args.max_timesteps} steps)")
        print("-" * 82)
        if in_box:
            print(">>> 🏆 [MISSION SUCCESS!] Model successfully detected, grasped, and placed cube in TargetBox!")
        else:
            print(">>> ❌ [MISSION FAILED] Cube was not placed inside the box.")
        print("=" * 82 + "\n")

    except asyncio.CancelledError:
        print("[CLEANUP] Policy evaluation task was cancelled.")
    except Exception as e:
        print(f"[ERROR in run_eval]: {e}")
        import traceback
        traceback.print_exc()


def main():
    global _active_eval_task
    if is_kit_gui:
        _active_eval_task = asyncio.ensure_future(run_eval())
    else:
        task = asyncio.ensure_future(run_eval())
        while not task.done() and sim_app.is_running():
            sim_app.update()
        sim_app.close()


if __name__ == "__main__" or is_kit_gui:
    main()

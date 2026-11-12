#!/usr/bin/env python3
"""
[Step 05] SO-101 Sim-to-Real: Closed-Loop Vision-Guided ACT Policy Deployment in Isaac Sim
================================================================================
[핵심 기능]
1. 하드코딩된 Lula IK 없이, 순수 학습된 DINOv2-ACT 신경망 정책 기반 실시간 자율 제어
2. 양안 카메라(cam_left, cam_right) 실시간 캡처 -> DINOv2 Bicubic (224x294) 전처리
3. CVAE Latent z=0 고정 결정론적 추론 + Temporal Ensembling (지수 가중치 결합)
4. INFERENCE_INTERVAL=10 (또는 2) 주기로 추론하여 50FPS 실시간 제어 달성

[실행 방법]
  1) Isaac Sim GUI Script Editor:
     - Isaac Sim 상단 메뉴 [Window] -> [Script Editor] 열기
     - 본 스크립트 내용을 붙여넣고 [Run (Ctrl+Enter)] 클릭
     - 코드 상단의 GUI_TEST_CUBE_X, GUI_TEST_CUBE_Y 좌표를 수정하여 임의 위치 테스트 가능

  2) 터미널 Standalone 실행:
     bash scripts/05_eval_sim_policy.sh
     # 또는 특정 위치 지정:
     bash scripts/05_eval_sim_policy.sh --cube_x 0.320 --cube_y -0.050

[CLI 주요 옵션]
  --headless   : 창 없이 백그라운드 고속 실행 (기본값: False)
  --usd        : 시뮬레이션 USD 스테이지 경로 (기본값: dual_cam_so101_redcube.usd)
  --cube_x     : 테스트 큐브 초기 X 좌표 [m] (기본값: 0.310, 범위: 0.28 ~ 0.34)
  --cube_y     : 테스트 큐브 초기 Y 좌표 [m] (기본값: 0.000, 범위: -0.10 ~ +0.10)
  --ckpt       : 사용할 가중치 파일명 또는 절대경로 (기본값: policy_best.pth)
"""

import os
import sys

# -----------------------------------------------------------------------------
# 0. PyTorch 번들 경로 및 CUDA 라이브러리 경로 최우선 설정 (torch 임포트 전)
# -----------------------------------------------------------------------------
WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
PROJECT_ROOT = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project")
SCRIPTS_DIR = os.path.join(WORKSPACE_DIR, "scripts")
NV_DIR = "/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR = "/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

for p in [
    ML_DIR,
    "/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle",
    WORKSPACE_DIR,
    PROJECT_ROOT,
    os.path.join(PROJECT_ROOT, "models"),
    SCRIPTS_DIR,
]:
    if p not in sys.path:
        sys.path.insert(0, p)

# CUDA 라이브러리 경로 동적 추가
if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = f"{NV_DIR}/cuda_runtime/lib:{NV_DIR}/cublas/lib:{NV_DIR}/cudnn/lib:{NV_DIR}/cusolver/lib:{os.environ['LD_LIBRARY_PATH']}"

# -----------------------------------------------------------------------------
# [GUI 테스트 설정] 스크립트 에디터에서 실행할 때 큐브 위치 및 제어 주기 설정
# -----------------------------------------------------------------------------
GUI_TEST_CUBE_X = 0.310  # X 좌표 (거리 R 범위: 0.28m ~ 0.34m)
GUI_TEST_CUBE_Y = 0.000  # Y 좌표 (좌우 범위: -0.10m ~ +0.10m, 좌측이 +, 우측이 -)

# 추론 주기 (Inference Interval):
# - 1: 매 스텝 추론 (매우 느림, 연산 과부하)
# - 10: 10스텝(0.2초)마다 1회 추론 (권장: 실시간 50FPS 달성, 연산량 90% 절감)
INFERENCE_INTERVAL = 2

# Standalone (터미널 실행) vs GUI Script Editor 분기
is_kit_gui = "omni.kit.app" in sys.modules
simulation_app = None

if not is_kit_gui:
    from isaacsim.simulation_app import SimulationApp
    import argparse
    parser = argparse.ArgumentParser(description="SO-101 Policy Inference")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless without GUI")
    parser.add_argument("--usd", type=str, default=os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"))
    parser.add_argument("--cube_x", type=float, default=0.310, help="Test Cube X position (meters)")
    parser.add_argument("--cube_y", type=float, default=0.000, help="Test Cube Y position (meters)")
    parser.add_argument("--ckpt", type=str, default=None, help="Custom policy checkpoint path or filename")
    args, _ = parser.parse_known_args()

    simulation_app = SimulationApp({"headless": args.headless})
    import omni.usd
    omni.usd.get_context().open_stage(args.usd)

import asyncio
import traceback
import pickle
import numpy as np
import torch
import torchvision.transforms as transforms

import omni.timeline
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, Gf

try:
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.utils.types import ArticulationAction

import importlib
import models.so101_dual_dinov2_act.config as cfg
importlib.reload(cfg)

from models.so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
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

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

# -----------------------------------------------------------------------------
# 1. 이전 실행 태스크 정리 및 정지 리스너
# -----------------------------------------------------------------------------
global _active_eval_policy_task, _eval_policy_stop_sub
for t_name in ["_active_eval_policy_task", "_active_collect_task", "_active_vision_pick_pipeline_task"]:
    if t_name in globals() and globals()[t_name] and not globals()[t_name].done():
        globals()[t_name].cancel()
        print(f"[CLEANUP] Cancelled previous task ({t_name}).")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        global _active_eval_policy_task
        if "_active_eval_policy_task" in globals() and _active_eval_policy_task and not _active_eval_policy_task.done():
            _active_eval_policy_task.cancel()
            print("[TIMELINE] Stop pressed: Closed-loop inference halted.")

_eval_policy_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
    _on_timeline_stop, name="eval_policy_stop_listener"
)

# -----------------------------------------------------------------------------
# 2. Stage Prims 및 드라이브 매핑
# -----------------------------------------------------------------------------
robot_art_prim = stage.GetPrimAtPath("/World/so101_dual_cam/Geometry/base_link")
if not (robot_art_prim.IsValid() and robot_art_prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
    for p in stage.Traverse():
        if "so101" in p.GetPath().pathString and p.HasAPI(UsdPhysics.ArticulationRootAPI):
            robot_art_prim = p
            break

cube_prim = stage.GetPrimAtPath("/World/RedCube")
box_prim = stage.GetPrimAtPath("/World/TargetBox")
left_cam_prim = stage.GetPrimAtPath(
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera"
)
right_cam_prim = stage.GetPrimAtPath(
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera"
)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
joint_drives = []
for jname in JOINT_NAMES:
    jp = stage.GetPrimAtPath(f"/World/so101_dual_cam/Physics/{jname}")
    drive = UsdPhysics.DriveAPI.Get(jp, "angular") if jp.IsValid() else None
    joint_drives.append(drive)

# -----------------------------------------------------------------------------
# 3. Replicator 카메라 Annotator (640x320)
# -----------------------------------------------------------------------------
IMG_W, IMG_H = 640, 320
rp_left = rep.create.render_product(str(left_cam_prim.GetPath()), (IMG_W, IMG_H))
rp_right = rep.create.render_product(str(right_cam_prim.GetPath()), (IMG_W, IMG_H))

annot_rgb_l = rep.AnnotatorRegistry.get_annotator("rgb")
annot_rgb_r = rep.AnnotatorRegistry.get_annotator("rgb")
annot_rgb_l.attach(rp_left)
annot_rgb_r.attach(rp_right)

robot_articulation = Articulation(str(robot_art_prim.GetPath()))
cube_rigid = RigidPrim(str(cube_prim.GetPath()))
ctrl = None

def get_rgb_frame(annotator):
    data = annotator.get_data()
    if data is None or len(data) == 0:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8).copy()
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

def reset_cube_pose(x_val, y_val, z_val=0.015):
    pos_np = np.array([float(x_val), float(y_val), float(z_val)], dtype=np.float32)
    quat_np = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    try:
        cube_rigid.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube_rigid.set_angular_velocity(np.zeros(3, dtype=np.float32))
        cube_rigid.set_world_pose(position=pos_np, orientation=quat_np)
    except Exception:
        pass

    t_attr = cube_prim.GetAttribute("xformOp:translate")
    if t_attr.IsValid():
        if "float3" in str(t_attr.GetTypeName()):
            t_attr.Set(Gf.Vec3f(float(x_val), float(y_val), float(z_val)))
        else:
            t_attr.Set(Gf.Vec3d(float(x_val), float(y_val), float(z_val)))
    o_attr = cube_prim.GetAttribute("xformOp:orient")
    if o_attr.IsValid():
        o_attr.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

def apply_full_joints(q_full_rad):
    if ctrl is not None:
        ctrl.apply_action(ArticulationAction(joint_positions=q_full_rad))
    q_deg = np.degrees(q_full_rad)
    for drive, deg in zip(joint_drives, q_deg):
        if drive:
            drive.GetTargetPositionAttr().Set(float(deg))

async def move_joints_smoothly(target_5dof, duration_sec=1.5, grip_deg=0.0):
    app = omni.kit.app.get_app()
    q_curr = robot_articulation.get_joint_positions()
    if q_curr is not None and len(q_curr) >= 5:
        q_start = np.array(q_curr[:5], dtype=np.float32)
    else:
        q_start = np.radians(np.array([d.GetTargetPositionAttr().Get() for d in joint_drives[:5]], dtype=np.float32))

    steps = max(int(duration_sec * 50), 1)
    for s in range(1, steps + 1):
        if not timeline.is_playing():
            return False
        alpha = 0.5 * (1.0 - np.cos(np.pi * (s / steps)))
        q_interp = q_start + alpha * (target_5dof - q_start)

        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_interp
        q_full[5] = np.radians(grip_deg)

        apply_full_joints(q_full)
        await app.next_update_async()
    return True

# -----------------------------------------------------------------------------
# 4. 정책 로드 및 실시간 폐루프 추론 엔진
# -----------------------------------------------------------------------------
def load_policy(ckpt_path=None, stats_path=None):
    if ckpt_path is None:
        ckpt_path = os.path.join(CKPT_DIR, "policy_best.pth")
    if stats_path is None:
        stats_path = os.path.join(CKPT_DIR, "dataset_stats.pkl")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"[ERROR] Checkpoint not found: {ckpt_path}")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"[ERROR] Stats not found: {stats_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[POLICY] Loading DINOv2-ACT model on {device}...")

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

    # 체크포인트 내부의 실제 청크 크기 자동 감지 (50 또는 100 등)
    if "query_embed.weight" in state_dict:
        actual_chunk_size = state_dict["query_embed.weight"].shape[0]
    else:
        actual_chunk_size = CHUNK_SIZE
    print(f"[POLICY] Detected Checkpoint Chunk Size: {actual_chunk_size}")

    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=actual_chunk_size,
        camera_names=CAMERA_NAMES,
        backbone_name=BACKBONE,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
        train_backbone=False,
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[POLICY] Policy loaded successfully from {ckpt_path}!")
    return model, transform, qpos_mean, qpos_std, action_mean, action_std, device, actual_chunk_size


async def run_closed_loop_eval(test_cube_xy=(0.310, 0.000), max_timesteps=700, ckpt_path=None):
    """
    학습된 DINOv2-ACT 신경망 정책만으로 자율 픽앤플레이스를 수행하는 폐루프 제어 루프.
    """
    global ctrl
    try:
        if not timeline.is_playing():
            timeline.play()

        app = omni.kit.app.get_app()
        for _ in range(20):
            await app.next_update_async()

        robot_articulation.initialize()
        cube_rigid.initialize()
        ctrl = robot_articulation.get_articulation_controller()

        # 1. 모델 로드 (체크포인트의 실제 chunk_size 동적 감지)
        model, transform, qpos_mean, qpos_std, action_mean, action_std, device, chunk_size = load_policy(ckpt_path=ckpt_path)

        # 2. 로봇을 관측 자세로 초기화 및 큐브 배치
        OBSERVATION_POSE = np.array([0.0, np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0], dtype=np.float32)
        print(f"\n[INIT] Moving to Observation Pose...")
        if not await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5, grip_deg=0.0):
            return

        print(f"[SPAWN] Placing RedCube at test position: ({test_cube_xy[0]:.3f}, {test_cube_xy[1]:.3f}, 0.015)...")
        reset_cube_pose(test_cube_xy[0], test_cube_xy[1], 0.015)
        for _ in range(25):
            await app.next_update_async()

        # 3. 실시간 Temporal Ensembling 버퍼 초기화
        all_time_actions = np.zeros((max_timesteps, max_timesteps + chunk_size, STATE_DIM), dtype=np.float32)
        weights = np.exp(-TEMPORAL_AGG_M * np.arange(chunk_size))

        print("\n" + "=" * 78)
        print(f">>> [SO-101 CLOSED-LOOP VISION INFERENCE] Autonomous Policy Control Started")
        print("=" * 78)

        box_mat = omni.usd.get_world_transform_matrix(box_prim)
        box_pos = np.array(box_mat.ExtractTranslation())

        success = False
        curr_pred_chunk = None  # (chunk_size, 6)
        pred_base_t = 0

        with torch.no_grad():
            for t in range(max_timesteps):
                if not timeline.is_playing():
                    print("[INFO] Simulation stopped by user.")
                    break

                # (1) INFERENCE_INTERVAL 스텝마다만 비전 신경망(DINOv2-ACT) 추론 실행 (연산 최적화)
                if t % INFERENCE_INTERVAL == 0 or curr_pred_chunk is None:
                    # 현재 관측값 획득 (로봇 관절 + 양안 이미지)
                    q_curr = robot_articulation.get_joint_positions()
                    if q_curr is None or len(q_curr) < 6:
                        q_curr_6 = np.zeros(6, dtype=np.float32)
                    else:
                        q_curr_6 = np.array(q_curr[:6], dtype=np.float32)

                    img_l = get_rgb_frame(annot_rgb_l)
                    img_r = get_rgb_frame(annot_rgb_r)

                    # 전처리 및 텐서 변환
                    q_tensor = torch.from_numpy(q_curr_6).float().to(device)
                    q_norm = ((q_tensor - qpos_mean) / qpos_std).unsqueeze(0)

                    cam_tensors = []
                    for img_raw in [img_l, img_r]:
                        t_img = torch.from_numpy(img_raw).permute(2, 0, 1).float() / 255.0
                        t_img = transform(t_img)
                        cam_tensors.append(t_img)
                    image_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(device)

                    # DINOv2-ACT 신경망 순전파 (CVAE Deterministic 추론)
                    a_hat, _, _ = model(qpos=q_norm, image=image_input)
                    curr_pred_chunk = a_hat[0].cpu().numpy() * action_std + action_mean  # (chunk_size, 6)
                    pred_base_t = t

                    all_time_actions[t, t:t + chunk_size] = curr_pred_chunk

                # (2) 현재 스텝 액션 결정 (앙상블 또는 최신 청크 추종)
                offset = t - pred_base_t
                if offset < chunk_size:
                    cmd_action = curr_pred_chunk[offset]
                else:
                    cmd_action = curr_pred_chunk[-1]

                # (3) 물리 드라이브 명령 인가
                apply_full_joints(cmd_action)
                await app.next_update_async()

                if t % 50 == 0 or t == max_timesteps - 1:
                    q_deg = np.degrees(cmd_action)
                    print(
                        f"  [Step {t:03d}/{max_timesteps}] Pan:{q_deg[0]:5.1f}° | Sh:{q_deg[1]:5.1f}° | "
                        f"Elb:{q_deg[2]:5.1f}° | Grip:{q_deg[5]:4.1f}°"
                    )

        # 4. 결과 검증 (상자 안착 여부)
        await asyncio.sleep(0.5)
        c_mat_final = omni.usd.get_world_transform_matrix(cube_prim)
        final_pos = np.array(c_mat_final.ExtractTranslation())
        dx_box = abs(float(final_pos[0]) - float(box_pos[0]))
        dy_box = abs(float(final_pos[1]) - float(box_pos[1]))
        in_box = (dx_box < 0.048) and (dy_box < 0.048) and (0.005 < float(final_pos[2]) < 0.060)

        print("\n" + "=" * 78)
        print(f"[EVALUATION RESULT] TargetBox Pos: ({box_pos[0]:.3f}, {box_pos[1]:.3f})")
        print(f"  └─ Final Cube Pos: ({final_pos[0]:.3f}, {final_pos[1]:.3f}, {final_pos[2]:.3f})")
        print(f"  └─ Box Distance: dX={dx_box*100:.1f}cm, dY={dy_box*100:.1f}cm")

        if in_box:
            print(">>> [SUCCESS!] DINOv2-ACT Policy autonomously completed Pick & Place inside TargetBox!")
        else:
            print(">>> [RESULT] Policy executed trajectory. Check final position.")
        print("=" * 78 + "\n")

    except asyncio.CancelledError:
        print("[CLEANUP] Inference task cancelled gracefully.")
    except Exception as e:
        print(f"\n[EXCEPTION ERROR in run_closed_loop_eval]: {e}")
        traceback.print_exc()

if simulation_app is not None:
    # Standalone 터미널 실행 모드
    test_xy = (args.cube_x, args.cube_y)
    eval_task = asyncio.ensure_future(run_closed_loop_eval(test_cube_xy=test_xy, ckpt_path=args.ckpt))
    while not eval_task.done():
        simulation_app.update()
    simulation_app.close()
else:
    # Isaac Sim GUI Script Editor 실행 모드
    _active_eval_policy_task = asyncio.ensure_future(
        run_closed_loop_eval(test_cube_xy=(GUI_TEST_CUBE_X, GUI_TEST_CUBE_Y))
    )


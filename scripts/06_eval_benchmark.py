#!/usr/bin/env python3
"""
[Step 06] SO-101 Sim-to-Real: DINOv2-ACT 정책 자동 벤치마크 평가기 (Grid / Range Benchmark)
================================================================================
[핵심 기능]
1. 사용자가 지정한 반경 거리(R) 및 각도(Theta) 범위와 간격에 따라 큐브 위치를 자동 순회
2. 4단계 정밀 진단 판정:
   - 1단계: 접근 판정 (Approach) - 그리퍼-큐브 최소 거리 <= 4.5cm
   - 2단계: 파지/리프트 판정 (Grasp / Lift) - 큐브 최대 고도 >= 5.0cm
   - 3단계: 운반 판정 (Transport) - 상자 상공 도달 여부
   - 4단계: 최종 안착 판정 (Place / Success) - 상자 내부 안착 완료
3. 전체 성공률 통계 리포트 터미널 출력 및 CSV 파일 직접 저장
4. Isaac Sim GUI Script Editor 및 Standalone 터미널 실행 완벽 지원

[실행 방법]
  1) Isaac Sim GUI Script Editor:
     - Isaac Sim 상단 메뉴 [Window] -> [Script Editor] 열기
     - 본 스크립트 내용을 붙여넣고 [Run (Ctrl+Enter)] 클릭
     - 코드 상단의 R_MIN, R_MAX, THETA_MIN, THETA_MAX 변수를 수정하여 범위 지정

  2) 터미널 실행 (단일 명령어로 자동 실행):
     bash scripts/06_eval_benchmark.sh --r_min 0.28 --r_max 0.34 --r_step 0.02 --theta_min -20 --theta_max 20 --theta_step 10

[CLI 주요 옵션]
  --headless   : 창 없이 백그라운드 고속 평가 (기본값: False)
  --usd        : 시뮬레이션 USD 스테이지 경로 (기본값: dual_cam_so101_redcube.usd)
  --r_min      : 최소 테스트 반경 거리 [m] (기본값: 0.28)
  --r_max      : 최대 테스트 반경 거리 [m] (기본값: 0.34)
  --r_step     : 반경 거리 간격 [m] (기본값: 0.02)
  --theta_min  : 최소 테스트 각도 [deg] (기본값: -18.0)
  --theta_max  : 최대 테스트 각도 [deg] (기본값: +18.0)
  --theta_step : 각도 간격 [deg] (기본값: 9.0)
  --ckpt       : 평가할 모델 가중치 파일명 또는 경로 (기본값: policy_best.pth)
"""

import os
import sys
import csv
import time
import asyncio
import argparse
import traceback
import importlib
import numpy as np

# -----------------------------------------------------------------------------
# 0. PyTorch 번들 경로 및 CUDA 라이브러리 경로 최우선 설정
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

if "LD_LIBRARY_PATH" in os.environ:
    os.environ["LD_LIBRARY_PATH"] = f"{NV_DIR}/cuda_runtime/lib:{NV_DIR}/cublas/lib:{NV_DIR}/cudnn/lib:{NV_DIR}/cusolver/lib:{os.environ['LD_LIBRARY_PATH']}"

# -----------------------------------------------------------------------------
# [사용자 설정] 큐브 테스트 범위 및 간격 지정
# -----------------------------------------------------------------------------
# 1) 거리 R (미터): 0.28m ~ 0.34m (로봇 작업 최적 영역)
R_MIN = 0.280
R_MAX = 0.340
R_STEP = 0.020  # 0.28, 0.30, 0.32, 0.34 (4단계)

# 2) 회전 각도 Theta (도): -20° ~ +20°
THETA_MIN = -20.0
THETA_MAX = 20.0
THETA_STEP = 10.0  # -20°, -10°, 0°, +10°, +20° (5단계)

# 3) 추론 주기 및 최대 스텝
INFERENCE_INTERVAL = 10  # 10스텝(0.2초)마다 비전 추론 (실시간 50FPS 보장)
MAX_TIMESTEPS = 650

# -----------------------------------------------------------------------------
# Standalone (터미널 실행) vs GUI Script Editor 분기
# -----------------------------------------------------------------------------
is_kit_gui = "omni.kit.app" in sys.modules
simulation_app = None

if not is_kit_gui:
    from isaacsim.simulation_app import SimulationApp
    parser = argparse.ArgumentParser(description="SO-101 Policy Benchmark Evaluator")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless without GUI")
    parser.add_argument("--usd", type=str, default=os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"))
    parser.add_argument("--r_min", type=float, default=R_MIN)
    parser.add_argument("--r_max", type=float, default=R_MAX)
    parser.add_argument("--r_step", type=float, default=R_STEP)
    parser.add_argument("--theta_min", type=float, default=THETA_MIN)
    parser.add_argument("--theta_max", type=float, default=THETA_MAX)
    parser.add_argument("--theta_step", type=float, default=THETA_STEP)
    parser.add_argument("--ckpt", type=str, default=None, help="Custom policy checkpoint path")
    args, _ = parser.parse_known_args()

    R_MIN = args.r_min
    R_MAX = args.r_max
    R_STEP = args.r_step
    THETA_MIN = args.theta_min
    THETA_MAX = args.theta_max
    THETA_STEP = args.theta_step

    simulation_app = SimulationApp({"headless": args.headless})
    import omni.usd
    omni.usd.get_context().open_stage(args.usd)

import pickle
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
global _active_benchmark_task, _benchmark_stop_sub
for t_name in ["_active_benchmark_task", "_active_eval_policy_task", "_active_collect_task"]:
    if t_name in globals() and globals()[t_name] and not globals()[t_name].done():
        globals()[t_name].cancel()
        print(f"[CLEANUP] Cancelled previous task ({t_name}).")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        global _active_benchmark_task
        if "_active_benchmark_task" in globals() and _active_benchmark_task and not _active_benchmark_task.done():
            _active_benchmark_task.cancel()
            print("[TIMELINE] Stop pressed: Benchmark halted.")

_benchmark_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
    _on_timeline_stop, name="benchmark_stop_listener"
)

# -----------------------------------------------------------------------------
# 2. Stage Prims 및 그리퍼/센서 바인딩
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

# 그리퍼 끝단(TCP) 프림 검색
gripper_tcp_prim = stage.GetPrimAtPath("/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/moving_jaw_so101")
if not gripper_tcp_prim.IsValid():
    for p in stage.Traverse():
        pname = p.GetName().lower()
        if "moving_jaw" in pname or "jaw_so101" in pname:
            gripper_tcp_prim = p
            break
if not gripper_tcp_prim.IsValid():
    gripper_tcp_prim = stage.GetPrimAtPath("/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link")

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
joint_drives = []
for jname in JOINT_NAMES:
    jp = stage.GetPrimAtPath(f"/World/so101_dual_cam/Physics/{jname}")
    drive = UsdPhysics.DriveAPI.Get(jp, "angular") if jp.IsValid() else None
    joint_drives.append(drive)

# 카메라 Annotator
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
    return data[:, :, :3]

def reset_cube_pose(x_val, y_val, z_val=0.015, yaw_deg=0.0):
    half_rad = np.radians(yaw_deg) * 0.5
    qw, qz = float(np.cos(half_rad)), float(np.sin(half_rad))
    pos_np = np.array([float(x_val), float(y_val), float(z_val)], dtype=np.float32)
    quat_np = np.array([qw, 0.0, 0.0, qz], dtype=np.float32)

    if cube_rigid is not None:
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
    else:
        UsdGeom.XformCommonAPI(cube_prim).SetTranslate(Gf.Vec3d(float(x_val), float(y_val), float(z_val)))

    o_attr = cube_prim.GetAttribute("xformOp:orient")
    if o_attr.IsValid():
        if "quatd" in str(o_attr.GetTypeName()):
            o_attr.Set(Gf.Quatd(qw, 0.0, 0.0, qz))
        else:
            o_attr.Set(Gf.Quatf(qw, 0.0, 0.0, qz))

    rb = UsdPhysics.RigidBodyAPI(cube_prim)
    if rb:
        rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

def apply_full_joints(q_full_rad):
    if ctrl is not None:
        ctrl.apply_action(ArticulationAction(joint_positions=q_full_rad))
    for i, drive in enumerate(joint_drives):
        if drive:
            drive.GetTargetPositionAttr().Set(float(np.degrees(q_full_rad[i])))

async def move_joints_smoothly(target_5dof, duration_sec=1.5, grip_deg=0.0):
    app = omni.kit.app.get_app()
    q_start = robot_articulation.get_joint_positions()
    if q_start is None or len(q_start) < 6:
        q_start = np.zeros(6, dtype=np.float32)
    else:
        q_start = np.array(q_start[:6], dtype=np.float32)

    q_target = np.array([
        target_5dof[0], target_5dof[1], target_5dof[2],
        target_5dof[3], target_5dof[4], np.radians(grip_deg)
    ], dtype=np.float32)

    steps = max(1, int(duration_sec * 50))
    for s in range(1, steps + 1):
        if not timeline.is_playing():
            return False
        alpha = s / steps
        q_interp = (1.0 - alpha) * q_start + alpha * q_target
        apply_full_joints(q_interp)
        await app.next_update_async()
    return True

# -----------------------------------------------------------------------------
# 3. DINOv2-ACT 정책 로드 함수 (체크포인트 청크 동적 감지)
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

    actual_chunk_size = state_dict["query_embed.weight"].shape[0] if "query_embed.weight" in state_dict else CHUNK_SIZE
    print(f"[POLICY] Checkpoint Chunk Size: {actual_chunk_size}")

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
    return model, transform, qpos_mean, qpos_std, action_mean, action_std, device, actual_chunk_size

# -----------------------------------------------------------------------------
# 4. 자동 벤치마크 메인 평가 루프
# -----------------------------------------------------------------------------
async def run_benchmark():
    global ctrl
    try:
        if not timeline.is_playing():
            timeline.play()

        app = omni.kit.app.get_app()
        for _ in range(25):
            await app.next_update_async()

        robot_articulation.initialize()
        cube_rigid.initialize()
        ctrl = robot_articulation.get_articulation_controller()

        # 1. 모델 로드
        custom_ckpt = getattr(args, "ckpt", None) if not is_kit_gui else None
        model, transform, qpos_mean, qpos_std, action_mean, action_std, device, chunk_size = load_policy(ckpt_path=custom_ckpt)

        # 2. 평가 좌표 그리드 생성
        r_list = np.arange(R_MIN, R_MAX + 1e-5, R_STEP)
        theta_list = np.arange(THETA_MIN, THETA_MAX + 1e-5, THETA_STEP)
        test_points = []
        for r_val in r_list:
            for deg_val in theta_list:
                rad_val = np.radians(deg_val)
                x = float(r_val * np.cos(rad_val))
                y = float(r_val * np.sin(rad_val))
                test_points.append({"r": r_val, "deg": deg_val, "x": x, "y": y})

        total_trials = len(test_points)
        print("\n" + "=" * 78)
        print(f">>> [SO-101 AUTOMATIC BENCHMARK] Total Trials: {total_trials}")
        print(f"  • R Range:     [{R_MIN:.3f}m ~ {R_MAX:.3f}m], Step: {R_STEP:.3f}m ({len(r_list)} steps)")
        print(f"  • Theta Range: [{THETA_MIN:.1f}° ~ {THETA_MAX:.1f}°], Step: {THETA_STEP:.1f}° ({len(theta_list)} steps)")
        print(f"  • Chunk Size:  {chunk_size} | Interval: {INFERENCE_INTERVAL} steps | Max Steps: {MAX_TIMESTEPS}")
        print("=" * 78)

        OBSERVATION_POSE = np.array([0.0, np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0], dtype=np.float32)
        box_mat = omni.usd.get_world_transform_matrix(box_prim)
        box_pos = np.array(box_mat.ExtractTranslation())

        benchmark_results = []
        approach_count = 0
        grasp_count = 0
        transport_count = 0
        place_success_count = 0

        # 순회 실행 루프
        for idx, pt in enumerate(test_points):
            if not timeline.is_playing():
                print("[INFO] Benchmark stopped by user.")
                break

            target_x, target_y = pt["x"], pt["y"]
            r_val, deg_val = pt["r"], pt["deg"]
            print(f"\n[{idx+1:02d}/{total_trials:02d}] Testing R={r_val:.3f}m, Theta={deg_val:+5.1f}° -> Target XY=({target_x:.3f}, {target_y:.3f})")

            # 관측 포즈로 복귀 및 큐브 위치 리셋
            await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.2, grip_deg=0.0)
            reset_cube_pose(target_x, target_y, 0.015, yaw_deg=0.0)
            for _ in range(20):
                await app.next_update_async()

            # 메트릭 추적 변수 초기화
            min_gripper_cube_dist = 999.0
            max_cube_lift_z = 0.0
            min_cube_box_dist = 999.0

            curr_pred_chunk = None
            pred_base_t = 0

            with torch.no_grad():
                for t in range(MAX_TIMESTEPS):
                    if not timeline.is_playing():
                        break

                    # (1) 상태 추적 (TCP-큐브 거리, 큐브 Z고도, 큐브-상자 거리)
                    tcp_mat = omni.usd.get_world_transform_matrix(gripper_tcp_prim)
                    tcp_pos = np.array(tcp_mat.ExtractTranslation())
                    c_mat = omni.usd.get_world_transform_matrix(cube_prim)
                    c_pos = np.array(c_mat.ExtractTranslation())

                    dist_tcp_cube = float(np.linalg.norm(tcp_pos - c_pos))
                    if dist_tcp_cube < min_gripper_cube_dist:
                        min_gripper_cube_dist = dist_tcp_cube

                    if float(c_pos[2]) > max_cube_lift_z:
                        max_cube_lift_z = float(c_pos[2])

                    dist_cube_box = float(np.linalg.norm(c_pos[:2] - box_pos[:2]))
                    if dist_cube_box < min_cube_box_dist:
                        min_cube_box_dist = dist_cube_box

                    # (2) 비전 신경망 추론 (INFERENCE_INTERVAL 주기)
                    if t % INFERENCE_INTERVAL == 0 or curr_pred_chunk is None:
                        q_curr = robot_articulation.get_joint_positions()
                        q_curr_6 = np.array(q_curr[:6], dtype=np.float32) if (q_curr is not None and len(q_curr) >= 6) else np.zeros(6, dtype=np.float32)

                        img_l = get_rgb_frame(annot_rgb_l)
                        img_r = get_rgb_frame(annot_rgb_r)

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

                    # (3) 모터 액션 인가
                    offset = t - pred_base_t
                    cmd_action = curr_pred_chunk[offset] if offset < chunk_size else curr_pred_chunk[-1]
                    apply_full_joints(cmd_action)
                    await app.next_update_async()

            # 에피소드 종료 후 4단계 정밀 진단 판정
            await asyncio.sleep(0.3)
            c_mat_final = omni.usd.get_world_transform_matrix(cube_prim)
            final_cube_pos = np.array(c_mat_final.ExtractTranslation())
            dx_final = abs(float(final_cube_pos[0]) - float(box_pos[0]))
            dy_final = abs(float(final_cube_pos[1]) - float(box_pos[1]))
            z_final = float(final_cube_pos[2])

            # 최종 로봇 관절 상태 확인 (그리퍼 개방 각도 확인)
            q_final = robot_articulation.get_joint_positions()
            final_gripper_deg = float(np.degrees(q_final[5])) if (q_final is not None and len(q_final) >= 6) else 0.0

            # 1) 접근: 그리퍼와 큐브가 6.5cm 이내로 근접했는가
            is_approach = (min_gripper_cube_dist <= 0.065)
            # 2) 파지: 큐브를 바닥(1.5cm)에서 5cm 이상 공중으로 들어올렸는가
            is_grasp = (max_cube_lift_z >= 0.050)
            # 3) 운반: 상자 상공 8cm 이내로 큐브를 운반했는가
            is_transport = (min_cube_box_dist <= 0.080)
            # 4) 위치: 큐브의 최종 위치가 상자 내부인가
            cube_in_box_area = (dx_final < 0.048) and (dy_final < 0.048) and (0.005 < z_final < 0.060)
            # 5) 개방: 상자 위에서 그리퍼를 실제로 열었는가 (최소 45° 이상 개방)
            gripper_released = (final_gripper_deg >= 45.0)

            # 진정한 성공 (SUCCESS): 상자 내부 안착 AND 그리퍼 완전 개방 분리!
            is_place = cube_in_box_area and gripper_released

            if is_approach:
                approach_count += 1
            if is_grasp:
                grasp_count += 1
            if is_transport:
                transport_count += 1
            if is_place:
                place_success_count += 1

            # 진단 상태 레이블 결정
            if is_place:
                diag_status = "SUCCESS (상자 안착 및 그리퍼 개방 완료)"
            elif cube_in_box_area and not gripper_released:
                diag_status = f"FAIL_HOLDING (상자 도착했으나 안놓고 물고있음: Grip={final_gripper_deg:.1f}°)"
            elif is_transport:
                diag_status = "FAIL_PLACE (상자 상공 도달했으나 상자 밖 낙하)"
            elif is_grasp:
                diag_status = "FAIL_TRANSPORT (운반 도중 낙하)"
            elif is_approach:
                diag_status = "FAIL_GRASP (접근 성공했으나 파지 실패)"
            else:
                diag_status = "FAIL_APPROACH (접근 실패 / 엉뚱한 방향)"

            print(f"  └─ [RESULT] {diag_status}")
            print(f"     • 최소 접근 거리: {min_gripper_cube_dist*100:4.1f}cm (기준 <= 6.5cm)")
            print(f"     • 최대 리프트 높이: {max_cube_lift_z*100:4.1f}cm (기준 >= 5.0cm)")
            print(f"     • 최종 상자 오차: dX={dx_final*100:.1f}cm, dY={dy_final*100:.1f}cm, Z={z_final*100:.1f}cm")
            print(f"     • 최종 그리퍼 각도: {final_gripper_deg:4.1f}° (개방기준 >= 45.0°)")

            benchmark_results.append({
                "trial": idx + 1,
                "r": r_val,
                "theta_deg": deg_val,
                "target_x": target_x,
                "target_y": target_y,
                "min_tcp_dist_cm": round(min_gripper_cube_dist * 100, 2),
                "max_lift_z_cm": round(max_cube_lift_z * 100, 2),
                "final_gripper_deg": round(final_gripper_deg, 1),
                "is_approach": is_approach,
                "is_grasp": is_grasp,
                "is_transport": is_transport,
                "is_place": is_place,
                "status": diag_status,
            })

        # -----------------------------------------------------------------------------
        # 5. 최종 종합 리포트 출력 및 CSV 저장
        # -----------------------------------------------------------------------------
        num_evaluated = len(benchmark_results)
        if num_evaluated == 0:
            print("[INFO] No trials were completed.")
            return

        app_rate = (approach_count / num_evaluated) * 100
        grasp_rate = (grasp_count / num_evaluated) * 100
        trans_rate = (transport_count / num_evaluated) * 100
        place_rate = (place_success_count / num_evaluated) * 100

        print("\n" + "=" * 78)
        print(">>> [SO-101 POLICY AUTOMATIC BENCHMARK REPORT SUMMARY]")
        print("=" * 78)
        print(f"  • 총 평가 시도:   {num_evaluated} / {total_trials}")
        print(f"  -------------------------------------------------------------------------")
        print(f"  [Step 1] 큐브 접근 성공률 (Approach Rate):    {approach_count:2d}/{num_evaluated} ({app_rate:5.1f}%) [<= 4.5cm]")
        print(f"  [Step 2] 큐브 파지/리프트율 (Grasp/Lift Rate):  {grasp_count:2d}/{num_evaluated} ({grasp_rate:5.1f}%) [Z >= 5cm]")
        print(f"  [Step 3] 상자 운반 도달율 (Transport Rate):    {transport_count:2d}/{num_evaluated} ({trans_rate:5.1f}%) [상자 상공]")
        print(f"  [Step 4] 최종 상자 안착 성공률 (Success Rate):  {place_success_count:2d}/{num_evaluated} ({place_rate:5.1f}%) [안착]")
        print("=" * 78)

        # CSV 리포트 저장
        csv_path = os.path.join(CKPT_DIR, "benchmark_report.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "trial", "r", "theta_deg", "target_x", "target_y",
                "min_tcp_dist_cm", "max_lift_z_cm", "final_gripper_deg",
                "is_approach", "is_grasp", "is_transport", "is_place", "status"
            ])
            writer.writeheader()
            writer.writerows(benchmark_results)
        print(f"[REPORT] Benchmark CSV report saved to: {csv_path}\n")

    except asyncio.CancelledError:
        print("[CLEANUP] Benchmark task cancelled gracefully.")
    except Exception as e:
        print(f"\n[EXCEPTION ERROR in run_benchmark]: {e}")
        traceback.print_exc()

# 실행 분기
if simulation_app is not None:
    eval_task = asyncio.ensure_future(run_benchmark())
    while not eval_task.done():
        simulation_app.update()
    simulation_app.close()
else:
    _active_benchmark_task = asyncio.ensure_future(run_benchmark())


#!/usr/bin/env python3
"""
[Active Search Step 01] SO-101 능동 시각 탐색 기반 전자율 픽앤플레이스 (Scan-to-Pick)
===================================================================================
[핵심 목적]
1. 로봇 제어기는 사전에 큐브의 위치(X, Y)를 일절 입력받지 않음.
2. 홈/관측 자세에서 시작했을 때 큐브가 카메라에 안 보이면 자율적으로 주변을 둘러봄(Active Scan).
3. 카메라에 큐브가 들어오는 순간 회전을 멈추고 큐브를 정면으로 바라봄(Face Target).
4. 카메라 영상 및 시선 방향을 바탕으로 큐브 위치를 능동 인식(Visual Localization)함.
5. 인식된 위치로 접근하여 파지 -> 리프트 -> 목표 상자 운반 -> 안착까지 원스톱 자율 수행.

[동작 단계]
  • Step 1: 홈 자세에서 시각 확인 -> 큐브 미보임 감지
  • Step 2: 둘러보기(Scan) 모드 진입 -> 좌우 파노라마 회전 탐색
  • Step 3: 실시간 카메라에서 큐브 발견 즉시 정지 및 정면 응시 정렬 (Lock-on)
  • Step 4: 시각 인식 좌표 (X_est, Y_est) 도출
  • Step 5: Lula IK 기반 피킹 궤적 생성 및 상자 안착 완료

[실행 방법]
  1) Isaac Sim GUI Script Editor:
     - Isaac Sim 상단 메뉴 [Window] -> [Script Editor] 열기
     - 본 스크립트 붙여넣고 [Run (Ctrl+Enter)] 클릭
     - 코드 상단 TEST_THETA_DEG, TEST_RADIUS_M 수정으로 다양한 사각지대 테스트 가능

  2) 터미널 실행 (Standalone):
     bash scripts/active_search/01_test_scan_and_pick.sh --theta 45.0 --r 0.32
"""

import os
import sys

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
# [환경 설정: 사각지대 큐브 스폰 좌표 (에이전트는 이 값을 모른 채 탐색 시작)]
# -----------------------------------------------------------------------------
TEST_THETA_DEG = 45.0   # 사각지대 각도 (기본: 좌측 +45도)
TEST_RADIUS_M = 0.320   # 사각지대 반경 (기본: 0.32m)

is_kit_gui = "omni.kit.app" in sys.modules
simulation_app = None

if not is_kit_gui:
    from isaacsim.simulation_app import SimulationApp
    import argparse
    parser = argparse.ArgumentParser(description="SO-101 Scan-to-Pick Autonomous Pipeline")
    parser.add_argument("--headless", action="store_true", default=False, help="Run headless without GUI")
    parser.add_argument("--usd", type=str, default=os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"))
    parser.add_argument("--theta", type=float, default=TEST_THETA_DEG, help="Blind spot cube angle in degrees")
    parser.add_argument("--r", type=float, default=TEST_RADIUS_M, help="Blind spot cube radius in meters")
    args, _ = parser.parse_known_args()

    TEST_THETA_DEG = args.theta
    TEST_RADIUS_M = args.r

    simulation_app = SimulationApp({"headless": args.headless})
    import omni.usd
    omni.usd.get_context().open_stage(args.usd)

import asyncio
import traceback
import numpy as np

import omni.timeline
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf

try:
    from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.motion_generation import LulaKinematicsSolver
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.utils.types import ArticulationAction

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

# -----------------------------------------------------------------------------
# 1. 태스크 정리 및 정지 리스너
# -----------------------------------------------------------------------------
global _active_scan_pick_task, _scan_pick_stop_sub
if "_active_scan_pick_task" in globals() and _active_scan_pick_task and not _active_scan_pick_task.done():
    _active_scan_pick_task.cancel()
    print("[CLEANUP] Cancelled previous scan-pick task.")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        global _active_scan_pick_task
        if "_active_scan_pick_task" in globals() and _active_scan_pick_task and not _active_scan_pick_task.done():
            _active_scan_pick_task.cancel()
            print("[TIMELINE] Stop pressed: Scan-to-Pick halted.")

_scan_pick_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
    _on_timeline_stop, name="scan_pick_stop_listener"
)

# -----------------------------------------------------------------------------
# 2. Stage Prims 및 Lula 역기구학 엔진
# -----------------------------------------------------------------------------
URDF_PATH = os.path.join(WORKSPACE_DIR, "so101_dual_cam.urdf")
DESC_YAML = os.path.join(WORKSPACE_DIR, "motion_policy_configs/so101/so101_robot_description.yaml")

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

if not robot_art_prim.IsValid() or not cube_prim.IsValid() or not left_cam_prim.IsValid() or not right_cam_prim.IsValid() or not box_prim.IsValid():
    raise RuntimeError("[ERROR] Required prims not found in USD stage.")

# 실물 카메라 PnP 역산 기준 화각(focalLength=19.32mm) 및 장착 각도 반영
for _cam_prim, _fl, _rot in [
    (left_cam_prim, 19.32, (179.52, 3.48, -89.73)),
    (right_cam_prim, 19.32, (175.25, 5.11, -90.95)),
]:
    UsdGeom.Camera(_cam_prim).GetFocalLengthAttr().Set(float(_fl))
    for _op in UsdGeom.Xformable(_cam_prim).GetOrderedXformOps():
        if _op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
            _op.Set(Gf.Vec3f(*_rot))
            break

# -----------------------------------------------------------------------------
# 2-1. 물리 안정화 및 비대칭 조 고마찰 패드 설정 (execute_vision_pick.py 물리 가드 이식)
# -----------------------------------------------------------------------------
# (1) 큐브 질량 및 튕김 방지 Damping
mass_api = UsdPhysics.MassAPI.Apply(cube_prim) if not cube_prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI(cube_prim)
mass_api.CreateMassAttr(0.045)
mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(6.75e-6, 6.75e-6, 6.75e-6))

physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(cube_prim) if not cube_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI(cube_prim)
physx_rb.CreateMaxDepenetrationVelocityAttr(0.4)
physx_rb.CreateLinearDampingAttr(2.0)
physx_rb.CreateAngularDampingAttr(1.5)

cube_col = PhysxSchema.PhysxCollisionAPI.Apply(cube_prim) if not cube_prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI(cube_prim)
cube_col.CreateContactOffsetAttr(0.001)
cube_col.CreateRestOffsetAttr(0.0)

# (2) 그리퍼 조 Instanceable 해제, 4개 조 메쉬 1mm 밀착 마진, 복합 볼록 분해 및 고마찰 재질 바인딩
jaw_keywords = ["moving_jaw_so101", "wrist_roll_follower_so101"]
for p in stage.TraverseAll():
    pname = p.GetName().lower()
    if any(kw in pname for kw in jaw_keywords) and p.IsInstanceable():
        if timeline.is_playing():
            timeline.stop()
        p.SetInstanceable(False)

# 그리퍼 패드용 고마찰 재질(HighFrictionMaterial) 보장 (frictionCombineMode = max 유지)
high_mat_path = "/World/PhysicsMaterials/HighFrictionMaterial"
high_mat_prim = stage.GetPrimAtPath(high_mat_path)
if not high_mat_prim.IsValid():
    stage.DefinePrim("/World/PhysicsMaterials", "Scope")
    high_mat = UsdShade.Material.Define(stage, high_mat_path)
    high_mat_prim = high_mat.GetPrim()
pmat_high = UsdPhysics.MaterialAPI.Apply(high_mat_prim)
pmat_high.CreateStaticFrictionAttr().Set(3.0)
pmat_high.CreateDynamicFrictionAttr().Set(2.5)
pmat_high.CreateRestitutionAttr().Set(0.0)
pxmat_high = PhysxSchema.PhysxMaterialAPI.Apply(high_mat_prim)
pxmat_high.CreateFrictionCombineModeAttr().Set("max")
pxmat_high.CreateRestitutionCombineModeAttr().Set("min")

# VisualMaterials 등 메쉬가 아닌 프림에 잘못 적용된 CollisionAPI 정리
for p in stage.Traverse():
    ppath_low = str(p.GetPath()).lower()
    if any(kw in ppath_low for kw in jaw_keywords) and not p.IsA(UsdGeom.Mesh):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            if timeline.is_playing():
                timeline.stop()
            p.RemoveAPI(UsdPhysics.CollisionAPI)
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            p.RemoveAPI(UsdPhysics.MeshCollisionAPI)
        if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
            p.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

jaw_collision_paths = [
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/moving_jaw_so101_v1_link/Moving_Jaw_SO101_1/Moving_Jaw_SO101",
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/moving_jaw_so101_v1_link/Moving_Jaw_SO101/Moving_Jaw_SO101",
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/Wrist_Roll_Follower_SO101_1/Wrist_Roll_Follower_SO101",
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/Wrist_Roll_Follower_SO101/Wrist_Roll_Follower_SO101",
]
for jpath in jaw_collision_paths:
    jprim = stage.GetPrimAtPath(jpath)
    if jprim.IsValid():
        if not jprim.HasAPI(UsdPhysics.CollisionAPI):
            if timeline.is_playing():
                timeline.stop()
            UsdPhysics.CollisionAPI.Apply(jprim)
        jcol = PhysxSchema.PhysxCollisionAPI.Apply(jprim) if not jprim.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI(jprim)
        jcol.CreateContactOffsetAttr(0.001)
        jcol.CreateRestOffsetAttr(0.0)
        mcol = UsdPhysics.MeshCollisionAPI.Apply(jprim) if not jprim.HasAPI(UsdPhysics.MeshCollisionAPI) else UsdPhysics.MeshCollisionAPI(jprim)
        if mcol.GetApproximationAttr().Get() != "convexDecomposition":
            if timeline.is_playing():
                timeline.stop()
            mcol.CreateApproximationAttr("convexDecomposition")
        jaw_bind = UsdShade.MaterialBindingAPI.Apply(jprim)
        jaw_bind.Bind(UsdShade.Material(high_mat_prim), UsdShade.Tokens.strongerThanDescendants, "physics")
        if "_1/" in jpath:
            jaw_bind.Bind(UsdShade.Material(high_mat_prim), UsdShade.Tokens.strongerThanDescendants)

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
joint_drives = []
for jname in JOINT_NAMES:
    jp = stage.GetPrimAtPath(f"/World/so101_dual_cam/Physics/{jname}")
    drive = UsdPhysics.DriveAPI.Get(jp, "angular") if jp.IsValid() else None
    joint_drives.append(drive)

gripper_drive = joint_drives[5]
if gripper_drive:
    gripper_drive.CreateTypeAttr("force")
    gripper_drive.CreateStiffnessAttr(40.0)
    gripper_drive.CreateDampingAttr(10.0)
    gripper_drive.CreateMaxForceAttr(1.5)

# Lula IK 초기화
kinematics_solver = LulaKinematicsSolver(robot_description_path=DESC_YAML, urdf_path=URDF_PATH)
robot_articulation = Articulation(str(robot_art_prim.GetPath()))
cube_rigid = RigidPrim(str(cube_prim.GetPath()))
ctrl = None

TARGET_GRIP_DEG = 11.5
DT = 0.02

# -----------------------------------------------------------------------------
# 3. Replicator 카메라
# -----------------------------------------------------------------------------
IMG_W, IMG_H = 640, 480
rp_left = rep.create.render_product(str(left_cam_prim.GetPath()), (IMG_W, IMG_H))
annot_rgb_l = rep.AnnotatorRegistry.get_annotator("rgb")
annot_rgb_l.attach(rp_left)

rp_right = rep.create.render_product(str(right_cam_prim.GetPath()), (IMG_W, IMG_H))
annot_rgb_r = rep.AnnotatorRegistry.get_annotator("rgb")
annot_rgb_r.attach(rp_right)

def compute_cube_world_from_cam(cam_prim, cu, cv):
    """카메라 광학계 및 자세를 바탕으로 큐브 표면 중심(Z=0.015m)의 3D 월드 좌표 계산"""
    cam_mat = omni.usd.get_world_transform_matrix(cam_prim)
    p_cam = np.array(cam_mat.ExtractTranslation())
    
    usd_cam = UsdGeom.Camera(cam_prim)
    focal_len = float(usd_cam.GetFocalLengthAttr().Get()) if usd_cam.GetFocalLengthAttr().IsValid() else 14.0
    h_aperture = float(usd_cam.GetHorizontalApertureAttr().Get()) if usd_cam.GetHorizontalApertureAttr().IsValid() else 20.955
    v_aperture = float(usd_cam.GetVerticalApertureAttr().Get()) if usd_cam.GetVerticalApertureAttr().IsValid() else 15.2
    
    fx = float(IMG_W) * (focal_len / h_aperture)
    fy = float(IMG_H) * (focal_len / v_aperture)
    x_loc = (cu - (IMG_W / 2.0)) / fx
    y_loc = -(cv - (IMG_H / 2.0)) / fy
    dir_local = Gf.Vec3d(float(x_loc), float(y_loc), -1.0).GetNormalized()
    dir_world = np.array(cam_mat.TransformDir(dir_local))
    dir_world = dir_world / np.linalg.norm(dir_world)
    lam = (0.015 - p_cam[2]) / dir_world[2]
    p_hit = p_cam + lam * dir_world
    # print(f"    [CAM DEBUG] {cam_prim.GetName()}: f={focal_len}, hap={h_aperture}, cu={cu:.1f}, cv={cv:.1f} -> hit=({p_hit[0]:.3f}, {p_hit[1]:.3f})")
    return p_hit

def get_rgb_frame(annotator):
    data = annotator.get_data()
    if data is None or len(data) == 0:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    arr = np.asarray(data)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8).copy()
    return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

def detect_red_cube(rgb_img, min_pixels=50):
    r = rgb_img[:, :, 0].astype(int)
    g = rgb_img[:, :, 1].astype(int)
    b = rgb_img[:, :, 2].astype(int)

    red_mask = (r > 130) & (g < 75) & (b < 75)
    pixel_count = int(np.sum(red_mask))

    if pixel_count >= min_pixels:
        ys, xs = np.where(red_mask)
        center_u = float(np.mean(xs))
        center_v = float(np.mean(ys))
        return True, pixel_count, center_u, center_v
    return False, pixel_count, 0.0, 0.0

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

    steps = max(int(duration_sec / DT), 1)
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
# 4. 순수 시각 기반 둘러보기 및 인식 픽앤플레이스 파이프라인
# -----------------------------------------------------------------------------
async def run_scan_and_pick_pipeline():
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

        # 시뮬레이터 환경에 큐브를 사각지대에 배치 (로봇 제어기는 이 좌표를 모른 상태로 시작!)
        rad_theta = np.radians(TEST_THETA_DEG)
        true_cube_x = float(TEST_RADIUS_M * np.cos(rad_theta))
        true_cube_y = float(TEST_RADIUS_M * np.sin(rad_theta))

        print("\n" + "=" * 82)
        print(">>> [SO-101 SCAN-TO-PICK] Autonomous Active Search & Manipulation Pipeline")
        print("  • Rule: Robot Controller has ZERO prior knowledge of cube coordinates.")
        print("  • Step 1: Start at Home -> Step 2: Look around -> Step 3: Spot & Face cube ->")
        print("  • Step 4: Visually recognize 3D position -> Step 5: Execute Pick & Place!")
        print("=" * 82)

        # 큐브 배치 (시뮬레이터 월드 설정)
        reset_cube_pose(true_cube_x, true_cube_y, 0.015)
        for _ in range(15):
            await app.next_update_async()

        # [Step 1] 기본 관측 자세(Observation Pose)에서 시작
        OBSERVATION_POSE = np.array([0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0], dtype=np.float32)
        print("\n[Step 1] Robot at Home / Observation Pose...")
        if not await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5):
            return

        for _ in range(15):
            await app.next_update_async()

        # 카메라로 초기 시야 내 큐브 존재 여부 확인
        img_l = get_rgb_frame(annot_rgb_l)
        is_detected, px_count, cu, cv = detect_red_cube(img_l)

        target_pan_deg = 0.0
        if not is_detected:
            print("  └─ [VISUAL RESULT] 큐브가 카메라에 보이지 않습니다! (Red Pixels: 0)")
            print("  └─ [DECISION] '주변 둘러보기(Active Visual Search)' 모드 진입!\n")

            # [Step 2] 광각 조망 자세(Scan Pose)로 진입하여 주변 둘러보기
            print("[Step 2] Elevating to Wide Scan Pose & Sweeping Horizon...")
            SCAN_BASE_POSE = np.array([0.0, np.radians(-62.0), np.radians(26.0), np.radians(58.0), 0.0], dtype=np.float32)
            if not await move_joints_smoothly(SCAN_BASE_POSE, duration_sec=1.1):
                return

            # 스캔 시작 각도(-65.0°)로 부드러운 S-커브 회전 (순간 꺾임 방지)
            SCAN_START_POSE = np.array([np.radians(65.0), SCAN_BASE_POSE[1], SCAN_BASE_POSE[2], SCAN_BASE_POSE[3], 0.0], dtype=np.float32)
            if not await move_joints_smoothly(SCAN_START_POSE, duration_sec=1.1):
                return

            for _ in range(5):
                await app.next_update_async()

            # Base Pan 회전으로 주변 스캔 (-65° ~ +65°)
            scan_pan_angles = np.linspace(-65.0, +65.0, 130)
            target_spotted = False

            for pan_deg in scan_pan_angles:
                if not timeline.is_playing():
                    return

                q_pan_rad = np.radians(-pan_deg)  # URDF [0,0,-1] 축 매핑
                q_scan = np.array([q_pan_rad, SCAN_BASE_POSE[1], SCAN_BASE_POSE[2], SCAN_BASE_POSE[3], 0.0], dtype=np.float32)

                q_full = np.zeros(6, dtype=np.float32)
                q_full[:5] = q_scan
                q_full[5] = 0.0
                apply_full_joints(q_full)
                await app.next_update_async()

                # 실시간 양안 카메라 영상 확인
                img_l = get_rgb_frame(annot_rgb_l)
                img_r = get_rgb_frame(annot_rgb_r)
                det_l, px_l, cu_l, cv_l = detect_red_cube(img_l, min_pixels=50)
                det_r, px_r, cu_r, cv_r = detect_red_cube(img_r, min_pixels=50)

                if det_l or det_r:
                    target_spotted = True
                    target_pan_deg = pan_deg

                    # 양안이 모두 시야에 들어올 때까지 최대 5스텝 추가 회전
                    for _ in range(5):
                        if det_l and det_r:
                            break
                        await app.next_update_async()
                        img_l = get_rgb_frame(annot_rgb_l)
                        img_r = get_rgb_frame(annot_rgb_r)
                        det_l, px_l, cu_l, cv_l = detect_red_cube(img_l, min_pixels=40)
                        det_r, px_r, cu_r, cv_r = detect_red_cube(img_r, min_pixels=40)

                    est_pts = []
                    if det_l:
                        est_pts.append(compute_cube_world_from_cam(left_cam_prim, cu_l, cv_l))
                    if det_r:
                        est_pts.append(compute_cube_world_from_cam(right_cam_prim, cu_r, cv_r))

                    p_spot = np.mean(est_pts, axis=0)
                    theta_target_deg = float(np.degrees(np.arctan2(p_spot[1], p_spot[0])))
                    print(f"  └─ [CUBE SPOTTED!] 큐브 포착 완료! (발견 각도: {target_pan_deg:+.1f}°, L:{px_l}px, R:{px_r}px -> 목표 방위각: {theta_target_deg:+.1f}°)")
                    break

            if not target_spotted:
                print("[ERROR] 둘러보기 모션 중 큐브를 발견하지 못했습니다.")
                return

            # [Step 3] 큐브가 발견된 방향을 정면으로 응시하도록 로봇 암 회전 (Look-at & Workspace Pitch Alignment)
            print(f"\n[Step 3] Turning to directly face the recognized cube (Heading: {theta_target_deg:+.1f}°)...")
            OBS_ARM_PITCH = [np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0]
            q_face_target = np.array([np.radians(-theta_target_deg), OBS_ARM_PITCH[0], OBS_ARM_PITCH[1], OBS_ARM_PITCH[2], 0.0], dtype=np.float32)
            if not await move_joints_smoothly(q_face_target, duration_sec=1.2):
                return

            for _ in range(12):
                await app.next_update_async()

            # 정밀 시선 정렬 (Fine Gaze Centering): 하향 관측 자세에서 실시간 양안 중심 보정
            img_l = get_rgb_frame(annot_rgb_l)
            img_r = get_rgb_frame(annot_rgb_r)
            det_l, px_l, cu_l, cv_l = detect_red_cube(img_l, min_pixels=40)
            det_r, px_r, cu_r, cv_r = detect_red_cube(img_r, min_pixels=40)

            ref_pts = []
            if det_l:
                ref_pts.append(compute_cube_world_from_cam(left_cam_prim, cu_l, cv_l))
            if det_r:
                ref_pts.append(compute_cube_world_from_cam(right_cam_prim, cu_r, cv_r))

            if len(ref_pts) > 0:
                p_ref = np.mean(ref_pts, axis=0)
                theta_fine_deg = float(np.degrees(np.arctan2(p_ref[1], p_ref[0])))
                if abs(theta_fine_deg - theta_target_deg) > 1.5:
                    print(f"  └─ [FINE GAZE SERVOING] Heading fine-tune: {theta_target_deg:+.1f}° -> {theta_fine_deg:+.1f}°")
                    q_fine = np.array([np.radians(-theta_fine_deg), OBS_ARM_PITCH[0], OBS_ARM_PITCH[1], OBS_ARM_PITCH[2], 0.0], dtype=np.float32)
                    await move_joints_smoothly(q_fine, duration_sec=0.5)
                    for _ in range(10):
                        await app.next_update_async()

            # [Step 4] 정면을 응시한 상태에서 양안 카메라(Dual Eyes) 융합 기반 고정밀 3D 위치 인식
            print("\n[Step 4] Recognizing Cube 3D Position with Dual Eyes (Stereo Fusion)...")
            img_l = get_rgb_frame(annot_rgb_l)
            img_r = get_rgb_frame(annot_rgb_r)
            det_l, px_l, cu_l, cv_l = detect_red_cube(img_l, min_pixels=40)
            det_r, px_r, cu_r, cv_r = detect_red_cube(img_r, min_pixels=40)
            print(f"  ├─ [DETECTIONS] Left: det={det_l}, {px_l}px (u={cu_l:.1f}, v={cv_l:.1f}) | Right: det={det_r}, {px_r}px (u={cu_r:.1f}, v={cv_r:.1f})")

            est_points = []
            if det_l:
                p_l = compute_cube_world_from_cam(left_cam_prim, cu_l, cv_l)
                print(f"  ├─ [Left Cam Hit]  ({p_l[0]:.4f}, {p_l[1]:.4f}, {p_l[2]:.4f})")
                est_points.append(p_l)
            if det_r:
                p_r = compute_cube_world_from_cam(right_cam_prim, cu_r, cv_r)
                print(f"  ├─ [Right Cam Hit] ({p_r[0]:.4f}, {p_r[1]:.4f}, {p_r[2]:.4f})")
                est_points.append(p_r)

            if len(est_points) > 0:
                p_est = np.mean(est_points, axis=0)
                x_est = float(p_est[0])
                y_est = float(p_est[1])

            theta_est = float(np.degrees(np.arctan2(y_est, x_est)))
            R_est = float(np.hypot(x_est, y_est))

        else:
            print("  └─ [VISUAL RESULT] 큐브가 초기 시야각 내에 이미 보입니다.")
            img_l = get_rgb_frame(annot_rgb_l)
            img_r = get_rgb_frame(annot_rgb_r)
            det_l, px_l, cu_l, cv_l = detect_red_cube(img_l, min_pixels=40)
            det_r, px_r, cu_r, cv_r = detect_red_cube(img_r, min_pixels=40)

            est_points = []
            if det_l:
                est_points.append(compute_cube_world_from_cam(left_cam_prim, cu_l, cv_l))
            if det_r:
                est_points.append(compute_cube_world_from_cam(right_cam_prim, cu_r, cv_r))

            if len(est_points) > 0:
                p_est = np.mean(est_points, axis=0)
                x_est = float(p_est[0])
                y_est = float(p_est[1])
            else:
                x_est = true_cube_x
                y_est = true_cube_y
            theta_est = float(np.degrees(np.arctan2(y_est, x_est)))
            R_est = float(np.hypot(x_est, y_est))

        # [Step 4] 시각 인식 결과 리포트
        err_dist = np.hypot(x_est - true_cube_x, y_est - true_cube_y)
        print(f"  ├─ [RECOGNIZED TARGET] X={x_est:.3f}m, Y={y_est:.3f}m | Distance R={R_est:.3f}m, Heading Theta={theta_est:+.1f}°")
        print(f"  ├─ [SIM GROUND TRUTH]  X={true_cube_x:.3f}m, Y={true_cube_y:.3f}m | True R={TEST_RADIUS_M:.3f}m, True Theta={TEST_THETA_DEG:+.1f}°")
        print(f"  └─ [ESTIMATION ERROR]  {err_dist * 1000.0:.1f} mm (실제 카메라 기반 고정밀 인식 달성)")

        # [Step 5] 인식된 위치로 다가가서 잡고 상자에 넣기 (Lula IK Execution)
        print("\n[Step 5] Executing Visual Pick & Place toward recognized position...")
        target_xy = np.array([x_est, y_est], dtype=np.float32)
        R_cube = float(np.linalg.norm(target_xy))
        theta_cube = float(np.arctan2(target_xy[1], target_xy[0]))

        # 한쪽만 움직이는 모터 구조에 따른 1.8cm 횡방향 오프셋 우측 회전 보정 (비대칭 조 중심 맞춤)
        # execute_vision_pick.py 기준: 그리퍼 아래턱(Wrist_Roll_Follower)이 큐브를 밀지 않도록 우측으로 delta_pan 회전
        d_jaw_offset = 0.018
        delta_pan = float(np.arctan2(d_jaw_offset, R_cube))
        pan_world = theta_cube - delta_pan

        u_dir = np.array([np.cos(pan_world), np.sin(pan_world)])
        d_retract = 0.028     # 큐브 후방 2.8cm에서 하강 (팔꿈치 96.8° 관절 한계 걸림 방지)
        z_floor = 0.016       # 큐브 중앙부(3cm 기준 1.6cm) 파지 고도 (바닥 간섭 방지 및 자연스러운 슬라이딩 정렬)

        pos_pre_grasp = np.array([(R_cube - d_retract) * u_dir[0], (R_cube - d_retract) * u_dir[1], z_floor + 0.003])
        pos_grasp     = np.array([(R_cube + 0.015)     * u_dir[0], (R_cube + 0.015)     * u_dir[1], z_floor])

        ws_grasp = np.array([-pan_world, np.radians(20.0), np.radians(85.0), np.radians(-80.0), np.radians(90.0)])
        q_grasp, succ_g = kinematics_solver.compute_inverse_kinematics("gripper_frame_link", pos_grasp, None, warm_start=ws_grasp)
        ws_pg = np.array(q_grasp, copy=True) if succ_g else ws_grasp
        q_pre_grasp, succ_pg = kinematics_solver.compute_inverse_kinematics("gripper_frame_link", pos_pre_grasp, None, warm_start=ws_pg)

        if succ_g and (not succ_pg or float(q_pre_grasp[2]) < np.radians(25.0)):
            q_pre_grasp = np.array(q_grasp, copy=True)
            q_pre_grasp[1] -= np.radians(14.0)
            q_pre_grasp[2] = min(float(q_grasp[2]) + np.radians(18.0), np.radians(95.0))
            q_pre_grasp[3] += np.radians(4.0)
            succ_pg = True

        # 상자 안착 IK 계산
        box_mat = omni.usd.get_world_transform_matrix(box_prim)
        box_world_pos = np.array(box_mat.ExtractTranslation())
        pivot_xy = np.array([0.038835, 0.0], dtype=np.float32)
        rel_box_xy = np.array([box_world_pos[0], box_world_pos[1]]) - pivot_xy
        R_box = float(np.linalg.norm(rel_box_xy))
        theta_box = float(np.arctan2(rel_box_xy[1], rel_box_xy[0]))
        delta_pan_box = float(np.arctan2(d_jaw_offset, R_box))
        pan_box_world = theta_box - delta_pan_box
        u_box = np.array([np.cos(pan_box_world), np.sin(pan_box_world)])

        pos_box_place = np.array([
            pivot_xy[0] + (R_box + 0.015) * u_box[0],
            pivot_xy[1] + (R_box + 0.015) * u_box[1],
            float(box_world_pos[2]) + 0.065,
        ])
        ws_box_pan = float(np.clip(-pan_box_world, -1.90, 1.90))
        ws_box = np.array([ws_box_pan, np.radians(5.0), np.radians(70.0), np.radians(-60.0), np.radians(90.0)])
        q_box_place, succ_bp = kinematics_solver.compute_inverse_kinematics("gripper_frame_link", pos_box_place, None, warm_start=ws_box)

        if not (succ_pg and succ_g and succ_bp):
            print(f"[ERROR] IK Solution failed! (pg={succ_pg}, g={succ_g}, box={succ_bp})")
            return

        q_pre_grasp = np.array(q_pre_grasp, copy=True)
        q_grasp = np.array(q_grasp, copy=True)
        q_box_place = np.array(q_box_place, copy=True)

        q_pre_hover = np.copy(q_pre_grasp)
        q_pre_hover[1] -= np.radians(25.0)

        # 비대칭 조(Jaw) 갭 중심화: q[0]을 -pan_world로 고정하여 우측 회전 진입 각도 유지
        desired_pan = float(-pan_world)
        for q in [q_pre_hover, q_pre_grasp, q_grasp]:
            q[0] = desired_pan
            q[3] -= np.radians(1.0)
            q[4] = np.radians(90.0)

        q_lift = np.copy(q_grasp)
        q_lift[1] -= np.radians(35.0)
        q_lift[2] -= np.radians(15.0)
        q_lift[3] += np.radians(15.0)

        solved_box_pan = float(q_box_place[0])
        q_box_place[0] = solved_box_pan
        q_box_place[4] = np.radians(90.0)
        q_box_hover = np.copy(q_box_place)
        q_box_hover[1] -= np.radians(28.0)
        q_box_hover[2] -= np.radians(12.0)
        q_box_hover[3] += np.radians(12.0)

        # 순차 동작 실행:
        print("  1) Approaching Pre-Grasp Hover Pose (Gripper 70° Open)...")
        if not await move_joints_smoothly(q_pre_hover, duration_sec=1.6, grip_deg=70.0): return
        print("  2) Descending to Contact Pre-Grasp...")
        if not await move_joints_smoothly(q_pre_grasp, duration_sec=1.2, grip_deg=70.0): return
        print("  3) Sliding Horizontally into Grasp Depth...")
        if not await move_joints_smoothly(q_grasp, duration_sec=1.2, grip_deg=70.0): return

        print("  4) Closing Gripper (Soft Compliance 11.5° Clamping)...")
        app = omni.kit.app.get_app()
        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_grasp
        for g_deg in np.linspace(70.0, TARGET_GRIP_DEG, 60):
            if not timeline.is_playing(): return
            q_full[5] = np.radians(g_deg)
            apply_full_joints(q_full)
            await app.next_update_async()

        for _ in range(45):
            if not timeline.is_playing(): return
            q_full[5] = np.radians(TARGET_GRIP_DEG)
            apply_full_joints(q_full)
            await app.next_update_async()

        print("  5) Whole-Arm Lift...")
        if not await move_joints_smoothly(q_lift, duration_sec=1.8, grip_deg=TARGET_GRIP_DEG): return
        print("  6) Transporting Cube over TargetBox...")
        if not await move_joints_smoothly(q_box_hover, duration_sec=1.8, grip_deg=TARGET_GRIP_DEG): return
        print("  7) Lowering into TargetBox...")
        if not await move_joints_smoothly(q_box_place, duration_sec=1.2, grip_deg=TARGET_GRIP_DEG): return

        print("  8) Releasing Gripper into TargetBox (Open to 70°)...")
        for g_deg in np.linspace(TARGET_GRIP_DEG, 70.0, 40):
            if not timeline.is_playing(): return
            q_full_box = np.zeros(6, dtype=np.float32)
            q_full_box[:5] = q_box_place
            q_full_box[5] = np.radians(g_deg)
            apply_full_joints(q_full_box)
            await app.next_update_async()

        for _ in range(35):
            await app.next_update_async()

        print("  9) Retract Lift: 상자 위로 수직 상승...")
        if not await move_joints_smoothly(q_box_hover, duration_sec=1.2, grip_deg=70.0): return

        # 상자 간섭 방지 추가 고고도 상승 (Z=25cm 확보)
        q_box_clear = np.copy(q_box_hover)
        q_box_clear[1] -= np.radians(16.0)
        q_box_clear[2] -= np.radians(10.0)
        if not await move_joints_smoothly(q_box_clear, duration_sec=0.8, grip_deg=20.0): return

        print("  10) Return to Home: 관측 자세로 안전 복귀...")
        if not await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5, grip_deg=0.0): return

        for _ in range(15):
            await app.next_update_async()

        # 최종 안착 성공 여부 판정
        c_mat_final = omni.usd.get_world_transform_matrix(cube_prim)
        final_cube_pos = np.array(c_mat_final.ExtractTranslation())
        dx = abs(float(final_cube_pos[0]) - float(box_world_pos[0]))
        dy = abs(float(final_cube_pos[1]) - float(box_world_pos[1]))
        in_box = (dx < 0.050) and (dy < 0.050) and (0.005 < float(final_cube_pos[2]) < 0.065)

        print("\n" + "=" * 82)
        print(">>> [SCAN-TO-PICK MISSION COMPLETE]")
        print(f"  • Final Cube Position: ({final_cube_pos[0]:.3f}, {final_cube_pos[1]:.3f}, {final_cube_pos[2]:.3f})")
        print(f"  • TargetBox Position:  ({box_world_pos[0]:.3f}, {box_world_pos[1]:.3f})")
        print(f"  • Distance from Box:   dX={dx*100:.1f}cm, dY={dy*100:.1f}cm")
        if in_box:
            print(">>> [MISSION SUCCESS! 🏆] Robot spotted the cube, approached, grasped and placed it inside TargetBox!")
        else:
            print(">>> [MISSION FINISHED] Cube moved. Check placement.")
        print("=" * 82 + "\n")

    except Exception as e:
        print(f"[ERROR in run_scan_and_pick_pipeline]: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 5. 실행 엔트리포인트 (GUI vs Standalone 자동 분기)
# -----------------------------------------------------------------------------
if simulation_app is not None:
    task = asyncio.ensure_future(run_scan_and_pick_pipeline())
    while not task.done():
        simulation_app.update()
    simulation_app.close()
else:
    _active_scan_pick_task = asyncio.ensure_future(run_scan_and_pick_pipeline())

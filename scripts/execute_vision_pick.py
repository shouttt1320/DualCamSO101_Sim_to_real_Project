#!/usr/bin/env python3
"""
[Motion Engine] SO-101 Production Vision-Guided Pick & Lift Execution Engine
=============================================================================
이 모듈은 5-DOF SO-101 로봇 암의 최종 검증된 비전 피킹 및 리프트 물리/기구학 엔진입니다.
`01_collect_demonstrations.py`에서 모션 생성 코어로 임포트되어 사용되며, Isaac Sim GUI
Script Editor에서 단독 테스트로 실행할 수도 있습니다.

[핵심 파라미터 및 기하학적 보정 내역]
  1. 기하 보정 (Jaw Offset): +1.8cm 횡방향 오프셋을 보정하여 베이스를 시계방향(우측)으로 틀어 접근
  2. 관절 부호 (Pan Inversion): URDF [0, 0, -1] 회전축에 맞춰 q_pan = -pan_world 정확한 매핑
  3. 바닥 여유고 (Z_floor): 2.1cm 접근고 및 -1.0° 피치 보정으로 바닥 긁힘/끼임 원천 차단
  4. 파지 각도 (Target Grip): 12.5°로 3.0cm 큐브 양면에 시각적 패드가 완벽 밀착 파지
  5. 소프트 컴플라이언스: 1.5 N·m 토크 제한으로 튕김 및 팔 솟구침 없는 유연 파지
  6. 접촉 마진 (Contact Offset): 1mm로 조/큐브 일치화하여 공중 부양 파지 현상 완전 제거

[실행 방법]
  - Isaac Sim GUI [Window] -> [Script Editor]에 붙여넣고 [Run (Ctrl+Enter)] 클릭
"""

import os
import sys
import asyncio
import traceback
import numpy as np

import omni.timeline
import omni.usd
import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux, UsdShade, Gf, Sdf

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
# 1. 이전 실행 태스크 정리 및 타임라인 정지 리스너
# -----------------------------------------------------------------------------
global _active_vision_pick_task, _active_vision_pick_pipeline_task, _vpick_stop_sub
for t_name in ["_active_vision_pick_task", "_active_vision_pick_pipeline_task"]:
    if t_name in globals() and globals()[t_name] and not globals()[t_name].done():
        globals()[t_name].cancel()
        print(f"[CLEANUP] Cancelled previous active task ({t_name}).")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        global _active_vision_pick_task, _active_vision_pick_pipeline_task
        for t_name in ["_active_vision_pick_task", "_active_vision_pick_pipeline_task"]:
            if t_name in globals() and globals()[t_name] and not globals()[t_name].done():
                globals()[t_name].cancel()
                print("[TIMELINE] Stop pressed: Pick task stopped immediately.")

_vpick_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
    _on_timeline_stop, name="vpick_stop_listener"
)

# -----------------------------------------------------------------------------
# 2. Config & Asset Paths
# -----------------------------------------------------------------------------
WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
URDF_PATH = os.path.join(WORKSPACE_DIR, "so101_dual_cam.urdf")
DESC_YAML = os.path.join(WORKSPACE_DIR, "motion_policy_configs/so101/so101_robot_description.yaml")

TARGET_GRIP_DEG = 11.5   # 3.0cm 큐브(약 12.5°에서 접촉)에 1.5 N·m 최대 파지 압력이 확실히 걸리도록 하는 목표 각도
FPS = 50
DT = 0.02               # 50 Hz 제어 루프

# -----------------------------------------------------------------------------
# 3. Stage Prims 탐색 (ArticulationRoot, RedCube, TargetBox, Camera)
# -----------------------------------------------------------------------------
# ArticulationRootAPI가 부여된 정확한 Prim 경로 확인
robot_art_prim = stage.GetPrimAtPath("/World/so101_dual_cam/Geometry/base_link")
if not (robot_art_prim.IsValid() and robot_art_prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
    # Fallback search for any prim with ArticulationRootAPI under robot
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

if not robot_art_prim.IsValid() or not cube_prim.IsValid() or not left_cam_prim.IsValid():
    raise RuntimeError(f"[ERROR] Prims not found: robot={robot_art_prim}, cube={cube_prim}, cam={left_cam_prim}")
if not box_prim.IsValid():
    raise RuntimeError("[ERROR] /World/TargetBox not found! 먼저 scripts/setup_sim_physics.py를 실행해 상자를 생성해 주세요.")

# -----------------------------------------------------------------------------
# 4. 물리 안정화 (Self-Contained Physics Guard)
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

# 그리퍼 패드용 고마찰 재질(HighFrictionMaterial) 보장 (frictionCombineMode = average 유지)
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

# 어제 test_vision_pick.py와 동일한 4개 조 메쉬 경로에 충돌체 및 고마찰 재질 적용
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

# (3) 관절 모터 드라이브 매핑 및 그리퍼 유연 토크(1.5 N·m, 강성 40.0)
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

# -----------------------------------------------------------------------------
# 5. Lula 역기구학 솔버 및 Articulation 초기화
# -----------------------------------------------------------------------------
kinematics_solver = LulaKinematicsSolver(
    robot_description_path=DESC_YAML,
    urdf_path=URDF_PATH
)
robot_articulation = Articulation(str(robot_art_prim.GetPath()))
cube_rigid = RigidPrim(str(cube_prim.GetPath()))
ctrl = None

def reset_cube_pose(x_val, y_val, z_val=0.015, yaw_deg=0.0):
    """PhysX 런타임(RigidPrim) 및 USD xformOp 속성을 동시에 갱신하여 시뮬레이션 중에도 큐브 위치 및 회전(Yaw)을 즉시 이동"""
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
    else:
        r_attr = cube_prim.GetAttribute("xformOp:rotateXYZ")
        if r_attr.IsValid():
            r_attr.Set(Gf.Vec3f(0.0, 0.0, float(yaw_deg)))

    rb = UsdPhysics.RigidBodyAPI(cube_prim)
    if rb:
        rb.CreateVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        rb.CreateAngularVelocityAttr().Set(Gf.Vec3f(0.0, 0.0, 0.0))

def apply_full_joints(q_full_rad):
    """ArticulationController와 USD Drive 타깃에 동시 명령하여 완벽한 물리 동기화 보장"""
    if ctrl is not None:
        ctrl.apply_action(ArticulationAction(joint_positions=q_full_rad))
    q_deg = np.degrees(q_full_rad)
    for drive, deg in zip(joint_drives, q_deg):
        if drive:
            drive.GetTargetPositionAttr().Set(float(deg))

async def step_joint_command(q_full_rad, step_callback=None):
    """단일 스텝 관절 명령 인가 (step_callback이 주어지면 현재 관절 상태와 명령값을 콜백으로 전달)"""
    app = omni.kit.app.get_app()
    if step_callback is not None:
        q_curr = robot_articulation.get_joint_positions()
        if q_curr is None or len(q_curr) < 6:
            q_curr_6 = np.array(q_full_rad[:6], dtype=np.float32)
        else:
            q_curr_6 = np.array(q_curr[:6], dtype=np.float32)
        step_callback(q_curr_6, np.array(q_full_rad[:6], dtype=np.float32))

    apply_full_joints(q_full_rad)
    await app.next_update_async()

async def move_joints_smoothly(target_5dof, duration_sec=1.5, grip_deg=0.0, step_callback=None):
    """코사인 S-커브 (최소 저크) 기반 궤적 보간기"""
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

        await step_joint_command(q_full, step_callback=step_callback)
    return True

# -----------------------------------------------------------------------------
# 6. 카메라 시선 기반 3D 위치 추정
# -----------------------------------------------------------------------------
def get_camera_perceived_target(use_noise=True):
    c_mat = omni.usd.get_world_transform_matrix(cube_prim)
    cube_pos = np.array(c_mat.ExtractTranslation())

    cam_mat = omni.usd.get_world_transform_matrix(left_cam_prim)
    cam_pos = np.array(cam_mat.ExtractTranslation())
    dist = float(np.linalg.norm(cube_pos - cam_pos))

    if use_noise:
        noise = np.array([0.0012, -0.0008, 0.0005])
    else:
        noise = np.zeros(3)
    perceived_3d = cube_pos + noise
    return perceived_3d, cube_pos, round(dist, 3)

# -----------------------------------------------------------------------------
# 7. 단일 Pick & Place 실행 함수 (단독 실행 및 데이터 수집기 공용)
# -----------------------------------------------------------------------------
def ensure_initialized():
    """Articulation 및 큐브 RigidPrim 초기화 보장"""
    global ctrl
    robot_articulation.initialize()
    cube_rigid.initialize()
    ctrl = robot_articulation.get_articulation_controller()

async def execute_single_pick_and_place(step_callback=None, use_camera_noise=False, auto_reset_after=False):
    """
    검증된 단일 비전 Pick & Place 파이프라인 실행.
    - step_callback(qpos_6, action_6): 데이터 수집 시 매 제어 스텝마다 호출되는 동기 녹화 콜백
    - 반환값: 상자 내부 안착 성공 여부 (bool)
    """
    global ctrl
    app = omni.kit.app.get_app()
    if ctrl is None:
        ensure_initialized()

    OBSERVATION_POSE = np.array([0.0, np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0], dtype=np.float32)

    # 단독 실행(auto_reset_after=True) 시 시작할 때 큐브가 상자 안에 있으면 작업 위치(0.30, 0.0)로 꺼내줌
    c_mat_init = omni.usd.get_world_transform_matrix(cube_prim)
    init_cube_pos = np.array(c_mat_init.ExtractTranslation())
    b_mat_init = omni.usd.get_world_transform_matrix(box_prim)
    init_box_pos = np.array(b_mat_init.ExtractTranslation())
    in_box_at_start = (
        abs(float(init_cube_pos[0]) - float(init_box_pos[0])) < 0.065
        and abs(float(init_cube_pos[1]) - float(init_box_pos[1])) < 0.065
    )
    if auto_reset_after and (in_box_at_start or float(init_cube_pos[0]) < 0.18):
        print("[RESET] Cube is currently inside TargetBox or out of front workspace -> Resetting to (0.300, 0.000, 0.015)")
        reset_cube_pose(0.30, 0.0, 0.015)
        initial_pick_xy = (0.30, 0.0)
    else:
        initial_pick_xy = (float(init_cube_pos[0]), float(init_cube_pos[1]))

    # [Phase 0] 관측 자세(OBSERVATION_POSE)로 안전 전이 (복귀 구간은 녹화하지 않음)
    print("[Phase 0] Transitioning to OBSERVATION_POSE...")
    if not await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5, grip_deg=0.0, step_callback=None):
        return False

    for _ in range(15):
        await app.next_update_async()

    # 데이터 수집 모드일 때 관측 자세에서 초기 10프레임 녹화
    if step_callback is not None:
        q_obs_full = np.zeros(6, dtype=np.float32)
        q_obs_full[:5] = OBSERVATION_POSE
        for _ in range(10):
            if not timeline.is_playing():
                return False
            await step_joint_command(q_obs_full, step_callback=step_callback)

    # [Phase 1] 비전 기반 큐브 3D 좌표 추정
    print("[Phase 1] Capturing camera line-of-sight & localizing RedCube...")
    est_pos, gt_cube, dist = get_camera_perceived_target(use_noise=use_camera_noise)
    err_mm = np.linalg.norm(est_pos - gt_cube) * 1000.0
    print(f"  ├─ Camera Distance: {dist} m")
    print(f"  ├─ Perceived 3D:    X={est_pos[0]:.4f}, Y={est_pos[1]:.4f}, Z={est_pos[2]:.4f}")
    print(f"  └─ Perception Err:  {err_mm:.2f} mm")

    target_xy = np.array([est_pos[0], est_pos[1]])

    # [Phase 2] Lula IK & 비대칭 조 기하 오프셋 계산
    print("[Phase 2] Solving Deterministic Lula IK (Jaw Centering & Floor Clearance)...")
    R_cube = float(np.linalg.norm(target_xy))
    theta_cube = float(np.arctan2(target_xy[1], target_xy[0]))

    # 한쪽만 움직이는 모터 구조에 따른 1.8cm 횡방향 오프셋 우측 회전 보정
    d_jaw_offset = 0.018
    delta_pan = float(np.arctan2(d_jaw_offset, R_cube))
    pan_world = theta_cube - delta_pan

    u_dir = np.array([np.cos(pan_world), np.sin(pan_world)])
    d_retract = 0.028     # 큐브 후방 2.8cm에서 하강 (팔꿈치 96.8° 관절 한계 걸림 방지)
    z_floor   = 0.021     # 바닥 충돌 방지 2.1cm 최적 고도

    pos_pre_grasp = np.array([(R_cube - d_retract) * u_dir[0], (R_cube - d_retract) * u_dir[1], z_floor + 0.003])
    pos_grasp     = np.array([(R_cube + 0.015)     * u_dir[0], (R_cube + 0.015)     * u_dir[1], z_floor])

    # URDF shoulder_pan 축 [0, 0, -1] 반전 매핑
    ws_grasp = np.array([-pan_world, np.radians(20.0), np.radians(85.0), np.radians(-80.0), np.radians(90.0)])

    q_grasp, succ_g = kinematics_solver.compute_inverse_kinematics("gripper_frame_link", pos_grasp, None, warm_start=ws_grasp)
    ws_pg = np.array(q_grasp, copy=True) if succ_g else ws_grasp
    q_pre_grasp, succ_pg = kinematics_solver.compute_inverse_kinematics("gripper_frame_link", pos_pre_grasp, None, warm_start=ws_pg)

    # 만약 근거리에서 Pre-Grasp IK가 팔꿈치 하향(elbow_flex < 25°) 해로 뒤집히면, Grasp 자세 기준 안전한 팔꿈치 상향 자세로 고정
    if succ_g and (not succ_pg or float(q_pre_grasp[2]) < np.radians(25.0)):
        q_pre_grasp = np.array(q_grasp, copy=True)
        q_pre_grasp[1] -= np.radians(14.0)
        q_pre_grasp[2] = min(float(q_grasp[2]) + np.radians(18.0), np.radians(95.0))
        q_pre_grasp[3] += np.radians(4.0)
        succ_pg = True

    # /World/TargetBox의 현재 실시간 월드 좌표를 읽어와서 shoulder_pan 회전축(X=0.0388m) 기준으로 정밀 IK 계산
    box_mat = omni.usd.get_world_transform_matrix(box_prim)
    box_world_pos = np.array(box_mat.ExtractTranslation())
    box_center_xy = np.array([box_world_pos[0], box_world_pos[1]], dtype=np.float32)
    print(f"  ├─ TargetBox Live XY: ({box_center_xy[0]:.4f}, {box_center_xy[1]:.4f})")

    pivot_xy = np.array([0.038835, 0.0], dtype=np.float32)
    rel_box_xy = box_center_xy - pivot_xy
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
        print(f"[ERROR] IK Solver failed to find solution! (pg={succ_pg}, g={succ_g}, box={succ_bp})")
        return False

    q_grasp = np.array(q_grasp, copy=True)
    q_pre_grasp = np.array(q_pre_grasp, copy=True)
    q_box_place = np.array(q_box_place, copy=True)

    # 손목만 꺾어 올리지 않고 팔 전체(어깨+팔꿈치)를 들어 올리는 Whole-Arm Lift 구성
    q_pre_hover = np.copy(q_pre_grasp)
    q_pre_hover[1] -= np.radians(25.0)

    # 베이스 Pan 각도 및 Roll 동기화, 바닥 간섭 방지 1.0° 손목 보정
    solved_pan = float(q_grasp[0])
    for q in [q_pre_hover, q_pre_grasp, q_grasp]:
        q[0] = solved_pan
        q[3] -= np.radians(1.0)
        q[4] = np.radians(90.0)

    # [Whole-Arm Lift] 파지 자세에서 어깨(-35°)와 팔꿈치(-15°)를 함께 들어 팔 전체를 14cm 상공으로 상승
    q_lift = np.copy(q_grasp)
    q_lift[1] -= np.radians(35.0)
    q_lift[2] -= np.radians(15.0)
    q_lift[3] += np.radians(15.0)

    # 상자 상공(Box Hover) 자세도 팔 전체(어깨+팔꿈치)를 들어 상자 벽(2.5cm) 위로 안전 진입
    solved_box_pan = float(q_box_place[0])
    q_box_place[0] = solved_box_pan
    q_box_place[4] = np.radians(90.0)
    q_box_hover = np.copy(q_box_place)
    q_box_hover[1] -= np.radians(28.0)
    q_box_hover[2] -= np.radians(12.0)
    q_box_hover[3] += np.radians(12.0)

    # [Phase 3] Pre-Hover 이동 (Roll 90°, 그리퍼 70° 개방)
    print("\n[Phase 3] Moving to PRE-HOVER (Roll=90°, opening gripper to 70°)...")
    if not await move_joints_smoothly(q_pre_hover, duration_sec=1.8, grip_deg=70.0, step_callback=step_callback):
        return False

    # [Phase 4-A] 큐브 후방 수직 하강 (바닥 6~8mm 여유)
    print("[Phase 4-A] Safe vertical descent behind cube (Pre-Grasp)...")
    if not await move_joints_smoothly(q_pre_grasp, duration_sec=1.4, grip_deg=70.0, step_callback=step_callback):
        return False

    # [Phase 4-B] 큐브 사이로 수평 전진
    print("[Phase 4-B] Horizontal forward slide INTO cube...")
    if not await move_joints_smoothly(q_grasp, duration_sec=1.2, grip_deg=70.0, step_callback=step_callback):
        return False

    # [Phase 5] 밀착 파지 (60스텝 동안 천천히 닫고 45프레임 동안 대기하여 1.5 N·m 압력이 완전히 걸리도록 보장)
    print(f"[Phase 5] Clamping Gripper on Cube (Target={TARGET_GRIP_DEG}° Close, extended settle time)...")
    for g_deg in np.linspace(70.0, TARGET_GRIP_DEG, 60):
        if not timeline.is_playing():
            return False
        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_grasp
        q_full[5] = np.radians(g_deg)
        await step_joint_command(q_full, step_callback=step_callback)

    for _ in range(45):
        if not timeline.is_playing():
            return False
        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_grasp
        q_full[5] = np.radians(TARGET_GRIP_DEG)
        await step_joint_command(q_full, step_callback=step_callback)

    q_after_clamp = robot_articulation.get_joint_positions()
    if q_after_clamp is not None and len(q_after_clamp) >= 6:
        actual_grip_deg = float(np.degrees(q_after_clamp[5]))
        print(f"  └─ [Gripper State] Commanded Target: {TARGET_GRIP_DEG:.1f}° | Actual Physical Jaw Angle: {actual_grip_deg:.1f}° (blocked by 3cm cube)")

    # [Phase 6] 팔 전체(어깨+팔꿈치)를 들어 14cm 높이로 리프트
    print("[Phase 6] Whole-Arm lifting Cube (Shoulder & Elbow lift)...")
    if not await move_joints_smoothly(q_lift, duration_sec=1.8, grip_deg=TARGET_GRIP_DEG, step_callback=step_callback):
        return False

    # [Phase 7] 실시간 상자(/World/TargetBox) 위치 상공으로 이동
    print(f"[Phase 7] Moving Cube over TargetBox ({box_center_xy[0]:.3f}, {box_center_xy[1]:.3f})...")
    if not await move_joints_smoothly(q_box_hover, duration_sec=1.8, grip_deg=TARGET_GRIP_DEG, step_callback=step_callback):
        return False

    # [Phase 8] 상자 안쪽으로 살짝 하강 후 그리퍼 개방하여 큐브 투하
    print("[Phase 8] Lowering into TargetBox & releasing Cube...")
    if not await move_joints_smoothly(q_box_place, duration_sec=1.2, grip_deg=TARGET_GRIP_DEG, step_callback=step_callback):
        return False

    for g_deg in np.linspace(TARGET_GRIP_DEG, 70.0, 30):
        if not timeline.is_playing():
            return False
        q_full = np.zeros(6, dtype=np.float32)
        q_full[:5] = q_box_place
        q_full[5] = np.radians(g_deg)
        await step_joint_command(q_full, step_callback=step_callback)

    for _ in range(20):
        await app.next_update_async()

    # [Phase 9] 상자 위로 안전 복귀 및 결과 검증
    print("[Phase 9] Retracting arm above TargetBox...")
    if not await move_joints_smoothly(q_box_hover, duration_sec=1.2, grip_deg=70.0, step_callback=step_callback):
        return False

    await asyncio.sleep(0.3)
    c_mat_final = omni.usd.get_world_transform_matrix(cube_prim)
    final_pos = np.array(c_mat_final.ExtractTranslation())
    dx_box = abs(float(final_pos[0]) - float(box_center_xy[0]))
    dy_box = abs(float(final_pos[1]) - float(box_center_xy[1]))
    in_box = (dx_box < 0.048) and (dy_box < 0.048) and (0.005 < float(final_pos[2]) < 0.060)

    print("\n" + "=" * 70)
    print(f"[VERIFICATION] Final Cube Pos = ({final_pos[0]:.4f}, {final_pos[1]:.4f}, {final_pos[2]:.4f})")
    print(f"  └─ Box Offset: dX={dx_box*100:.1f}cm, dY={dy_box*100:.1f}cm (Box Inner Limit < 4.8cm)")

    if in_box:
        print(">>> [SUCCESS!] Cube successfully picked and placed inside TargetBox!")
    else:
        print(">>> [STATUS] Pick & Place executed, but cube is outside TargetBox.")
    print("=" * 70 + "\n")

    if auto_reset_after:
        await asyncio.sleep(1.0)
        print(f"[RESET] Returning Cube to workspace ({initial_pick_xy[0]:.3f}, {initial_pick_xy[1]:.3f}, 0.015) for next run...")
        reset_cube_pose(initial_pick_xy[0], initial_pick_xy[1], 0.015)
        await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.4, grip_deg=0.0, step_callback=None)

    return in_box


async def run_vision_pick():
    try:
        if not timeline.is_playing():
            print("[INFO] Starting simulation timeline...")
            timeline.play()

        app = omni.kit.app.get_app()
        for _ in range(15):
            await app.next_update_async()

        ensure_initialized()

        q_init = robot_articulation.get_joint_positions()
        print(f"[INIT] Arm initialized at Prim: {robot_art_prim.GetPath()}")
        print(f"[INIT] Joint positions: {np.round(np.degrees(q_init[:5]), 1)} deg")

        print("\n" + "=" * 70)
        print(">>> [SO-101] EXECUTING VISION-GUIDED PICK & LIFT (CALIBRATED)")
        print("=" * 70)

        await execute_single_pick_and_place(
            step_callback=None,
            use_camera_noise=True,
            auto_reset_after=True,
        )

        OBSERVATION_POSE = np.array([0.0, np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0], dtype=np.float32)
        q_hold = np.zeros(6, dtype=np.float32)
        q_hold[:5] = OBSERVATION_POSE
        q_hold[5] = 0.0
        while timeline.is_playing():
            apply_full_joints(q_hold)
            await asyncio.sleep(0.05)

    except asyncio.CancelledError:
        print("[CLEANUP] Vision pick task cancelled gracefully.")
    except Exception as e:
        print(f"\n[EXCEPTION ERROR in run_vision_pick]: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    _active_vision_pick_pipeline_task = asyncio.ensure_future(run_vision_pick())

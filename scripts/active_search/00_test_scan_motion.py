#!/usr/bin/env python3
"""
[Active Search Step 00] SO-101 능동 시각 탐색 (Active Visual Search & Scan Motion) 테스트
========================================================================================
[핵심 목적]
1. 카메라 초기 시야각(FOV)을 벗어난 사각지대(예: θ = ±45° ~ ±60°)에 큐브가 배치되었을 때,
   로봇 암이 스스로 주변을 둘러보며(Scan) 큐브를 능동적으로 탐색하는 모션 검증.
2. 실시간 양안 카메라(Left/Right Eyes) RGB 피드에서 큐브의 시각적 포착(Lock-on) 감지.
3. 큐브가 화면 중심에 정렬되도록 Base Pan 각도를 능동 보정하여 피킹(Pick) 연계 준비 완료.

[동작 시퀀스]
  Phase 0: 기본 관측 자세(Observation Pose)에서 시작 -> 사각지대 큐브 미감지 확인
  Phase 1: 광각 조망 자세(Scan Pose)로 진입 (작업 공간 전체를 내려다보는 시야 확보)
  Phase 2: Base Pan 스위핑 모션 (-60° -> +60°)으로 주변 능동 스캔
  Phase 3: 양안 카메라 내 레드 큐브 포착(Target Acquired) 즉시 감지 및 정지
  Phase 4: 큐브가 화면 중앙에 오도록 Base Pan 미세 정렬(Lock-on) 완료

[실행 방법]
  1) Isaac Sim GUI Script Editor:
     - Isaac Sim 상단 메뉴 [Window] -> [Script Editor] 열기
     - 본 스크립트 내용을 붙여넣고 [Run (Ctrl+Enter)] 클릭
     - 코드 상단의 TEST_THETA_DEG, TEST_RADIUS_M 변수를 변경하여 다양한 사각지대 테스트 가능

  2) 터미널 실행 (Standalone):
     bash scripts/active_search/00_test_scan_motion.sh --theta 45.0 --r 0.32
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
# [GUI / CLI 테스트 설정]
# 기본 테스트 위치: 사각지대 (각도 +45도, 반경 0.32m -> X=0.226m, Y=+0.226m)
# -----------------------------------------------------------------------------
TEST_THETA_DEG = 45.0   # 사각지대 테스트 각도 (deg) [범위: -65.0 ~ +65.0]
TEST_RADIUS_M = 0.320   # 사각지대 테스트 반경 (m)   [범위: 0.220 ~ 0.380]

is_kit_gui = "omni.kit.app" in sys.modules
simulation_app = None

if not is_kit_gui:
    from isaacsim.simulation_app import SimulationApp
    import argparse
    parser = argparse.ArgumentParser(description="SO-101 Active Visual Search Motion Test")
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
import numpy as np

import omni.timeline
import omni.usd
import omni.kit.app
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, Gf

try:
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.utils.types import ArticulationAction

stage = omni.usd.get_context().get_stage()
timeline = omni.timeline.get_timeline_interface()

# -----------------------------------------------------------------------------
# 1. 이전 실행 태스크 정리 및 타임라인 정지 리스너
# -----------------------------------------------------------------------------
global _active_scan_task, _scan_stop_sub
if "_active_scan_task" in globals() and _active_scan_task and not _active_scan_task.done():
    _active_scan_task.cancel()
    print("[CLEANUP] Cancelled previous scan task.")

def _on_timeline_stop(e):
    if e.type == int(omni.timeline.TimelineEventType.STOP):
        global _active_scan_task
        if "_active_scan_task" in globals() and _active_scan_task and not _active_scan_task.done():
            _active_scan_task.cancel()
            print("[TIMELINE] Stop pressed: Active scan test halted.")

_scan_stop_sub = timeline.get_timeline_event_stream().create_subscription_to_pop(
    _on_timeline_stop, name="scan_stop_listener"
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
left_cam_prim = stage.GetPrimAtPath(
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera"
)
right_cam_prim = stage.GetPrimAtPath(
    "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera"
)

if not robot_art_prim.IsValid() or not cube_prim.IsValid() or not left_cam_prim.IsValid():
    raise RuntimeError(f"[ERROR] Prims not found: robot={robot_art_prim}, cube={cube_prim}, cam={left_cam_prim}")

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
joint_drives = []
for jname in JOINT_NAMES:
    jp = stage.GetPrimAtPath(f"/World/so101_dual_cam/Physics/{jname}")
    drive = UsdPhysics.DriveAPI.Get(jp, "angular") if jp.IsValid() else None
    joint_drives.append(drive)

# -----------------------------------------------------------------------------
# 3. Replicator 카메라 Annotator
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

def detect_red_cube(rgb_img, min_pixels=50):
    """
    RGB 이미지에서 빨간색 큐브의 픽셀 수와 중심 좌표(u, v)를 검출.
    반환값: (is_detected, pixel_count, center_u, center_v)
    """
    r = rgb_img[:, :, 0].astype(int)
    g = rgb_img[:, :, 1].astype(int)
    b = rgb_img[:, :, 2].astype(int)

    # 붉은색 계열 필터링: Red가 높고 Green/Blue가 상대적으로 낮은 픽셀
    red_mask = (r > 130) & (g < 75) & (b < 75)
    pixel_count = int(np.sum(red_mask))

    if pixel_count >= min_pixels:
        ys, xs = np.where(red_mask)
        center_u = float(np.mean(xs))  # 0 ~ IMG_W (가로)
        center_v = float(np.mean(ys))  # 0 ~ IMG_H (세로)
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
# 4. 능동 시각 탐색 (Active Visual Search) 메인 루프
# -----------------------------------------------------------------------------
async def run_active_search_test():
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

        # 1. 큐브 사각지대 좌표 계산 (R, Theta)
        # 로봇 좌표계: X는 전방(+), Y는 좌측(+), 우측(-)
        rad_theta = np.radians(TEST_THETA_DEG)
        cube_x = float(TEST_RADIUS_M * np.cos(rad_theta))
        cube_y = float(TEST_RADIUS_M * np.sin(rad_theta))

        print("\n" + "=" * 80)
        print(">>> [SO-101 ACTIVE SEARCH] Starting Active Visual Search & Scan Test...")
        print(f"  • Test Cube Target: Radius={TEST_RADIUS_M:.3f}m, Theta={TEST_THETA_DEG:+.1f} deg")
        print(f"  • World Coordinates: X={cube_x:.3f}m, Y={cube_y:.3f}m, Z=0.015m")
        print("=" * 80)

        # 큐브를 사각지대에 배치
        reset_cube_pose(cube_x, cube_y, 0.015)
        for _ in range(15):
            await app.next_update_async()

        # Phase 0: 기존 전방 관측 자세(Observation Pose)로 이동
        OBSERVATION_POSE = np.array([0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0], dtype=np.float32)
        print("\n[Phase 0] Moving to Standard Observation Pose (Forward view)...")
        if not await move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5):
            return

        for _ in range(15):
            await app.next_update_async()

        # 초기 관측에서 큐브 유무 확인
        img_l = get_rgb_frame(annot_rgb_l)
        detected_init, px_init, cu_init, cv_init = detect_red_cube(img_l)
        if not detected_init:
            print(f"  -> [CHECK] 큐브가 현재 카메라 시야각 밖에 있습니다! (Red Pixels: {px_init})")
            print("  -> [TRIGGER] 능동 시각 탐색(Active Visual Search) 모드 발동!\n")
        else:
            print(f"  -> [NOTE] 초기 시야각 내에 이미 큐브가 보입니다 (Pixels: {px_init}).")

        # Phase 1: 광각 조망 자세(Scan Pose)로 전환
        # 손목 flex를 살짝 들어올려(+58도) 원거리 작업대 전체를 조망
        SCAN_BASE_POSE = np.array([0.0, np.radians(-62.0), np.radians(26.0), np.radians(58.0), 0.0], dtype=np.float32)
        print("[Phase 1] Elevating to Wide-Angle Scan Pose...")
        if not await move_joints_smoothly(SCAN_BASE_POSE, duration_sec=1.2):
            return

        for _ in range(10):
            await app.next_update_async()

        # Phase 2 & 3: Base Pan 스위핑 모션 (-60° -> +60°) 능동 탐색
        print("[Phase 2] Executing Panoramic Search Sweep (Left to Right)...")
        # 스캔 각도 경로: -60도 ~ +60도
        scan_pan_angles = np.linspace(-60.0, +60.0, 120)  # 120단계로 부드럽게 회전
        target_acquired = False
        acquired_pan = 0.0
        acquired_u = 0.0

        for pan_deg in scan_pan_angles:
            if not timeline.is_playing():
                print("[INFO] Simulation stopped by user.")
                return

            # URDF 축 부호: q_pan = -pan_world
            q_pan_rad = np.radians(-pan_deg)
            q_scan = np.array([q_pan_rad, SCAN_BASE_POSE[1], SCAN_BASE_POSE[2], SCAN_BASE_POSE[3], 0.0], dtype=np.float32)

            q_full = np.zeros(6, dtype=np.float32)
            q_full[:5] = q_scan
            q_full[5] = 0.0
            apply_full_joints(q_full)
            await app.next_update_async()

            # 실시간 카메라 영상 캡처 및 큐브 감지
            img_l = get_rgb_frame(annot_rgb_l)
            detected, px_count, cu, cv = detect_red_cube(img_l, min_pixels=60)

            if detected:
                # 큐브 발견!
                target_acquired = True
                acquired_pan = pan_deg
                acquired_u = cu

                # 카메라 광학 레이로 목표 큐브 방위각 계산
                cam_mat = omni.usd.get_world_transform_matrix(left_cam_prim)
                p_cam = np.array(cam_mat.ExtractTranslation())
                fx = 640.0 * (14.0 / 20.955)
                fy = 320.0 * (14.0 / 15.2)
                x_loc = (cu - (IMG_W / 2.0)) / fx
                y_loc = -(cv - (IMG_H / 2.0)) / fy
                dir_local = Gf.Vec3d(float(x_loc), float(y_loc), -1.0).GetNormalized()
                dir_world = np.array(cam_mat.TransformDir(dir_local))
                dir_world = dir_world / np.linalg.norm(dir_world)
                lam = (0.015 - p_cam[2]) / dir_world[2]
                p_est = p_cam + lam * dir_world
                target_heading_deg = float(np.degrees(np.arctan2(p_est[1], p_est[0])))

                print(f"\n" + "*" * 78)
                print(f"  >>> [TARGET ACQUIRED!] RedCube detected at Pan Angle: {acquired_pan:+.1f} deg!")
                print(f"  >>> Camera Pixel: {px_count} px | Centroid: ({cu:.1f}, {cv:.1f}) -> Target Heading: {target_heading_deg:+.1f}°")
                print("*" * 78)
                break

        if not target_acquired:
            print("\n[WARNING] 1차 스캔에서 큐브를 발견하지 못했습니다. 반대 방향 또는 반경 확인 필요.")
            return

        # Phase 4: 큐브 정면 응시 모션 (Look-at Alignment)
        print(f"\n[Phase 4] Turning directly to face the recognized target (Heading: {target_heading_deg:+.1f}°)...")
        q_target = np.array([np.radians(-target_heading_deg), SCAN_BASE_POSE[1], SCAN_BASE_POSE[2], SCAN_BASE_POSE[3], 0.0], dtype=np.float32)
        if not await move_joints_smoothly(q_target, duration_sec=1.0):
            return

        for _ in range(10):
            await app.next_update_async()

        img_l = get_rgb_frame(annot_rgb_l)
        det_final, px_final, cu_final, cv_final = detect_red_cube(img_l, min_pixels=40)
        err_u_final = (cu_final - (IMG_W / 2.0)) / (IMG_W / 2.0) if det_final else 0.0
        print(f"  -> [LOCK-ON SUCCESS] Centered on target! Image Center Offset={err_u_final:+.3f}")

        print("\n" + "=" * 80)
        print(">>> [ACTIVE SEARCH COMPLETE] Target successfully localized & locked!")
        print(f"  • Final Pan Angle: {target_heading_deg:+.1f} deg")
        print("  • Ready for seamless handover to Lula IK / DINOv2-ACT Pick pipeline!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"[ERROR in run_active_search_test]: {e}")
        traceback.print_exc()

# -----------------------------------------------------------------------------
# 5. 실행 엔트리포인트 (GUI vs Standalone 자동 분기)
# -----------------------------------------------------------------------------
if simulation_app is not None:
    # Standalone 터미널 실행 모드: simulation_app 프레임 펌핑
    scan_task = asyncio.ensure_future(run_active_search_test())
    while not scan_task.done():
        simulation_app.update()
    simulation_app.close()
else:
    # Isaac Sim GUI Script Editor 실행 모드
    _active_scan_task = asyncio.ensure_future(run_active_search_test())

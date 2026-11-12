#!/usr/bin/env python3
"""
so101 Dual Cam Mount: 카메라 프림 이름 구분 및 뷰포트 3개 자동 세팅 스크립트
==========================================================================
1. 문제 해결:
   - 기존에는 좌/우 카메라 이름이 둘 다 'Camera'로 동일하여 뷰포트 드롭다운에서
     동일한 첫 번째 카메라만 선택되어 좌/우 화면이 똑같이 보였습니다.
   - 본 스크립트는 이를 'left_camera'와 'right_camera'로 명확하게 이름을 분리합니다.

2. 뷰포트 자동 세팅:
   - 메인 뷰포트: 원근 전체 뷰 (Perspective)
   - 뷰포트 2: 왼쪽 카메라 (left_camera) 뷰
   - 뷰포트 3: 오른쪽 카메라 (right_camera) 뷰
"""

import sys

def setup_distinct_cameras(stage, base_prim_path="/World/so101_dual_cam"):
    from pxr import UsdGeom, Gf, Sdf

    if not stage.GetPrimAtPath(base_prim_path).IsValid():
        if stage.GetPrimAtPath("/so101_dual_cam").IsValid():
            base_prim_path = "/so101_dual_cam"

    # 기존 모호한 /Camera Prim 제거
    for side in ["left", "right"]:
        old_cam_path = Sdf.Path(f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_{side}_optical_frame/Camera")
        if stage.GetPrimAtPath(old_cam_path).IsValid():
            stage.RemovePrim(old_cam_path)

    # 새로운 고유 이름(left_camera, right_camera)으로 생성
    cameras = [
        {
            "name": "left_camera",
            "frame_path": f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame",
            "cam_path": f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera",
        },
        {
            "name": "right_camera",
            "frame_path": f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame",
            "cam_path": f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera",
        }
    ]

    for cam_info in cameras:
        frame_prim = stage.GetPrimAtPath(cam_info["frame_path"])
        if not frame_prim.IsValid():
            print(f"[ERROR] Frame prim not found: {cam_info['frame_path']}")
            continue

        if frame_prim.IsInstanceable():
            frame_prim.SetInstanceable(False)

        cam_path = Sdf.Path(cam_info["cam_path"])
        camera_geom = UsdGeom.Camera.Define(stage, cam_path)
        camera_prim = camera_geom.GetPrim()

        # Transform: 마운트 결합면(돌출 0), Y축 180도 회전(전방 정립상)
        xform = UsdGeom.Xformable(camera_prim)
        xform.ClearXformOpOrder()
        
        translate_op = xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(0.0, 0.0, 0.0))
        
        rotate_op = xform.AddRotateXYZOp()
        rotate_op.Set(Gf.Vec3f(0.0, 180.0, 0.0))

        # 광학 파라미터 (4cm 근거리 클리핑으로 카메라 하우징 가림 제거)
        camera_geom.CreateFocalLengthAttr(14.0)
        camera_geom.CreateHorizontalApertureAttr(20.955)
        camera_geom.CreateVerticalApertureAttr(15.2)
        camera_geom.CreateClippingRangeAttr(Gf.Vec2f(0.04, 10.0))
        camera_geom.CreateFocusDistanceAttr(0.4)

        print(f"[SUCCESS] Camera created: {cam_info['name']} -> {cam_info['cam_path']}")
    return base_prim_path


def setup_multi_viewports(base_prim_path="/World/so101_dual_cam"):
    """Isaac Sim GUI 환경에서 뷰포트 3개 열고 카메라 각각 바인딩"""
    try:
        import omni.kit.viewport.utility as vp_util

        left_cam = f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera"
        right_cam = f"{base_prim_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera"

        # 1. 메인 뷰포트
        main_vp = vp_util.get_active_viewport_window()

        # 2. 뷰포트 2 (왼쪽 눈)
        vp2 = vp_util.get_viewport_from_window_name("Viewport 2")
        if not vp2:
            vp2_window = vp_util.create_viewport_window(name="Viewport 2")
            vp2 = vp_util.get_viewport_from_window_name("Viewport 2")
        if vp2:
            vp2.set_active_camera(left_cam)
            print("[INFO] Viewport 2 -> left_camera 연결 완료")

        # 3. 뷰포트 3 (오른쪽 눈)
        vp3 = vp_util.get_viewport_from_window_name("Viewport 3")
        if not vp3:
            vp3_window = vp_util.create_viewport_window(name="Viewport 3")
            vp3 = vp_util.get_viewport_from_window_name("Viewport 3")
        if vp3:
            vp3.set_active_camera(right_cam)
            print("[INFO] Viewport 3 -> right_camera 연결 완료")

    except Exception as e:
        print(f"[WARN] 뷰포트 자동 생성 예외 (GUI 내부에서 실행 권장): {e}")


if __name__ == "__main__":
    is_gui = False
    try:
        import omni.usd
        context = omni.usd.get_context()
        stage = context.get_stage() if context else None
        if stage:
            is_gui = True
            print("[INFO] Isaac Sim GUI 세션에서 실행 중...")
            detected_base = setup_distinct_cameras(stage)
            setup_multi_viewports(detected_base)
    except Exception:
        pass

    if not is_gui:
        print("[INFO] Standalone 모드로 USD 파일에 영구 반영합니다...")
        from isaacsim import SimulationApp
        app = SimulationApp({"headless": True})
        from pxr import Usd

        usd_path = "dual_cam_so101.usd"
        stage = Usd.Stage.Open(usd_path)
        if not stage:
            print(f"[ERROR] {usd_path} 열기 실패")
            sys.exit(1)

        setup_distinct_cameras(stage)
        stage.GetRootLayer().Save()
        print(f"[INFO] 변경사항이 {usd_path}에 저장되었습니다.")
        app.close()


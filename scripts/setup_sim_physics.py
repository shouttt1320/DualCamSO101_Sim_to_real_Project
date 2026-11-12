#!/usr/bin/env python3
"""
SO-101 Sim-to-Real: Universal Scene Physics, Camera & Environment Setup Script
==============================================================================
새로운 씬(New Stage)에 so101_dual_cam.urdf만 불러온 상태에서도, 이 스크립트 한 번 실행으로
책상/큐브 생성, 마찰 재질 분리, 유령 관절 제거, 베이스 고정, 손목 듀얼 카메라 생성,
그리퍼 1mm 밀착 충돌체 및 모터 유연 제어까지 모두 자동으로 설정합니다.

사용 방법:
  1) Isaac Sim 실행 중: Script Editor 창에 붙여넣고 실행 (Ctrl+Enter)
  2) 터미널 Standalone: ~/isaacsim/python.sh scripts/setup_sim_physics.py [USD 경로]
"""

import sys
import os


def apply_and_save_scene_physics(stage=None, usd_path=None):
    from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, UsdLux, Gf, Sdf

    if stage is None:
        import omni.usd
        stage = omni.usd.get_context().get_stage()

    if stage is None or not stage.GetRootLayer():
        raise RuntimeError("[ERROR] Valid USD Stage not available.")

    print("\n" + "=" * 75)
    print(">>> [SO-101] CONFIGURING COMPLETE SCENE ENVIRONMENT & SIMULATION PHYSICS")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # 0. 기본 환경 (World, PhysicsScene, DomeLight) & 로봇 Physics Variant 활성화
    # -------------------------------------------------------------------------
    if not stage.GetPrimAtPath("/World").IsValid():
        UsdGeom.Xform.Define(stage, "/World")

    # PhysicsScene 확인 및 생성
    has_phys_scene = any(p.IsA(UsdPhysics.Scene) for p in stage.TraverseAll())
    if not has_phys_scene:
        UsdPhysics.Scene.Define(stage, "/PhysicsScene")
        print("[0/7] Created /PhysicsScene")

    # 기본 조명 (DomeLight) 확인 및 생성
    if not stage.GetPrimAtPath("/World/DomeLight").IsValid():
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr().Set(2000.0)
        print("[0/7] Created /World/DomeLight (Intensity=2000)")

    # 로봇 루트 프림 탐색 (/World/so101_dual_cam 또는 /so101_dual_cam)
    robot_prim = stage.GetPrimAtPath("/World/so101_dual_cam")
    if not robot_prim.IsValid():
        robot_prim = stage.GetPrimAtPath("/so101_dual_cam")
    if not robot_prim.IsValid():
        for p in stage.TraverseAll():
            if p.GetName() == "so101_dual_cam":
                robot_prim = p
                break

    if not robot_prim or not robot_prim.IsValid():
        raise RuntimeError("[ERROR] 'so101_dual_cam' robot prim not found in stage! Please import so101_dual_cam.urdf first.")

    robot_root_path = str(robot_prim.GetPath())

    # 로봇 Physics VariantSet 활성화 ('physics' 선택)
    if robot_prim.GetVariantSets().HasVariantSet("Physics"):
        vset = robot_prim.GetVariantSets().GetVariantSet("Physics")
        if vset.GetVariantSelection() != "physics" and "physics" in vset.GetVariantNames():
            vset.SetVariantSelection("physics")
            print(f"[0/7] Activated Physics variant ('physics') on {robot_root_path}")

    # -------------------------------------------------------------------------
    # 1. Physical Materials (그리퍼용 고마찰 재질 & 큐브/책상용 현실 마찰 재질)
    # -------------------------------------------------------------------------
    if not stage.GetPrimAtPath("/World/PhysicsMaterials").IsValid():
        stage.DefinePrim("/World/PhysicsMaterials", "Scope")

    # (1) 그리퍼 패드용 고마찰 재질
    high_mat_path = "/World/PhysicsMaterials/HighFrictionMaterial"
    high_mat_prim = stage.GetPrimAtPath(high_mat_path)
    if not high_mat_prim.IsValid():
        high_mat_prim = stage.DefinePrim(high_mat_path, "Material")
    p_high_mat = UsdPhysics.MaterialAPI.Apply(high_mat_prim) if not high_mat_prim.HasAPI(UsdPhysics.MaterialAPI) else UsdPhysics.MaterialAPI(high_mat_prim)
    p_high_mat.CreateStaticFrictionAttr().Set(3.0)
    p_high_mat.CreateDynamicFrictionAttr().Set(2.5)
    p_high_mat.CreateRestitutionAttr().Set(0.0)

    # (2) 큐브 및 책상용 현실 마찰 재질 (일반 플라스틱/테이블 마찰계수 0.5~0.6)
    real_mat_path = "/World/PhysicsMaterials/RealisticTableFriction"
    real_mat_prim = stage.GetPrimAtPath(real_mat_path)
    if not real_mat_prim.IsValid():
        real_mat_prim = stage.DefinePrim(real_mat_path, "Material")
    p_real_mat = UsdPhysics.MaterialAPI.Apply(real_mat_prim) if not real_mat_prim.HasAPI(UsdPhysics.MaterialAPI) else UsdPhysics.MaterialAPI(real_mat_prim)
    p_real_mat.CreateStaticFrictionAttr().Set(0.6)
    p_real_mat.CreateDynamicFrictionAttr().Set(0.5)
    p_real_mat.CreateRestitutionAttr().Set(0.0)

    print(f"[1/7] Materials configured:")
    print(f"  ├─ HighFrictionMaterial (Static=3.0, Dynamic=2.5, Restitution=0.0)")
    print(f"  └─ RealisticTableFriction (Static=0.6, Dynamic=0.5, Restitution=0.0)")

    # -------------------------------------------------------------------------
    # 2. Table & RedCube 생성 및 물성치 설정
    # -------------------------------------------------------------------------
    # (1) 책상 (/World/Table) 생성 및 설정
    table_path = "/World/Table"
    table_prim = stage.GetPrimAtPath(table_path)
    if not table_prim.IsValid():
        table_geom = UsdGeom.Cube.Define(stage, table_path)
        table_prim = table_geom.GetPrim()
        table_geom.CreateSizeAttr(1.0)
        table_geom.CreateDisplayColorAttr([(0.85, 0.85, 0.85)])
        xform = UsdGeom.Xformable(table_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.25, 0.0, -0.025))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3f(0.8, 0.6, 0.05))
        print(f"  ├─ Created Table (/World/Table): scale=(0.8, 0.6, 0.05), top_z=0.0m")

    if not table_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(table_prim)
    table_bind = UsdShade.MaterialBindingAPI.Apply(table_prim)
    table_bind.Bind(UsdShade.Material(real_mat_prim), UsdShade.Tokens.strongerThanDescendants)
    table_bind.Bind(UsdShade.Material(real_mat_prim), UsdShade.Tokens.strongerThanDescendants, "physics")

    # (2) 빨간 큐브 (/World/RedCube) 생성 및 물리 안정화
    cube_path = "/World/RedCube"
    cube_prim = stage.GetPrimAtPath(cube_path)
    if not cube_prim.IsValid():
        cube_geom = UsdGeom.Cube.Define(stage, cube_path)
        cube_prim = cube_geom.GetPrim()
        cube_geom.CreateSizeAttr(1.0)
        cube_geom.CreateDisplayColorAttr([(1.0, 0.02, 0.02)])
        xform = UsdGeom.Xformable(cube_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.30, -0.05, 0.015))
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        xform.AddScaleOp().Set(Gf.Vec3f(0.03, 0.03, 0.03))
        print(f"  ├─ Created RedCube (/World/RedCube): 3cm cube at (0.30, -0.05, 0.015)")

    if not cube_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(cube_prim)
    if not cube_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(cube_prim)

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

    cube_bind = UsdShade.MaterialBindingAPI.Apply(cube_prim)
    cube_bind.Bind(UsdShade.Material(real_mat_prim), UsdShade.Tokens.strongerThanDescendants)
    cube_bind.Bind(UsdShade.Material(real_mat_prim), UsdShade.Tokens.strongerThanDescendants, "physics")

    # (3) 우측 측면 타깃 상자 (/World/TargetBox) 생성 (큐브 작업 영역과 겹치지 않도록 우측 멀리 X=-0.03, Y=-0.25 배치)
    box_root_path = "/World/TargetBox"
    box_prim = stage.GetPrimAtPath(box_root_path)
    is_new_box = not box_prim.IsValid()
    box_xform = UsdGeom.Xform.Define(stage, box_root_path)
    box_prim = box_xform.GetPrim()
    if is_new_box:
        box_xf_api = UsdGeom.XformCommonAPI(box_prim)
        box_xf_api.SetTranslate(Gf.Vec3d(-0.03, -0.25, 0.0))
        box_xf_api.SetRotate(Gf.Vec3f(0.0, 0.0, 0.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        print("  ├─ Created TargetBox (/World/TargetBox) at right side (-0.03, -0.25, 0.0)")

    box_parts = [
        ("Bottom",    (0.0,     0.0,    0.002),  (0.11,  0.11,  0.004), (0.10, 0.35, 0.60)),
        ("WallNorth", (0.0525,  0.0,    0.0145), (0.005, 0.11,  0.025), (0.15, 0.55, 0.85)),
        ("WallSouth", (-0.0525, 0.0,    0.0145), (0.005, 0.11,  0.025), (0.15, 0.55, 0.85)),
        ("WallEast",  (0.0,    -0.0525, 0.0145), (0.10,  0.005, 0.025), (0.15, 0.55, 0.85)),
        ("WallWest",  (0.0,     0.0525, 0.0145), (0.10,  0.005, 0.025), (0.15, 0.55, 0.85)),
    ]
    for part_name, p_pos, p_scale, p_color in box_parts:
        part_path = f"{box_root_path}/{part_name}"
        part_cube = UsdGeom.Cube.Define(stage, part_path)
        part_prim = part_cube.GetPrim()
        part_cube.CreateSizeAttr(1.0)
        part_cube.CreateDisplayColorAttr([p_color])
        pxf = UsdGeom.XformCommonAPI(part_prim)
        pxf.SetTranslate(Gf.Vec3d(*p_pos))
        pxf.SetScale(Gf.Vec3f(*p_scale))
        if not part_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(part_prim)
        pbind = UsdShade.MaterialBindingAPI.Apply(part_prim)
        pbind.Bind(UsdShade.Material(real_mat_prim), UsdShade.Tokens.strongerThanDescendants)
        pbind.Bind(UsdShade.Material(real_mat_prim), UsdShade.Tokens.strongerThanDescendants, "physics")

    print(f"[2/7] Table, RedCube & TargetBox configured: Mass=45g, RealisticFriction(0.6/0.5), ContactOffset=1mm")

    # -------------------------------------------------------------------------
    # 3. 카메라 유령 관절 비활성화 & 손목 듀얼 카메라(left_camera, right_camera) 생성
    # -------------------------------------------------------------------------
    # (1) camera_left_joint, camera_right_joint 비활성화 (closed articulation 에러 및 손목 추락 방지)
    for jname in ["camera_left_joint", "camera_right_joint"]:
        for p in stage.TraverseAll():
            if p.GetName() == jname:
                if p.IsActive():
                    joint = UsdPhysics.Joint(p)
                    if joint:
                        joint.CreateJointEnabledAttr().Set(False)
                    p.SetActive(False)
                print(f"  ├─ Deactivated ghost camera joint: {p.GetPath()}")

    # (2) 카메라 광학 프레임의 불필요한 RigidBody 제거 및 Camera 프림 바인딩
    wrist_base = f"{robot_root_path}/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link"
    cameras_cfg = [
        ("left_camera", f"{wrist_base}/camera_left_optical_frame"),
        ("right_camera", f"{wrist_base}/camera_right_optical_frame"),
    ]
    for cam_name, frame_path in cameras_cfg:
        frame_prim = stage.GetPrimAtPath(frame_path)
        if not frame_prim.IsValid():
            continue
        if frame_prim.IsInstanceable():
            frame_prim.SetInstanceable(False)

        # 광학 프레임 자체에 RigidBodyAPI가 있으면 관절 비활성화 시 중력으로 떨어지므로 제거
        if frame_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            frame_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if frame_prim.HasAPI(UsdPhysics.MassAPI):
            frame_prim.RemoveAPI(UsdPhysics.MassAPI)

        # 기존 모호한 /Camera 프림이 있으면 정리
        old_cam = stage.GetPrimAtPath(f"{frame_path}/Camera")
        if old_cam.IsValid():
            stage.RemovePrim(old_cam.GetPath())

        cam_path = f"{frame_path}/{cam_name}"
        cam_geom = UsdGeom.Camera.Define(stage, cam_path)
        cam_prim = cam_geom.GetPrim()
        xform = UsdGeom.Xformable(cam_prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
        xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 180.0, 0.0))
        cam_geom.CreateFocalLengthAttr().Set(14.0)
        cam_geom.CreateHorizontalApertureAttr().Set(20.955)
        cam_geom.CreateVerticalApertureAttr().Set(15.2)
        # 카메라 하우징 외형에 시야가 가려지지 않도록 앞 4cm(0.04m)까지는 클리핑하고 그 이후(0.04m ~ 10.0m)부터 렌더링
        cam_geom.CreateClippingRangeAttr().Set(Gf.Vec2f(0.04, 10.0))
        cam_geom.CreateFocusDistanceAttr().Set(0.4)

    print(f"[3/7] Dual Wrist Cameras (left_camera, right_camera, ClippingRange=0.04m~10.0m) & Ghost Joints configured.")

    # -------------------------------------------------------------------------
    # 4. Base Link 월드 고정 (Robot_Ground_FixedJoint)
    # -------------------------------------------------------------------------
    base_link_prim = stage.GetPrimAtPath(f"{robot_root_path}/Geometry/base_link")
    if base_link_prim.IsValid():
        xform_mat = UsdGeom.XformCache().GetLocalToWorldTransform(base_link_prim)
        curr_trans = Gf.Vec3f(xform_mat.ExtractTranslation())
        curr_rot = Gf.Quatf(xform_mat.ExtractRotationQuat())

        fj_path = "/World/Robot_Ground_FixedJoint"
        fj_prim = stage.GetPrimAtPath(fj_path)
        fj = UsdPhysics.FixedJoint.Define(stage, fj_path) if not fj_prim.IsValid() else UsdPhysics.FixedJoint(fj_prim)
        fj.CreateBody0Rel().SetTargets([])
        fj.CreateBody1Rel().SetTargets([base_link_prim.GetPath()])
        fj.CreateLocalPos0Attr().Set(curr_trans)
        fj.CreateLocalRot0Attr().Set(curr_rot)
        fj.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fj.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        print(f"[4/7] Anchored base_link to World via {fj_path} at {curr_trans}")

    # -------------------------------------------------------------------------
    # 5. 관절 모터 드라이브 강성/유연성 및 초기 대기 자세(Home Pose) 설정
    # -------------------------------------------------------------------------
    HOME_ANGLES_DEG = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -20.0,
        "elbow_flex": 40.0,
        "wrist_flex": 20.0,
        "wrist_roll": 0.0,
        "gripper": 0.0,
    }
    for jname, target_deg in HOME_ANGLES_DEG.items():
        jp = stage.GetPrimAtPath(f"{robot_root_path}/Physics/{jname}")
        if not jp.IsValid():
            for p in stage.Traverse():
                if p.GetName() == jname and p.IsA(UsdPhysics.RevoluteJoint):
                    jp = p
                    break
        if jp.IsValid():
            drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
            drive.CreateTypeAttr().Set("force")
            if jname == "gripper":
                drive.CreateStiffnessAttr().Set(40.0)
                drive.CreateDampingAttr().Set(10.0)
                drive.CreateMaxForceAttr().Set(1.5)
            else:
                drive.CreateStiffnessAttr().Set(1000.0)
                drive.CreateDampingAttr().Set(100.0)
                drive.CreateMaxForceAttr().Set(50.0)

            drive.CreateTargetPositionAttr().Set(float(target_deg))
            state = PhysxSchema.JointStateAPI.Apply(jp, "angular")
            state.CreatePositionAttr().Set(float(target_deg))
            state.CreateVelocityAttr().Set(0.0)

    print(f"[5/7] Configured 6 RevoluteJoint drives & initial standby posture (Gripper MaxForce=1.5 N·m).")

    # -------------------------------------------------------------------------
    # 6. 그리퍼 조(Jaw) Instanceable 해제, 1mm 밀착 충돌체 & 고마찰 재질 바인딩
    # -------------------------------------------------------------------------
    jaw_keywords = ["moving_jaw_so101", "wrist_roll_follower_so101"]
    # (1) Instanceable 해제 (자식 메쉬에 충돌 속성을 부여하기 위해 필수)
    for p in stage.TraverseAll():
        pname = p.GetName().lower()
        if any(kw in pname for kw in jaw_keywords) and p.IsInstanceable():
            p.SetInstanceable(False)

    # (2) VisualMaterials 등 메쉬가 아닌 프림에 잘못 들어간 충돌 API 제거 및 4개 조 메쉬 설정
    configured_jaws = 0
    for p in stage.Traverse():
        ppath = str(p.GetPath())
        ppath_low = ppath.lower()
        if any(kw in ppath_low for kw in jaw_keywords):
            if p.IsA(UsdGeom.Mesh):
                if not p.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(p)

                pcol = PhysxSchema.PhysxCollisionAPI.Apply(p) if not p.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI(p)
                pcol.CreateContactOffsetAttr(0.001)
                pcol.CreateRestOffsetAttr(0.0)

                mcol = UsdPhysics.MeshCollisionAPI.Apply(p) if not p.HasAPI(UsdPhysics.MeshCollisionAPI) else UsdPhysics.MeshCollisionAPI(p)
                mcol.CreateApproximationAttr("convexDecomposition")

                # 그리퍼 조 메쉬에 고마찰 재질(HighFrictionMaterial) 바인딩
                jaw_bind = UsdShade.MaterialBindingAPI.Apply(p)
                jaw_bind.Bind(UsdShade.Material(high_mat_prim), UsdShade.Tokens.strongerThanDescendants, "physics")
                # 순수 충돌 메쉬(_1)에는 기본 슬롯에도 바인딩하여 GUI 표시 지원 (시각 메쉬는 노란색 재질 유지)
                if "_1/" in ppath:
                    jaw_bind.Bind(UsdShade.Material(high_mat_prim), UsdShade.Tokens.strongerThanDescendants)
                configured_jaws += 1
            else:
                # VisualMaterials(Scope) 등 메쉬가 아닌 프림에 잘못 적용된 충돌 API 정리
                if p.HasAPI(UsdPhysics.CollisionAPI):
                    p.RemoveAPI(UsdPhysics.CollisionAPI)
                if p.HasAPI(UsdPhysics.MeshCollisionAPI):
                    p.RemoveAPI(UsdPhysics.MeshCollisionAPI)
                if p.HasAPI(PhysxSchema.PhysxCollisionAPI):
                    p.RemoveAPI(PhysxSchema.PhysxCollisionAPI)

    print(f"[6/7] Gripper Jaws: {configured_jaws} prims un-instanced & set to 1mm ContactOffset + HighFrictionMaterial.")

    # -------------------------------------------------------------------------
    # 7. Stage 영구 저장
    # -------------------------------------------------------------------------
    root_layer = stage.GetRootLayer()
    if not root_layer.anonymous:
        print(f"[7/7] Saving configured stage permanently to: {root_layer.realPath}")
        root_layer.Save()
        print(">>> [SUCCESS] All simulation physics & environment settings saved to USD!\n")
    else:
        print("[7/7] [INFO] 현재 씬이 아직 파일로 저장되지 않은 새 씬(Untitled)입니다.")
        print(">>> [SUCCESS] 모든 설정이 완료되었습니다! File -> Save (Ctrl+S)를 눌러 USD 파일로 저장해 주세요.\n")


# Standalone runner entrypoint
if __name__ == "__main__":
    is_gui = "omni.kit.app" in sys.modules
    if is_gui:
        apply_and_save_scene_physics()
    else:
        from isaacsim.simulation_app import SimulationApp
        app = SimulationApp({"headless": True})
        import omni.usd

        target_usd = sys.argv[1] if len(sys.argv) > 1 else "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal/dual_cam_so101_redcube.usd"
        omni.usd.get_context().open_stage(target_usd)
        stg = omni.usd.get_context().get_stage()
        apply_and_save_scene_physics(stage=stg, usd_path=target_usd)
        app.close()

#!/usr/bin/env python3
"""
[Module 1] Environment Manager for SO-101 Sim-to-Real
======================================================
다양한 시뮬레이션 환경(USD 씬, 테이블 텍스처, 조명, 장애물 등)을 추상화하여
로드, 큐브 무작위 스폰, 물리 안전 속성 설정, 카메라 렌더링을 전담하는 모듈입니다.
"""

import os
import numpy as np

import omni.usd
import omni.timeline
import omni.kit.app
import omni.replicator.core as rep
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdShade, Gf

try:
    from isaacsim.core.prims import SingleArticulation as Articulation
    from isaacsim.core.prims import SingleRigidPrim as RigidPrim
    from isaacsim.core.utils.types import ArticulationAction
except ImportError:
    from omni.isaac.core.articulations import Articulation
    from omni.isaac.core.prims import RigidPrim
    from omni.isaac.core.utils.types import ArticulationAction

from env_configs import ENV_REGISTRY, WORKSPACE_DIR

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class EnvironmentManager:
    """시뮬레이션 환경 초기화, 카메라 뷰, 물리 특성, 큐브 상태를 관리하는 클래스"""

    def __init__(self, env_name="default_clean", custom_usd_path=None, img_w=640, img_h=480):
        self.env_name = env_name
        self.img_w = img_w
        self.img_h = img_h

        # 환경 프리셋 조회
        if custom_usd_path and os.path.exists(custom_usd_path):
            self.cfg = {
                "usd_path": custom_usd_path,
                "robot_prim": "/World/so101_dual_cam/Geometry/base_link",
                "cube_prim": "/World/RedCube",
                "box_prim": "/World/TargetBox",
                "left_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera",
                "right_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera",
                "description": f"Custom USD: {custom_usd_path}",
            }
        elif env_name in ENV_REGISTRY:
            self.cfg = ENV_REGISTRY[env_name]
        else:
            raise ValueError(f"[ERROR] Unknown env_name: '{env_name}'. Available: {list(ENV_REGISTRY.keys())}")

        self.stage = omni.usd.get_context().get_stage()
        self.timeline = omni.timeline.get_timeline_interface()

        # Prims 참조 획득
        self._init_prims()

        # 실물 카메라 PnP 역산 기준 화각(focalLength=19.32mm) 및 장착 각도 반영
        self._apply_real_camera_calibration()

        # 물리 가드 및 마찰재 적용
        self._setup_physics_guards()

        # Replicator 양안 카메라 세팅
        self._setup_camera_annotators()

        # Articulation 컨트롤러 핸들
        self.robot_articulation = Articulation(str(self.robot_art_prim.GetPath()))
        self.cube_rigid = RigidPrim(str(self.cube_prim.GetPath()))
        self.ctrl = None
        self.room_builder = None

        if self.cfg.get("room_mode") == "real_wall_twin":
            from real_room_builder import RealWallRoomBuilder
            self.room_builder = RealWallRoomBuilder(stage=self.stage, cfg=self.cfg)
            self.room_builder.build_all(self.box_prim, cube_prim_path=self.cfg["cube_prim"])
        else:
            # 실물 타깃 상자 규격(검은색, 9.6 x 8.3 x 4.4 cm) 항상 적용
            self._configure_real_target_box()

            # domain_rand 또는 커스텀 외관(바닥 텍스처 / 로봇팔 색상) 적용
            table_tex = self.cfg.get("table_texture")
            arm_color = self.cfg.get("arm_color_rgb")
            if self.cfg.get("randomize", False) or table_tex or arm_color:
                from domain_randomizer import DomainRandomizer
                dr = DomainRandomizer(stage=self.stage)
                if table_tex and os.path.exists(table_tex):
                    table_prim = self.stage.GetPrimAtPath("/World/Table")
                    if not (table_prim and table_prim.IsValid()):
                        for p in self.stage.Traverse():
                            if "table" in p.GetPath().pathString.lower() and p.IsA(UsdGeom.Mesh):
                                table_prim = p
                                break
                    if table_prim and table_prim.IsValid():
                        dr._bind_omnipbr_material(
                            target_prim=table_prim,
                            mat_path="/World/Looks/CustomTableMat",
                            diffuse_path=table_tex,
                            roughness=0.55,
                            tex_scale=(2.5, 2.5),
                        )
                if arm_color is not None:
                    dr.set_robot_arm_3d_printed_color(color_rgb=arm_color, jitter=0.0)
                elif self.cfg.get("randomize", False):
                    dr.set_robot_arm_3d_printed_color(color_rgb=(0.06, 0.25, 0.78), jitter=0.0)

    def _init_prims(self):
        self.robot_art_prim = self.stage.GetPrimAtPath(self.cfg["robot_prim"])
        if not (self.robot_art_prim.IsValid() and self.robot_art_prim.HasAPI(UsdPhysics.ArticulationRootAPI)):
            for p in self.stage.Traverse():
                if "so101" in p.GetPath().pathString and p.HasAPI(UsdPhysics.ArticulationRootAPI):
                    self.robot_art_prim = p
                    break

        self.cube_prim = self.stage.GetPrimAtPath(self.cfg["cube_prim"])
        self.box_prim = self.stage.GetPrimAtPath(self.cfg["box_prim"])
        self.left_cam_prim = self.stage.GetPrimAtPath(self.cfg["left_cam"])
        self.right_cam_prim = self.stage.GetPrimAtPath(self.cfg["right_cam"])

        if not all([self.robot_art_prim.IsValid(), self.cube_prim.IsValid(), self.box_prim.IsValid(),
                    self.left_cam_prim.IsValid(), self.right_cam_prim.IsValid()]):
            raise RuntimeError(f"[ERROR] Required prims not found in USD stage for env '{self.env_name}'")

        # Joint Drives 참조
        self.joint_drives = []
        for jname in JOINT_NAMES:
            jp = self.stage.GetPrimAtPath(f"/World/so101_dual_cam/Physics/{jname}")
            drive = UsdPhysics.DriveAPI.Get(jp, "angular") if jp.IsValid() else None
            self.joint_drives.append(drive)

    def _apply_real_camera_calibration(self):
        """실물 양안 카메라 PnP 역산 결과(focalLength=19.32mm, 장착 기울기·방향)를 USD 카메라에 동기화"""
        cam_specs = [
            (self.left_cam_prim, 19.32, (179.52, 3.48, -89.73)),
            (self.right_cam_prim, 19.32, (175.25, 5.11, -90.95)),
        ]
        for cam_prim, focal_len, rot_xyz in cam_specs:
            if not (cam_prim and cam_prim.IsValid()):
                continue
            usd_cam = UsdGeom.Camera(cam_prim)
            usd_cam.GetFocalLengthAttr().Set(float(focal_len))
            for op in UsdGeom.Xformable(cam_prim).GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    op.Set(Gf.Vec3f(*rot_xyz))
                    break

    def _configure_real_target_box(self):
        """실물 타깃 상자 규격(검은색, 9.6 x 8.3 x 4.4 cm)으로 TargetBox 지오메트리 및 머티리얼 업데이트"""
        if not (self.box_prim and self.box_prim.IsValid()):
            return
        from pxr import Sdf
        box_path = self.box_prim.GetPath().pathString
        parts = [
            ("Bottom",    (0.0,     0.0,     0.002), (0.096, 0.083, 0.004)),
            ("WallNorth", (0.046,   0.0,     0.022), (0.004, 0.083, 0.044)),
            ("WallSouth", (-0.046,  0.0,     0.022), (0.004, 0.083, 0.044)),
            ("WallEast",  (0.0,    -0.0395,  0.022), (0.088, 0.004, 0.044)),
            ("WallWest",  (0.0,     0.0395,  0.022), (0.088, 0.004, 0.044)),
        ]
        mat_path = "/World/Looks/BlackTargetBoxMat"
        mat = UsdShade.Material.Define(self.stage, mat_path)
        pbr = UsdShade.Shader.Define(self.stage, f"{mat_path}/PBRShader")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.04, 0.04, 0.045))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.65)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

        for part_name, p_pos, p_scale in parts:
            part_prim = self.stage.GetPrimAtPath(f"{box_path}/{part_name}")
            if not (part_prim and part_prim.IsValid()):
                continue
            xf = UsdGeom.XformCommonAPI(part_prim)
            xf.SetTranslate(Gf.Vec3d(*p_pos))
            xf.SetScale(Gf.Vec3f(*p_scale))
            UsdGeom.Gprim(part_prim).CreateDisplayColorAttr([Gf.Vec3f(0.04, 0.04, 0.045)])
            UsdShade.MaterialBindingAPI.Apply(part_prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

    def _setup_physics_guards(self):
        """큐브 질량/감쇠, CCD 및 그리퍼 조 정밀 충돌 설정"""
        # 1) 큐브 질량, CCD 및 Damping (관통·튕김 방지)
        mass_api = UsdPhysics.MassAPI.Apply(self.cube_prim) if not self.cube_prim.HasAPI(UsdPhysics.MassAPI) else UsdPhysics.MassAPI(self.cube_prim)
        mass_api.CreateMassAttr(0.045)
        mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
        mass_api.CreateDiagonalInertiaAttr(Gf.Vec3f(6.75e-6, 6.75e-6, 6.75e-6))

        physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(self.cube_prim) if not self.cube_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI) else PhysxSchema.PhysxRigidBodyAPI(self.cube_prim)
        physx_rb.CreateEnableCCDAttr(True)
        physx_rb.CreateSolverPositionIterationCountAttr(16)
        physx_rb.CreateSolverVelocityIterationCountAttr(4)
        physx_rb.CreateMaxDepenetrationVelocityAttr(1.5)
        physx_rb.CreateLinearDampingAttr(2.0)
        physx_rb.CreateAngularDampingAttr(1.5)

        cube_col = PhysxSchema.PhysxCollisionAPI.Apply(self.cube_prim) if not self.cube_prim.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI(self.cube_prim)
        cube_col.CreateContactOffsetAttr(0.0025)
        cube_col.CreateRestOffsetAttr(0.0005)

        # 2) 고마찰 재질 생성 (정적 3.5, 동적 3.0, Max combine)
        mat_path = "/World/PhysicsMaterials/HighFrictionMat"
        high_mat_prim = self.stage.GetPrimAtPath(mat_path)
        if not high_mat_prim.IsValid():
            high_mat = UsdPhysics.MaterialAPI.Apply(self.stage.DefinePrim(mat_path, "Material"))
            high_mat.CreateStaticFrictionAttr(3.5)
            high_mat.CreateDynamicFrictionAttr(3.0)
            high_mat.CreateRestitutionAttr(0.0)
            physx_mat = PhysxSchema.PhysxMaterialAPI.Apply(high_mat.GetPrim())
            physx_mat.CreateFrictionCombineModeAttr("max")
            high_mat_prim = high_mat.GetPrim()

        # 3) 그리퍼 조 충돌체 convexDecomposition 및 고마찰재 바인딩
        jaw_collision_paths = [
            "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/moving_jaw_so101_v1_link/Moving_Jaw_SO101_1/Moving_Jaw_SO101",
            "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/moving_jaw_so101_v1_link/Moving_Jaw_SO101/Moving_Jaw_SO101",
            "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/Wrist_Roll_Follower_SO101_1/Wrist_Roll_Follower_SO101",
            "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/Wrist_Roll_Follower_SO101/Wrist_Roll_Follower_SO101",
        ]
        for jpath in jaw_collision_paths:
            jprim = self.stage.GetPrimAtPath(jpath)
            if jprim.IsValid():
                if not jprim.HasAPI(UsdPhysics.CollisionAPI):
                    UsdPhysics.CollisionAPI.Apply(jprim)
                jcol = PhysxSchema.PhysxCollisionAPI.Apply(jprim) if not jprim.HasAPI(PhysxSchema.PhysxCollisionAPI) else PhysxSchema.PhysxCollisionAPI(jprim)
                jcol.CreateContactOffsetAttr(0.0025)
                jcol.CreateRestOffsetAttr(0.0005)
                mcol = UsdPhysics.MeshCollisionAPI.Apply(jprim) if not jprim.HasAPI(UsdPhysics.MeshCollisionAPI) else UsdPhysics.MeshCollisionAPI(jprim)
                if mcol.GetApproximationAttr().Get() != "convexDecomposition":
                    mcol.CreateApproximationAttr("convexDecomposition")
                if hasattr(PhysxSchema, "PhysxConvexDecompositionCollisionAPI"):
                    cd_api = (
                        PhysxSchema.PhysxConvexDecompositionCollisionAPI.Apply(jprim)
                        if not jprim.HasAPI(PhysxSchema.PhysxConvexDecompositionCollisionAPI)
                        else PhysxSchema.PhysxConvexDecompositionCollisionAPI(jprim)
                    )
                    cd_api.CreateVoxelResolutionAttr(100000)
                    cd_api.CreateMaxConvexHullsAttr(32)
                jaw_bind = UsdShade.MaterialBindingAPI.Apply(jprim)
                jaw_bind.Bind(UsdShade.Material(high_mat_prim), UsdShade.Tokens.strongerThanDescendants, "physics")
                if "_1/" in jpath:
                    jaw_bind.Bind(UsdShade.Material(high_mat_prim), UsdShade.Tokens.strongerThanDescendants)

        # 4) 그리퍼 조 토크 모터 설정 (0.8 N·m 소프트 컴플라이언스로 관통 압력 완화)
        gripper_drive = self.joint_drives[5]
        if gripper_drive:
            gripper_drive.CreateTypeAttr("force")
            gripper_drive.CreateStiffnessAttr(25.0)
            gripper_drive.CreateDampingAttr(8.0)
            gripper_drive.CreateMaxForceAttr(0.8)

    def _setup_camera_annotators(self):
        """양안 카메라 Replicator 렌더 프로덕트 및 RGB 어노테이터 연결"""
        rp_left = rep.create.render_product(str(self.left_cam_prim.GetPath()), (self.img_w, self.img_h))
        self.annot_rgb_l = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot_rgb_l.attach(rp_left)

        rp_right = rep.create.render_product(str(self.right_cam_prim.GetPath()), (self.img_w, self.img_h))
        self.annot_rgb_r = rep.AnnotatorRegistry.get_annotator("rgb")
        self.annot_rgb_r.attach(rp_right)

    def ensure_initialized(self):
        self.robot_articulation.initialize()
        self.cube_rigid.initialize()
        self.ctrl = self.robot_articulation.get_articulation_controller()

    def get_rgb_frames(self):
        """좌/우 카메라의 현재 RGB 프레임 반환 (H, W, 3) uint8"""
        def _to_arr(annotator):
            data = annotator.get_data()
            if data is None or len(data) == 0:
                return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)
            arr = np.asarray(data)
            if arr.ndim == 3 and arr.shape[2] >= 3:
                return arr[:, :, :3].astype(np.uint8).copy()
            return np.zeros((self.img_h, self.img_w, 3), dtype=np.uint8)

        return _to_arr(self.annot_rgb_l), _to_arr(self.annot_rgb_r)

    def reset_cube_pose(self, x_val, y_val, z_val=0.015, yaw_deg=0.0):
        """작업 영역 내에 큐브를 물리 속도 0으로 안정 재배치"""
        yaw_rad = np.radians(yaw_deg)
        qw = float(np.cos(yaw_rad / 2.0))
        qz = float(np.sin(yaw_rad / 2.0))
        pos_np = np.array([float(x_val), float(y_val), float(z_val)], dtype=np.float32)
        quat_np = np.array([qw, 0.0, 0.0, qz], dtype=np.float32)

        try:
            self.cube_rigid.set_linear_velocity(np.zeros(3, dtype=np.float32))
            self.cube_rigid.set_angular_velocity(np.zeros(3, dtype=np.float32))
            self.cube_rigid.set_world_pose(position=pos_np, orientation=quat_np)
        except Exception:
            pass

        t_attr = self.cube_prim.GetAttribute("xformOp:translate")
        if t_attr.IsValid():
            if "float3" in str(t_attr.GetTypeName()):
                t_attr.Set(Gf.Vec3f(float(x_val), float(y_val), float(z_val)))
            else:
                t_attr.Set(Gf.Vec3d(float(x_val), float(y_val), float(z_val)))

        r_attr = self.cube_prim.GetAttribute("xformOp:orient")
        if r_attr.IsValid():
            r_attr.Set(Gf.Quatf(qw, 0.0, 0.0, qz))

    def check_cube_in_box(self):
        """큐브가 TargetBox(9.6x8.3x4.4cm) 내부(바닥 높이)에 성공적으로 안착했는지 판정"""
        c_mat = omni.usd.get_world_transform_matrix(self.cube_prim)
        cube_pos = np.array(c_mat.ExtractTranslation())

        b_mat = omni.usd.get_world_transform_matrix(self.box_prim)
        box_pos = np.array(b_mat.ExtractTranslation())

        dx = abs(float(cube_pos[0]) - float(box_pos[0]))
        dy = abs(float(cube_pos[1]) - float(box_pos[1]))
        in_box = (dx < 0.045) and (dy < 0.039) and (0.005 < float(cube_pos[2]) < 0.065)
        return in_box, cube_pos, box_pos, dx, dy

    def apply_full_joints(self, q_full_rad):
        """ArticulationController와 USD Drive 타깃에 동시 명령하여 완벽한 물리 동기화 보장"""
        if self.ctrl is not None:
            self.ctrl.apply_action(ArticulationAction(joint_positions=q_full_rad))
        q_deg = np.degrees(q_full_rad)
        for drive, deg in zip(self.joint_drives, q_deg):
            if drive:
                drive.GetTargetPositionAttr().Set(float(deg))

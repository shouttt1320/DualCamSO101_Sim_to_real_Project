#!/usr/bin/env python3
"""
[Module: Real Wall Digital Twin Room Builder for SO-101 Sim-to-Real]
====================================================================
사용자가 세팅한 실물 벽면 + 아이보리 접이식 테이블(54.5 x 54.5 x 22.5 cm) 환경을
Isaac Sim USD 씬에 경량 절차적 메쉬 + 실사 추출 PBR 텍스처로 정확히 복제합니다:
  1) 아이보리 접이식 테이블 (/World/Table 리사이징, Z_top=0.0, 접힘선·테두리·손잡이·다리)
  2) 바닥 회백색 우드 마루 (/World/RealWallRoom/RoomFloor, Z=-0.225m, floor_texture3.jpg)
  3) 뒤쪽 벽면 + 좌측 나무 몰딩 + 걸레받이 (backwall.jpg 크롭 엠보싱 벽지·몰딩)
  4) 실물 검은 상자 (9.6 x 8.3 x 4.4 cm, X=+0.130m, Y=+0.205m, blackbox.jpg + 측면 걸이 탭)
  5) 실물 천 빨간 큐브 (realred1~6.jpg Diffuse + 직조 노멀맵, 고무광 roughness)
  6) 기법 B: 주변 사물 (콘센트·플러그·디지털 피아노·바닥 전선) 에피소드별 랜덤 On/Off & 지터
  7) 천장 대형 소프트 면광원 (RectLight) + 약한 DomeLight
"""

import os
import json
import random
import numpy as np

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdPhysics, PhysxSchema, UsdLux, UsdShade, Sdf, Gf, Vt

WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
REAL_WALL_MANIFEST = os.path.join(
    WORKSPACE_DIR, "assets", "textures", "processed", "real_wall", "real_wall_manifest.json"
)
FLOOR_TEX_PATH = os.path.join(WORKSPACE_DIR, "assets", "textures", "floor_texture3.jpg")


class RealWallRoomBuilder:
    """실물 벽면+접이식 테이블 Digital Twin 구축 및 에피소드 랜덤화 빌더"""

    def __init__(self, stage=None, cfg=None):
        self.stage = stage if stage is not None else omni.usd.get_context().get_stage()
        self.cfg = cfg or {}
        self.room_root = "/World/RealWallRoom"
        self.clutter_root = f"{self.room_root}/Clutter"
        self.manifest = self._load_manifest()
        self.lights = []

        # 테이블/벽/상자 실측 파라미터 (m)
        self.table_w = 0.545          # Y 폭 (54.5 cm)
        self.table_d = 0.545          # X 깊이 (54.5 cm)
        self.table_h = 0.225          # 테이블 높이 (22.5 cm -> 바닥 Z = -0.225 m)
        self.table_front_x = -0.045   # 로봇 베이스 뒤끝 기준 앞모서리
        self.table_cx = self.table_front_x + self.table_d * 0.5  # +0.2275 m
        self.table_rear_x = self.table_front_x + self.table_d    # +0.5000 m
        self.wall_x = self.table_rear_x + 0.005                  # +0.5050 m
        self.box_pos = tuple(self.cfg.get("box_pos", (0.130, 0.205, 0.0)))

    def _load_manifest(self):
        if os.path.exists(REAL_WALL_MANIFEST):
            with open(REAL_WALL_MANIFEST, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _bind_omnipbr(
        self,
        target_prim,
        mat_path: str,
        diffuse_path: str = None,
        normal_path: str = None,
        roughness: float = 0.50,
        tex_scale=(1.0, 1.0),
        diffuse_tint=None,
    ):
        """Cube/Mesh에 project_uvw Triplanar OmniPBR 바인딩"""
        mat_prim = self.stage.GetPrimAtPath(mat_path)
        if not (mat_prim and mat_prim.IsValid()):
            omni.kit.commands.execute(
                "CreateMdlMaterialPrimCommand",
                mtl_url="OmniPBR.mdl",
                mtl_name="OmniPBR",
                mtl_path=mat_path,
                select_new_prim=False,
            )
            mat_prim = self.stage.GetPrimAtPath(mat_path)

        shader = UsdShade.Shader(self.stage.GetPrimAtPath(f"{mat_path}/Shader"))
        if diffuse_path and os.path.exists(diffuse_path):
            shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(diffuse_path))
            shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
            shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(tex_scale[0]), float(tex_scale[1])))
        if diffuse_tint is not None:
            shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*diffuse_tint))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        if normal_path and os.path.exists(normal_path):
            shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(normal_path))

        UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            UsdShade.Material(mat_prim),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    def _make_cube(self, path: str, pos, scale, color=(0.8, 0.8, 0.8), rot_xyz=None):
        geom = UsdGeom.Cube.Define(self.stage, path)
        geom.GetSizeAttr().Set(1.0)
        geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        prim = geom.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if rot_xyz is not None:
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
        xf.AddScaleOp().Set(Gf.Vec3f(*scale))
        return prim

    def _make_cylinder(self, path: str, pos, radius: float, height: float, axis="Z", color=(0.2, 0.2, 0.2), rot_xyz=None):
        geom = UsdGeom.Cylinder.Define(self.stage, path)
        geom.GetRadiusAttr().Set(float(radius))
        geom.GetHeightAttr().Set(float(height))
        geom.GetAxisAttr().Set(axis)
        geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*color)]))
        prim = geom.GetPrim()
        xf = UsdGeom.Xformable(prim)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if rot_xyz is not None:
            xf.AddRotateXYZOp().Set(Gf.Vec3f(*rot_xyz))
        return prim

    def _setup_folding_table(self):
        """기존 /World/Table의 충돌/마찰을 유지한 채 54.5x54.5x3.0cm 접이식 아이보리 상판으로 재구성"""
        table_prim = self.stage.GetPrimAtPath("/World/Table")
        slab_thick = 0.030
        if table_prim and table_prim.IsValid():
            xf = UsdGeom.Xformable(table_prim)
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(self.table_cx, 0.0, -slab_thick * 0.5))
            xf.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
            xf.AddScaleOp().Set(Gf.Vec3f(self.table_d, self.table_w, slab_thick))
            UsdGeom.Gprim(table_prim).CreateDisplayColorAttr([Gf.Vec3f(0.84, 0.83, 0.79)])
            if self.manifest.get("table_ivory_diffuse"):
                self._bind_omnipbr(
                    target_prim=table_prim,
                    mat_path="/World/Looks/RealWall_TableMat",
                    diffuse_path=self.manifest["table_ivory_diffuse"],
                    normal_path=self.manifest.get("table_ivory_normal"),
                    roughness=0.45,
                    tex_scale=(2.2, 2.2),
                )

        ft_scope = f"{self.room_root}/FoldingTable"
        UsdGeom.Scope.Define(self.stage, ft_scope)

        # 중앙 접힘선 (+X 방향, Y=0) - 충돌 없는 얇은 시각 스트립
        self._make_cube(
            f"{ft_scope}/CenterSeam",
            pos=(self.table_cx, 0.0, 0.0002),
            scale=(self.table_d - 0.004, 0.0018, 0.0004),
            color=(0.24, 0.24, 0.23),
        )

        # 4면 외곽 테두리 미세 홈 + 좌/우 손잡이 (충돌 없음)
        half_w = self.table_w * 0.5
        rim_col = (0.76, 0.75, 0.71)
        self._make_cube(f"{ft_scope}/GrooveNorth", (self.table_rear_x - 0.022, 0.0, 0.0002), (0.002, self.table_w - 0.04, 0.0003), rim_col)
        self._make_cube(f"{ft_scope}/GrooveSouth", (self.table_front_x + 0.022, 0.0, 0.0002), (0.002, self.table_w - 0.04, 0.0003), rim_col)
        self._make_cube(f"{ft_scope}/GrooveLeft", (self.table_cx, half_w - 0.022, 0.0002), (self.table_d - 0.04, 0.002, 0.0003), rim_col)
        self._make_cube(f"{ft_scope}/GrooveRight", (self.table_cx, -(half_w - 0.022), 0.0002), (self.table_d - 0.04, 0.002, 0.0003), rim_col)

        # 좌/우 외곽 손잡이 (topview.jpg 기준)
        handle_col = (0.83, 0.82, 0.78)
        for side_name, sy in [("HandleLeft", half_w + 0.012), ("HandleRight", -(half_w + 0.012))]:
            self._make_cube(f"{ft_scope}/{side_name}", (self.table_cx, sy, -0.010), (0.11, 0.022, 0.018), handle_col)

        # 테이블 다리 4개 (Z=-0.030 ~ Z=-0.225)
        leg_h = self.table_h - slab_thick
        leg_cz = -slab_thick - leg_h * 0.5
        for idx, (lx, ly) in enumerate([
            (self.table_front_x + 0.04, half_w - 0.04),
            (self.table_front_x + 0.04, -(half_w - 0.04)),
            (self.table_rear_x - 0.04, half_w - 0.04),
            (self.table_rear_x - 0.04, -(half_w - 0.04)),
        ]):
            self._make_cylinder(f"{ft_scope}/Leg_{idx}", (lx, ly, leg_cz), radius=0.012, height=leg_h, color=(0.14, 0.14, 0.15))

    def _setup_floor_and_walls(self):
        """실제 우드 마루 바닥(Z=-0.225, floor_texture3.jpg) + 엠보싱 벽지 + 좌측 나무 몰딩 + 걸레받이"""
        floor_prim = self._make_cube(
            f"{self.room_root}/RoomFloor",
            pos=(0.20, 0.0, -self.table_h - 0.010),
            scale=(3.0, 3.0, 0.020),
            color=(0.72, 0.70, 0.67),
        )
        if not floor_prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI.Apply(floor_prim)
        if os.path.exists(FLOOR_TEX_PATH):
            self._bind_omnipbr(
                target_prim=floor_prim,
                mat_path="/World/Looks/RealWall_FloorMat",
                diffuse_path=FLOOR_TEX_PATH,
                roughness=0.38,
                tex_scale=(2.2, 2.2),
            )

        walls_scope = f"{self.room_root}/Walls"
        UsdGeom.Scope.Define(self.stage, walls_scope)
        wall_h = 1.80
        wall_cz = -self.table_h + wall_h * 0.5

        wall_specs = [
            ("BackWall", (self.wall_x + 0.010, 0.0, wall_cz), (0.020, 2.6, wall_h)),
            ("LeftWall", (0.20, 1.25, wall_cz), (2.6, 0.020, wall_h)),
            ("RightWall", (0.20, -1.25, wall_cz), (2.6, 0.020, wall_h)),
            ("RearWall", (-1.05, 0.0, wall_cz), (0.020, 2.6, wall_h)),
        ]
        for wname, wpos, wscale in wall_specs:
            wp = self._make_cube(f"{walls_scope}/{wname}", pos=wpos, scale=wscale, color=(0.88, 0.88, 0.87))
            if self.manifest.get("wallpaper_diffuse"):
                self._bind_omnipbr(
                    target_prim=wp,
                    mat_path="/World/Looks/RealWall_WallpaperMat",
                    diffuse_path=self.manifest["wallpaper_diffuse"],
                    normal_path=self.manifest.get("wallpaper_normal"),
                    roughness=0.72,
                    tex_scale=(1.35, 1.35),
                )

        # 좌측 세로 나무 몰딩 (setup1 / topview 기준, Y = +0.285m 부근) & 하단 걸레받이
        molding_v = self._make_cube(
            f"{walls_scope}/LeftWoodMolding",
            pos=(self.wall_x - 0.008, 0.285, wall_cz),
            scale=(0.016, 0.068, wall_h),
            color=(0.60, 0.52, 0.40),
        )
        if self.manifest.get("molding_vert"):
            self._bind_omnipbr(
                target_prim=molding_v,
                mat_path="/World/Looks/RealWall_MoldingVertMat",
                diffuse_path=self.manifest["molding_vert"],
                roughness=0.58,
                tex_scale=(1.2, 1.2),
            )

        baseboard = self._make_cube(
            f"{walls_scope}/Baseboard",
            pos=(self.wall_x - 0.006, 0.0, -self.table_h + 0.040),
            scale=(0.012, 2.6, 0.080),
            color=(0.60, 0.52, 0.40),
        )
        if self.manifest.get("molding_horiz"):
            self._bind_omnipbr(
                target_prim=baseboard,
                mat_path="/World/Looks/RealWall_MoldingHorizMat",
                diffuse_path=self.manifest["molding_horiz"],
                roughness=0.58,
                tex_scale=(1.4, 1.4),
            )

    def _setup_real_blackbox(self, box_prim):
        """실물 검은 상자(9.6x8.3x4.4cm)를 도면 위치 (X=+0.130, Y=+0.205)로 이동하고 blackbox.jpg 질감 + 측면 걸이 탭 적용"""
        if not (box_prim and box_prim.IsValid()):
            return

        # 루트 Transform 이동
        xf_root = UsdGeom.Xformable(box_prim)
        xf_root.ClearXformOpOrder()
        xf_root.AddTranslateOp().Set(Gf.Vec3d(*self.box_pos))
        xf_root.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))

        box_path = box_prim.GetPath().pathString
        # 9.6 x 8.3 x 4.4 cm (벽 두께 3mm)
        parts = [
            ("Bottom",    (0.0,     0.0,     0.002), (0.096, 0.083, 0.004)),
            ("WallNorth", (0.0465,  0.0,     0.022), (0.003, 0.083, 0.044)),
            ("WallSouth", (-0.0465, 0.0,     0.022), (0.003, 0.083, 0.044)),
            ("WallEast",  (0.0,    -0.0400,  0.022), (0.090, 0.003, 0.044)),
            ("WallWest",  (0.0,     0.0400,  0.022), (0.090, 0.003, 0.044)),
        ]
        bb_col = (0.04, 0.04, 0.045)
        for part_name, p_pos, p_scale in parts:
            part_prim = self.stage.GetPrimAtPath(f"{box_path}/{part_name}")
            if not (part_prim and part_prim.IsValid()):
                continue
            xf = UsdGeom.XformCommonAPI(part_prim)
            xf.SetTranslate(Gf.Vec3d(*p_pos))
            xf.SetScale(Gf.Vec3f(*p_scale))
            UsdGeom.Gprim(part_prim).CreateDisplayColorAttr([Gf.Vec3f(*bb_col)])
            if self.manifest.get("blackbox_diffuse"):
                self._bind_omnipbr(
                    target_prim=part_prim,
                    mat_path="/World/Looks/RealWall_BlackBoxMat",
                    diffuse_path=self.manifest["blackbox_diffuse"],
                    normal_path=self.manifest.get("blackbox_normal"),
                    roughness=0.80,
                    tex_scale=(8.0, 8.0),
                )

        # blackbox.jpg 우측 돌출 걸이 탭(SideTab) - 충돌 없음
        tab_prim = self._make_cube(
            f"{box_path}/SideTab",
            pos=(0.0, -0.049, 0.030),
            scale=(0.036, 0.016, 0.003),
            color=bb_col,
        )
        if self.manifest.get("blackbox_diffuse"):
            self._bind_omnipbr(
                target_prim=tab_prim,
                mat_path="/World/Looks/RealWall_BlackBoxMat",
                diffuse_path=self.manifest["blackbox_diffuse"],
                normal_path=self.manifest.get("blackbox_normal"),
                roughness=0.80,
                tex_scale=(8.0, 8.0),
            )

    def _find_or_create_lights(self):
        self.lights = []
        for prim in self.stage.Traverse():
            if prim.IsA(UsdLux.DomeLight) or prim.IsA(UsdLux.RectLight) or prim.IsA(UsdLux.DistantLight) or prim.IsA(UsdLux.SphereLight):
                self.lights.append(prim)

        key_path = "/World/KeyRectLight"
        key_prim = self.stage.GetPrimAtPath(key_path)
        if not (key_prim and key_prim.IsValid()):
            rect_light = UsdLux.RectLight.Define(self.stage, key_path)
            rect_light.CreateWidthAttr().Set(1.1)
            rect_light.CreateHeightAttr().Set(0.8)
            rect_light.CreateIntensityAttr().Set(3000.0)
            self.lights.append(rect_light.GetPrim())

    def _hide_default_ground_planes(self):
        """기존 USD에 포함된 Z=0 기본 회색 바닥 평면(GroundPlane/Plane 등)을 숨겨 책상 높이(22.5cm) 및 마루 바닥 노출"""
        protected_prefixes = (
            "/World/RealWallRoom",
            "/World/so101_dual_cam",
            "/World/RedCube",
            "/World/TargetBox",
            "/World/Table",
            "/World/Looks",
            "/World/PhysicsMaterials",
        )
        hidden = []
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
        for prim in self.stage.Traverse():
            pstr = prim.GetPath().pathString
            if pstr in ("/", "/World", "/PhysicsScene") or any(pstr.startswith(pref) for pref in protected_prefixes):
                continue
            if (
                prim.IsA(UsdLux.LightAPI)
                or prim.IsA(UsdLux.DomeLight)
                or prim.IsA(UsdLux.RectLight)
                or prim.IsA(UsdLux.DistantLight)
                or prim.IsA(UsdLux.SphereLight)
                or prim.IsA(UsdPhysics.Joint)
                or "joint" in pstr.lower()
            ):
                continue
            name_l = prim.GetName().lower()
            is_named_ground = any(k in name_l for k in ("ground", "defaultground", "grid", "environment"))
            is_flat_floor = False
            if prim.IsA(UsdGeom.Plane) or prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cube):
                try:
                    wb = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
                    sz = wb.GetSize()
                    mid_z = wb.GetMidpoint()[2]
                    if (sz[0] > 1.0 or sz[1] > 1.0) and sz[2] < 0.15 and abs(mid_z) < 0.08:
                        is_flat_floor = True
                except Exception:
                    pass
            if is_named_ground or is_flat_floor:
                try:
                    UsdGeom.Imageable(prim).MakeInvisible()
                    if prim.HasAPI(UsdPhysics.CollisionAPI):
                        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                    hidden.append(pstr)
                except Exception:
                    pass
        if hidden:
            print(f"[RealWallRoom] Hidden default ground prims ({len(hidden)}): {hidden[:6]}")

    def build_all(self, box_prim, cube_prim_path="/World/RedCube"):
        """최초 1회 Digital Twin 고정 구조 생성 (테이블·바닥·벽·몰딩·검은상자·파란팔)"""
        if not self.stage.GetPrimAtPath(self.room_root).IsValid():
            UsdGeom.Scope.Define(self.stage, self.room_root)

        self._hide_default_ground_planes()
        self._setup_folding_table()
        self._setup_floor_and_walls()
        self._setup_real_blackbox(box_prim)
        self._find_or_create_lights()

        from domain_randomizer import DomainRandomizer
        self._dr = DomainRandomizer(stage=self.stage)
        self._dr.set_robot_arm_3d_printed_color(color_rgb=(0.08, 0.33, 0.82), jitter=0.0)
        self.randomize_real_red_cube(cube_prim_path)
        self.randomize_soft_ceiling_light()

    def randomize_real_red_cube(self, cube_prim_path="/World/RedCube"):
        """realred1~6.jpg 천 큐브 표면 Diffuse + 직조 노멀맵 무작위 바인딩"""
        cube_prim = self.stage.GetPrimAtPath(cube_prim_path)
        faces = self.manifest.get("real_red_faces", [])
        if not (cube_prim and cube_prim.IsValid() and faces):
            return
        face = random.choice(faces)
        rough_val = float(np.random.uniform(0.80, 0.92))
        self._bind_omnipbr(
            target_prim=cube_prim,
            mat_path="/World/Looks/RealWall_RedClothMat",
            diffuse_path=face["diffuse"],
            normal_path=face.get("normal"),
            roughness=rough_val,
            tex_scale=(33.0, 33.0),
        )

    def randomize_soft_ceiling_light(self):
        """topview.jpg 기준 부드러운 천장등 소프트 섀도우 (DomeLight + 대형 RectLight)"""
        if not self.lights:
            self._find_or_create_lights()
        temp_val = float(np.random.uniform(4200.0, 6200.0))

        for lp in self.lights:
            if lp.IsA(UsdLux.DomeLight):
                intensity = float(np.random.uniform(420.0, 720.0))
            elif lp.IsA(UsdLux.RectLight):
                intensity = float(np.random.uniform(2200.0, 4000.0))
                w_attr = lp.GetAttribute("inputs:width")
                h_attr = lp.GetAttribute("inputs:height")
                if w_attr and w_attr.IsValid():
                    w_attr.Set(float(np.random.uniform(0.90, 1.40)))
                if h_attr and h_attr.IsValid():
                    h_attr.Set(float(np.random.uniform(0.60, 1.00)))

                lx = 0.20 + float(np.random.uniform(-0.35, 0.35))
                ly = float(np.random.uniform(-0.35, 0.35))
                lz = float(np.random.uniform(1.50, 2.20))
                look_mat = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(lx, ly, lz),
                    Gf.Vec3d(0.20, 0.0, 0.0),
                    Gf.Vec3d(0.0, 0.0, 1.0),
                ).GetInverse()
                rot_q = look_mat.ExtractRotationQuat()
                img = rot_q.GetImaginary()
                xf = UsdGeom.Xformable(lp)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, lz))
                xf.AddOrientOp().Set(Gf.Quatf(float(rot_q.GetReal()), float(img[0]), float(img[1]), float(img[2])))
            else:
                intensity = float(np.random.uniform(200.0, 500.0))

            ia = lp.GetAttribute("inputs:intensity")
            if ia and ia.IsValid():
                ia.Set(intensity)
            et = lp.GetAttribute("inputs:enableColorTemperature")
            if et and et.IsValid():
                et.Set(True)
            ta = lp.GetAttribute("inputs:colorTemperature")
            if ta and ta.IsValid():
                ta.Set(temp_val)

    def clear_room_clutter(self):
        parent = self.stage.GetPrimAtPath(self.clutter_root)
        if parent and parent.IsValid():
            for child in list(parent.GetChildren()):
                self.stage.RemovePrim(child.GetPath())

    def randomize_room_clutter(self):
        """기법 B: 콘센트·플러그·디지털 피아노·바닥 전선을 50% 안팎 확률로 On/Off & 위치 지터"""
        self.clear_room_clutter()
        if not self.stage.GetPrimAtPath(self.clutter_root).IsValid():
            UsdGeom.Scope.Define(self.stage, self.clutter_root)

        # 1) 우측 뒤 벽 콘센트 + 검은 플러그/얇은 사선 전선 (setup1.jpg 기준)
        if np.random.rand() < 0.55:
            oy = float(np.random.uniform(-0.26, -0.15))
            oz = float(np.random.uniform(0.16, 0.26))
            ox = self.wall_x - 0.004
            self._make_cube(f"{self.clutter_root}/OutletPlate", (ox, oy, oz), (0.008, 0.085, 0.060), (0.92, 0.92, 0.90))
            self._make_cylinder(
                f"{self.clutter_root}/OutletPlug",
                (ox - 0.012, oy + 0.010, oz),
                radius=0.012,
                height=0.022,
                axis="X",
                color=(0.06, 0.06, 0.07),
            )
            # 플러그 중심(p0)에서 우측 뒤 구석 아래(p1)까지 양 끝점 쿼터니언 정렬로 정확히 연결
            p0 = Gf.Vec3d(ox - 0.014, oy + 0.010, oz)
            p1 = Gf.Vec3d(ox - 0.012, oy - float(np.random.uniform(0.14, 0.22)), -0.02)
            seg_vec = p1 - p0
            seg_len = float(seg_vec.GetLength())
            mid_pt = (p0 + p1) * 0.5
            seg_dir = seg_vec.GetNormalized()
            rot_q = Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), seg_dir).GetQuat()
            q_im = rot_q.GetImaginary()

            cord_geom = UsdGeom.Cylinder.Define(self.stage, f"{self.clutter_root}/OutletCord")
            cord_geom.GetRadiusAttr().Set(0.0022)
            cord_geom.GetHeightAttr().Set(seg_len)
            cord_geom.GetAxisAttr().Set("Z")
            cord_geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([Gf.Vec3f(0.05, 0.05, 0.05)]))
            cord_xf = UsdGeom.Xformable(cord_geom.GetPrim())
            cord_xf.ClearXformOpOrder()
            cord_xf.AddTranslateOp().Set(mid_pt)
            cord_xf.AddOrientOp().Set(Gf.Quatf(float(rot_q.GetReal()), float(q_im[0]), float(q_im[1]), float(q_im[2])))

        # 2) 우측 외곽 검은 디지털 피아노 + 건반 스트립 (setup1.jpg 기준, 테이블 밖 우측 멀리)
        if np.random.rand() < 0.50:
            px = float(np.random.uniform(0.28, 0.46))
            py = float(np.random.uniform(-0.68, -0.50))
            pz = float(np.random.uniform(0.16, 0.30))
            self._make_cube(f"{self.clutter_root}/PianoBody", (px, py, pz), (0.26, 0.24, 0.11), (0.05, 0.05, 0.06))
            self._make_cube(f"{self.clutter_root}/PianoKeys", (px, py + 0.095, pz + 0.052), (0.24, 0.05, 0.012), (0.93, 0.93, 0.91))

        # 3) 테이블 아래 바닥 전선 (좌측/우측 마루 위, Z=-0.220)
        if np.random.rand() < 0.50:
            for ci in range(2):
                side = 1.0 if ci == 0 else -1.0
                cx = float(np.random.uniform(-0.08, 0.35))
                cy = side * float(np.random.uniform(0.30, 0.48))
                col = (0.05, 0.05, 0.05) if np.random.rand() < 0.7 else (0.78, 0.78, 0.78)
                self._make_cylinder(
                    f"{self.clutter_root}/FloorCable_{ci}",
                    (cx, cy, -self.table_h + 0.004),
                    radius=0.0025,
                    height=0.32,
                    axis="X",
                    color=col,
                    rot_xyz=(0.0, 0.0, float(np.random.uniform(-65.0, 65.0))),
                )

        # 4) 테이블 바깥 마루 위 비적색 실내 소품 (책, 소형 박스, 통 등) 랜덤 배치
        floor_palette = [
            (0.14, 0.15, 0.18),
            (0.38, 0.28, 0.18),
            (0.24, 0.30, 0.22),
            (0.55, 0.54, 0.50),
            (0.12, 0.20, 0.34),
        ]
        if np.random.rand() < 0.65:
            num_floor = int(np.random.randint(1, 4))
            for fi in range(num_floor):
                side = 1.0 if np.random.rand() < 0.5 else -1.0
                fx = float(np.random.uniform(-0.05, self.wall_x - 0.14))
                fy = side * float(np.random.uniform(0.34, 0.58))
                sx = float(np.random.uniform(0.08, 0.18))
                sy = float(np.random.uniform(0.07, 0.15))
                sz = float(np.random.uniform(0.04, 0.12))
                col = floor_palette[int(np.random.randint(0, len(floor_palette)))]
                self._make_cube(
                    f"{self.clutter_root}/FloorProp_{fi}",
                    (fx, fy, -self.table_h + sz * 0.5),
                    (sx, sy, sz),
                    color=col,
                    rot_xyz=(0.0, 0.0, float(np.random.uniform(-45.0, 45.0))),
                )

        # 5) 접이식 테이블 뒤쪽/우측 모서리 안전 외곽(스폰 영역 밖) 비적색 소형 방해물
        if np.random.rand() < 0.45:
            col = floor_palette[int(np.random.randint(0, len(floor_palette)))]
            dx = float(np.random.uniform(0.435, 0.465))
            dy = float(np.random.uniform(-0.22, 0.22))
            sx = float(np.random.uniform(0.025, 0.045))
            sy = float(np.random.uniform(0.025, 0.055))
            sz = float(np.random.uniform(0.018, 0.038))
            self._make_cube(
                f"{self.clutter_root}/TableEdgeDistractor",
                (dx, dy, sz * 0.5),
                (sx, sy, sz),
                color=col,
                rot_xyz=(0.0, 0.0, float(np.random.uniform(-40.0, 40.0))),
            )

    def randomize_episode(self, cube_prim_path="/World/RedCube"):
        """매 에피소드 실행: 천장 조명 + 천 큐브 면 텍스처 + 테이블 거칠기/색온도 + 주변 사물 On/Off + 팔 색 지터"""
        self.randomize_soft_ceiling_light()
        self.randomize_real_red_cube(cube_prim_path=cube_prim_path)
        self.randomize_room_clutter()

        # 테이블 거칠기(0.32 ~ 0.60) 및 색온도(diffuse_tint) 변동
        table_mat_shader = self.stage.GetPrimAtPath("/World/Looks/RealWall_TableMat/Shader")
        if table_mat_shader and table_mat_shader.IsValid():
            sh = UsdShade.Shader(table_mat_shader)
            r_in = sh.GetInput("reflection_roughness_constant")
            if r_in:
                r_in.Set(float(np.random.uniform(0.32, 0.60)))
            t_in = sh.GetInput("diffuse_tint")
            if t_in:
                t_in.Set(
                    Gf.Vec3f(
                        float(np.random.uniform(0.92, 1.04)),
                        float(np.random.uniform(0.90, 1.03)),
                        float(np.random.uniform(0.85, 1.02)),
                    )
                )

        if hasattr(self, "_dr"):
            self._dr.set_robot_arm_3d_printed_color(color_rgb=(0.08, 0.33, 0.82), jitter=0.05)


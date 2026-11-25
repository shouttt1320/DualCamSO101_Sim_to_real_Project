#!/usr/bin/env python3
"""
[Module: Domain Randomizer for SO-101 Sim-to-Real]
===================================================
실제 로봇 환경 배포 시의 시각적/물리적 도메인 갭(Sim-to-Real Reality Gap)을 극복하기 위한
Isaac Sim 씬 절차적 도메인 랜덤화 도구:
  1) 방해물(Distractor Objects): 펜, 다른 색 블록, 구슬 등 무작위 스폰/배치/물리 적용
  2) 조명 및 그림자(Lighting): 광원 각도(회전), 조도(Intensity), 색온도(Color Temp) 무작위 변경
  3) 바닥 머티리얼(Tabletop Texture): 실제 하얀 책상 변형 12종 및 우드/스투코 PBR 무작위 바인딩
  4) 큐브 실사 플라스틱/PLA 머티리얼(Cube PBR): 3D 프린팅 적층 결, 노멀맵, 스펙큘러 하이라이트 적용
  5) 카메라 장착 오차 지터(Camera Extrinsics Jitter): 회전(±0.5°) 및 위치(±1.0mm) 미세 지터링
"""

import os
import glob
import json
import random
import numpy as np

import omni.usd
import omni.kit.commands
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, UsdShade, Sdf, Gf, Vt

WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "assets", "textures", "texture_manifest.json")
TEXTURE_PROCESSED_DIR = os.path.join(WORKSPACE_DIR, "assets", "textures", "processed")
POLYHAVEN_CATALOG_PATH = os.path.join(TEXTURE_PROCESSED_DIR, "polyhaven_pbr", "pbr_catalog.json")

# 빨간색(타겟 큐브)을 제외한 방해물 색상 팔레트
NON_RED_COLORS = [
    ("Blue", Gf.Vec3f(0.12, 0.35, 0.88)),
    ("Cyan", Gf.Vec3f(0.05, 0.75, 0.82)),
    ("Green", Gf.Vec3f(0.15, 0.72, 0.22)),
    ("Yellow", Gf.Vec3f(0.92, 0.82, 0.08)),
    ("Purple", Gf.Vec3f(0.58, 0.22, 0.78)),
    ("Charcoal", Gf.Vec3f(0.22, 0.22, 0.24)),
    ("Orange", Gf.Vec3f(0.90, 0.52, 0.10)),
]


class DomainRandomizer:
    """Isaac Sim USD 씬의 시각적 요소를 실시간 랜덤화하는 통합 헬퍼 클래스"""

    def __init__(self, stage=None, distractor_parent="/World/Distractors", bg_parent="/World/BackgroundProps"):
        self.stage = stage if stage is not None else omni.usd.get_context().get_stage()
        self.distractor_parent_path = distractor_parent
        self.bg_parent_path = bg_parent
        self.distractor_prims = []
        self.bg_prims = []
        self._base_cam_poses = {}
        self._find_scene_lights()
        self._load_manifest()

    def _load_manifest(self):
        """텍스처 매니페스트 및 PolyHaven PBR 카탈로그 로드"""
        self.manifest = None
        if os.path.exists(MANIFEST_PATH):
            try:
                with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                    self.manifest = json.load(f)
            except Exception as e:
                print(f"[WARN] Failed to load texture manifest: {e}")

        self.pbr_catalog = []
        if os.path.exists(POLYHAVEN_CATALOG_PATH):
            try:
                with open(POLYHAVEN_CATALOG_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                for item in raw:
                    diff = os.path.join(WORKSPACE_DIR, item["diffuse"]) if not os.path.isabs(item["diffuse"]) else item["diffuse"]
                    nor = item.get("nor_gl")
                    if nor and not os.path.isabs(nor):
                        nor = os.path.join(WORKSPACE_DIR, nor)
                    rough = item.get("rough")
                    if rough and not os.path.isabs(rough):
                        rough = os.path.join(WORKSPACE_DIR, rough)
                    if os.path.exists(diff):
                        self.pbr_catalog.append({
                            "slug": item.get("slug", "pbr"),
                            "diffuse": diff,
                            "normal": nor if (nor and os.path.exists(nor)) else None,
                            "rough": rough if (rough and os.path.exists(rough)) else None,
                        })
            except Exception as e:
                print(f"[WARN] Failed to load PolyHaven PBR catalog: {e}")

    def _find_scene_lights(self):
        """씬 내의 DistantLight / DomeLight / SphereLight / RectLight 탐색"""
        self.lights = []
        for prim in self.stage.Traverse():
            if (
                prim.IsA(UsdLux.LightAPI)
                or prim.IsA(UsdLux.DistantLight)
                or prim.IsA(UsdLux.DomeLight)
                or prim.IsA(UsdLux.SphereLight)
                or prim.IsA(UsdLux.RectLight)
            ):
                self.lights.append(prim)

    def _ensure_key_rect_light(self):
        """실내 천장/스탠드처럼 부드러운 방향성 그림자를 만드는 상단 사선 RectLight 확보"""
        key_path = "/World/KeyRectLight"
        key_prim = self.stage.GetPrimAtPath(key_path)
        if not (key_prim and key_prim.IsValid()):
            rect_light = UsdLux.RectLight.Define(self.stage, key_path)
            rect_light.CreateWidthAttr().Set(0.7)
            rect_light.CreateHeightAttr().Set(0.45)
            rect_light.CreateIntensityAttr().Set(3200.0)
            key_prim = rect_light.GetPrim()
            self._find_scene_lights()
        return key_prim

    # -------------------------------------------------------------------------
    # 1. 조명 및 그림자 각도 랜덤화 (약한 DomeLight + 상단 사선 RectLight 소프트 섀도우)
    # -------------------------------------------------------------------------
    def randomize_lighting(self, intensity_range=(800.0, 2500.0), temp_range=(3500.0, 6500.0), rotate=True):
        """조명 세기, 색온도, 각도(소프트 그림자 방향) 무작위 변경"""
        self._ensure_key_rect_light()
        if not self.lights:
            self._find_scene_lights()

        shared_temp = float(np.random.uniform(temp_range[0], temp_range[1]))

        for light_prim in self.lights:
            if light_prim.IsA(UsdLux.DomeLight):
                # DomeLight는 약한 주변 반사광(Ambient Fill)으로 낮춰 그림자가 씻기지 않게 유지
                intensity_val = float(np.random.uniform(300.0, 650.0))
            elif light_prim.IsA(UsdLux.RectLight):
                intensity_val = float(np.random.uniform(2400.0, 5200.0))
                w_attr = light_prim.GetAttribute("inputs:width")
                h_attr = light_prim.GetAttribute("inputs:height")
                if w_attr and w_attr.IsValid():
                    w_attr.Set(float(np.random.uniform(0.45, 0.90)))
                if h_attr and h_attr.IsValid():
                    h_attr.Set(float(np.random.uniform(0.30, 0.60)))
            else:
                intensity_val = float(np.random.uniform(intensity_range[0], intensity_range[1]))

            intensity_attr = light_prim.GetAttribute("inputs:intensity")
            if intensity_attr and intensity_attr.IsValid():
                intensity_attr.Set(intensity_val)

            enable_temp_attr = light_prim.GetAttribute("inputs:enableColorTemperature")
            if enable_temp_attr and enable_temp_attr.IsValid():
                enable_temp_attr.Set(True)
            temp_attr = light_prim.GetAttribute("inputs:colorTemperature")
            if temp_attr and temp_attr.IsValid():
                temp_attr.Set(shared_temp)

            if rotate and light_prim.IsA(UsdLux.RectLight):
                # 작업 영역 중심(0.15, 0.0, 0.0)을 향하도록 상단 사선 배치
                horiz_r = float(np.random.uniform(0.55, 1.25))
                azim = float(np.random.uniform(0.0, 2.0 * np.pi))
                lx = 0.15 + horiz_r * float(np.cos(azim))
                ly = horiz_r * float(np.sin(azim))
                lz = float(np.random.uniform(1.05, 1.65))

                look_mat = Gf.Matrix4d().SetLookAt(
                    Gf.Vec3d(lx, ly, lz),
                    Gf.Vec3d(0.15, 0.0, 0.0),
                    Gf.Vec3d(0.0, 0.0, 1.0),
                ).GetInverse()
                rot_q = look_mat.ExtractRotationQuat()
                img = rot_q.GetImaginary()

                xf = UsdGeom.Xformable(light_prim)
                xf.ClearXformOpOrder()
                xf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, lz))
                xf.AddOrientOp().Set(Gf.Quatf(float(rot_q.GetReal()), float(img[0]), float(img[1]), float(img[2])))

            elif rotate and (light_prim.IsA(UsdLux.DistantLight) or "distant" in light_prim.GetPath().pathString.lower()):
                xformable = UsdGeom.Xformable(light_prim)
                rot_x = float(np.random.uniform(-65.0, -35.0))
                rot_y = float(np.random.uniform(-45.0, 45.0))
                rot_z = float(np.random.uniform(0.0, 360.0))

                rot_op = None
                for op in xformable.GetOrderedXformOps():
                    if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                        rot_op = op
                        break
                if not rot_op:
                    rot_op = xformable.AddRotateXYZOp()
                rot_op.Set(Gf.Vec3f(rot_x, rot_y, rot_z))

    # -------------------------------------------------------------------------
    # 2. 방해물 (Distractor Objects) 무작위 스폰
    # -------------------------------------------------------------------------
    def spawn_distractors(self, num_objects=2, cube_pos=(0.24, 0.0), box_pos=(0.0, 0.28), min_dist=0.065):
        """큐브 및 상자와 충돌하지 않는 안전 위치에 비(非)빨간색 잡동사니 스폰"""
        self.clear_distractors()

        parent_prim = self.stage.GetPrimAtPath(self.distractor_parent_path)
        if not parent_prim.IsValid():
            UsdGeom.Scope.Define(self.stage, self.distractor_parent_path)

        spawned = 0
        max_attempts = 30

        for i in range(num_objects):
            valid_pos = None
            for _ in range(max_attempts):
                r = np.random.uniform(0.18, 0.32)
                theta = np.radians(np.random.uniform(-60.0, 60.0))
                cand_x = r * np.cos(theta)
                cand_y = r * np.sin(theta)

                d_cube = np.hypot(cand_x - cube_pos[0], cand_y - cube_pos[1])
                d_box = np.hypot(cand_x - box_pos[0], cand_y - box_pos[1])

                if d_cube >= min_dist and d_box >= min_dist:
                    valid_pos = (cand_x, cand_y)
                    break

            if valid_pos is None:
                continue

            shape_type = np.random.choice(["cylinder", "cuboid", "sphere"])
            prim_path = f"{self.distractor_parent_path}/obj_{i}_{shape_type}"
            color_name, color_vec = random.choice(NON_RED_COLORS)

            if shape_type == "cylinder":
                radius = float(np.random.uniform(0.008, 0.012))
                height = float(np.random.uniform(0.040, 0.075))
                geom = UsdGeom.Cylinder.Define(self.stage, prim_path)
                geom.GetRadiusAttr().Set(radius)
                geom.GetHeightAttr().Set(height)
                geom.GetAxisAttr().Set("Z")
                z_pos = height / 2.0
            elif shape_type == "cuboid":
                sx = float(np.random.uniform(0.020, 0.035))
                sy = float(np.random.uniform(0.020, 0.035))
                sz = float(np.random.uniform(0.015, 0.028))
                geom = UsdGeom.Cube.Define(self.stage, prim_path)
                geom.GetSizeAttr().Set(1.0)
                z_pos = sz / 2.0
                xformable = UsdGeom.Xformable(geom.GetPrim())
                scale_op = xformable.AddScaleOp()
                scale_op.Set(Gf.Vec3f(sx, sy, sz))
            else:
                radius = float(np.random.uniform(0.012, 0.018))
                geom = UsdGeom.Sphere.Define(self.stage, prim_path)
                geom.GetRadiusAttr().Set(radius)
                z_pos = radius

            prim = geom.GetPrim()
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(valid_pos[0], valid_pos[1], z_pos))
            xformable.AddRotateZOp().Set(float(np.random.uniform(-180.0, 180.0)))
            geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([color_vec]))

            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).GetMassAttr().Set(0.03)

            self.distractor_prims.append(prim)
            spawned += 1

    def clear_distractors(self):
        """스폰된 방해물 제거"""
        for prim in self.distractor_prims:
            if prim.IsValid():
                self.stage.RemovePrim(prim.GetPath())
        self.distractor_prims.clear()

        parent = self.stage.GetPrimAtPath(self.distractor_parent_path)
        if parent.IsValid():
            for child in parent.GetChildren():
                self.stage.RemovePrim(child.GetPath())

    # -------------------------------------------------------------------------
    # 내부 헬퍼: OmniPBR Triplanar 머티리얼 생성 및 바인딩 (Cube Prim UV 지원)
    # -------------------------------------------------------------------------
    def _bind_omnipbr_material(self, target_prim, mat_path, diffuse_path, normal_path=None, roughness_path=None, roughness=0.45, tex_scale=(1.0, 1.0)):
        """UsdGeom.Cube처럼 UV(st)가 없는 Prim에도 텍스처가 투영되도록 OmniPBR(project_uvw=True) 바인딩"""
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
        shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(diffuse_path))
        shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("world_or_object", Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(float(tex_scale[0]), float(tex_scale[1])))
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0.0)
        if normal_path and os.path.exists(normal_path):
            shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(normal_path))
        if roughness_path and os.path.exists(roughness_path):
            shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(roughness_path))
            shader.CreateInput("roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)

        UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(
            UsdShade.Material(mat_prim),
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        )

    # -------------------------------------------------------------------------
    # 2-B. 외곽 배경 오브젝트 (Background Props) 무작위 스폰 (비충돌)
    # -------------------------------------------------------------------------
    def clear_background_props(self):
        """스폰된 외곽 배경 오브젝트 제거"""
        for prim in self.bg_prims:
            if prim.IsValid():
                self.stage.RemovePrim(prim.GetPath())
        self.bg_prims.clear()

        parent = self.stage.GetPrimAtPath(self.bg_parent_path)
        if parent.IsValid():
            for child in parent.GetChildren():
                self.stage.RemovePrim(child.GetPath())

    def spawn_background_props(self, num_props=6, safe_radius=0.40):
        """로봇 팔 작업 반경 바깥(r >= 0.40m)에 책상 위 배경 물건(책, 머그컵, 상자, 모니터 패널 등) 배치"""
        self.clear_background_props()

        parent_prim = self.stage.GetPrimAtPath(self.bg_parent_path)
        if not parent_prim.IsValid():
            UsdGeom.Scope.Define(self.stage, self.bg_parent_path)

        prop_palette = [
            ("DarkSlate", Gf.Vec3f(0.08, 0.09, 0.11)),
            ("WarmGray",  Gf.Vec3f(0.42, 0.40, 0.38)),
            ("WoodBrown", Gf.Vec3f(0.38, 0.24, 0.14)),
            ("Navy",      Gf.Vec3f(0.10, 0.18, 0.34)),
            ("Beige",     Gf.Vec3f(0.76, 0.70, 0.60)),
            ("Olive",     Gf.Vec3f(0.24, 0.32, 0.20)),
            ("OffWhite",  Gf.Vec3f(0.84, 0.83, 0.80)),
            ("Charcoal",  Gf.Vec3f(0.16, 0.16, 0.18)),
        ]

        prop_types = ["book_stack", "mug", "cardboard_box", "desk_mat", "bottle", "monitor_panel"]
        placed_xy = []

        for i in range(num_props):
            ptype = prop_types[i] if i < len(prop_types) and np.random.rand() < 0.7 else random.choice(prop_types)
            if ptype == "monitor_panel":
                r = float(np.random.uniform(0.62, 0.92))
                theta = np.radians(np.random.uniform(-55.0, 55.0))
            elif ptype == "desk_mat":
                r = float(np.random.uniform(0.44, 0.65))
                theta = np.radians(np.random.uniform(-110.0, 110.0))
            else:
                r = float(np.random.uniform(safe_radius, 0.78))
                theta = np.radians(np.random.uniform(-125.0, 125.0))

            px = r * float(np.cos(theta))
            py = r * float(np.sin(theta))
            if any(np.hypot(px - ox, py - oy) < 0.12 for ox, oy in placed_xy):
                continue
            placed_xy.append((px, py))

            prim_path = f"{self.bg_parent_path}/bg_{i}_{ptype}"
            _, color_vec = random.choice(prop_palette)

            if ptype == "book_stack":
                sx = float(np.random.uniform(0.14, 0.22))
                sy = float(np.random.uniform(0.10, 0.16))
                sz = float(np.random.uniform(0.025, 0.065))
                geom = UsdGeom.Cube.Define(self.stage, prim_path)
                geom.GetSizeAttr().Set(1.0)
                z_pos = sz / 2.0
                xf = UsdGeom.Xformable(geom.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(px, py, z_pos))
                xf.AddRotateZOp().Set(float(np.random.uniform(-180.0, 180.0)))
                xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            elif ptype == "mug":
                rad = float(np.random.uniform(0.036, 0.048))
                h = float(np.random.uniform(0.085, 0.115))
                geom = UsdGeom.Cylinder.Define(self.stage, prim_path)
                geom.GetRadiusAttr().Set(rad)
                geom.GetHeightAttr().Set(h)
                geom.GetAxisAttr().Set("Z")
                xf = UsdGeom.Xformable(geom.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(px, py, h / 2.0))
                xf.AddRotateZOp().Set(float(np.random.uniform(-180.0, 180.0)))
            elif ptype == "cardboard_box":
                sx = float(np.random.uniform(0.12, 0.22))
                sy = float(np.random.uniform(0.10, 0.18))
                sz = float(np.random.uniform(0.07, 0.14))
                geom = UsdGeom.Cube.Define(self.stage, prim_path)
                geom.GetSizeAttr().Set(1.0)
                color_vec = Gf.Vec3f(0.55, 0.42, 0.28)
                xf = UsdGeom.Xformable(geom.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(px, py, sz / 2.0))
                xf.AddRotateZOp().Set(float(np.random.uniform(-180.0, 180.0)))
                xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            elif ptype == "desk_mat":
                sx = float(np.random.uniform(0.24, 0.36))
                sy = float(np.random.uniform(0.18, 0.28))
                sz = 0.004
                geom = UsdGeom.Cube.Define(self.stage, prim_path)
                geom.GetSizeAttr().Set(1.0)
                xf = UsdGeom.Xformable(geom.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(px, py, sz / 2.0))
                xf.AddRotateZOp().Set(float(np.random.uniform(-45.0, 45.0)))
                xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            elif ptype == "bottle":
                rad = float(np.random.uniform(0.028, 0.036))
                h = float(np.random.uniform(0.14, 0.22))
                geom = UsdGeom.Cylinder.Define(self.stage, prim_path)
                geom.GetRadiusAttr().Set(rad)
                geom.GetHeightAttr().Set(h)
                geom.GetAxisAttr().Set("Z")
                xf = UsdGeom.Xformable(geom.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(px, py, h / 2.0))
            else:  # monitor_panel (책상 뒤쪽 모니터/파티션 실루엣)
                sx = 0.03
                sy = float(np.random.uniform(0.38, 0.56))
                sz = float(np.random.uniform(0.26, 0.38))
                geom = UsdGeom.Cube.Define(self.stage, prim_path)
                geom.GetSizeAttr().Set(1.0)
                color_vec = Gf.Vec3f(0.05, 0.05, 0.06)
                xf = UsdGeom.Xformable(geom.GetPrim())
                xf.AddTranslateOp().Set(Gf.Vec3d(px, py, sz / 2.0 + 0.04))
                xf.AddRotateZOp().Set(float(np.degrees(theta)))
                xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))

            prim = geom.GetPrim()
            geom.GetDisplayColorAttr().Set(Vt.Vec3fArray([color_vec]))

            # 일부 배경 물체에 실제 PolyHaven PBR 텍스처 바인딩 (단조로운 단색 탈피)
            if self.pbr_catalog and ptype in ("book_stack", "cardboard_box", "desk_mat") and np.random.rand() < 0.7:
                pbr = random.choice(self.pbr_catalog)
                self._bind_omnipbr_material(
                    target_prim=prim,
                    mat_path=f"/World/Looks/BgPropMat_{i}",
                    diffuse_path=pbr["diffuse"],
                    normal_path=pbr.get("normal"),
                    roughness_path=pbr.get("rough"),
                    roughness=float(np.random.uniform(0.40, 0.80)),
                    tex_scale=(1.5, 1.5),
                )

            self.bg_prims.append(prim)

    # -------------------------------------------------------------------------
    # 3. 테이블 바닥 PBR 텍스처 무작위 바인딩 (실물 바닥 + PolyHaven PBR + 기존 매니페스트 혼합)
    # -------------------------------------------------------------------------
    def randomize_table_texture(self, table_prim_path="/World/Table", mode="white_dominant"):
        """
        테이블 바닥 Prim(/World/Table)에 실물 바닥 + PolyHaven CC0 PBR(Diffuse/Normal/Roughness) +
        기존 하얀 책상/우드 텍스처를 혼합하여 무작위 바인딩
        """
        table_prim = self.stage.GetPrimAtPath(table_prim_path)
        if not table_prim.IsValid():
            for p in self.stage.Traverse():
                name_l = p.GetName().lower()
                if "table" in name_l or "ground" in name_l:
                    table_prim = p
                    break

        if not (table_prim and table_prim.IsValid()):
            return

        chosen_tex = None
        chosen_nor = None
        chosen_rough = None

        real_floor_textures = [
            p for p in [
                os.path.join(WORKSPACE_DIR, "assets", "textures", "floor_texture3.jpg"),
                os.path.join(WORKSPACE_DIR, "assets", "textures", "floor_texture3_rot90.jpg"),
            ]
            if os.path.exists(p)
        ]

        roll = float(np.random.rand())
        if mode == "white_dominant":
            if real_floor_textures and roll < 0.30:
                chosen_tex = random.choice(real_floor_textures)
            elif self.pbr_catalog and roll < 0.75:
                pbr = random.choice(self.pbr_catalog)
                chosen_tex = pbr["diffuse"]
                chosen_nor = pbr.get("normal")
                chosen_rough = pbr.get("rough")
        elif mode == "all" and self.pbr_catalog and roll < 0.55:
            pbr = random.choice(self.pbr_catalog)
            chosen_tex = pbr["diffuse"]
            chosen_nor = pbr.get("normal")
            chosen_rough = pbr.get("rough")

        if not chosen_tex and self.manifest:
            whites = [p for p in self.manifest.get("white_desk_textures", []) if "_nor_" not in os.path.basename(p)]
            generals = [p for p in self.manifest.get("general_tabletop_textures", []) if "_nor_" not in os.path.basename(p)]

            if mode == "white_only" and whites:
                chosen_tex = random.choice(whites)
            elif mode == "white_dominant" and whites:
                if np.random.rand() < 0.7 or not generals:
                    chosen_tex = random.choice(whites)
                else:
                    chosen_tex = random.choice(generals)
            elif whites or generals:
                chosen_tex = random.choice(whites + generals)

            if chosen_tex:
                cand_nor = chosen_tex.replace("_diff_", "_nor_").replace(".jpg", ".png")
                if os.path.exists(cand_nor):
                    chosen_nor = cand_nor

        if not chosen_tex:
            all_files = [
                p for p in glob.glob(os.path.join(TEXTURE_PROCESSED_DIR, "**", "*.jpg"), recursive=True)
                if "_nor_" not in os.path.basename(p)
            ]
            if not all_files:
                return
            chosen_tex = random.choice(all_files)

        scale_uv = float(np.random.uniform(1.4, 3.4))
        roughness_val = float(np.random.uniform(0.30, 0.80))
        self._bind_omnipbr_material(
            target_prim=table_prim,
            mat_path="/World/Looks/RandomTableMat",
            diffuse_path=chosen_tex,
            normal_path=chosen_nor,
            roughness_path=chosen_rough,
            roughness=roughness_val,
            tex_scale=(scale_uv, scale_uv),
        )

    # -------------------------------------------------------------------------
    # 4. 큐브 실사 플라스틱/PLA 머티리얼 적용 (3D 프린팅 결, 노멀맵, 스펙큘러)
    # -------------------------------------------------------------------------
    def randomize_cube_material(self, cube_prim_path="/World/RedCube"):
        """
        단순 무광 빨강 대신 실제 3D 프린터 출력물(PLA)의 층결, 노멀맵, 하이라이트 반사 적용
        """
        cube_prim = self.stage.GetPrimAtPath(cube_prim_path)
        if not (cube_prim and cube_prim.IsValid()):
            return

        diff_files = []
        nor_file = None
        if self.manifest and "cube_materials" in self.manifest:
            cm = self.manifest["cube_materials"]
            diff_files = cm.get("diffuse_maps", [])
            nor_files = cm.get("normal_maps", [])
            if nor_files:
                nor_file = nor_files[0]

        if not diff_files:
            return

        chosen_diff = random.choice(diff_files)
        roughness_val = float(np.random.uniform(0.25, 0.45))
        self._bind_omnipbr_material(
            target_prim=cube_prim,
            mat_path="/World/Looks/RealisticCubeMat",
            diffuse_path=chosen_diff,
            normal_path=nor_file,
            roughness=roughness_val,
            tex_scale=(33.0, 33.0),
        )

    # -------------------------------------------------------------------------
    # 5. 카메라 장착 각도 오차 (Extrinsics Jittering)
    # -------------------------------------------------------------------------
    def jitter_camera_pose(self, cam_prim, rot_deg_range=0.5, trans_mm_range=1.0):
        """
        손목 카메라의 3D 프린터 조립 오차 모사 (Roll/Pitch/Yaw ±0.5°, XYZ ±1.0mm)
        - 최초 호출 시 기준(base) 포즈를 캐시하여 에피소드 반복 시 오차가 누적되지 않도록 방지
        """
        if not (cam_prim and cam_prim.IsValid()):
            return

        cam_path = cam_prim.GetPath().pathString
        xformable = UsdGeom.Xformable(cam_prim)
        ops = xformable.GetOrderedXformOps()

        t_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeTranslate), None)
        r_op = next((op for op in ops if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ), None)

        if cam_path not in self._base_cam_poses:
            base_t = tuple(t_op.Get()) if t_op is not None else None
            base_r = tuple(r_op.Get()) if r_op is not None else None
            self._base_cam_poses[cam_path] = {"t": base_t, "r": base_r}

        base = self._base_cam_poses[cam_path]

        d_roll = float(np.random.uniform(-rot_deg_range, rot_deg_range))
        d_pitch = float(np.random.uniform(-rot_deg_range, rot_deg_range))
        d_yaw = float(np.random.uniform(-rot_deg_range, rot_deg_range))
        d_x = float(np.random.uniform(-trans_mm_range, trans_mm_range)) * 0.001
        d_y = float(np.random.uniform(-trans_mm_range, trans_mm_range)) * 0.001
        d_z = float(np.random.uniform(-trans_mm_range, trans_mm_range)) * 0.001

        if t_op is not None and base["t"] is not None:
            bt = base["t"]
            t_op.Set(Gf.Vec3d(bt[0] + d_x, bt[1] + d_y, bt[2] + d_z))
        if r_op is not None and base["r"] is not None:
            br = base["r"]
            r_op.Set(Gf.Vec3f(br[0] + d_roll, br[1] + d_pitch, br[2] + d_yaw))

    # -------------------------------------------------------------------------
    # 6. 로봇 팔 3D 프린트 파트 색상 변경 (모터 sts3215 제외)
    # -------------------------------------------------------------------------
    def set_robot_arm_3d_printed_color(self, color_rgb=(0.06, 0.25, 0.78), jitter=0.03, robot_root="/World/so101_dual_cam"):
        """
        로봇 팔에서 모터(sts3215) 및 카메라 마운트 등을 제외하고
        3D 프린팅된 파트(tn__3d_printed_ 머티리얼 바인딩 대상)만 지정 색상(기본 파란색)으로 변경
        """
        if not hasattr(self, "_printed_arm_targets"):
            self._printed_arm_targets = []
            for p in self.stage.Traverse(Usd.TraverseInstanceProxies()):
                if p.IsA(UsdGeom.Mesh) and robot_root in p.GetPath().pathString:
                    m, _ = UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial()
                    if m and "tn__3d_printed_" in m.GetPath().pathString:
                        cur = p
                        while cur and cur.IsInstanceProxy():
                            cur = cur.GetParent()
                        if cur and cur.IsValid():
                            self._printed_arm_targets.append(cur)
            # 중복 제거
            seen = set()
            unique_targets = []
            for prim in self._printed_arm_targets:
                ps = prim.GetPath().pathString
                if ps not in seen:
                    seen.add(ps)
                    unique_targets.append(prim)
            self._printed_arm_targets = unique_targets

        if not self._printed_arm_targets:
            return

        r = float(np.clip(color_rgb[0] + np.random.uniform(-jitter, jitter), 0.0, 1.0)) if jitter > 0 else float(color_rgb[0])
        g = float(np.clip(color_rgb[1] + np.random.uniform(-jitter, jitter), 0.0, 1.0)) if jitter > 0 else float(color_rgb[1])
        b = float(np.clip(color_rgb[2] + np.random.uniform(-jitter, jitter), 0.0, 1.0)) if jitter > 0 else float(color_rgb[2])

        mat_path = "/World/Looks/Blue3DPrintedMat"
        mat = UsdShade.Material.Define(self.stage, mat_path)
        pbr = UsdShade.Shader.Define(self.stage, f"{mat_path}/PBRShader")
        pbr.CreateIdAttr("UsdPreviewSurface")
        pbr.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(r, g, b))
        pbr.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(np.random.uniform(0.35, 0.55)) if jitter > 0 else 0.45)
        pbr.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
        mat.CreateSurfaceOutput().ConnectToSource(pbr.ConnectableAPI(), "surface")

        for target_prim in self._printed_arm_targets:
            UsdShade.MaterialBindingAPI.Apply(target_prim).Bind(mat, bindingStrength=UsdShade.Tokens.strongerThanDescendants)

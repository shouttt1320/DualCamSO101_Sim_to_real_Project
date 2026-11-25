#!/usr/bin/env python3
"""
[Texture Preprocessing & Manifest Pipeline]
============================================
1. 다운로드된 고해상도(4K~7K) 및 EXR 텍스처를 Isaac Sim 실시간 렌더링에 최적화된 1024x1024 / 2048x2048로 변환.
2. 실제 하얀 책상 기반의 다양한 변형(얼룩, 연필자국, 스크래치, 커피링) 텍스처 세트 자동 생성.
3. 3D 프린팅(PLA) 적층 결, 미세 스크래치, 스펙큘러 하이라이트를 모사한 큐브 전용 PBR 텍스처 생성.
4. 모든 텍스처를 분류하여 JSON 매니페스트(texture_manifest.json)로 자동 인덱싱.
"""

import os
import glob
import json
import random
import re
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFilter

WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
RAW_DIR = os.path.join(WORKSPACE_DIR, "assets", "textures")
PROCESSED_DIR = os.path.join(RAW_DIR, "processed")
TABLETOP_DIR = os.path.join(PROCESSED_DIR, "tabletop")
WHITE_DESK_DIR = os.path.join(PROCESSED_DIR, "white_desk")
CUBE_DIR = os.path.join(PROCESSED_DIR, "cube")
MANIFEST_PATH = os.path.join(RAW_DIR, "texture_manifest.json")


def ensure_dirs():
    for d in [TABLETOP_DIR, WHITE_DESK_DIR, CUBE_DIR]:
        os.makedirs(d, exist_ok=True)


def crop_and_resize(img_pil, target_size=(1024, 1024)):
    """가운데 중심 정방형 크롭 및 리사이즈"""
    w, h = img_pil.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = img_pil.crop((left, top, left + min_dim, top + min_dim))
    return cropped.resize(target_size, Image.Resampling.LANCZOS)


def add_subtle_desk_imperfections(img_pil, seed=None):
    """하얀 책상 위에 자연스러운 미세 얼룩(연필선, 먼지, 연한 컵자국) 오버레이"""
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        
    img = img_pil.convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # 1. 옅은 커피/물컵 링 자국 (0~2개)
    num_rings = np.random.choice([0, 1, 2], p=[0.2, 0.5, 0.3])
    for _ in range(num_rings):
        cx = np.random.randint(150, w - 150)
        cy = np.random.randint(150, h - 150)
        radius = np.random.randint(70, 140)
        thickness = np.random.randint(4, 9)
        
        mask = Image.new("L", (w, h), 0)
        mdraw = ImageDraw.Draw(mask)
        bbox = [cx - radius, cy - radius, cx + radius, cy + radius]
        ring_alpha = np.random.randint(40, 110)
        mdraw.ellipse(bbox, outline=ring_alpha, width=thickness)
        
        # 내부 물얼룩
        if np.random.rand() > 0.4:
            inner_bbox = [cx - radius + 10, cy - radius + 10, cx + radius - 10, cy + radius - 10]
            mdraw.ellipse(inner_bbox, fill=int(ring_alpha * 0.35))
            
        mask = mask.filter(ImageFilter.GaussianBlur(radius=2.5))
        stain_color = (110, 80, 50) if np.random.rand() > 0.3 else (140, 140, 145)  # 커피 or 물때
        stain_layer = Image.new("RGB", (w, h), stain_color)
        img.paste(stain_layer, (0, 0), mask)

    # 2. 연필/볼펜 미세 스크래치 선 (3~15개)
    num_lines = np.random.randint(3, 16)
    for _ in range(num_lines):
        x1 = np.random.randint(50, w - 50)
        y1 = np.random.randint(50, h - 50)
        length = np.random.randint(15, 70)
        angle = np.random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        gray = np.random.randint(90, 180)
        draw.line([x1, y1, x2, y2], fill=(gray, gray, gray), width=1)

    # 3. 미세 먼지/스머지 (10~40개)
    num_dots = np.random.randint(10, 45)
    for _ in range(num_dots):
        px = np.random.randint(0, w)
        py = np.random.randint(0, h)
        r = np.random.choice([1, 2])
        g = np.random.randint(120, 190)
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(g, g, g))

    return img


def process_downloaded_textures():
    """사용자가 다운로드한 원본 텍스처(JPG, JPEG, PNG, WEBP)를 큐브와 테이블로 자동 분류하여 변환"""
    print(">>> [1/4] 다운로드 텍스처(큐브 및 테이블 텍스처) 분류 및 변환 중...")
    
    # 1. assets/textures 루트의 모든 이미지 파일 검색
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
    raw_files = []
    for ext in extensions:
        raw_files.extend(glob.glob(os.path.join(RAW_DIR, ext)))
        raw_files.extend(glob.glob(os.path.join(RAW_DIR, ext.upper())))

    for f in raw_files:
        basename = os.path.basename(f)
        name_lower = basename.lower()
        
        # 안전한 파일명 슬러그 생성 (공백 제거)
        safe_name = name_lower.replace(" ", "_").replace("-", "_")
        for ext in [".jpeg", ".jpg", ".png", ".webp"]:
            safe_name = safe_name.replace(ext, "")
            
        try:
            with Image.open(f) as img:
                img_rgb = img.convert("RGB")
                
                # (A) 큐브 텍스처 판별 ('cube' 또는 단어 단위의 'red' 포함, 단 wood/floor/table/desk 제외)
                is_cube = bool(re.search(r'\b(cube|red)\b', name_lower) or "cube" in name_lower)
                if any(k in name_lower for k in ["wood", "floor", "table", "desk", "wall", "stucco"]):
                    is_cube = False

                if is_cube:
                    resized = crop_and_resize(img_rgb, (512, 512))
                    out_path = os.path.join(CUBE_DIR, f"cube_diffuse_user_{safe_name}.png")
                    resized.save(out_path, quality=95)
                    print(f"  🟥 [Cube Texture] 변환 완료: cube_diffuse_user_{safe_name}.png (512x512)")
                
                # (B) 테이블 텍스처 (하얀 책상 또는 일반 테이블)
                else:
                    resized = crop_and_resize(img_rgb, (1024, 1024))
                    arr = np.array(resized)
                    # 밝기 판별 (평균 210 이상이거나 white 포함 시 하얀 책상으로 분류)
                    if "white" in name_lower or np.mean(arr) > 210:
                        out_path = os.path.join(WHITE_DESK_DIR, f"white_desk_{safe_name}.jpg")
                        resized.save(out_path, quality=92)
                        print(f"  ⚪ [White Desk] 변환 완료: white_desk_{safe_name}.jpg (1024x1024)")
                    else:
                        out_path = os.path.join(TABLETOP_DIR, f"tabletop_{safe_name}.jpg")
                        resized.save(out_path, quality=92)
                        print(f"  🪵 [Tabletop] 변환 완료: tabletop_{safe_name}.jpg (1024x1024)")
        except Exception as e:
            print(f"  ⚠️ 변환 실패 ({f}): {e}")

    # 2. Poly Haven 폴더 처리 (white_stucco, polystyrene)
    blend_dirs = glob.glob(os.path.join(RAW_DIR, "*_4k.blend", "textures"))
    for bdir in blend_dirs:
        prefix = os.path.basename(os.path.dirname(bdir)).replace("_4k.blend", "")
        diff_files = glob.glob(os.path.join(bdir, "*diff*.jpg")) + glob.glob(os.path.join(bdir, "*diff*.png"))
        for df in diff_files:
            try:
                with Image.open(df) as img:
                    resized = crop_and_resize(img.convert("RGB"), (1024, 1024))
                    out_path = os.path.join(WHITE_DESK_DIR, f"white_desk_{prefix}_diff_1k.jpg")
                    resized.save(out_path, quality=92)
                    print(f"  ⚪ [White Desk - Poly Haven] 변환: white_desk_{prefix}_diff_1k.jpg")
            except Exception as e:
                print(f"  ⚠️ {df} 실패: {e}")

        nor_files = glob.glob(os.path.join(bdir, "*nor*.exr"))
        for nf in nor_files:
            try:
                img_exr = cv2.imread(nf, cv2.IMREAD_UNCHANGED)
                if img_exr is not None:
                    img_exr = cv2.cvtColor(img_exr, cv2.COLOR_BGR2RGB)
                    if img_exr.dtype != np.uint8:
                        img_exr = np.clip(img_exr * 255.0, 0, 255).astype(np.uint8)
                    resized = cv2.resize(img_exr, (1024, 1024), interpolation=cv2.INTER_AREA)
                    out_path = os.path.join(WHITE_DESK_DIR, f"white_desk_{prefix}_nor_1k.png")
                    Image.fromarray(resized).save(out_path)
                    print(f"  ⚪ [White Desk Normal] 변환: white_desk_{prefix}_nor_1k.png")
            except Exception as e:
                print(f"  ⚠️ {nf} 실패: {e}")


def generate_white_desk_variations(num_variations=12):
    """실제 하얀 책상 환경을 위한 12가지 고유 변형 텍스처 생성"""
    print(f">>> [2/4] 실제 하얀 책상 기반 랜덤 변형 텍스처 {num_variations}종 생성 중...")
    
    # 베이스 이미지 후보: white_stucco 또는 절차적 화이트
    base_stucco_path = os.path.join(TABLETOP_DIR, "white_stucco_diff_1k.jpg")
    has_base = os.path.exists(base_stucco_path)

    for i in range(1, num_variations + 1):
        if has_base and i % 2 == 0:
            base_img = Image.open(base_stucco_path).copy()
        else:
            # 베이스 화이트(RGB: 232~244) + 가우시안 텍스처
            arr = np.random.normal(loc=np.random.randint(230, 242), scale=2.5, size=(1024, 1024, 3))
            arr = np.clip(arr, 215, 250).astype(np.uint8)
            base_img = Image.fromarray(arr)

        var_img = add_subtle_desk_imperfections(base_img, seed=100 + i)
        out_name = f"white_desk_pattern_{i:02d}.jpg"
        out_path = os.path.join(WHITE_DESK_DIR, out_name)
        var_img.save(out_path, quality=94)
        print(f"  ✅ 생성 완료: {out_name}")


def generate_cube_realistic_materials():
    """3D 프린팅(PLA) 적층 결, 미세 스크래치, 스펙큘러 노멀맵 생성"""
    print(">>> [3/4] 3D 프린팅(PLA) 큐브 전용 실사 PBR 텍스처 생성 중...")
    w, h = 512, 512

    # 1. 기본 레드 변형들 (Crimson, Coral Red, Scarlet, Tomato Red)
    red_palette = [
        ("pure_red", [225, 25, 20]),
        ("crimson", [185, 20, 25]),
        ("scarlet", [240, 45, 15]),
        ("coral_red", [220, 55, 40]),
    ]
    for name, col in red_palette:
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[:, :] = col
        img = Image.fromarray(base)
        img.save(os.path.join(CUBE_DIR, f"cube_diffuse_{name}.png"))

    # 2. PLA 적층 결(Layer Lines) 디퓨즈 + 노멀 맵
    # 0.2mm 레이어 결을 모사하는 가로 줄무늬 사인파
    y = np.linspace(0, 120, h)
    lines = (np.sin(y * 2 * np.pi) + 1.0) * 0.5  # 0~1
    
    # PLA Layer Diffuse (기본 레드에 미세 명암)
    pla_diff = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(3):
        base_val = red_palette[0][1][c]
        pla_diff[:, :, c] = np.clip(base_val * (0.92 + 0.16 * lines[:, None]), 0, 255).astype(np.uint8)
    Image.fromarray(pla_diff).save(os.path.join(CUBE_DIR, "cube_diffuse_pla_layers.png"))

    # PLA Layer Normal Map (R=128, G=y-slope, B=255)
    slope = np.cos(y * 2 * np.pi) * 0.4  # 기울기
    nor_map = np.zeros((h, w, 3), dtype=np.uint8)
    nor_map[:, :, 0] = 128  # X normal
    nor_map[:, :, 1] = np.clip((slope[:, None] * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)  # Y normal
    nor_map[:, :, 2] = 255  # Z normal (위쪽)
    Image.fromarray(nor_map).save(os.path.join(CUBE_DIR, "cube_normal_pla_layers.png"))

    # 3. 큐브 표면 거칠기(Roughness Map: 0.25 ~ 0.45 플라스틱 반사)
    rough = np.random.normal(loc=90, scale=8, size=(h, w)).clip(65, 125).astype(np.uint8)
    # 모서리나 일부에 스크래치
    r_img = Image.fromarray(rough)
    r_draw = ImageDraw.Draw(r_img)
    for _ in range(12):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2 = x1 + np.random.randint(-30, 30)
        y2 = y1 + np.random.randint(-30, 30)
        r_draw.line([x1, y1, x2, y2], fill=np.random.randint(140, 200), width=1)
    r_img.save(os.path.join(CUBE_DIR, "cube_roughness_plastic.png"))

    print("  ✅ 큐브 PBR 텍스처 (디퓨즈 5종, 노멀맵, 거칠기맵) 생성 완료!")


def build_manifest():
    """모든 텍스처를 단일 JSON 매니페스트로 인덱싱"""
    print(">>> [4/4] 텍스처 매니페스트(JSON) 생성 중...")
    
    white_desks = sorted(glob.glob(os.path.join(WHITE_DESK_DIR, "*.jpg")) + glob.glob(os.path.join(WHITE_DESK_DIR, "*.png")))
    tabletops = sorted(glob.glob(os.path.join(TABLETOP_DIR, "*.jpg")) + glob.glob(os.path.join(TABLETOP_DIR, "*.png")))
    cube_diffuses = sorted(glob.glob(os.path.join(CUBE_DIR, "cube_diffuse_*.png")))
    cube_normals = sorted(glob.glob(os.path.join(CUBE_DIR, "cube_normal_*.png")))
    cube_roughness = sorted(glob.glob(os.path.join(CUBE_DIR, "cube_roughness_*.png")))

    manifest = {
        "white_desk_textures": [os.path.abspath(p) for p in white_desks],
        "general_tabletop_textures": [os.path.abspath(p) for p in tabletops],
        "cube_materials": {
            "diffuse_maps": [os.path.abspath(p) for p in cube_diffuses],
            "normal_maps": [os.path.abspath(p) for p in cube_normals],
            "roughness_maps": [os.path.abspath(p) for p in cube_roughness],
        },
        "stats": {
            "num_white_desks": len(white_desks),
            "num_general_tabletops": len(tabletops),
            "num_cube_variants": len(cube_diffuses),
        }
    }

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print(f"🎉 텍스처 매니페스트 저장 완료: {MANIFEST_PATH}")
    print(f"  • 하얀 책상 변형: {manifest['stats']['num_white_desks']}개")
    print(f"  • 일반 테이블(우드/매트/스투코): {manifest['stats']['num_general_tabletops']}개")
    print(f"  • 큐브 PBR 머티리얼 세트: {manifest['stats']['num_cube_variants']}개")
    print("=" * 70)


def main():
    ensure_dirs()
    process_downloaded_textures()
    generate_white_desk_variations(num_variations=12)
    generate_cube_realistic_materials()
    build_manifest()


if __name__ == "__main__":
    main()


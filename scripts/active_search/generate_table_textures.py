#!/usr/bin/env python3
"""
[Procedural Texture Generator]
Sim-to-Real 도메인 랜덤화를 위한 테이블 상판 텍스처(우드, 얼룩, 스크래치, 매트) 자동 생성기.
외부 다운로드 없이 NumPy와 PIL만으로 고해상도(1024x1024) 텍스처를 즉시 생성합니다.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def create_wood_texture(width=1024, height=1024):
    """자연스러운 원목(Wood Grain) 텍스처 생성"""
    x = np.linspace(0, 10, width)
    y = np.linspace(0, 40, height)
    xx, yy = np.meshgrid(x, y)
    
    # 나이테 및 결 생성 (주기적 정현파 + 노이즈 왜곡)
    noise = np.sin(xx * 2.5 + np.sin(yy * 0.4) * 3.0) + 0.3 * np.sin(xx * 10.0 + yy * 0.8)
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    
    # 오크 원목 컬러 매핑 (Base: 따뜻한 브라운 #c29b68 ~ 어두운 결 #7a5832)
    base_col = np.array([194, 155, 104], dtype=np.float32)
    grain_col = np.array([115, 80, 45], dtype=np.float32)
    
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for c in range(3):
        ch = base_col[c] * (1.0 - 0.5 * noise) + grain_col[c] * (0.5 * noise)
        rgb[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
        
    img = Image.fromarray(rgb)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.8))
    return img


def add_coffee_ring(img, center=(512, 512), radius=120, thickness=12, intensity=0.75):
    """커피/음료수 컵 바닥 링 얼룩 합성"""
    mask = Image.new("L", img.size, 0)
    draw = ImageDraw.Draw(mask)
    
    # 컵 테두리 원형 고리
    bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.ellipse(bbox, outline=int(255 * intensity), width=thickness)
    
    # 내부 연한 고임 얼룩
    inner_bbox = [center[0] - radius + 15, center[1] - radius + 15, center[0] + radius - 15, center[1] + radius - 15]
    draw.ellipse(inner_bbox, fill=int(80 * intensity))
    
    # 자연스러운 번짐 블러 효과
    mask = mask.filter(ImageFilter.GaussianBlur(radius=3.5))
    
    # 커피 갈색 색상 (#5a381e) 오버레이
    stain_layer = Image.new("RGB", img.size, (90, 56, 30))
    img.paste(stain_layer, (0, 0), mask)
    return img


def add_scratches_and_dust(img, num_scratches=25, num_specks=200):
    """책상 표면 미세 스크래치 및 먼지 점 합성"""
    draw = ImageDraw.Draw(img)
    w, h = img.size
    
    # 스크래치 선
    for _ in range(num_scratches):
        x1 = np.random.randint(0, w)
        y1 = np.random.randint(0, h)
        length = np.random.randint(20, 140)
        angle = np.random.uniform(0, 2 * np.pi)
        x2 = int(x1 + length * np.cos(angle))
        y2 = int(y1 + length * np.sin(angle))
        
        # 밝은 긁힘 또는 어두운 긁힘
        gray = np.random.choice([60, 210])
        draw.line([x1, y1, x2, y2], fill=(gray, gray, gray), width=np.random.choice([1, 2]))
        
    # 작은 먼지/오염 점
    for _ in range(num_specks):
        px = np.random.randint(0, w)
        py = np.random.randint(0, h)
        r = np.random.choice([1, 2])
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(80, 80, 80))
        
    return img


def create_white_desk_texture(width=1024, height=1024):
    """흰색/아이보리 사무용 책상 (커피 링 얼룩 + 스크래치)"""
    # 기본 베이지 화이트 (#e8e6e1) + 미세 노이즈
    base = np.random.normal(loc=232, scale=3, size=(height, width, 3))
    base = np.clip(base, 215, 245).astype(np.uint8)
    img = Image.fromarray(base)
    
    # 커피 자국 2개 랜덤 배치
    img = add_coffee_ring(img, center=(320, 420), radius=110, thickness=10, intensity=0.65)
    img = add_coffee_ring(img, center=(740, 680), radius=85, thickness=8, intensity=0.45)
    img = add_scratches_and_dust(img, num_scratches=35, num_specks=150)
    return img


def create_dark_desk_mat(width=1024, height=1024):
    """어두운 차콜 장패드/고무 매트 (마모 및 물자국 얼룩)"""
    # 짙은 차콜 그레이 (#2c2e33)
    base = np.random.normal(loc=46, scale=4, size=(height, width, 3))
    base = np.clip(base, 30, 65).astype(np.uint8)
    img = Image.fromarray(base)
    
    # 먼지/마모 자국
    img = add_coffee_ring(img, center=(600, 350), radius=130, thickness=14, intensity=0.40)
    img = add_scratches_and_dust(img, num_scratches=20, num_specks=250)
    return img


def create_light_wood_craft(width=1024, height=1024):
    """밝은 자작나무 합판 (Plywood / Craft Table)"""
    wood = create_wood_texture(width, height)
    # 톤을 밝게 조절
    arr = np.array(wood, dtype=np.float32) * 1.15
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    
    img = add_coffee_ring(img, center=(450, 520), radius=95, thickness=11, intensity=0.55)
    img = add_scratches_and_dust(img, num_scratches=25, num_specks=100)
    return img


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(os.path.dirname(script_dir))
    out_dir = os.path.join(workspace_dir, "assets", "textures", "tabletop")
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 70)
    print(">>> [TEXTURE GEN] Sim-to-Real 절차적 테이블 상판 텍스처 생성 시작...")
    print(f"  출력 폴더: {out_dir}")
    print("=" * 70)
    
    textures = [
        ("wood_oak.png", create_wood_texture),
        ("white_desk_stained.png", create_white_desk_texture),
        ("dark_mat_stained.png", create_dark_desk_mat),
        ("plywood_craft.png", create_light_wood_craft),
    ]
    
    for filename, gen_fn in textures:
        path = os.path.join(out_dir, filename)
        img = gen_fn()
        img.save(path, quality=95)
        print(f"  ✅ 생성 완료: {filename} (1024x1024)")
        
    print("=" * 70)
    print(">>> 모든 텍스처가 성공적으로 생성되었습니다!")
    print(f"    위치: {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()


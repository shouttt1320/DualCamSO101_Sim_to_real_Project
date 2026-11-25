#!/usr/bin/env python3
"""
[Step 0 - Real Wall Digital Twin Texture Extractor]
====================================================
사용자가 촬영한 실물 사진(backwall.jpg, topview.jpg, blackbox.jpg, realred1~6.jpg)에서
Isaac Sim OmniPBR용 타일형 Diffuse/Normal 텍스처를 자동 추출하고 매니페스트를 저장합니다.
"""

import os
import json
import glob
import numpy as np
import cv2
from PIL import Image

WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"
TEX_DIR = os.path.join(WORKSPACE_DIR, "assets", "textures")
OUT_DIR = os.path.join(TEX_DIR, "processed", "real_wall")
MANIFEST_PATH = os.path.join(OUT_DIR, "real_wall_manifest.json")


def _make_normal_map(rgb_arr: np.ndarray, strength: float = 2.0, blur_ksize: int = 3) -> np.ndarray:
    """그레이스케일 휘도 기반 Sobel 노멀맵 (OpenGL/USD RGB uint8) 생성"""
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) * strength
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) * strength
    nx = -dx
    ny = -dy
    nz = np.ones_like(gray)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx /= norm
    ny /= norm
    nz /= norm
    nor = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    return np.clip(nor * 255.0, 0, 255).astype(np.uint8)


def _save_tex_pair(rgb_arr: np.ndarray, prefix: str, normal_strength: float = 1.8, blur_ksize: int = 3):
    diff_path = os.path.join(OUT_DIR, f"{prefix}_diff.jpg")
    nor_path = os.path.join(OUT_DIR, f"{prefix}_nor.png")
    Image.fromarray(rgb_arr).save(diff_path, quality=95)
    nor_arr = _make_normal_map(rgb_arr, strength=normal_strength, blur_ksize=blur_ksize)
    Image.fromarray(nor_arr).save(nor_path)
    return diff_path, nor_path


def extract_backwall_and_molding():
    src = os.path.join(TEX_DIR, "backwall.jpg")
    img = np.array(Image.open(src).convert("RGB"))
    h, w, _ = img.shape

    # 갈색 나무 몰딩 행 자동 탐색
    row_rgb = img.mean(axis=1)
    molding_rows = np.where((row_rgb[:, 0] - row_rgb[:, 2]) > 15)[0]
    m_top = int(molding_rows[0]) if len(molding_rows) > 0 else int(h * 0.44)
    m_bot = int(molding_rows[-1]) if len(molding_rows) > 0 else int(h * 0.62)

    # 1) 하단 순수 엠보싱 벽지 영역 (몰딩 아래쪽 여유 마진 포함)
    wp_y0 = min(h - 1024, m_bot + 60)
    wp_y1 = h - 40
    side = min(1024, wp_y1 - wp_y0, w - 200)
    cx = w // 2
    wp_crop = img[wp_y0 : wp_y0 + side, cx - side // 2 : cx + side // 2]
    wp_res = cv2.resize(wp_crop, (1024, 1024), interpolation=cv2.INTER_AREA)
    wp_diff, wp_nor = _save_tex_pair(wp_res, "wallpaper", normal_strength=2.4, blur_ksize=3)

    # 2) 연갈색 나무 몰딩 영역 (수직 몰딩용으로 90도 회전한 버전 포함)
    pad = 45
    my0 = max(0, m_top + pad)
    my1 = min(h, m_bot - pad)
    if my1 - my0 < 100:
        my0, my1 = m_top, m_bot
    mw_crop = img[my0:my1, 600:3400]
    mw_res = cv2.resize(mw_crop, (512, 512), interpolation=cv2.INTER_AREA)
    mw_diff_h = os.path.join(OUT_DIR, "molding_wood_horiz.jpg")
    mw_diff_v = os.path.join(OUT_DIR, "molding_wood_vert.jpg")
    Image.fromarray(mw_res).save(mw_diff_h, quality=94)
    Image.fromarray(np.rot90(mw_res)).save(mw_diff_v, quality=94)

    return wp_diff, wp_nor, mw_diff_h, mw_diff_v


def extract_topview_table():
    src = os.path.join(TEX_DIR, "topview.jpg")
    img = np.array(Image.open(src).convert("RGB"))
    # 중앙 접힘선 및 로봇팔을 피한 좌상단 평탄 아이보리 상판 영역 [650:1200, 1250:1800]
    crop = img[650:1200, 1250:1800]
    res = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    # 엣지 이음새 완화용 미러 타일링 방지: 중심 90%만 재크롭해 매끈한 미세 입자 유지
    m = 24
    res = cv2.resize(res[m:-m, m:-m], (512, 512), interpolation=cv2.INTER_AREA)
    med_rgb = np.median(res, axis=(0, 1)).astype(float).tolist()
    t_diff, t_nor = _save_tex_pair(res, "table_ivory", normal_strength=1.2, blur_ksize=3)
    return t_diff, t_nor, med_rgb


def extract_blackbox():
    src = os.path.join(TEX_DIR, "blackbox.jpg")
    img = np.array(Image.open(src).convert("RGB"))
    # 상자 안쪽 순수 매트 블랙 바닥 중심부 [1100:1900, 1700:2500]
    crop = img[1100:1900, 1700:2500]
    res = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)
    # 사진 노출로 약간 밝게 찍힌 어두운 회색을 실물 무광 블랙/차콜 톤으로 미세 보정
    res_dark = np.clip(res.astype(np.float32) * 0.55, 10, 120).astype(np.uint8)
    b_diff, b_nor = _save_tex_pair(res_dark, "blackbox", normal_strength=2.2, blur_ksize=3)
    return b_diff, b_nor


def extract_real_red_cubes():
    paths = sorted(glob.glob(os.path.join(TEX_DIR, "realred*.jpg")))
    faces = []
    for idx, p in enumerate(paths, start=1):
        img = np.array(Image.open(p).convert("RGB")).astype(np.int16)
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        mask = ((r > 90) & ((r - g) > 35) & ((r - b) > 35)).astype(np.uint8) * 255
        # 모폴로지로 외곽 잡음 제거 후 가장 큰 연결 요소 바운딩 박스 탐색
        kernel = np.ones((15, 15), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if num_labels > 1:
            best_lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, bw, bh, _ = stats[best_lbl]
        else:
            x, y, bw, bh = 1400, 900, 1200, 1200

        # 흰 테이블 배경이 1픽셀도 섞이지 않도록 사방 18% 안쪽으로 안전 수축
        mx = int(bw * 0.18)
        my = int(bh * 0.18)
        x0, x1 = x + mx, x + bw - mx
        y0, y1 = y + my, y + bh - my
        crop = img[y0:y1, x0:x1].astype(np.uint8)
        res = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA)

        diff_p, nor_p = _save_tex_pair(res, f"realred_face_{idx}", normal_strength=3.2, blur_ksize=3)
        faces.append({"diffuse": diff_p, "normal": nor_p, "source": p})
    return faces


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    wp_diff, wp_nor, mw_h, mw_v = extract_backwall_and_molding()
    t_diff, t_nor, med_rgb = extract_topview_table()
    b_diff, b_nor = extract_blackbox()
    red_faces = extract_real_red_cubes()

    manifest = {
        "wallpaper_diffuse": wp_diff,
        "wallpaper_normal": wp_nor,
        "molding_horiz": mw_h,
        "molding_vert": mw_v,
        "table_ivory_diffuse": t_diff,
        "table_ivory_normal": t_nor,
        "table_ivory_median_rgb": med_rgb,
        "blackbox_diffuse": b_diff,
        "blackbox_normal": b_nor,
        "real_red_faces": red_faces,
    }
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[prepare_wall_textures] Saved manifest: {MANIFEST_PATH} ({len(red_faces)} red cube faces)")


if __name__ == "__main__":
    main()


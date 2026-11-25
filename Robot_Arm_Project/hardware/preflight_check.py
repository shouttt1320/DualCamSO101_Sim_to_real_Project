#!/usr/bin/env python3
"""SO-101 실물 로봇 추론 배포 전 종합 사전 점검(Pre-flight Check) 스크립트.

점검 항목:
  1) calibration.json 및 시리얼 포트 / 서보 모터(ID 1~6) Ping·전압·온도·현재 각도 확인 (토크 OFF 안전 모드)
  2) 양안 손목 카메라(left/right) 연결 및 실시간 프레임 캡처 확인
  3) 학습된 DINOv2-ACT 체크포인트 및 정규화 통계 로드 + 1회 Dry-Run 추론 검증 (모터 구동 없음)
  4) (선택) --move-obs 지정 시 사용자 확인 후 OBSERVATION_POSE([0, -68.3°, 32.8°, 74.8°, 0, 0])로 부드럽게 이동 테스트
"""

import argparse
import json
import os
import pickle
import sys
import time
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

HARDWARE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HARDWARE_DIR)
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
for p in (PROJECT_ROOT, MODELS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from hardware.camera_interface import MultiCameraCapture
from hardware.dual_arm_controller import rad_to_raw, raw_to_rad
from hardware.identify_arms import find_port_by_serial
from hardware.sts3215_driver import STS3215Driver
from so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    FIXED_FOCUS_VALUE,
    HIDDEN_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
)
from so101_dual_dinov2_act.models import build_act_model

OBSERVATION_POSE_RAD = np.array(
    [0.0, np.radians(-68.3), np.radians(32.8), np.radians(74.8), 0.0, 0.0],
    dtype=np.float32,
)


def _clamp(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))


def check_motors(config: dict) -> Dict[str, object]:
    """시리얼 포트 및 1~6번 모터 상태 점검 (토크 OFF 유지)."""
    print("\n" + "=" * 72)
    print("[1/3] 로봇팔 시리얼 포트 및 서보 모터(ID 1~6) 점검")
    print("=" * 72)

    raw_port = config.get("follower_port", "/dev/ttyACM0")
    serial_num = config.get("follower_serial", "")
    port = find_port_by_serial(serial_num, default_port=raw_port) if serial_num else raw_port
    print(f"  • 대상 포트: {port} (시리얼: {serial_num or '미지정'})")

    if not os.path.exists(port):
        print(f"  ❌ [FAIL] 포트 {port} 가 존재하지 않습니다. USB 케이블/전원을 확인하세요.")
        return {"ok": False, "driver": None, "qpos_rad": None}

    driver = STS3215Driver(port=port, baudrate=1000000, timeout=0.05)
    if not driver.connect():
        print(f"  ❌ [FAIL] 포트 {port} 연결 실패 (권한 확인: sudo chmod 666 {port})")
        return {"ok": False, "driver": None, "qpos_rad": None}

    joints_cfg = config["joints"]
    qpos_rad = []
    all_ok = True

    for j_id in range(1, 7):
        cfg = joints_cfg[f"joint_{j_id}"]
        zero_pos = int(cfg.get("follower_zero", 2048))
        direction = int(cfg.get("direction", 1))
        min_lim = int(cfg.get("follower_min", cfg.get("min", 0)))
        max_lim = int(cfg.get("follower_max", cfg.get("max", 4095)))

        alive = driver.ping(j_id)
        status = driver.read_status(j_id) if alive else None
        if not alive or status is None:
            print(f"  ❌ J{j_id} ({cfg.get('name', '')}): 응답 없음 (Ping/Status 실패)")
            all_ok = False
            qpos_rad.append(0.0)
            continue

        raw_pos = int(status["position"])
        rad_val = raw_to_rad(raw_pos, zero_pos=zero_pos, direction=direction)
        deg_val = float(np.degrees(rad_val))
        in_range = min_lim <= raw_pos <= max_lim
        range_str = "OK" if in_range else f"범위초과({min_lim}~{max_lim})"
        print(
            f"  ✅ J{j_id}: raw={raw_pos:4d} (zero={zero_pos:4d}, dir={direction:+d}) "
            f"| {rad_val:+6.3f} rad ({deg_val:+6.1f}°) "
            f"| {status['voltage']:.1f}V, {status['temperature']}°C | {range_str}"
        )
        qpos_rad.append(rad_val)

    return {
        "ok": all_ok,
        "driver": driver,
        "qpos_rad": np.array(qpos_rad, dtype=np.float32),
    }


def check_cameras(left_cam: int, right_cam: int, focus: int) -> Dict[str, object]:
    """양안 카메라 연결 및 실시간 프레임 캡처 점검."""
    print("\n" + "=" * 72)
    print(f"[2/3] 양안 손목 카메라 점검 (Left=/dev/video{left_cam}, Right=/dev/video{right_cam})")
    print("=" * 72)

    cam_dict = {"cam_left": left_cam, "cam_right": right_cam}
    focus_dict = {"cam_left": focus, "cam_right": focus}
    cap: Optional[MultiCameraCapture] = None

    try:
        cap = MultiCameraCapture(
            camera_dict=cam_dict,
            width=640,
            height=480,
            fps=30,
            disable_autofocus=True,
            focus_dict=focus_dict,
        )
        cap.start()
        time.sleep(0.1)
        frames = cap.read_all()

        ok = True
        for name in ("cam_left", "cam_right"):
            frame = frames.get(name)
            if frame is None or frame.size == 0 or np.mean(frame) < 1.0:
                print(f"  ❌ [{name}] 유효한 프레임 캡처 실패")
                ok = False
            else:
                print(
                    f"  ✅ [{name}] 프레임 정상: shape={frame.shape}, "
                    f"mean_brightness={frame.mean():.1f}"
                )
        return {"ok": ok, "frames": frames}
    except Exception as err:
        print(f"  ❌ [FAIL] 카메라 초기화 실패: {err}")
        return {"ok": False, "frames": {}}
    finally:
        if cap is not None:
            cap.stop()


def check_model_dry_run(
    ckpt_dir: str,
    ckpt_name: str,
    config: dict,
    qpos_rad: Optional[np.ndarray],
    frames: Dict[str, np.ndarray],
) -> bool:
    """체크포인트 로드 및 1-step Dry-Run 추론 점검 (모터 전송 없음)."""
    print("\n" + "=" * 72)
    print("[3/3] 모델 로드 및 1-Step Dry-Run 추론 점검")
    print("=" * 72)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")

    if not os.path.exists(ckpt_path):
        print(f"  ❌ [FAIL] 체크포인트가 없습니다: {ckpt_path}")
        return False
    if not os.path.exists(stats_path):
        print(f"  ❌ [FAIL] 정규화 통계가 없습니다: {stats_path}")
        return False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  • Device: {device}")
    print(f"  • Checkpoint: {ckpt_path}")

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    qpos_mean = torch.from_numpy(stats["qpos_mean"]).float()
    qpos_std = torch.from_numpy(stats["qpos_std"]).float()
    action_mean = torch.from_numpy(stats["action_mean"]).float()
    action_std = torch.from_numpy(stats["action_std"]).float()

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    chunk_size = state_dict["query_embed.weight"].shape[0] if "query_embed.weight" in state_dict else CHUNK_SIZE

    model = build_act_model(
        state_dim=STATE_DIM,
        num_queries=chunk_size,
        camera_names=CAMERA_NAMES,
        backbone_name=BACKBONE,
        hidden_dim=HIDDEN_DIM,
        dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS,
        dec_layers=DEC_LAYERS,
        nheads=NHEADS,
        latent_dim=LATENT_DIM,
        train_backbone=False,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    used_qpos = qpos_rad if qpos_rad is not None else OBSERVATION_POSE_RAD.copy()
    qpos_source = "실물 모터 측정값" if qpos_rad is not None else "OBSERVATION_POSE 더미값"
    qpos_norm = ((torch.from_numpy(used_qpos).float() - qpos_mean) / qpos_std).unsqueeze(0).to(device)

    cam_tensors = []
    for name in CAMERA_NAMES:
        frame = frames.get(name)
        if frame is None:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        cam_tensors.append(transform(img_t))
    img_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(device)

    t0 = time.time()
    with torch.inference_mode():
        a_hat, _, _ = model(qpos=qpos_norm, image=img_input)
    inf_ms = (time.time() - t0) * 1000.0

    action_0 = (a_hat.squeeze(0).cpu() * action_std + action_mean)[0].numpy()
    joints_cfg = config["joints"]
    target_raws = []
    for i in range(1, 7):
        cfg = joints_cfg[f"joint_{i}"]
        raw_t = rad_to_raw(
            float(action_0[i - 1]),
            zero_pos=int(cfg.get("follower_zero", 2048)),
            direction=int(cfg.get("direction", 1)),
        )
        lo = int(cfg.get("follower_min", cfg.get("min", 0)))
        hi = int(cfg.get("follower_max", cfg.get("max", 4095)))
        target_raws.append(_clamp(raw_t, lo, hi))

    print(f"  ✅ 모델 로드 및 1회 추론 성공! (Chunk={chunk_size}, 추론시간={inf_ms:.1f}ms)")
    print(f"  • 입력 qpos 출처: {qpos_source}")
    print(f"  • 예측 Action[0] (deg): {[round(float(np.degrees(x)), 1) for x in action_0]}")
    print(f"  • 변환 Target Raw (1~6): {target_raws}")
    return True


def run_move_observation_test(driver: STS3215Driver, config: dict, duration_sec: float = 2.5):
    """사용자 동의 하에 OBSERVATION_POSE로 천천히 이동해 영점·관절 방향 검증."""
    print("\n" + "=" * 72)
    print("[선택 테스트] OBSERVATION_POSE 부드러운 이동 검증")
    print("  목표 각도(deg): [Pan=0.0, Lift=-68.3, Elbow=+32.8, W_Flex=+74.8, W_Roll=0.0, Grip=0.0]")
    print("=" * 72)
    ans = input("  ⚠️ 로봇 팔이 실제로 움직입니다. 주변 장애물을 치우고 진행할까요? (y/N): ").strip().lower()
    if ans != "y":
        print("  ℹ️ 이동 테스트를 건너뛰었습니다.")
        return

    joints_cfg = config["joints"]
    start_raw = {}
    goal_raw = {}
    for i in range(1, 7):
        cfg = joints_cfg[f"joint_{i}"]
        zero_pos = int(cfg.get("follower_zero", 2048))
        direction = int(cfg.get("direction", 1))
        lo = int(cfg.get("follower_min", cfg.get("min", 0)))
        hi = int(cfg.get("follower_max", cfg.get("max", 4095)))

        cur = driver.read_position(i)
        start_raw[i] = cur if cur is not None else zero_pos
        goal_raw[i] = _clamp(rad_to_raw(float(OBSERVATION_POSE_RAD[i - 1]), zero_pos=zero_pos, direction=direction), lo, hi)

    try:
        driver.set_all_torques(range(1, 7), True)
        driver.set_torque_limit(6, 0.45)

        steps = max(1, int(duration_sec * 40))
        for s in range(1, steps + 1):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (s / steps)))
            cmd = {
                i: int(round(start_raw[i] + alpha * (goal_raw[i] - start_raw[i])))
                for i in range(1, 7)
            }
            driver.sync_write_positions(cmd, speed=0, acc=0)
            time.sleep(duration_sec / steps)

        print("  ✅ OBSERVATION_POSE 이동 완료! 시뮬레이션의 초기 관측 자세와 같은지 확인하세요.")
        input("  👉 확인이 끝났으면 [Enter]를 누르세요 (토크가 안전하게 해제됩니다)...")
    finally:
        driver.set_all_torques(range(1, 7), False)
        print("  💤 모터 토크 해제 완료.")


def main():
    default_ckpt_dir = os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", "domain_rand")
    default_config = os.path.join(PROJECT_ROOT, "configs", "calibration.json")

    parser = argparse.ArgumentParser(description="SO-101 실물 로봇 배포 전 Pre-flight Check")
    parser.add_argument("--config", type=str, default=default_config, help="calibration.json 경로")
    parser.add_argument("--ckpt-dir", type=str, default=default_ckpt_dir, help="체크포인트 디렉토리")
    parser.add_argument("--ckpt", type=str, default="policy_best.pth", help="체크포인트 파일명")
    parser.add_argument("--left-cam", type=int, default=2, help="좌안 카메라 인덱스 (기본: 2)")
    parser.add_argument("--right-cam", type=int, default=0, help="우안 카메라 인덱스 (기본: 0)")
    parser.add_argument("--focus", type=int, default=FIXED_FOCUS_VALUE, help="고정 초점 값 (기본: 25)")
    parser.add_argument("--move-obs", action="store_true", help="OBSERVATION_POSE 이동 테스트까지 수행")
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f"calibration 파일을 찾을 수 없습니다: {args.config}")
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    motor_res = check_motors(config)
    cam_res = check_cameras(args.left_cam, args.right_cam, args.focus)
    model_ok = check_model_dry_run(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt,
        config=config,
        qpos_rad=motor_res["qpos_rad"],
        frames=cam_res["frames"],
    )

    if args.move_obs and motor_res["ok"] and motor_res["driver"] is not None:
        run_move_observation_test(motor_res["driver"], config)

    if motor_res["driver"] is not None:
        motor_res["driver"].close()

    print("\n" + "=" * 72)
    print(">>> [PRE-FLIGHT CHECK SUMMARY]")
    print(f"  • 1. 모터/시리얼 점검 : {'PASS ✅' if motor_res['ok'] else 'FAIL ❌'}")
    print(f"  • 2. 양안 카메라 점검 : {'PASS ✅' if cam_res['ok'] else 'FAIL ❌'}")
    print(f"  • 3. 모델 Dry-Run 추론: {'PASS ✅' if model_ok else 'FAIL ❌'}")
    print("=" * 72)

    if not (motor_res["ok"] and cam_res["ok"] and model_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()


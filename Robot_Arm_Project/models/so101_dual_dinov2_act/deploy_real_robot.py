"""SO-101 양안(Dual Eyes) DINOv2 ACT 실시간 자율 추론 및 로봇 배포 스크립트."""

import argparse
import csv
import json
import os
import pickle
import signal
import sys
import time
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

try:
    from hardware.camera_interface import MultiCameraCapture
    from hardware.sts3215_driver import STS3215Driver
    from hardware.identify_arms import find_port_by_serial
except ImportError:
    try:
        from camera_interface import MultiCameraCapture
        from sts3215_driver import STS3215Driver
        from identify_arms import find_port_by_serial
    except ImportError:
        find_port_by_serial = None

from so101_dual_dinov2_act.config import (
    BACKBONE,
    CAMERA_NAMES,
    CHUNK_SIZE,
    CKPT_DIR,
    DEC_LAYERS,
    DIM_FEEDFORWARD,
    ENC_LAYERS,
    FIXED_FOCUS_VALUE,
    HIDDEN_DIM,
    IMAGE_SIZE,
    LATENT_DIM,
    NHEADS,
    STATE_DIM,
    TEMPORAL_AGG_M,
)
from so101_dual_dinov2_act.models import build_act_model


def clamp(val, min_val, max_val):
    return max(min_val, min(max_val, val))


def raw_to_rad(raw_val: int, zero_pos: int = 2048, direction: int = 1) -> float:
    dir_sign = -1 if direction < 0 else 1
    return (raw_val - zero_pos) * dir_sign * (2.0 * np.pi / 4096.0)


def rad_to_raw(rad_val: float, zero_pos: int = 2048, direction: int = 1) -> int:
    dir_sign = -1 if direction < 0 else 1
    return int(round(float(zero_pos) + rad_val * dir_sign * (4096.0 / (2.0 * np.pi))))


OBSERVATION_POSE_RAD = np.array(
    [0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0, 0.0],
    dtype=np.float32,
)


class RealRobotDeployer:
    """학습된 양안 DINOv2 ACT 정책을 실제 SO-101 팔로워 로봇에 배포하는 실시간 자율 제어기."""

    def __init__(
        self,
        ckpt_dir: str = CKPT_DIR,
        ckpt_name: str = "policy_best.pth",
        config_path: str = "configs/calibration.json",
        left_cam_id: int = 2,
        right_cam_id: int = 0,
        focus: int = FIXED_FOCUS_VALUE,
        fps: int = 30,
        temporal_agg: bool = True,
        effective_chunk: int = None,
        show_preview: bool = True,
        start_obs: bool = False,
        max_step_deg: float = 5.0,
        use_vis_gate: bool = False,
        vis_gate_thr: float = 0.5,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.temporal_agg = temporal_agg
        self.effective_chunk = effective_chunk if effective_chunk is not None else CHUNK_SIZE
        if self.effective_chunk > CHUNK_SIZE:
            self.effective_chunk = CHUNK_SIZE
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))
        self.start_obs = start_obs
        self.max_step_deg = max_step_deg
        self.use_vis_gate = bool(use_vis_gate)
        self.vis_gate_thr = float(vis_gate_thr)
        self.ckpt_dir = ckpt_dir
        self.ckpt_name = ckpt_name
        self.running = False
        self._stopped = False

        # 1. 캘리브레이션 및 한계치 로드
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"{config_path} 파일이 없습니다.")
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        raw_follower_port = self.config.get("follower_port", "/dev/ttyACM0")
        follower_serial = self.config.get("follower_serial", "")
        if find_port_by_serial and follower_serial:
            self.follower_port = find_port_by_serial(follower_serial, default_port=raw_follower_port)
            if self.follower_port != raw_follower_port:
                print(f"🔍 [포트 자동 매칭] 고유 시리얼({follower_serial}) 기반 팔로워 포트: {self.follower_port}")
        else:
            self.follower_port = raw_follower_port

        self.joints_cfg = self.config["joints"]
        self.gripper_cfg = self.config["gripper"]

        # 2. 정규화 통계 로드
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"{stats_path} 가 없습니다. 먼저 모델을 학습시키세요.")
        with open(stats_path, "rb") as f:
            self.stats = pickle.load(f)

        self.qpos_mean = torch.from_numpy(self.stats["qpos_mean"]).float()
        self.qpos_std = torch.from_numpy(self.stats["qpos_std"]).float()
        self.action_mean = torch.from_numpy(self.stats["action_mean"]).float()
        self.action_std = torch.from_numpy(self.stats["action_std"]).float()

        # 3. DINOv2 ACT 모델 빌드 및 가중치 로드
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} 가중치 파일이 없습니다.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚙️ 추론 연산 장치: {self.device}")

        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        ckpt_backbone = checkpoint.get("backbone_name", "dinov2_vits14") if isinstance(checkpoint, dict) else "dinov2_vits14"
        actual_chunk = state_dict["query_embed.weight"].shape[0] if "query_embed.weight" in state_dict else CHUNK_SIZE

        self.model = build_act_model(
            state_dim=STATE_DIM,
            num_queries=actual_chunk,
            camera_names=CAMERA_NAMES,
            backbone_name=ckpt_backbone,
            hidden_dim=HIDDEN_DIM,
            dim_feedforward=DIM_FEEDFORWARD,
            enc_layers=ENC_LAYERS,
            dec_layers=DEC_LAYERS,
            nheads=NHEADS,
            latent_dim=LATENT_DIM,
            train_backbone=False,
        )
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        print(f"✅ 양안 DINOv2({ckpt_backbone}) 가중치 로드 완료 ➔ {ckpt_path}")

        # 4. DINOv2 이미지 전처리 파이프라인 (Bicubic Resize + ImageNet Normalization)
        self.image_transform = transforms.Compose([
            transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 5. 카메라 인터페이스 및 드라이버 설정 (초점 고정)
        self.cam_ids = {CAMERA_NAMES[0]: left_cam_id, CAMERA_NAMES[1]: right_cam_id}
        focus_dict = {name: focus for name in self.cam_ids}
        self.camera_cap = MultiCameraCapture(
            camera_dict=self.cam_ids,
            width=640,
            height=480,
            fps=fps,
            disable_autofocus=True,
            focus_dict=focus_dict,
        )
        self.driver = STS3215Driver(self.follower_port, baudrate=1000000, timeout=0.03)

        # 6. Temporal Ensembling 버퍼
        self.temporal_weights = np.exp(-TEMPORAL_AGG_M * np.arange(self.effective_chunk))
        self.home_pos = None

    def connect(self):
        print("\n" + "=" * 68)
        print(" 🤖 [SO-101 양안(Dual Eyes) DINOv2 ACT 실시간 자율 추론]")
        print(f" 🦾 대상 로봇   : Follower ({self.follower_port})")
        print(f" 👁️ 좌안 카메라 : /dev/video{self.cam_ids[CAMERA_NAMES[0]]}")
        print(f" 👁️ 우안 카메라 : /dev/video{self.cam_ids[CAMERA_NAMES[1]]}")
        print(f" 🧠 모델 백본   : DINOv2 ViT-S/14 (Frozen) + ACT CVAE-Transformer")
        print(f" ⏱️ 제어 주기   : {self.fps} Hz | Temporal Ensembling: {'ON' if self.temporal_agg else 'OFF'}")
        print("=" * 68)

        if not self.driver.connect():
            raise ConnectionError(f"팔로워 로봇({self.follower_port}) 연결 실패!")

        # 시작 전 현재 위치를 읽어 홈 포즈로 저장
        current_pos = self.driver.read_positions(range(1, 7))
        if len(current_pos) == 6:
            self.home_pos = current_pos.copy()
            print(f"🏠 홈 포즈 등록 완료 (현재 로봇 시작 위치): {self.home_pos}")
        else:
            self.home_pos = {
                1: self.joints_cfg["joint_1"]["follower_zero"],
                2: self.joints_cfg["joint_2"]["follower_zero"],
                3: self.joints_cfg["joint_3"]["follower_zero"],
                4: self.joints_cfg["joint_4"]["follower_zero"],
                5: self.joints_cfg["joint_5"]["follower_zero"],
                6: self.gripper_cfg["follower_close"],
            }
            print(f"🏠 홈 포즈 등록 완료 (캘리브레이션 기본 영점): {self.home_pos}")

        self.driver.set_all_torques(range(1, 7), True)
        self.driver.set_torque_limit(6, 0.45)  # 그리퍼(ID 6) 과부하 방지용 소프트 토크 제한
        print("✅ 팔로워 로봇 토크 활성화 완료 (자율 주행 준비)")

        self.camera_cap.start()
        print("✅ 양안 카메라 실시간 캡처 준비 완료")

    def return_to_home(self, duration: float = 2.5):
        """추론 종료 후 현재 위치에서 홈 포즈로 부드럽게 S-Curve 보간 이동."""
        if self.home_pos is None or not (self.driver and self.driver.ser and self.driver.ser.is_open):
            return

        print("\n🏠 [홈 복귀] 추론 완료 후 시작 홈 포즈로 부드럽게 복귀합니다...")
        start_pos = self.driver.read_positions(range(1, 7))
        if len(start_pos) < 6:
            self.driver.sync_write_positions(self.home_pos, speed=0, acc=0)
            time.sleep(duration)
            return

        steps = max(10, int(duration * self.fps))
        dt = duration / steps

        for step in range(1, steps + 1):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step / steps)))
            interp_goals = {}
            for i in range(1, 7):
                p0 = start_pos.get(i, self.home_pos[i])
                p1 = self.home_pos[i]
                interp_goals[i] = int(round(p0 + (p1 - p0) * alpha))

            self.driver.sync_write_positions(interp_goals, speed=0, acc=0)
            time.sleep(dt)

        print("✅ 홈 포즈 복귀 완료.")
        time.sleep(0.3)

    def _render_preview(
        self,
        cam_frames: Dict[str, Optional[np.ndarray]],
        hud_text: str,
        bar_color: Tuple[int, int, int] = (0, 140, 0),
    ) -> int:
        """양안 프리뷰 창을 렌더링하고 눌린 키 코드를 반환합니다."""
        if not self.show_preview:
            return -1

        left_img = cam_frames.get("cam_left")
        right_img = cam_frames.get("cam_right")
        target_w, target_h = 480, 360
        left_vis = cv2.resize(left_img, (target_w, target_h)) if left_img is not None else np.zeros((target_h, target_w, 3), dtype=np.uint8)
        right_vis = cv2.resize(right_img, (target_w, target_h)) if right_img is not None else np.zeros((target_h, target_w, 3), dtype=np.uint8)

        cv2.rectangle(left_vis, (0, 0), (target_w, 28), (40, 160, 40), -1)
        cv2.putText(left_vis, "LEFT EYE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.rectangle(right_vis, (0, 0), (target_w, 28), (20, 120, 240), -1)
        cv2.putText(right_vis, "RIGHT EYE", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        combined = np.hstack([left_vis, right_vis])
        h, w = combined.shape[:2]
        cv2.rectangle(combined, (0, h - 38), (w, h), bar_color, -1)
        cv2.putText(combined, hud_text, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2)

        win_name = "SO-101 Dual-Eyes DINOv2 ACT [Autonomous]"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.imshow(win_name, combined)
        return cv2.waitKey(1) & 0xFF

    def move_to_observation_pose(self, duration: float = 2.0, settle_sec: float = 0.5):
        """Active Search 추론 시작 전 시뮬레이션 OBSERVATION_POSE로 부드럽게 이동."""
        if not (self.driver and self.driver.ser and self.driver.ser.is_open):
            return

        obs_targets = {}
        for i in range(1, 7):
            cfg = self.joints_cfg[f"joint_{i}"]
            zero_pos = int(cfg.get("follower_zero", 2048))
            direction = int(cfg.get("direction", 1))
            min_lim = int(cfg.get("follower_min", cfg.get("min", 0)))
            max_lim = int(cfg.get("follower_max", cfg.get("max", 4095)))
            raw_t = rad_to_raw(float(OBSERVATION_POSE_RAD[i - 1]), zero_pos=zero_pos, direction=direction)
            obs_targets[i] = int(clamp(raw_t, min_lim, max_lim))

        self.home_pos = obs_targets.copy()
        print(f"\n🎯 [관측 자세 이동] OBSERVATION_POSE로 {duration:.1f}초간 부드럽게 이동합니다...")
        start_pos = self.driver.read_positions(range(1, 7))
        steps = max(10, int(duration * self.fps))
        dt = duration / steps

        for step in range(1, steps + 1):
            if not self.running:
                return
            alpha = 0.5 * (1.0 - np.cos(np.pi * (step / steps)))
            interp_goals = {}
            for i in range(1, 7):
                p0 = start_pos.get(i)
                if p0 is None:
                    p0 = obs_targets[i]
                p1 = obs_targets[i]
                interp_goals[i] = int(round(p0 + (p1 - p0) * alpha))
            self.driver.sync_write_positions(interp_goals, speed=0, acc=0)
            key = self._render_preview(
                self.camera_cap.read_all(),
                "MOVING TO OBSERVATION POSE... [Q/ESC]=Quit",
                (180, 110, 0),
            )
            if key in (ord("q"), 27):
                self.running = False
                return
            time.sleep(dt)

        settle_steps = max(1, int(settle_sec * self.fps))
        for _ in range(settle_steps):
            if not self.running:
                return
            key = self._render_preview(
                self.camera_cap.read_all(),
                "SETTLING AT OBSERVATION POSE... [Q/ESC]=Quit",
                (180, 110, 0),
            )
            if key in (ord("q"), 27):
                self.running = False
                return
            time.sleep(dt)
        print("✅ OBSERVATION_POSE 도착 완료.")

    def _wait_for_start_or_quit(self, ep_idx: int) -> bool:
        """대기 상태(STANDBY)에서 [SPACE] 입력 시 True, [Q/ESC] 종료 시 False 반환."""
        print(f"\n⏸️  [대기 상태 - Ep #{ep_idx}] 프리뷰 창에서 [SPACE]를 누르면 추론을 시작합니다. ([Q / ESC] 또는 [Ctrl+C] 종료)")
        if not self.show_preview:
            try:
                ans = input("  ▶ [Enter] 추론 시작 | [q] 종료: ").strip().lower()
                return ans != "q"
            except EOFError:
                return False

        while self.running:
            cam_frames = self.camera_cap.read_all()
            key = self._render_preview(
                cam_frames,
                f"STANDBY [Ep #{ep_idx}] | [SPACE]=Start Inference | [Q/ESC]=Quit",
                (180, 100, 0),
            )
            if key == ord(" "):
                return True
            if key in (ord("q"), 27):
                self.running = False
                return False
            time.sleep(0.02)
        return False

    def _run_episode(self, max_timesteps: int, ep_idx: int) -> str:
        """단일 에피소드 자율 추론 실행. 반환값: 'done' | 'standby' | 'quit'."""
        total_time = max_timesteps + self.effective_chunk
        all_time_actions = torch.zeros([total_time, total_time, STATE_DIM])
        weights = torch.from_numpy(self.temporal_weights).float()

        loop_start_time = time.time()
        last_raw_qpos = {}
        last_sent_raw = {}
        in_search_phase = self.use_vis_gate
        vis_hit_streak = 0

        for t in range(max_timesteps):
            if not self.running:
                return "quit"
            t0 = time.time()

            # 1. 팔로워 현재 관절값 읽기 (qpos)
            f_pos = self.driver.read_positions(range(1, 7))
            curr_qpos = []
            for i in range(1, 7):
                cfg = self.joints_cfg[f"joint_{i}"]
                zero_pos = int(cfg.get("follower_zero", 2048))
                direction = int(cfg.get("direction", 1))
                val = f_pos.get(i)
                if val is None and last_raw_qpos.get(i) is not None:
                    val = last_raw_qpos[i]
                if val is None:
                    val = zero_pos
                last_raw_qpos[i] = val
                curr_qpos.append(raw_to_rad(val, zero_pos=zero_pos, direction=direction))

            qpos_tensor = torch.from_numpy(np.array(curr_qpos, dtype=np.float32))
            qpos_norm = (qpos_tensor - self.qpos_mean) / self.qpos_std
            qpos_input = qpos_norm.unsqueeze(0).to(self.device)

            # 2. 실시간 양안 카메라 프레임 캡처 및 DINOv2 전처리
            cam_frames = self.camera_cap.read_all()
            cam_tensors = []
            for name in CAMERA_NAMES:
                frame = cam_frames.get(name)
                if frame is None:
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
                img_t = self.image_transform(img_t)
                cam_tensors.append(img_t)

            img_input = torch.stack(cam_tensors, dim=0).unsqueeze(0).to(self.device)

            # 3. ACT 모델 추론 (z=0 Prior 모드)
            t_inf_start = time.time()
            with torch.inference_mode():
                a_hat, _, _ = self.model(qpos=qpos_input, image=img_input)
            inf_ms = (time.time() - t_inf_start) * 1000.0

            p_vis = None
            if getattr(self.model, "last_aux", None) is not None and "vis_logit" in self.model.last_aux:
                p_vis = float(torch.sigmoid(self.model.last_aux["vis_logit"]).item())

            # 역정규화
            a_hat_cpu = a_hat.squeeze(0).cpu()
            action_chunk = a_hat_cpu * self.action_std + self.action_mean

            # 4. 액션 결정 (Temporal Aggregation 적용)
            if self.temporal_agg:
                all_time_actions[[t], t : t + self.effective_chunk] = action_chunk[: self.effective_chunk]
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]

                k = len(actions_for_curr_step)
                step_weights = weights[:k]
                step_weights = step_weights / step_weights.sum()

                target_action = (actions_for_curr_step * step_weights.unsqueeze(-1)).sum(axis=0).numpy()
            else:
                target_action = action_chunk[0].numpy()

            # 4.5 선택적 탐색 단계 한정 Grasp 게이트 (--vis-gate 활성화 시에만 동작, 기본 OFF)
            gate_active = False
            if in_search_phase and p_vis is not None:
                if p_vis >= self.vis_gate_thr:
                    vis_hit_streak += 1
                    if vis_hit_streak >= 3:
                        in_search_phase = False
                        print(f"\n🔓 [vis-gate 해제] 큐브 포착 확정 (p_vis={p_vis:.2f}, t={t+1})")
                else:
                    vis_hit_streak = 0
                    gate_active = True
                    # Pan(0) 스캔 회전은 유지하고 Lift/Elbow 하강 및 Gripper 개방만 차단
                    target_action = target_action.copy()
                    target_action[1] = min(float(target_action[1]), float(np.radians(-58.0)))
                    target_action[2] = min(float(target_action[2]), float(np.radians(42.0)))
                    target_action[3] = max(float(target_action[3]), float(np.radians(55.0)))
                    target_action[5] = 0.0

            # 5. 관절 안전 한계값 클램핑 및 하드웨어 목표각 전송
            follower_goals = {}
            for i in range(1, 6):
                cfg = self.joints_cfg[f"joint_{i}"]
                raw_target = rad_to_raw(
                    target_action[i - 1],
                    zero_pos=int(cfg.get("follower_zero", 2048)),
                    direction=int(cfg.get("direction", 1)),
                )
                min_lim = cfg.get("follower_min", cfg.get("min", 0))
                max_lim = cfg.get("follower_max", cfg.get("max", 4095))
                follower_goals[i] = int(clamp(raw_target, min_lim, max_lim))

            # 그리퍼 클램핑
            grip_j_cfg = self.joints_cfg["joint_6"]
            grip_raw = rad_to_raw(
                target_action[5],
                zero_pos=int(grip_j_cfg.get("follower_zero", 2048)),
                direction=int(grip_j_cfg.get("direction", 1)),
            )
            f_open = self.gripper_cfg["follower_open"]
            f_close = self.gripper_cfg["follower_close"]
            grip_min = min(f_open, f_close)
            grip_max = max(f_open, f_close)
            follower_goals[6] = int(clamp(grip_raw, grip_min, grip_max))

            # 스텝당 최대 관절 변화량 제한 (OOD 급발진/바닥 충돌 방지 안전 가드)
            if self.max_step_deg > 0:
                arm_max_ticks = max(1, int(round(self.max_step_deg * (4096.0 / 360.0))))
                grip_max_ticks = max(1, int(round(self.max_step_deg * 2.0 * (4096.0 / 360.0))))
                for i in range(1, 7):
                    prev = last_sent_raw.get(i, last_raw_qpos.get(i, follower_goals[i]))
                    max_delta = grip_max_ticks if i == 6 else arm_max_ticks
                    delta = follower_goals[i] - prev
                    if abs(delta) > max_delta:
                        follower_goals[i] = int(prev + np.sign(delta) * max_delta)
                    last_sent_raw[i] = follower_goals[i]

            self.driver.sync_write_positions(follower_goals, speed=0, acc=0)

            # 6. 실시간 양안 프리뷰 및 HUD
            elapsed = time.time() - loop_start_time
            vis_str = f" | Vis:{p_vis:.2f}{'[GATE]' if gate_active else ''}" if p_vis is not None else ""
            hud = f"RUNNING Ep #{ep_idx} [Step {t+1:3d}/{max_timesteps} | {elapsed:4.1f}s | {inf_ms:4.1f}ms{vis_str}] - [R]=Standby, [Q/ESC]=Quit"
            key = self._render_preview(cam_frames, hud, (0, 140, 0))
            if key == ord("r"):
                print("\n↩️  현재 추론을 중단하고 대기 상태로 복귀합니다.")
                return "standby"
            if key in (ord("q"), 27):
                print("\n⏹️  사용자가 프리뷰 창에서 Q/ESC를 눌러 프로그램을 종료합니다.")
                self.running = False
                return "quit"

            sys.stdout.write(f"\r  🤖 자율 주행 실행 중... 스텝: {t+1:3d}/{max_timesteps} ({elapsed:4.1f}초, 추론: {inf_ms:4.1f}ms)")
            sys.stdout.flush()

            # 제어 주기 유지
            dt_step = time.time() - t0
            sleep_time = self.dt - dt_step
            if sleep_time > 0:
                time.sleep(sleep_time)

        print(f"\n✅ [Ep #{ep_idx}] 추론 완료! 대기 상태로 복귀합니다 (토크 유지).")
        return "done"

    def run(self, max_timesteps: int = 600):
        self.connect()
        self.running = True

        print("\n" + "-" * 68)
        print(" 🟢 [자율 주행 준비] 카메라 프리뷰 확인 후 [SPACE]로 추론을 시작합니다.")
        print(f" 📦 [청크 설정] 모델 청크: {CHUNK_SIZE}스텝 | 실효 윈도우: {self.effective_chunk}스텝 ({self.effective_chunk * self.dt:.2f}초)")
        print("    ([SPACE]: 추론 시작 | [R]: 추론 중 대기 복귀 | [Q/ESC / Ctrl+C]: 종료 및 토크 해제)")
        print("-" * 68 + "\n")

        # 모델 GPU 웜업
        print("⚡ [모델 웜업] 첫 스텝 지연 방지를 위해 사전 웜업을 수행합니다...")
        with torch.inference_mode():
            dummy_qpos = torch.zeros(1, STATE_DIM, device=self.device)
            dummy_img = torch.zeros(1, len(CAMERA_NAMES), 3, IMAGE_SIZE[0], IMAGE_SIZE[1], device=self.device)
            self.model(qpos=dummy_qpos, image=dummy_img)
        print("✅ 모델 GPU 웜업 완료.")

        # 로봇 이동 전부터 카메라 프리뷰 창 즉시 표시
        if self.show_preview:
            self._render_preview(self.camera_cap.read_all(), "INITIALIZING...", (100, 100, 100))

        if self.start_obs and self.running:
            self.move_to_observation_pose(duration=2.0, settle_sec=0.5)

        ep_idx = 1
        try:
            while self.running:
                if not self._wait_for_start_or_quit(ep_idx):
                    break

                print(f"\n🚀 [Ep #{ep_idx}] 자율 추론 시작!")
                outcome = self._run_episode(max_timesteps=max_timesteps, ep_idx=ep_idx)
                if outcome == "quit" or not self.running:
                    break

                # 추론 종료 후 토크는 유지한 채 관측/홈 자세로 복귀하여 다음 [SPACE] 대기
                if self.start_obs:
                    self.move_to_observation_pose(duration=1.8, settle_sec=0.3)
                else:
                    self.return_to_home(duration=1.8)
                ep_idx += 1
        finally:
            self.stop()

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        self.running = False

        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        try:
            self.return_to_home()
        except Exception as e:
            print(f"[경고] 홈 복귀 중 예외: {e}")

        try:
            # 안전하게 모터 토크 해제
            self.driver.set_all_torques(range(1, 7), False)
            print("💤 팔로워 모터 토크 해제 완료 (대기 모드)")
        except Exception:
            pass

        self.driver.close()
        self.camera_cap.stop()
        print("\n🏁 자율 주행 세션이 정상 종료되었습니다.")


def main():
    parser = argparse.ArgumentParser(description="SO-101 양안 DINOv2 ACT 실 로봇 자율 배포")
    parser.add_argument("--ckpt", type=str, default="policy_best.pth", help="가중치 파일 이름 (기본값: policy_best.pth)")
    parser.add_argument("--ckpt-dir", type=str, default=CKPT_DIR, help=f"체크포인트 디렉토리 (기본값: {CKPT_DIR})")
    parser.add_argument("--config", type=str, default="configs/calibration.json", help="캘리브레이션 JSON 경로")
    parser.add_argument("--left-cam", type=int, default=2, help="좌안 카메라 인덱스 (기본값: 2)")
    parser.add_argument("--right-cam", type=int, default=0, help="우안 카메라 인덱스 (기본값: 0)")
    parser.add_argument("--focus", type=int, default=FIXED_FOCUS_VALUE, help="초점 고정 값 (0~255, 기본값: 25)")
    parser.add_argument("--fps", type=int, default=30, help="제어 주기 FPS (기본값: 30)")
    parser.add_argument("--max-steps", type=int, default=600, help="최대 실행 스텝 수 (기본값: 600, 약 20초)")
    parser.add_argument("--no-agg", action="store_true", help="Temporal Aggregation 비활성화 (단일 청크 스텝)")
    parser.add_argument("--chunk", type=int, default=None, help="실효 청크 윈도우 크기 (기본값: 전체 CHUNK_SIZE)")
    parser.add_argument("--no-preview", action="store_true", help="프리뷰 창 끄기 (터미널 모드)")
    parser.add_argument("--start-obs", action="store_true", help="추론 시작 전 OBSERVATION_POSE로 먼저 이동")
    parser.add_argument("--max-step-deg", type=float, default=5.0, help="스텝당 최대 관절 변화량 [deg] (0=제한 없음, 기본: 5.0)")
    parser.add_argument("--vis-gate", action="store_true", default=False, help="초기 탐색 중 보조 헤드 p_vis < thr일 때 하강/개방 차단 게이트 활성화 (기본: OFF)")
    parser.add_argument("--vis-gate-thr", type=float, default=0.5, help="vis-gate 큐브 보임 판정 임계값 (기본: 0.5)")
    args = parser.parse_args()

    deployer = RealRobotDeployer(
        ckpt_dir=args.ckpt_dir,
        ckpt_name=args.ckpt,
        config_path=args.config,
        left_cam_id=args.left_cam,
        right_cam_id=args.right_cam,
        focus=args.focus,
        fps=args.fps,
        temporal_agg=not args.no_agg,
        effective_chunk=args.chunk,
        show_preview=not args.no_preview,
        start_obs=args.start_obs,
        max_step_deg=args.max_step_deg,
        use_vis_gate=args.vis_gate,
        vis_gate_thr=args.vis_gate_thr,
    )

    def sigint_handler(sig, frame):
        deployer.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)
    deployer.run(max_timesteps=args.max_steps)


if __name__ == "__main__":
    main()

"""SO-101 양팔(Dual SO-101, 총 12 DOF) 로봇 제어기.

왼팔 6 DOF (관절 1~5 + 그리퍼 6) + 오른팔 6 DOF (관절 1~5 + 그리퍼 6)
리더 암(Leader) 및 팔로워 암(Follower) 시리얼 동기화 제어.
"""

import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from .sts3215_driver import STS3215Driver


def raw_to_rad(raw_val: Optional[int], zero_pos: int = 2048, direction: int = 1) -> float:
    """STS3215 Raw (0~4095) -> 라디안 (영점 오프셋 및 회전 방향 반영)"""
    if raw_val is None:
        raw_val = zero_pos
    dir_sign = -1 if direction < 0 else 1
    return float((raw_val - zero_pos) * dir_sign * (2.0 * np.pi / 4096.0))


def rad_to_raw(
    rad_val: float,
    zero_pos: int = 2048,
    direction: int = 1,
    min_pos: int = 0,
    max_pos: int = 4095,
) -> int:
    """라디안 -> STS3215 Raw (영점 오프셋, 회전 방향 및 안전 한계 클램핑 반영)"""
    dir_sign = -1 if direction < 0 else 1
    raw = int(round(rad_val * dir_sign * (4096.0 / (2.0 * np.pi)) + zero_pos))
    lo = max(0, min(int(min_pos), int(max_pos)))
    hi = min(4095, max(int(min_pos), int(max_pos)))
    return max(lo, min(hi, raw))


def load_calibration_params(calibration_path: str) -> Dict[int, Dict[str, int]]:
    """calibration.json에서 관절별(1~6) 영점·방향·한계치를 로드"""
    with open(calibration_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    joints = data.get("joints", {})
    params: Dict[int, Dict[str, int]] = {}
    for j_id in range(1, 7):
        j_cfg = joints.get(f"joint_{j_id}", {})
        params[j_id] = {
            "follower_zero": int(j_cfg.get("follower_zero", 2048)),
            "leader_zero": int(j_cfg.get("leader_zero", 2048)),
            "direction": int(j_cfg.get("direction", 1)),
            "follower_min": int(j_cfg.get("follower_min", j_cfg.get("min", 0))),
            "follower_max": int(j_cfg.get("follower_max", j_cfg.get("max", 4095))),
        }
    return params


class DualArmController:
    """양팔 SO-101 (총 12개 모터) 실시간 제어기"""

    def __init__(
        self,
        left_follower_port: str = "/dev/ttyACM0",
        right_follower_port: str = "/dev/ttyACM1",
        left_leader_port: Optional[str] = "/dev/ttyACM2",
        right_leader_port: Optional[str] = "/dev/ttyACM3",
        baudrate: int = 1000000,
        joint_ids: Optional[List[int]] = None,
        calibration_path: Optional[str] = None,
    ):
        self.joint_ids = joint_ids if joint_ids is not None else [1, 2, 3, 4, 5, 6]
        self.calib_params = (
            load_calibration_params(calibration_path)
            if calibration_path and os.path.exists(calibration_path)
            else {}
        )
        self._last_follower_raw: Dict[int, int] = {}
        self._last_leader_raw: Dict[int, int] = {}

        # 팔로워 드라이버 (실제 작업 수행 암)
        self.l_follower = STS3215Driver(left_follower_port, baudrate=baudrate, timeout=0.03)
        self.r_follower = STS3215Driver(right_follower_port, baudrate=baudrate, timeout=0.03)

        # 리더 드라이버 (교시 조작용 암)
        self.l_leader = (
            STS3215Driver(left_leader_port, baudrate=baudrate, timeout=0.03) if left_leader_port else None
        )
        self.r_leader = (
            STS3215Driver(right_leader_port, baudrate=baudrate, timeout=0.03) if right_leader_port else None
        )

        self.is_connected = False

    def connect(self) -> bool:
        """모든 연결된 포트 초기화"""
        lf_ok = self.l_follower.connect()
        rf_ok = self.r_follower.connect()

        ll_ok = self.l_leader.connect() if self.l_leader else True
        rl_ok = self.r_leader.connect() if self.r_leader else True

        if not (lf_ok and rf_ok):
            print(f"⚠️ [DualArm] 팔로워 로봇 암 연결 실패: Left={lf_ok}, Right={rf_ok}")
            self.is_connected = False
            return False

        # 리더 암 토크 해제 (수동 조작 가능하도록)
        if self.l_leader and ll_ok:
            for j_id in self.joint_ids:
                self.l_leader.set_torque(j_id, False)
        if self.r_leader and rl_ok:
            for j_id in self.joint_ids:
                self.r_leader.set_torque(j_id, False)

        # 팔로워 암 토크 인가
        for j_id in self.joint_ids:
            self.l_follower.set_torque(j_id, True)
            self.r_follower.set_torque(j_id, True)

        self.is_connected = True
        print(f"✅ [DualArm] 양팔 팔로워 로봇 암 연결 성공! (총 12 DOF)")
        return True

    def _get_joint_cfg(self, j_id: int) -> Dict[str, int]:
        return self.calib_params.get(
            j_id,
            {
                "follower_zero": 2048,
                "leader_zero": 2048,
                "direction": 1,
                "follower_min": 0,
                "follower_max": 4095,
            },
        )

    def read_follower_qpos(self) -> np.ndarray:
        """팔로워 암 현재 12개 관절 각도(라디안) 읽기: [Left 6개, Right 6개]"""
        if not self.is_connected:
            return np.zeros(12, dtype=np.float32)

        qpos = []
        for idx_offset, drv in ((0, self.l_follower), (6, self.r_follower)):
            for i, j_id in enumerate(self.joint_ids):
                cfg = self._get_joint_cfg(j_id)
                zero_pos = cfg["follower_zero"]
                raw = drv.read_position(j_id)
                cache_key = idx_offset + i
                if raw is None:
                    raw = self._last_follower_raw.get(cache_key, zero_pos)
                else:
                    self._last_follower_raw[cache_key] = raw
                qpos.append(raw_to_rad(raw, zero_pos=zero_pos, direction=cfg["direction"]))

        return np.array(qpos, dtype=np.float32)

    def read_leader_qpos(self) -> np.ndarray:
        """리더 암 현재 12개 관절 각도(라디안) 읽기"""
        if not self.is_connected or not self.l_leader or not self.r_leader:
            return self.read_follower_qpos()

        qpos = []
        for idx_offset, drv in ((0, self.l_leader), (6, self.r_leader)):
            for i, j_id in enumerate(self.joint_ids):
                cfg = self._get_joint_cfg(j_id)
                zero_pos = cfg["leader_zero"]
                raw = drv.read_position(j_id)
                cache_key = idx_offset + i
                if raw is None:
                    raw = self._last_leader_raw.get(cache_key, zero_pos)
                else:
                    self._last_leader_raw[cache_key] = raw
                qpos.append(raw_to_rad(raw, zero_pos=zero_pos, direction=cfg["direction"]))

        return np.array(qpos, dtype=np.float32)

    def send_action(self, action_12d: np.ndarray):
        """12차원 목표 각도(라디안)를 양팔 팔로워에 동기 전송"""
        if not self.is_connected:
            return

        left_targets = {}
        right_targets = {}
        for i, j_id in enumerate(self.joint_ids):
            cfg = self._get_joint_cfg(j_id)
            left_targets[j_id] = rad_to_raw(
                action_12d[i],
                zero_pos=cfg["follower_zero"],
                direction=cfg["direction"],
                min_pos=cfg["follower_min"],
                max_pos=cfg["follower_max"],
            )
            right_targets[j_id] = rad_to_raw(
                action_12d[i + 6],
                zero_pos=cfg["follower_zero"],
                direction=cfg["direction"],
                min_pos=cfg["follower_min"],
                max_pos=cfg["follower_max"],
            )

        self.l_follower.sync_write_positions(left_targets)
        self.r_follower.sync_write_positions(right_targets)

    def teleop_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """텔레옵 1스텝: 리더 암 각도를 읽어 팔로워 암에 전달하고 qpos와 action 반환"""
        leader_qpos = self.read_leader_qpos()
        self.send_action(leader_qpos)
        follower_qpos = self.read_follower_qpos()
        return follower_qpos, leader_qpos

    def disconnect(self):
        """종료 시 안전하게 토크 끄고 포트 해제"""
        if self.is_connected:
            try:
                for j_id in self.joint_ids:
                    self.l_follower.set_torque(j_id, False)
                    self.r_follower.set_torque(j_id, False)
            except Exception:
                pass
            self.l_follower.close()
            self.r_follower.close()
            if self.l_leader:
                self.l_leader.close()
            if self.r_leader:
                self.r_leader.close()
            self.is_connected = False
            print("🛑 [DualArm] 양팔 연결 해제 완료.")

#!/usr/bin/env python3
"""실물 SO-101 로봇팔 URDF 기반 IK Active Search 데이터 수집기.

시뮬레이션(Isaac Sim) 없이 순수 Python + NumPy URDF 기구학(FK/DLS-IK)으로
실물 로봇을 구동해 Active Search -> 정렬 -> 파지 -> 상자 안착 -> 복귀 궤적을 실행하고,
각 에피소드 완료 후 사용자 확인(y/n)을 거쳐 HDF5 데이터셋으로 저장합니다.
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project")

for p in (SCRIPT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "models")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_recorder import HDF5DatasetRecorder
from hardware.camera_interface import MultiCameraCapture
from hardware.dual_arm_controller import rad_to_raw, raw_to_rad
from hardware.identify_arms import find_port_by_serial
from hardware.sts3215_driver import STS3215Driver

URDF_PATH = os.path.join(WORKSPACE_DIR, "so101_dual_cam.urdf")
CALIB_PATH = os.path.join(PROJECT_ROOT, "configs", "calibration.json")

OBSERVATION_POSE = np.array([0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0], dtype=np.float32)
SCAN_BASE_POSE = np.array([0.0, np.radians(-62.0), np.radians(26.0), np.radians(58.0), 0.0], dtype=np.float32)
TARGET_GRIP_DEG = 11.5

# 실물 PnP 역산 기준 카메라 파라미터 (env_manager.py와 동일)
CAM_FOCAL_LEN = 19.32
CAM_H_APERTURE = 20.955
CAM_V_APERTURE = 15.2
LEFT_CAM_ROT_XYZ = (179.52, 3.48, -89.73)
RIGHT_CAM_ROT_XYZ = (175.25, 5.11, -90.95)


def _rpy_to_mat(r: float, p: float, y: float) -> np.ndarray:
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _xyz_euler_deg_to_mat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """USD xformOp:rotateXYZ (extrinsic XYZ = Rz @ Ry @ Rx in column-vector convention)."""
    return _rpy_to_mat(np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg))


def _axis_angle_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = axis / np.linalg.norm(axis)
    x, y, z = ax
    c, s = np.cos(angle), np.sin(angle)
    C = 1.0 - c
    return np.array([
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
    ], dtype=np.float64)


class SO101URDFKinematics:
    """so101_dual_cam.urdf 기반 순수 NumPy 순기구학(FK) 및 DLS 역기구학(IK) 솔버."""

    ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def __init__(self, urdf_path: str = URDF_PATH):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        self.joints: Dict[str, dict] = {}
        for j in root.findall("joint"):
            name = j.attrib["name"]
            parent = j.find("parent").attrib["link"]
            child = j.find("child").attrib["link"]
            orig = j.find("origin")
            xyz = np.array([float(v) for v in orig.attrib.get("xyz", "0 0 0").split()], dtype=np.float64)
            rpy = np.array([float(v) for v in orig.attrib.get("rpy", "0 0 0").split()], dtype=np.float64)
            ax_el = j.find("axis")
            axis = np.array([float(v) for v in ax_el.attrib["xyz"].split()], dtype=np.float64) if ax_el is not None else np.array([0.0, 0.0, 1.0])
            lim_el = j.find("limit")
            lo = float(lim_el.attrib.get("lower", -np.pi)) if lim_el is not None else -np.pi
            hi = float(lim_el.attrib.get("upper", np.pi)) if lim_el is not None else np.pi
            self.joints[name] = {
                "parent": parent,
                "child": child,
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "lower": lo,
                "upper": hi,
            }

        self.lower = np.array([self.joints[n]["lower"] for n in self.ARM_JOINTS], dtype=np.float64)
        self.upper = np.array([self.joints[n]["upper"] for n in self.ARM_JOINTS], dtype=np.float64)

        # 실물 카메라 로컬 회전 (USD rotateXYZ)
        self.T_cam_left_local = np.eye(4, dtype=np.float64)
        self.T_cam_left_local[:3, :3] = _xyz_euler_deg_to_mat(*LEFT_CAM_ROT_XYZ)
        self.T_cam_right_local = np.eye(4, dtype=np.float64)
        self.T_cam_right_local[:3, :3] = _xyz_euler_deg_to_mat(*RIGHT_CAM_ROT_XYZ)

    def forward_kinematics(self, q5: np.ndarray) -> Dict[str, np.ndarray]:
        """base_link 기준 각 주요 링크의 4x4 동차변환행렬 계산."""
        T: Dict[str, np.ndarray] = {"base_link": np.eye(4, dtype=np.float64)}
        q_map = {n: float(q5[i]) for i, n in enumerate(self.ARM_JOINTS)}
        chain = self.ARM_JOINTS + ["gripper_frame_joint", "camera_left_joint", "camera_right_joint"]
        for jname in chain:
            j = self.joints[jname]
            T_loc = np.eye(4, dtype=np.float64)
            T_loc[:3, :3] = _rpy_to_mat(*j["rpy"])
            T_loc[:3, 3] = j["xyz"]
            if jname in q_map:
                T_q = np.eye(4, dtype=np.float64)
                T_q[:3, :3] = _axis_angle_mat(j["axis"], q_map[jname])
                T_loc = T_loc @ T_q
            T[j["child"]] = T[j["parent"]] @ T_loc

        T["left_camera"] = T["camera_left_optical_frame"] @ self.T_cam_left_local
        T["right_camera"] = T["camera_right_optical_frame"] @ self.T_cam_right_local
        return T

    def get_ee_pos(self, q5: np.ndarray) -> np.ndarray:
        return self.forward_kinematics(q5)["gripper_frame_link"][:3, 3]

    def compute_ik(
        self,
        target_pos: np.ndarray,
        warm_start: np.ndarray,
        max_iters: int = 120,
        tol_m: float = 0.0015,
    ) -> Tuple[np.ndarray, bool]:
        """수치 야코비안 + Damped Least Squares(LM) 역기구학."""
        q = np.clip(np.array(warm_start, dtype=np.float64), self.lower, self.upper)
        target = np.array(target_pos, dtype=np.float64)
        eps = 1e-4
        lam = 1e-3

        for _ in range(max_iters):
            pos = self.get_ee_pos(q)
            err = target - pos
            if np.linalg.norm(err) < tol_m:
                return q.astype(np.float32), True

            J = np.zeros((3, 5), dtype=np.float64)
            for k in range(5):
                qp = q.copy()
                qp[k] += eps
                J[:, k] = (self.get_ee_pos(qp) - pos) / eps

            # Damped Least Squares: dq = J^T (J J^T + lam^2 I)^-1 err
            jjt = J @ J.T + (lam ** 2) * np.eye(3)
            dq = J.T @ np.linalg.solve(jjt, err)
            step_norm = np.linalg.norm(dq)
            if step_norm > 0.25:
                dq *= 0.25 / step_norm
            q = np.clip(q + dq, self.lower, self.upper)

        final_err = np.linalg.norm(target - self.get_ee_pos(q))
        return q.astype(np.float32), (final_err < 0.008)

    def project_pixel_to_table(
        self,
        q5: np.ndarray,
        cam_side: str,
        cu: float,
        cv: float,
        img_w: int = 640,
        img_h: int = 480,
        z_plane: float = 0.015,
    ) -> Optional[np.ndarray]:
        """실물 카메라 픽셀(cu, cv)을 테이블 평면(Z=0.015m) 위 3D 월드 좌표로 역투영."""
        T_map = self.forward_kinematics(q5)
        T_cam = T_map["left_camera" if cam_side == "left" else "right_camera"]
        p_cam = T_cam[:3, 3]
        R_cam = T_cam[:3, :3]  # columns: +X right, +Y up, +Z back (-Z look)

        fx = float(img_w) * (CAM_FOCAL_LEN / CAM_H_APERTURE)
        fy = float(img_h) * (CAM_FOCAL_LEN / CAM_V_APERTURE)
        x_loc = (cu - img_w / 2.0) / fx
        y_loc = -(cv - img_h / 2.0) / fy
        dir_local = np.array([x_loc, y_loc, -1.0], dtype=np.float64)
        dir_local /= np.linalg.norm(dir_local)
        dir_world = R_cam @ dir_local
        dir_world /= np.linalg.norm(dir_world)

        if abs(dir_world[2]) < 1e-4:
            return None
        lam = (z_plane - p_cam[2]) / dir_world[2]
        if lam <= 0.0:
            return None
        return p_cam + lam * dir_world


def detect_red_cube_rgb(rgb_img: np.ndarray, min_pixels: int = 60) -> Tuple[bool, int, float, float]:
    """RGB 이미지에서 빨간색 큐브 중심(cu, cv) 검출."""
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 95, 55]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([168, 95, 55]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return False, 0, 0.0, 0.0

    best_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[best_idx, cv2.CC_STAT_AREA])
    if area < min_pixels:
        return False, area, 0.0, 0.0
    cu, cv = centroids[best_idx]
    return True, area, float(cu), float(cv)


class RealRobotIKCollector:
    """실물 SO-101 Active Search IK 실행 및 에피소드 수집기."""

    def __init__(
        self,
        env_name: str = "real_robot",
        left_cam_id: int = 2,
        right_cam_id: int = 0,
        focus: int = 25,
        fps: float = 50.0,
        box_xyz: Tuple[float, float, float] = (0.0, 0.28, 0.0),
        show_preview: bool = True,
    ):
        self.fps = fps
        self.dt = 1.0 / fps
        self.box_xyz = np.array(box_xyz, dtype=np.float64)
        self.show_preview = show_preview and bool(os.environ.get("DISPLAY"))

        with open(CALIB_PATH, "r", encoding="utf-8") as f:
            self.calib = json.load(f)
        self.joints_cfg = self.calib["joints"]
        self.gripper_cfg = self.calib["gripper"]

        raw_port = self.calib.get("follower_port", "/dev/ttyACM0")
        serial_num = self.calib.get("follower_serial", "")
        self.port = find_port_by_serial(serial_num, default_port=raw_port) if serial_num else raw_port

        self.driver = STS3215Driver(self.port, baudrate=1000000, timeout=0.03)
        self.cam_cap = MultiCameraCapture(
            camera_dict={"cam_left": left_cam_id, "cam_right": right_cam_id},
            width=640,
            height=480,
            fps=30,
            disable_autofocus=True,
            focus_dict={"cam_left": focus, "cam_right": focus},
        )
        self.kin = SO101URDFKinematics(URDF_PATH)

        output_dir = os.path.join(PROJECT_ROOT, "data", "so101_active_search", env_name)
        self.recorder = HDF5DatasetRecorder(output_dir=output_dir, img_w=640, img_h=480, fps=fps)
        self.output_dir = output_dir
        self._last_q_rad = np.zeros(6, dtype=np.float32)

    def _get_next_ep_idx(self) -> int:
        existing = [f for f in os.listdir(self.output_dir) if f.startswith("episode_") and f.endswith(".hdf5")]
        max_idx = -1
        for fname in existing:
            try:
                max_idx = max(max_idx, int(fname[len("episode_") : -len(".hdf5")]))
            except ValueError:
                pass
        return max_idx + 1

    def connect(self):
        if not self.driver.connect():
            raise RuntimeError(f"시리얼 포트 연결 실패: {self.port}")
        self.driver.set_all_torques(range(1, 7), True)
        self.driver.set_torque_limit(6, 0.45)
        self.cam_cap.start()
        print(f"✅ 실물 로봇({self.port}) 및 양안 카메라 연결 완료.")

    def disconnect(self):
        try:
            self.driver.set_all_torques(range(1, 7), False)
        except Exception:
            pass
        self.driver.close()
        self.cam_cap.stop()
        if self.show_preview:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def read_qpos_rad(self) -> np.ndarray:
        raws = self.driver.read_positions(range(1, 7))
        q = np.zeros(6, dtype=np.float32)
        for i in range(1, 7):
            cfg = self.joints_cfg[f"joint_{i}"]
            zero_pos = int(cfg.get("follower_zero", 2048))
            direction = int(cfg.get("direction", 1))
            val = raws.get(i)
            if val is None:
                q[i - 1] = self._last_q_rad[i - 1]
            else:
                q[i - 1] = raw_to_rad(val, zero_pos=zero_pos, direction=direction)
        self._last_q_rad = q.copy()
        return q

    def get_rgb_frames(self) -> Tuple[np.ndarray, np.ndarray]:
        frames = self.cam_cap.read_all()
        l_bgr = frames.get("cam_left")
        r_bgr = frames.get("cam_right")
        l_rgb = cv2.cvtColor(l_bgr, cv2.COLOR_BGR2RGB) if l_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        r_rgb = cv2.cvtColor(r_bgr, cv2.COLOR_BGR2RGB) if r_bgr is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        return l_rgb, r_rgb

    def _show_preview(self, img_l_rgb: np.ndarray, img_r_rgb: np.ndarray, det_l=None, det_r=None, label: str = ""):
        if not self.show_preview:
            return
        vis_l = cv2.cvtColor(cv2.resize(img_l_rgb, (480, 360)), cv2.COLOR_RGB2BGR)
        vis_r = cv2.cvtColor(cv2.resize(img_r_rgb, (480, 360)), cv2.COLOR_RGB2BGR)
        if det_l and det_l[0]:
            cv2.circle(vis_l, (int(det_l[2] * 0.75), int(det_l[3] * 0.75)), 8, (0, 255, 0), 2)
        if det_r and det_r[0]:
            cv2.circle(vis_r, (int(det_r[2] * 0.75), int(det_r[3] * 0.75)), 8, (0, 255, 0), 2)
        combo = np.hstack([vis_l, vis_r])
        if label:
            cv2.putText(combo, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Real Robot IK Collector", combo)
        cv2.waitKey(1)

    def send_q_rad(self, q_full_rad: np.ndarray):
        goals = {}
        for i in range(1, 6):
            cfg = self.joints_cfg[f"joint_{i}"]
            raw = rad_to_raw(
                float(q_full_rad[i - 1]),
                zero_pos=int(cfg.get("follower_zero", 2048)),
                direction=int(cfg.get("direction", 1)),
                min_pos=int(cfg.get("follower_min", cfg.get("min", 0))),
                max_pos=int(cfg.get("follower_max", cfg.get("max", 4095))),
            )
            goals[i] = raw
        g_cfg = self.joints_cfg["joint_6"]
        g_raw = rad_to_raw(
            float(q_full_rad[5]),
            zero_pos=int(g_cfg.get("follower_zero", 2048)),
            direction=int(g_cfg.get("direction", 1)),
        )
        f_open, f_close = int(self.gripper_cfg["follower_open"]), int(self.gripper_cfg["follower_close"])
        goals[6] = max(min(f_open, f_close), min(max(f_open, f_close), g_raw))
        self.driver.sync_write_positions(goals, speed=0, acc=0)

    def step_joint_command(self, q_full_rad: np.ndarray, record: bool = True):
        t0 = time.time()
        q_curr = self.read_qpos_rad()
        img_l, img_r = self.get_rgb_frames()
        if record:
            self.recorder.record_step(q_curr, np.array(q_full_rad[:6], dtype=np.float32), img_l, img_r)
        self.send_q_rad(q_full_rad)
        self._show_preview(img_l, img_r, label=f"Steps: {self.recorder.get_step_count()}")
        elapsed = time.time() - t0
        if elapsed < self.dt:
            time.sleep(self.dt - elapsed)

    def move_joints_smoothly(self, target_5dof: np.ndarray, duration_sec: float = 1.5, grip_deg: float = 0.0, record: bool = True):
        steps = max(1, int(duration_sec * self.fps))
        q_curr = self.read_qpos_rad()
        q_start = np.array(q_curr[:5], dtype=np.float32)
        t_rad = np.array(target_5dof[:5], dtype=np.float32)
        g_rad = np.radians(grip_deg)
        for i in range(1, steps + 1):
            alpha = 0.5 * (1.0 - np.cos(np.pi * (i / steps)))
            q_full = np.zeros(6, dtype=np.float32)
            q_full[:5] = q_start + alpha * (t_rad - q_start)
            q_full[5] = g_rad
            self.step_joint_command(q_full, record=record)

    def run_single_episode(self) -> Tuple[bool, bool, Tuple[float, float]]:
        self.recorder.reset_buffer()
        self.move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5, grip_deg=0.0, record=False)
        time.sleep(0.3)

        q_curr = self.read_qpos_rad()
        img_l, img_r = self.get_rgb_frames()
        det_l = detect_red_cube_rgb(img_l, min_pixels=60)
        det_r = detect_red_cube_rgb(img_r, min_pixels=60)
        self._show_preview(img_l, img_r, det_l, det_r, "Detecting @ OBS")

        search_triggered = False
        if not (det_l[0] and det_r[0]):
            search_triggered = True
            print("  🔍 정면 양안 미검출 -> 능동 스캔(Active Search) 시작")
            self.move_joints_smoothly(SCAN_BASE_POSE, duration_sec=1.0, grip_deg=0.0, record=True)
            scan_start = np.array([np.radians(65.0), SCAN_BASE_POSE[1], SCAN_BASE_POSE[2], SCAN_BASE_POSE[3], 0.0], dtype=np.float32)
            self.move_joints_smoothly(scan_start, duration_sec=1.1, grip_deg=0.0, record=True)

            spotted = False
            theta_target_deg = 0.0
            for pan_deg in np.linspace(-65.0, +65.0, 130):
                q_scan = np.array([np.radians(-pan_deg), SCAN_BASE_POSE[1], SCAN_BASE_POSE[2], SCAN_BASE_POSE[3], 0.0], dtype=np.float32)
                q_full = np.zeros(6, dtype=np.float32)
                q_full[:5] = q_scan
                self.step_joint_command(q_full, record=True)

                img_l, img_r = self.get_rgb_frames()
                dl = detect_red_cube_rgb(img_l, min_pixels=45)
                dr = detect_red_cube_rgb(img_r, min_pixels=45)
                self._show_preview(img_l, img_r, dl, dr, f"Scanning pan={-pan_deg:+.1f}")
                if dl[0] or dr[0]:
                    q5_now = self.read_qpos_rad()[:5]
                    pts = []
                    if dl[0]:
                        pt = self.kin.project_pixel_to_table(q5_now, "left", dl[2], dl[3])
                        if pt is not None:
                            pts.append(pt)
                    if dr[0]:
                        pt = self.kin.project_pixel_to_table(q5_now, "right", dr[2], dr[3])
                        if pt is not None:
                            pts.append(pt)
                    if pts:
                        p_spot = np.mean(pts, axis=0)
                        theta_target_deg = float(np.degrees(np.arctan2(p_spot[1], p_spot[0])))
                        spotted = True
                        break

            if not spotted:
                print("  ❌ 스캔 중 빨간 큐브를 찾지 못했습니다.")
                self.move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.2, grip_deg=0.0, record=False)
                return False, True, (0.0, 0.0)

            obs_pitch = [np.radians(-68.3), np.radians(32.8), np.radians(74.8)]
            q_face = np.array([np.radians(-theta_target_deg), obs_pitch[0], obs_pitch[1], obs_pitch[2], 0.0], dtype=np.float32)
            self.move_joints_smoothly(q_face, duration_sec=1.1, grip_deg=0.0, record=True)
            time.sleep(0.2)

            # Fine gaze servoing
            img_l, img_r = self.get_rgb_frames()
            dl = detect_red_cube_rgb(img_l, min_pixels=40)
            dr = detect_red_cube_rgb(img_r, min_pixels=40)
            q5_now = self.read_qpos_rad()[:5]
            ref_pts = []
            if dl[0]:
                pt = self.kin.project_pixel_to_table(q5_now, "left", dl[2], dl[3])
                if pt is not None:
                    ref_pts.append(pt)
            if dr[0]:
                pt = self.kin.project_pixel_to_table(q5_now, "right", dr[2], dr[3])
                if pt is not None:
                    ref_pts.append(pt)
            if ref_pts:
                p_ref = np.mean(ref_pts, axis=0)
                theta_fine = float(np.degrees(np.arctan2(p_ref[1], p_ref[0])))
                if abs(theta_fine - theta_target_deg) > 1.5:
                    q_fine = np.array([np.radians(-theta_fine), obs_pitch[0], obs_pitch[1], obs_pitch[2], 0.0], dtype=np.float32)
                    self.move_joints_smoothly(q_fine, duration_sec=0.5, grip_deg=0.0, record=True)
                    time.sleep(0.15)

        # 정면 응시 3D 좌표 추정
        img_l, img_r = self.get_rgb_frames()
        dl = detect_red_cube_rgb(img_l, min_pixels=40)
        dr = detect_red_cube_rgb(img_r, min_pixels=40)
        q5_now = self.read_qpos_rad()[:5]
        est_pts = []
        if dl[0]:
            pt = self.kin.project_pixel_to_table(q5_now, "left", dl[2], dl[3])
            if pt is not None:
                est_pts.append(pt)
        if dr[0]:
            pt = self.kin.project_pixel_to_table(q5_now, "right", dr[2], dr[3])
            if pt is not None:
                est_pts.append(pt)
        if not est_pts:
            print("  ❌ 정면 응시 상태에서 큐브 3D 위치 추정 실패.")
            self.move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.2, grip_deg=0.0, record=False)
            return False, search_triggered, (0.0, 0.0)

        p_est = np.mean(est_pts, axis=0)
        x_est, y_est = float(p_est[0]), float(p_est[1])
        print(f"  🎯 [Vision 3D] 추정 위치: X={x_est:.3f}m, Y={y_est:.3f}m (L:{dl[1]}px, R:{dr[1]}px)")

        # IK 계산 (search_and_pick_policy.py와 동일 기하)
        pivot_xy = np.array([0.038835, 0.0], dtype=np.float64)
        rel_xy = np.array([x_est, y_est], dtype=np.float64) - pivot_xy
        R_cube = float(np.linalg.norm(rel_xy))
        if not (0.20 <= R_cube <= 0.38):
            print(f"  ⚠️ 작업 반경 초과 (R={R_cube:.3f}m). 안전을 위해 중단합니다.")
            return False, search_triggered, (x_est, y_est)

        theta_cube = float(np.arctan2(rel_xy[1], rel_xy[0]))
        pan_world = theta_cube - float(np.arctan2(0.018, R_cube))
        u_dir = np.array([np.cos(pan_world), np.sin(pan_world)])
        z_floor = 0.016

        pos_pre_grasp = np.array([pivot_xy[0] + (R_cube - 0.028) * u_dir[0], pivot_xy[1] + (R_cube - 0.028) * u_dir[1], z_floor + 0.003])
        pos_grasp = np.array([pivot_xy[0] + (R_cube + 0.015) * u_dir[0], pivot_xy[1] + (R_cube + 0.015) * u_dir[1], z_floor])

        ws_grasp = np.array([-pan_world, np.radians(20.0), np.radians(85.0), np.radians(-80.0), np.radians(90.0)])
        q_grasp, succ_g = self.kin.compute_ik(pos_grasp, ws_grasp)
        q_pre_grasp, succ_pg = self.kin.compute_ik(pos_pre_grasp, q_grasp if succ_g else ws_grasp)
        if succ_g and (not succ_pg or float(q_pre_grasp[2]) < np.radians(25.0)):
            q_pre_grasp = q_grasp.copy()
            q_pre_grasp[1] -= np.radians(14.0)
            q_pre_grasp[2] = min(float(q_grasp[2]) + np.radians(18.0), np.radians(95.0))
            q_pre_grasp[3] += np.radians(4.0)
            succ_pg = True

        rel_box = self.box_xyz[:2] - pivot_xy
        R_box = float(np.linalg.norm(rel_box))
        pan_box = float(np.arctan2(rel_box[1], rel_box[0])) - float(np.arctan2(0.018, R_box))
        u_box = np.array([np.cos(pan_box), np.sin(pan_box)])
        pos_box = np.array([pivot_xy[0] + (R_box + 0.015) * u_box[0], pivot_xy[1] + (R_box + 0.015) * u_box[1], float(self.box_xyz[2]) + 0.065])
        ws_box = np.array([np.clip(-pan_box, -1.90, 1.90), np.radians(5.0), np.radians(70.0), np.radians(-60.0), np.radians(90.0)])
        q_box_place, succ_bp = self.kin.compute_ik(pos_box, ws_box)

        if not (succ_pg and succ_g and succ_bp):
            print(f"  ❌ IK 해를 찾지 못했습니다 (pg={succ_pg}, g={succ_g}, box={succ_bp}).")
            return False, search_triggered, (x_est, y_est)

        desired_pan = float(-pan_world)
        q_pre_hover = q_pre_grasp.copy()
        q_pre_hover[1] -= np.radians(25.0)
        for q in (q_pre_hover, q_pre_grasp, q_grasp):
            q[0] = desired_pan
            q[3] -= np.radians(1.0)
            q[4] = np.radians(90.0)

        q_lift = q_grasp.copy()
        q_lift[1] -= np.radians(35.0)
        q_lift[2] -= np.radians(15.0)
        q_lift[3] += np.radians(15.0)

        q_box_place[4] = np.radians(90.0)
        q_box_hover = q_box_place.copy()
        q_box_hover[1] -= np.radians(28.0)
        q_box_hover[2] -= np.radians(12.0)
        q_box_hover[3] += np.radians(12.0)

        # 순차 실행 및 녹화
        self.move_joints_smoothly(q_pre_hover, duration_sec=1.5, grip_deg=70.0, record=True)
        self.move_joints_smoothly(q_pre_grasp, duration_sec=1.2, grip_deg=70.0, record=True)
        self.move_joints_smoothly(q_grasp, duration_sec=1.2, grip_deg=70.0, record=True)

        for g_deg in np.linspace(70.0, TARGET_GRIP_DEG, 60):
            q_clamp = np.zeros(6, dtype=np.float32)
            q_clamp[:5] = q_grasp
            q_clamp[5] = np.radians(g_deg)
            self.step_joint_command(q_clamp, record=True)
        for _ in range(35):
            q_hold = np.zeros(6, dtype=np.float32)
            q_hold[:5] = q_grasp
            q_hold[5] = np.radians(TARGET_GRIP_DEG)
            self.step_joint_command(q_hold, record=True)

        self.move_joints_smoothly(q_lift, duration_sec=1.8, grip_deg=TARGET_GRIP_DEG, record=True)
        self.move_joints_smoothly(q_box_hover, duration_sec=1.8, grip_deg=TARGET_GRIP_DEG, record=True)
        self.move_joints_smoothly(q_box_place, duration_sec=1.2, grip_deg=TARGET_GRIP_DEG, record=True)

        for g_deg in np.linspace(TARGET_GRIP_DEG, 70.0, 40):
            q_rel = np.zeros(6, dtype=np.float32)
            q_rel[:5] = q_box_place
            q_rel[5] = np.radians(g_deg)
            self.step_joint_command(q_rel, record=True)
        time.sleep(0.5)

        self.move_joints_smoothly(q_box_hover, duration_sec=1.2, grip_deg=70.0, record=True)
        q_box_clear = q_box_hover.copy()
        q_box_clear[1] -= np.radians(16.0)
        q_box_clear[2] -= np.radians(10.0)
        self.move_joints_smoothly(q_box_clear, duration_sec=0.8, grip_deg=20.0, record=True)
        self.move_joints_smoothly(OBSERVATION_POSE, duration_sec=1.5, grip_deg=0.0, record=True)
        return True, search_triggered, (x_est, y_est)

    def run_interactive(self):
        self.connect()
        saved_count = 0
        try:
            self.move_joints_smoothly(OBSERVATION_POSE, duration_sec=2.0, grip_deg=0.0, record=False)
            while True:
                print("\n" + "=" * 72)
                ans = input("👉 큐브를 원하는 위치에 놓고 [Enter]를 누르세요 (종료: q + Enter): ").strip().lower()
                if ans == "q":
                    break

                ok, search_trig, est_xy = self.run_single_episode()
                steps = self.recorder.get_step_count()
                print(f"\n[실행 종료] 완료={ok}, Steps={steps}, Search={search_trig}, 추정XY=({est_xy[0]:.3f}, {est_xy[1]:.3f})")

                save_ans = input("💾 이 에피소드를 저장할까요? [y(저장) / n(버리기) / q(저장 안 하고 종료)]: ").strip().lower()
                if save_ans == "y":
                    ep_idx = self._get_next_ep_idx()
                    meta = {
                        "env_name": "real_robot",
                        "cube_spawn_xy": np.array(est_xy, dtype=np.float32),
                        "cube_yaw_deg": 0.0,
                        "search_triggered": bool(search_trig),
                        "stereo_error_mm": 0.0,
                        "success": True,
                    }
                    res = self.recorder.save_to_hdf5(ep_idx, metadata=meta)
                    if res:
                        save_path, n_steps, mb = res
                        saved_count += 1
                        print(f"  ✅ 저장 완료: {os.path.basename(save_path)} ({n_steps} steps, {mb:.1f}MB) | 총 저장: {saved_count}개")
                elif save_ans == "q":
                    self.recorder.reset_buffer()
                    break
                else:
                    self.recorder.reset_buffer()
                    print("  🗑️ 에피소드를 버렸습니다. 다음 위치를 준비하세요.")
        finally:
            self.disconnect()
            print(f"\n🏁 실물 IK 데이터 수집 종료 (이번 세션 저장: {saved_count}개, 경로: {self.output_dir})")


def main():
    parser = argparse.ArgumentParser(description="실물 SO-101 URDF IK 기반 Active Search 데이터 수집기")
    parser.add_argument("--env", type=str, default="real_robot", help="저장할 데이터셋 폴더명 (기본: real_robot, 기존 합본 시 domain_rand)")
    parser.add_argument("--left-cam", type=int, default=2, help="좌안 카메라 인덱스 (기본: 2)")
    parser.add_argument("--right-cam", type=int, default=0, help="우안 카메라 인덱스 (기본: 0)")
    parser.add_argument("--focus", type=int, default=25, help="고정 초점 값 (기본: 25)")
    parser.add_argument("--fps", type=float, default=50.0, help="제어/녹화 주기 Hz (기본: 50.0)")
    parser.add_argument("--box-x", type=float, default=0.0, help="목표 상자 중심 X [m] (기본: 0.0)")
    parser.add_argument("--box-y", type=float, default=0.28, help="목표 상자 중심 Y [m] (기본: 0.28)")
    parser.add_argument("--box-z", type=float, default=0.0, help="목표 상자 바닥 Z [m] (기본: 0.0)")
    parser.add_argument("--no-preview", action="store_true", help="프리뷰 창 끄기")
    parser.add_argument("--test-ik", action="store_true", help="하드웨어 연결 없이 URDF IK 계산만 오프라인 검증")
    args = parser.parse_args()

    if args.test_ik:
        kin = SO101URDFKinematics(URDF_PATH)
        for pt in [(0.30, 0.0, 0.016), (0.25, -0.15, 0.016), (0.0, 0.28, 0.065)]:
            ws = np.array([0.0, np.radians(15.0), np.radians(80.0), np.radians(-70.0), np.radians(90.0)])
            q, ok = kin.compute_ik(np.array(pt), ws)
            err_mm = np.linalg.norm(kin.get_ee_pos(q) - np.array(pt)) * 1000.0
            print(f"Target {pt} -> IK ok={ok}, err={err_mm:.2f}mm, q_deg={np.round(np.degrees(q), 1)}")
        return

    collector = RealRobotIKCollector(
        env_name=args.env,
        left_cam_id=args.left_cam,
        right_cam_id=args.right_cam,
        focus=args.focus,
        fps=args.fps,
        box_xyz=(args.box_x, args.box_y, args.box_z),
        show_preview=not args.no_preview,
    )
    collector.run_interactive()


if __name__ == "__main__":
    main()


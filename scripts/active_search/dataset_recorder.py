#!/usr/bin/env python3
"""
[Module 3] HDF5 Dataset Recorder for SO-101 Sim-to-Real
========================================================
ACT (Action Chunking with Transformers) 및 Diffusion Policy의 표준 규격에 맞추어
매 제어 스텝(50Hz)마다 양안 영상과 6-DOF 관절 궤적을 버퍼에 수집하고 HDF5 파일로 저장합니다.
"""

import os
import h5py
import numpy as np

FPS = 50.0
IMG_W, IMG_H = 640, 480


class HDF5DatasetRecorder:
    """에피소드 단위 데이터 수집 및 HDF5 영구 저장 모듈"""

    def __init__(self, output_dir, img_w=IMG_W, img_h=IMG_H, fps=FPS):
        self.output_dir = output_dir
        self.img_w = img_w
        self.img_h = img_h
        self.fps = fps
        os.makedirs(self.output_dir, exist_ok=True)

        self.reset_buffer()

    def reset_buffer(self):
        """에피소드 시작 시 메모리 버퍼 초기화"""
        self.buffer = {
            "qpos": [],
            "action": [],
            "cam_left": [],
            "cam_right": [],
        }

    def record_step(self, qpos_6, action_6, img_l, img_r):
        """
        단일 제어 스텝(50Hz) 데이터 기록
        - qpos_6: 현재 6-DOF 관절 위치 (rad)
        - action_6: 목표 제어 명령 (rad)
        - img_l, img_r: (H, W, 3) uint8 RGB 이미지
        """
        self.buffer["qpos"].append(np.array(qpos_6, dtype=np.float32))
        self.buffer["action"].append(np.array(action_6, dtype=np.float32))
        self.buffer["cam_left"].append(img_l)
        self.buffer["cam_right"].append(img_r)

    def get_step_count(self):
        return len(self.buffer["qpos"])

    def save_to_hdf5(self, ep_idx, metadata=None):
        """
        수집된 에피소드 버퍼를 HDF5 파일로 압축 저장
        """
        if self.get_step_count() == 0:
            print(f"[RECORDER WARNING] Episode {ep_idx:04d} buffer is empty. Skipping save.")
            return None

        save_path = os.path.join(self.output_dir, f"episode_{ep_idx:04d}.hdf5")

        qpos_arr = np.array(self.buffer["qpos"], dtype=np.float32)
        action_arr = np.array(self.buffer["action"], dtype=np.float32)

        # 관절 속도(qvel) 수치 미분 계산
        qvel_arr = np.zeros_like(qpos_arr)
        if len(qpos_arr) > 1:
            qvel_arr[1:] = (qpos_arr[1:] - qpos_arr[:-1]) * self.fps

        total_steps = len(qpos_arr)

        with h5py.File(save_path, "w", rdcc_nbytes=1024**2 * 4) as root:
            # 기본 메타데이터
            root.attrs["sim"] = True
            root.attrs["total_timesteps"] = total_steps
            root.attrs["fps"] = self.fps
            root.attrs["image_width"] = self.img_w
            root.attrs["image_height"] = self.img_h

            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (np.ndarray, list, tuple)):
                        root.attrs[k] = np.array(v)
                    else:
                        root.attrs[k] = v

            # 1) Action 데이터셋
            root.create_dataset("action", data=action_arr, dtype=np.float32)

            # 2) Observations 그룹
            obs = root.create_group("observations")
            obs.create_dataset("qpos", data=qpos_arr, dtype=np.float32)
            obs.create_dataset("qvel", data=qvel_arr, dtype=np.float32)

            # 3) Images 그룹 (Gzip 압축)
            img_grp = obs.create_group("images")
            for cam_key in ["cam_left", "cam_right"]:
                img_data = np.array(self.buffer[cam_key], dtype=np.uint8)
                img_grp.create_dataset(
                    cam_key,
                    data=img_data,
                    chunks=(1, self.img_h, self.img_w, 3),
                    compression="gzip",
                    compression_opts=2,
                )

        file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
        return save_path, total_steps, file_size_mb


"""양안(Dual Eyes) 카메라 HDF5 데이터 로더 및 전처리 모듈."""

import glob
import os
from typing import Tuple
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class DualEyesEpisodicDataset(Dataset):
    """양안(좌/우) 카메라 및 6-DOF 로봇 데모 에피소드 데이터셋."""

    def __init__(
        self,
        episode_paths: list,
        camera_names: list,
        norm_stats: dict,
        chunk_size: int = 100,
        image_size: Tuple[int, int] = (182, 364),
        steps_per_episode: int = 50,
        is_train: bool = True,
        weighted_sampling: bool = True,
        start_weight_mult: float = 5.0,
        grasp_weight_mult: float = 3.0,
        release_weight_mult: float = 4.0,
        fine_tune_gripper: bool = False,
        qpos_noise_std: float = 0.02,
        qpos_dropout_prob: float = 0.20,
    ):
        super().__init__()
        self.episode_paths = episode_paths
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.chunk_size = chunk_size
        self.image_size = image_size
        self.steps_per_episode = steps_per_episode
        self.is_train = is_train
        self.weighted_sampling = weighted_sampling
        self.start_weight_mult = start_weight_mult
        self.grasp_weight_mult = grasp_weight_mult
        self.release_weight_mult = release_weight_mult
        self.fine_tune_gripper = fine_tune_gripper
        self.qpos_noise_std = qpos_noise_std
        self.qpos_dropout_prob = qpos_dropout_prob

        # DINOv2 표준 전처리 파이프라인 (Bicubic Resize + ImageNet Normalization)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 각 에피소드의 길이 및 가중치 확률 분포 사전 계산 (빠른 샘플링용)
        self.episode_lengths = []
        self.sampling_probs = []
        for path in self.episode_paths:
            with h5py.File(path, "r") as f:
                ep_len = f["action"].shape[0]
                self.episode_lengths.append(ep_len)
                
                max_start = max(1, ep_len - self.chunk_size)
                if self.weighted_sampling and self.is_train:
                    # 가중치 배열 생성
                    weights = np.ones(max_start, dtype=np.float32)
                    # 1) 초기 탐색 구간 (t=0 ~ 40)
                    w_start_end = min(40, max_start)
                    weights[0:w_start_end] *= self.start_weight_mult
                    # 2) 파지 구간 (t=120 ~ 240)
                    w_grasp_start = min(120, max_start)
                    w_grasp_end = min(240, max_start)
                    if w_grasp_start < w_grasp_end:
                        weights[w_grasp_start:w_grasp_end] *= self.grasp_weight_mult
                    # 3) 상자 위 큐브 놓기/개방 구간 (t=550 ~ max_start)
                    w_release_start = min(550, max_start)
                    if w_release_start < max_start:
                        rel_mult = 15.0 if self.fine_tune_gripper else self.release_weight_mult
                        weights[w_release_start:max_start] *= rel_mult
                    
                    probs = weights / weights.sum()
                else:
                    probs = np.ones(max_start, dtype=np.float32) / max_start
                self.sampling_probs.append(probs)

    def __len__(self):
        # 1 에포크당 (에피소드 개수 * steps_per_episode) 번 샘플링
        return len(self.episode_paths) * self.steps_per_episode

    def _get_camera_image(self, root: h5py.File, cam_name: str, step: int) -> np.ndarray:
        """HDF5 내부 경로 호환 처리 (cam_left, left, cam_0 등 유연 대응)."""
        possible_keys = [
            f"observations/images/{cam_name}",
            f"observations/images/{cam_name.replace('cam_', '')}",
            f"observations/images/cam_{cam_name}",
        ]
        for key in possible_keys:
            if key in root:
                return root[key][step]
        # 키를 못 찾은 경우 images 하위의 사용 가능한 키 확인
        if "observations/images" in root:
            available = list(root["observations/images"].keys())
            raise KeyError(f"카메라 '{cam_name}'를 찾을 수 없습니다. 사용 가능한 키: {available}")
        raise KeyError(f"HDF5에 'observations/images' 경로가 없습니다.")

    def __getitem__(self, index):
        # index를 에피소드 인덱스로 변환 (0 ~ len(episode_paths) - 1)
        ep_idx = index % len(self.episode_paths)
        hdf5_path = self.episode_paths[ep_idx]
        probs = self.sampling_probs[ep_idx]
        max_start = len(probs)

        # 1. 가중치 확률 분포 기반 시작 시점 샘플링
        if self.weighted_sampling and self.is_train:
            start_ts = np.random.choice(max_start, p=probs)
        else:
            start_ts = np.random.randint(0, max_start)

        with h5py.File(hdf5_path, "r") as root:
            actions = root["action"][()]  # (T, 6)
            qpos_all = root["observations/qpos"][()]  # (T, 6)

            # 2. start_ts 시점의 관측값 (Observation)
            qpos = qpos_all[start_ts].copy()

            # 양안 카메라 이미지 로드 (RGB, uint8)
            cam_images = []
            for cam_name in self.camera_names:
                img = self._get_camera_image(root, cam_name, start_ts)  # (H, W, 3)
                # (H, W, C) -> (C, H, W) float [0, 1]
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                # Bicubic Resize (14배수) + Normalize
                img_t = self.transform(img_t)
                cam_images.append(img_t)

            # 양안 텐서 결합 (num_cam, 3, H, W)
            image_data = torch.stack(cam_images, dim=0)

            # 3. start_ts 부터 미래 chunk_size 스텝 액션 슬라이싱
            end_ts = start_ts + self.chunk_size
            action_slice = actions[start_ts:end_ts]
            actual_len = action_slice.shape[0]

            # 미래 스텝이 부족한 경우 마지막 액션으로 패딩하고 is_pad 마스킹
            padded_action = np.zeros((self.chunk_size, 6), dtype=np.float32)
            padded_action[:actual_len] = action_slice

            is_pad = np.zeros(self.chunk_size, dtype=np.float32)
            if actual_len < self.chunk_size:
                padded_action[actual_len:] = action_slice[-1]
                is_pad[actual_len:] = 1.0

        # 4. 학습 중 관절값 노이즈 주입 (가우시안 노이즈)
        # 단, 정답 action_norm에는 절대 노이즈를 주지 않음!
        if self.is_train and self.qpos_noise_std > 0.0:
            noise = np.random.normal(0.0, self.qpos_noise_std, size=qpos.shape).astype(np.float32)
            noise[-1] = 0.0  # 그리퍼(마지막 관절)는 정밀해야 하므로 노이즈 제외
            qpos = qpos + noise

        # 5. 정규화 적용 (qpos 및 action)
        qpos_norm = (qpos - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_norm = (padded_action - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # 6. 학습 중 조건부 관절값 드롭아웃 (Proprioceptive Dropout)
        # 중요: 손목이 꺾여 카메라에 빈 바닥만 보이는 사각지대 구간(t=20~90)에서는
        # 카메라에 큐브가 없으므로 관절값을 가리면 불가능한 문제가 됨 -> 사각지대 구간은 드롭아웃 제외!
        # 큐브가 시야에 확실히 들어와 있는 구간(t < 20 탐색 구간, t >= 90 상공 도착 및 파지 구간)에서만
        # 20% 확률로 qpos 입력을 0으로 마스킹하여 순수 비전(카메라)에만 의존하도록 강제
        is_blind_spot = (20 <= start_ts < 90)
        if self.is_train and self.qpos_dropout_prob > 0.0 and not is_blind_spot:
            if np.random.rand() < self.qpos_dropout_prob:
                qpos_norm = np.zeros_like(qpos_norm)

        qpos_data = torch.from_numpy(qpos_norm).float()
        action_data = torch.from_numpy(action_norm).float()
        is_pad_data = torch.from_numpy(is_pad).bool()

        return image_data, qpos_data, action_data, is_pad_data


def get_dataset_norm_stats(episode_paths: list) -> dict:
    """모든 에피소드 데이터로부터 관절 각도 및 액션의 평균/표준편차 통계 산출."""
    all_qpos = []
    all_actions = []

    for path in episode_paths:
        with h5py.File(path, "r") as root:
            q = root["observations/qpos"][()]  # (T, 6)
            a = root["action"][()]  # (T, 6)
            all_qpos.append(q)
            all_actions.append(a)

    all_qpos = np.concatenate(all_qpos, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)

    stats = {
        "qpos_mean": np.mean(all_qpos, axis=0).astype(np.float32),
        "qpos_std": np.clip(np.std(all_qpos, axis=0), 1e-4, np.inf).astype(np.float32),
        "action_mean": np.mean(all_actions, axis=0).astype(np.float32),
        "action_std": np.clip(np.std(all_actions, axis=0), 1e-4, np.inf).astype(np.float32),
    }
    return stats


def load_dual_eyes_dataloaders(
    data_dir: str,
    camera_names: list,
    batch_size: int = 4,
    chunk_size: int = 100,
    image_size: Tuple[int, int] = (182, 364),
    steps_per_episode: int = 50,
    weighted_sampling: bool = True,
    start_weight_mult: float = 5.0,
    grasp_weight_mult: float = 3.0,
    release_weight_mult: float = 4.0,
    fine_tune_gripper: bool = False,
    qpos_noise_std: float = 0.02,
    qpos_dropout_prob: float = 0.20,
    val_ratio: float = 0.0,
    num_workers: int = 4,
):
    """양안 데이터셋 로더 및 통계 반환 함수."""
    episode_files = sorted(glob.glob(os.path.join(data_dir, "*.hdf5")))
    if not episode_files:
        raise FileNotFoundError(f"'{data_dir}' 경로에서 HDF5 파일을 찾을 수 없습니다.")

    n_episodes = len(episode_files)
    if val_ratio <= 0.0:
        # 모든 에피소드를 100% 학습에 사용
        train_episodes = episode_files
        val_episodes = episode_files
        print(f"📊 [Dataset] 모든 에피소드({n_episodes}개, 100%)를 학습(Train)에 사용합니다.")
    else:
        # Train / Val 분할
        n_val = max(1, int(n_episodes * val_ratio))
        val_episodes = episode_files[:n_val]
        train_episodes = episode_files[n_val:]
        print(f"[Dataset] 총 {n_episodes}개 에피소드 중 {len(train_episodes)}개 Train, {len(val_episodes)}개 Val 분할")

    # 통계치 산출
    norm_stats = get_dataset_norm_stats(train_episodes)

    train_dataset = DualEyesEpisodicDataset(
        train_episodes,
        camera_names,
        norm_stats,
        chunk_size=chunk_size,
        image_size=image_size,
        steps_per_episode=steps_per_episode,
        is_train=True,
        weighted_sampling=weighted_sampling,
        start_weight_mult=start_weight_mult,
        grasp_weight_mult=grasp_weight_mult,
        release_weight_mult=release_weight_mult,
        fine_tune_gripper=fine_tune_gripper,
        qpos_noise_std=qpos_noise_std,
        qpos_dropout_prob=qpos_dropout_prob,
    )
    val_dataset = DualEyesEpisodicDataset(
        val_episodes,
        camera_names,
        norm_stats,
        chunk_size=chunk_size,
        image_size=image_size,
        steps_per_episode=max(1, steps_per_episode // 5),
        is_train=False,
        weighted_sampling=False,
        qpos_noise_std=0.0,
        qpos_dropout_prob=0.0,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, norm_stats

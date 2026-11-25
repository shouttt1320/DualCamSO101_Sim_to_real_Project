"""양안(Dual Eyes) 카메라 HDF5 데이터 로더 및 전처리 모듈."""

import glob
import os
from typing import Tuple
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class HorizontalMotionBlur:
    """회전(Pan) 탐색 중 웹캠에 발생하는 수평 방향 모션 블러 모사 (C, H, W float [0, 1] 텐서용)."""

    def __init__(self, kernels=(5, 7, 9), p: float = 0.30):
        self.kernels = kernels
        self.p = float(p)

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return img
        k = int(np.random.choice(self.kernels))
        c = img.shape[0]
        weight = torch.full((c, 1, 1, k), 1.0 / k, dtype=img.dtype, device=img.device)
        x = img.unsqueeze(0)
        pad_l = k // 2
        pad_r = k - 1 - pad_l
        x = F.pad(x, (pad_l, pad_r, 0, 0), mode="replicate")
        return F.conv2d(x, weight, groups=c).squeeze(0).clamp(0.0, 1.0)


def _detect_red_mask(img_rgb: torch.Tensor) -> torch.Tensor:
    """(3, H, W) [0, 1] 텐서에서 빨간 물체 픽셀 마스크 반환."""
    r, g, b = img_rgb[0], img_rgb[1], img_rgb[2]
    return (r > 0.35) & ((r - g) > 0.12) & ((r - b) > 0.12)


class RedProtectedBorderErasing:
    """기법 A: 빨간 큐브 및 하단 중앙 그리퍼를 보호하면서 화면 상단/좌우 외곽에 무작위 사각형 마스킹."""

    def __init__(self, p: float = 0.35, area_range=(0.02, 0.10)):
        self.p = float(p)
        self.area_range = area_range

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if np.random.rand() >= self.p:
            return img
        _, h, w = img.shape
        red_mask = _detect_red_mask(img)
        out = img.clone()

        for _ in range(12):
            target_area = float(np.random.uniform(*self.area_range)) * (h * w)
            aspect = float(np.random.uniform(0.5, 2.2))
            eh = int(np.clip(round(np.sqrt(target_area * aspect)), 12, int(h * 0.45)))
            ew = int(np.clip(round(np.sqrt(target_area / aspect)), 12, int(w * 0.45)))

            zone = int(np.random.choice([0, 1, 2]))  # 0: top, 1: left, 2: right
            if zone == 0:
                y0 = int(np.random.randint(0, max(1, int(h * 0.35))))
                x0 = int(np.random.randint(0, max(1, w - ew)))
            elif zone == 1:
                y0 = int(np.random.randint(0, max(1, h - eh)))
                x0 = int(np.random.randint(0, max(1, int(w * 0.25))))
            else:
                y0 = int(np.random.randint(0, max(1, h - eh)))
                x0 = int(np.random.randint(max(0, int(w * 0.75) - ew), max(1, w - ew)))

            y1 = min(h, y0 + eh)
            x1 = min(w, x0 + ew)

            # 하단 중앙 그리퍼 영역 보호
            if y1 > int(h * 0.65) and x1 > int(w * 0.30) and x0 < int(w * 0.70):
                continue
            # 빨간 물체 보호 (단 1픽셀도 덮지 않음)
            if red_mask[y0:y1, x0:x1].any():
                continue

            fill_rgb = torch.tensor(
                np.random.uniform(0.03, 0.55, size=(3, 1, 1)),
                dtype=img.dtype,
                device=img.device,
            )
            out[:, y0:y1, x0:x1] = fill_rgb
            return out
        return img


def compute_red_protected_patch_mask(
    img_rgb_resized: torch.Tensor,
    drop_prob: float,
    patch_size: int = 14,
) -> torch.Tensor:
    """
    기법 C: (3, H, W) [0, 1] 리사이즈 완료 이미지에서 DINOv2 패치 격자(H_p, W_p) 단위로
    빨간 큐브(+주변 1패치 팽창)는 절대 보호하고 순수 배경 패치만 drop_prob 확률로 마스킹.
    """
    _, h, w = img_rgb_resized.shape
    hp, wp = h // patch_size, w // patch_size
    if drop_prob <= 0.0:
        return torch.zeros((hp, wp), dtype=torch.bool)

    red_mask = _detect_red_mask(img_rgb_resized).float()
    # 패치당 빨간 픽셀 수가 3개 이상이면 큐브 포함 패치로 판정
    red_counts = F.avg_pool2d(red_mask[None, None], kernel_size=patch_size, stride=patch_size)[0, 0] * (
        patch_size * patch_size
    )
    protected = (red_counts >= 3.0).float()
    # 3x3 팽창(1패치 여유 마진)으로 큐브 경계 패치까지 보호
    protected_dilated = F.max_pool2d(protected[None, None], kernel_size=3, stride=1, padding=1)[0, 0] > 0.5

    drop = (torch.rand((hp, wp)) < float(drop_prob)) & (~protected_dilated)
    if drop.all():
        drop[0, 0] = False
    return drop


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
        qpos_dropout_prob: float = 0.05,
        patch_dropout_prob: float = 0.15,
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
        self.patch_dropout_prob = patch_dropout_prob

        # Normalize 전 증강(기법 A + C용 [0, 1] RGB 유지)과 Normalize 분리
        if self.is_train:
            self.pre_norm_transform = transforms.Compose([
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.04),
                HorizontalMotionBlur(kernels=(5, 7, 9), p=0.30),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.20),
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
                RedProtectedBorderErasing(p=0.35, area_range=(0.02, 0.10)),
            ])
        else:
            self.pre_norm_transform = transforms.Compose([
                transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            ])
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 각 에피소드의 길이, 가중치 확률 분포 및 작업 단계 경계(Phase Boundaries) 사전 계산
        self.episode_lengths = []
        self.sampling_probs = []
        self.phase_boundaries = []
        for path in self.episode_paths:
            with h5py.File(path, "r") as f:
                actions_arr = f["action"][()]
                raw_len = actions_arr.shape[0]

                # 물리적 그리퍼 이벤트 기반 4단계 경계 자동 감지 (Search / Grasp / Transport / Place)
                # 그리퍼 개방(>50°)이 전혀 없는 에피소드는 No-Cube 순수 탐색 데모로 처리
                grip_deg = np.degrees(actions_arr[:, 5])
                idx1 = np.where(grip_deg > 50.0)[0]
                is_no_cube = (len(idx1) == 0)

                if is_no_cube:
                    t_search_end = raw_len
                    t_grasp_done = raw_len
                    t_place_start = raw_len
                else:
                    t_search_end = int(idx1[0])
                    idx2 = np.where(grip_deg[t_search_end:] < 20.0)[0]
                    t_grasp_done = int(t_search_end + idx2[0]) if len(idx2) > 0 else int(raw_len * 0.4)
                    idx3 = np.where(grip_deg[t_grasp_done:] > 50.0)[0]
                    t_place_start = int(t_grasp_done + idx3[0]) if len(idx3) > 0 else int(raw_len * 0.8)

                # Active Search(균일 모드): 상자 위 투하 + 살짝 상승(q_box_hover) 이후의 불필요한 홈 복귀 궤적 논리적 제외
                if not self.weighted_sampling:
                    ep_len = raw_len if is_no_cube else min(raw_len, t_place_start + 75)
                    max_start = max(1, ep_len - 20)
                else:
                    ep_len = raw_len
                    max_start = max(1, ep_len - self.chunk_size)

                self.episode_lengths.append(ep_len)
                self.phase_boundaries.append((t_search_end, t_grasp_done, t_place_start))

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
                elif self.is_train:
                    # Active Search 모드: 초기 분기(t=0~45) 2.0x 및 탐색->파지 전환 구간 2.0x (원거리 파지 비중 보존)
                    weights = np.ones(max_start, dtype=np.float32)
                    w_start_end = min(45, max_start)
                    weights[0:w_start_end] *= min(float(self.start_weight_mult), 2.0)
                    if (not is_no_cube) and t_search_end > 0:
                        s_lo = max(0, t_search_end - 15)
                        s_hi = min(max_start, t_search_end + 15)
                        if s_lo < s_hi:
                            weights[s_lo:s_hi] *= 2.0
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

        ep_len = self.episode_lengths[ep_idx]
        with h5py.File(hdf5_path, "r") as root:
            actions = root["action"][:ep_len]  # (ep_len, 6)
            qpos_all = root["observations/qpos"][:ep_len]  # (ep_len, 6)

            # 2. start_ts 시점의 관측값 (Observation)
            qpos = qpos_all[start_ts].copy()

            # 양안 카메라 이미지 로드 (RGB, uint8) + Red-Protected 패치 마스크 + 15x20 보조 히트맵 라벨 산출
            cam_images = []
            cam_patch_masks = []
            cam_gt_heatmaps = []
            for cam_name in self.camera_names:
                img = self._get_camera_image(root, cam_name, start_ts)  # (H, W, 3)
                # (H, W, C) -> (C, H, W) float [0, 1]
                img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                img_pre = self.pre_norm_transform(img_t).clamp(0.0, 1.0)

                # 보조 헤드용 15x20 패치 히트맵 라벨 (패치당 빨간 픽셀 >= 3이면 1.0)
                red_mask_f = _detect_red_mask(img_pre).float()
                red_counts = F.avg_pool2d(red_mask_f[None, None], kernel_size=14, stride=14)[0, 0] * (14 * 14)
                gt_hm = (red_counts >= 3.0).float()
                cam_gt_heatmaps.append(gt_hm)

                if self.is_train and self.patch_dropout_prob > 0.0:
                    p_mask = compute_red_protected_patch_mask(img_pre, self.patch_dropout_prob, patch_size=14)
                else:
                    p_mask = torch.zeros((self.image_size[0] // 14, self.image_size[1] // 14), dtype=torch.bool)
                cam_patch_masks.append(p_mask)
                cam_images.append(self.normalize(img_pre))

            # 양안 텐서 결합 (num_cam, 3, H, W) 및 (num_cam, H_p, W_p)
            image_data = torch.stack(cam_images, dim=0)
            patch_mask_data = torch.stack(cam_patch_masks, dim=0)
            gt_heatmap_data = torch.stack(cam_gt_heatmaps, dim=0)
            gt_vis_data = gt_heatmap_data.max().float()

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

        # 6. 물리적 경계 기반 현재 샘플의 작업 단계 판별 (0: Search, 1: Grasp, 2: Transport, 3: Place)
        t_search_end, t_grasp_done, t_place_start = self.phase_boundaries[ep_idx]
        if start_ts < t_search_end:
            phase_int = 0
        elif start_ts < t_grasp_done:
            phase_int = 1
        elif start_ts < t_place_start:
            phase_int = 2
        else:
            phase_int = 3

        # 7. 학습 중 단계별 조건부 관절값 드롭아웃 (Selective Proprioceptive Dropout)
        # - Active Search(균일 모드): 큐브가 시야에 들어와 있는 'Grasp(phase_int == 1)' 구간에만 선택적 드롭아웃 적용.
        #   (빈 바닥만 보는 Search 구간 및 운반/안착 구간은 qpos를 보존하여 방향 상실 및 평균 행동 붕괴 방지)
        # - Legacy 가중 모드: 기존 사각지대(t=20~90) 제외 규칙 유지.
        if self.is_train and self.qpos_dropout_prob > 0.0:
            if not self.weighted_sampling:
                allow_dropout = (phase_int == 1)
            else:
                allow_dropout = not (20 <= start_ts < 90)

            if allow_dropout and np.random.rand() < self.qpos_dropout_prob:
                qpos_norm = np.zeros_like(qpos_norm)

        qpos_data = torch.from_numpy(qpos_norm).float()
        action_data = torch.from_numpy(action_norm).float()
        is_pad_data = torch.from_numpy(is_pad).bool()
        phase_id = torch.tensor(phase_int, dtype=torch.long)

        return image_data, qpos_data, action_data, is_pad_data, phase_id, patch_mask_data, gt_heatmap_data, gt_vis_data


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
    qpos_dropout_prob: float = 0.05,
    patch_dropout_prob: float = 0.15,
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
        patch_dropout_prob=patch_dropout_prob,
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
        patch_dropout_prob=0.0,
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

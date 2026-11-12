"""양안(Dual Eyes) 카메라 및 DINOv2 기반 ACT 모델 및 학습 설정 (Config)."""

import os

def _find_project_root():
    cur = os.path.abspath(__file__)
    for _ in range(5):
        cur = os.path.dirname(cur)
        if os.path.exists(os.path.join(cur, "configs", "calibration.json")) or os.path.exists(os.path.join(cur, "hardware")):
            return cur
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = _find_project_root()
DATA_DIR = os.path.join(BASE_DIR, "data", "so101_dual_eyes")
CKPT_DIR = os.path.join(BASE_DIR, "ckpt", "so101_dual_dinov2_act")

# 1. 로봇 및 센서 스펙
TASK_NAME = "so101_dual_eyes_pick_place"
STATE_DIM = 6  # 6-DOF (1: Base, 2: Shoulder, 3: Elbow, 4: Wrist Pitch, 5: Wrist Roll, 6: Gripper)
CAMERA_NAMES = ["cam_left", "cam_right"]  # 좌안 / 우안 스테레오 듀얼 카메라
CAMERA_PORTS = {"cam_left": 4, "cam_right": 6}  # 기본 하드웨어 포트 (video4: 좌안, video6: 우안)
DISABLE_AUTOFOCUS = True  # 학습 데이터 일관성을 위한 오토포커스 끄기 및 고정
FIXED_FOCUS_VALUE = 25    # 작업대 픽앤플레이스 거리에 최적화된 고정 초점 값 (0~255)
CONTROL_FPS = 30  # 30Hz

# 2. 이미지 전처리 및 해상도 (DINOv2 패치 크기 14의 배수)
# 640x320(2:1 비율)에 최적화된 14배수: 182(=14x13) x 364(=14x26), 2:1 비율 완전 보존 및 12GB VRAM 안정 구동
IMAGE_SIZE = (182, 364)  # (Height, Width)

# 3. DINOv2 백본 및 ACT 모델 하이퍼파라미터
BACKBONE = "dinov2_vits14"     # "dinov2_vits14", "dinov2_vitb14"
TRAIN_BACKBONE = False         # DINOv2 사전학습 가중치 고정(Freeze) 여부
CHUNK_SIZE = 100               # 미래 액션 예측 청크 크기 (50Hz 기준 2.0초 분량)
HIDDEN_DIM = 512               # 트랜스포머 임베딩 차원
DIM_FEEDFORWARD = 3200         # 피드포워드 레이어 차원
ENC_LAYERS = 6                 # 인코더 레이어 수
DEC_LAYERS = 7                 # 디코더 레이어 수
NHEADS = 8                     # 어텐션 헤드 수
LATENT_DIM = 32                # CVAE 잠재 변수 z 차원
KL_WEIGHT = 0.1                # CVAE KL 발산 손실 가중치

# 4. 강인한 비전 정책 학습을 위한 정규화 및 증강 설정
STEPS_PER_EPISODE_PER_EPOCH = 50  # 에포크당 에피소드별 샘플링 수 (29 x 50 = 1,450 샘플/에포크)
QPOS_NOISE_STD = 0.02             # 관절 입력 가우시안 노이즈 표준편차 (rad, 약 1.15도)
QPOS_DROPOUT_PROB = 0.20          # 관절 입력 드롭아웃 확률 (20% 확률로 qpos를 0으로 마스킹하여 순수 비전 의존)
WEIGHTED_SAMPLING = True          # 초기 관측(t=0~40) 및 파지(t=120~240) 가중 샘플링 활성화
START_WEIGHT_MULTIPLIER = 5.0     # 초기 관측 구간(t=0~40) 가중치 배수
GRASP_WEIGHT_MULTIPLIER = 3.0     # 파지 구간(t=120~240) 가중치 배수

# 5. 학습 최적화 설정
LR = 1e-5                      # 트랜스포머 및 프로젝션 학습률
LR_BACKBONE = 1e-5             # 백본 학습률 (TRAIN_BACKBONE=True 시 적용)
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 4                 # Isaac Sim 동시 실행 시 VRAM 안정성 확보를 위한 배치 4
NUM_EPOCHS = 1000              # 1,450 samples/epoch 기준 1000 epochs = 1,450,000 steps
SAVE_EVERY = 100
SEED = 0

# 6. 실 로봇 추론(Deployment) 파라미터
TEMPORAL_AGG_M = 0.02          # Temporal Ensembling 지수 가중치 m (exp(-m * i))

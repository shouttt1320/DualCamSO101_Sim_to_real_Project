#!/usr/bin/env bash
# =============================================================================
# [Step 04] SO-101 DINOv2-ACT 정책 학습 및 파인튜닝 실행 스크립트
#
# [실행 방법]
#   1) 표준 지도학습 (처음부터 300 에포크 학습):
#      bash scripts/04_train_policy.sh --num_epochs 300 --batch_size 4 --save_every 20
#
#   2) 백그라운드 무중단 학습 (nohup):
#      nohup bash scripts/04_train_policy.sh --num_epochs 500 --batch_size 4 --save_every 50 > train.log 2>&1 &
#      tail -f train.log
#
#   3) 기존 가중치 이어받아 파인튜닝 (상자 위 그리퍼 개방 집중 학습):
#      bash scripts/04_train_policy.sh --resume --fine_tune_gripper --num_epochs 150 --lr 5e-5 --save_name policy_finetuned.pth
#
# [CLI 주요 옵션]
#   --num_epochs          : 총 학습 에포크 수 (기본값: 300)
#   --batch_size          : 미니배치 크기 (기본값: 4)
#   --lr                  : 학습률 Learning Rate (기본값: 1e-4)
#   --save_every          : 체크포인트 저장 주기 [epochs] (기본값: 20)
#   --save_name           : 저장할 최고 성능 모델 가중치 파일명 (기본값: policy_best.pth)
#   --resume              : 기존 policy_best.pth 가중치를 불러와 이어서 학습
#   --fine_tune_gripper   : 상자 도달 구간에서 그리퍼 손실함수 가중치 부여 모드
#   --gripper_loss_weight : 그리퍼 관절 Loss 가중치 배수 (기본값: 1.0, 파인튜닝 시 3.0 권장)
#   --chunk_size          : Action Chunking 예측 길이 (기본값: 100)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="${WORKSPACE_DIR}/Robot_Arm_Project"

cd "${PROJECT_ROOT}"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================================="
echo ">>> [SO-101 DINOv2-ACT TRAINING] Starting training pipeline..."
echo "  • Working Dir: ${PROJECT_ROOT}"
echo "  • Dataset:     ${PROJECT_ROOT}/data/so101_dual_eyes"
echo "  • Checkpoint:  ${PROJECT_ROOT}/ckpt/so101_dual_dinov2_act"
echo "============================================================================="

~/isaacsim/python.sh models/so101_dual_dinov2_act/train.py "$@"


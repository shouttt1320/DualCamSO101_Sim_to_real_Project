#!/usr/bin/env bash
# =============================================================================
# [Step 03] SO-101 능동 탐색 (Active Search) DINOv2-ACT 정책 학습 스크립트
# =============================================================================
# [특징]
#   - 능동 탐색 가변 에피소드 데이터셋에 최적화된 완전 균일 샘플링(--no_weighted_sampling) 기본 적용
#   - 환경 프리셋별 데이터셋 및 체크포인트 디렉토리 자동 분리 관리
#   - 베이스라인(01~06) 설정과 완전 격리되어 안전하게 독립 실행
#
# [실행 방법]
#   1) 표준 학습 (기본 clean 환경, 300 에포크, 배치 8):
#      bash scripts/active_search/03_train_active_search.sh --num_epochs 300 --batch_size 8 --save_every 50
#
#   2) 백그라운드 무중단 학습 (nohup):
#      nohup bash scripts/active_search/03_train_active_search.sh --resume --num_epochs 300 --batch_size 8 --save_every 50 > train_resume.log 2>&1 &
#      tail -f train_resume.log
#
#   3) 특정 시뮬레이션 환경 데이터셋 학습 (예: dark, warm_light):
#      bash scripts/active_search/03_train_active_search.sh --env default_clean --num_epochs 300 --batch_size 8
#
#   4) 기존 가중치 이어받아 추가 학습 (Resume):
#      bash scripts/active_search/03_train_active_search.sh --resume --num_epochs 200 --batch_size 8 --lr 5e-5
#
# [주요 파라미터 옵션 가이드]
#   • --batch_size <N>   : 한 번의 역전파에 사용할 샘플 수 (기본 8 권장).
#                          - 8 : RTX 5070 (12GB) 기준 VRAM ~6.1GB 점유로 최적 속도 및 안정성 보장.
#                          - 4 : VRAM 여유가 적거나 다른 그래픽 작업 병행 시 안전하게 사용.
#   • --num_epochs <N>   : 총 학습 에포크 수 (능동 탐색은 150~300 에포크 권장).
#   • --save_every <N>   : N 에포크마다 중간 체크포인트(policy_epoch_N.pth) 및 손실 곡선 그래프 저장.
#   • --resume           : 이전 최고 가중치(policy_best.pth)를 로드하여 이어서 학습 (파인튜닝).
#   • --lr <float>       : 학습률 (신규 학습 시 1e-4 기본, Resume 시 5e-5 이하 권장).
#   • --env <name>       : 사용할 데이터셋 환경 폴더명 (기본: default_clean).
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT_ROOT="${WORKSPACE_DIR}/Robot_Arm_Project"

ENV_NAME="default_clean"
PASSTHROUGH_ARGS=()

# CLI 인자 중 --env 추출 및 나머지 인자 전달
while [[ $# -gt 0 ]]; do
  case $1 in
    --env)
      ENV_NAME="$2"
      shift 2
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

DATA_DIR="${PROJECT_ROOT}/data/so101_active_search/${ENV_NAME}"
CKPT_DIR="${PROJECT_ROOT}/ckpt/so101_active_search/${ENV_NAME}"

# 데이터셋 디렉토리 및 파일 존재 여부 검증
if [ ! -d "${DATA_DIR}" ]; then
  echo "❌ [ERROR] Dataset directory not found: ${DATA_DIR}"
  echo "먼저 데모 수집 스크립트를 실행해주세요: bash scripts/active_search/02_collect_search_demos.sh --env ${ENV_NAME}"
  exit 1
fi

HDF5_COUNT=$(find "${DATA_DIR}" -maxdepth 1 -name "*.hdf5" | wc -l)
if [ "${HDF5_COUNT}" -eq 0 ]; then
  echo "❌ [ERROR] No .hdf5 files found in ${DATA_DIR}"
  echo "수집된 에피소드가 없습니다. 데모 수집을 먼저 실행해주세요."
  exit 1
fi

mkdir -p "${CKPT_DIR}"

cd "${PROJECT_ROOT}"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================================="
echo ">>> [SO-101 ACTIVE SEARCH] Starting DINOv2-ACT Training..."
echo "  • Environment:     ${ENV_NAME}"
echo "  • Dataset Dir:     ${DATA_DIR} (${HDF5_COUNT} episodes found)"
echo "  • Checkpoint Dir:  ${CKPT_DIR}"
echo "  • Sampling Mode:   Uniform Random (--no_weighted_sampling)"
echo "============================================================================="

~/isaacsim/python.sh models/so101_dual_dinov2_act/train.py \
  --data_dir "${DATA_DIR}" \
  --ckpt_dir "${CKPT_DIR}" \
  --no_weighted_sampling \
  "${PASSTHROUGH_ARGS[@]}"


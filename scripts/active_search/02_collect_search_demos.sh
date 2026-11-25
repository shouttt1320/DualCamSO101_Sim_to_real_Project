#!/usr/bin/env bash
# =============================================================================
# [Active Search Step 02] SO-101 능동 시각 탐색 멀티 환경 데이터 수집기 실행 스크립트
#
# [실행 방법]
#   1) 기본 실행 (기본 환경 50개 에피소드 고속 수집, 무화면 권장):
#      bash scripts/active_search/02_collect_search_demos.sh --headless --num_episodes 50
#
#   2) 샘플 5개 테스트 수집:
#      bash scripts/active_search/02_collect_search_demos.sh --headless --num_episodes 5
#
#   3) 환경 변경 지정 실행 (추후 신규 환경 추가 시):
#      bash scripts/active_search/02_collect_search_demos.sh --env default_clean --num_episodes 50
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT_ROOT="${WORKSPACE_DIR}/Robot_Arm_Project"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${SCRIPT_DIR}:${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================================="
echo ">>> [SO-101 ACTIVE SEARCH] Starting Modular Dataset Collection..."
echo "  • Main Script:     ${SCRIPT_DIR}/02_collect_search_demos.py"
echo "  • Modules:         env_manager.py, search_and_pick_policy.py, dataset_recorder.py"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/02_collect_search_demos.py" "$@"


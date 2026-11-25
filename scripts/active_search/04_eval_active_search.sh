#!/usr/bin/env bash
# =============================================================================
# [Step 04] SO-101 능동 탐색 (Active Search) DINOv2-ACT 단일 추론 셸 스크립트
# =============================================================================
# [실행 방법]
#   1) GUI 창으로 사각지대(-45도) 자율 추론 관찰 (기본값):
#      bash scripts/active_search/04_eval_active_search.sh
#
#   2) 특정 각도 및 거리 테스트 (예: +40도 사각지대, 0.31m 거리):
#      bash scripts/active_search/04_eval_active_search.sh --cube_r 0.31 --cube_theta 40.0
#
#   3) 정면 큐브 테스트 (0도, 0.30m):
#      bash scripts/active_search/04_eval_active_search.sh --cube_r 0.30 --cube_theta 0.0
#
#   4) 화면 없이 백그라운드 고속 실행:
#      bash scripts/active_search/04_eval_active_search.sh --headless --cube_theta -50.0
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT_ROOT="${WORKSPACE_DIR}/Robot_Arm_Project"

cd "${PROJECT_ROOT}"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${SCRIPT_DIR}:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================================="
echo ">>> [SO-101 ACTIVE SEARCH EVALUATION] Starting closed-loop simulation..."
echo "  • Script:     ${SCRIPT_DIR}/04_eval_active_search.py"
echo "  • Model:      ${PROJECT_ROOT}/ckpt/so101_active_search/default_clean/policy_best.pth"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/04_eval_active_search.py" "$@"


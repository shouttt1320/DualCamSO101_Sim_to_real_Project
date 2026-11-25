#!/usr/bin/env bash
# =============================================================================
# [Active Search Step 01] SO-101 능동 시각 탐색 기반 전자율 픽앤플레이스 실행기
#
# [실행 방법]
#   1) 기본 실행 (사각지대 theta=+45도, r=0.32m 둘러보고 집기):
#      bash scripts/active_search/01_test_scan_and_pick.sh
#
#   2) 반대쪽 사각지대 (theta=-45도, r=0.30m 둘러보고 집기):
#      bash scripts/active_search/01_test_scan_and_pick.sh --theta -45.0 --r 0.30
#
#   3) 무화면(Headless) 고속 실행:
#      bash scripts/active_search/01_test_scan_and_pick.sh --headless --theta 40.0
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PROJECT_ROOT="${WORKSPACE_DIR}/Robot_Arm_Project"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================================="
echo ">>> [SO-101 SCAN-TO-PICK] Starting Autonomous Active Search & Pick..."
echo "  • Script: ${SCRIPT_DIR}/01_test_scan_and_pick.py"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/01_test_scan_and_pick.py" "$@"


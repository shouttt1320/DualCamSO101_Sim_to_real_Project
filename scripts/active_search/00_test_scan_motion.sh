#!/usr/bin/env bash
# =============================================================================
# [Active Search Step 00] SO-101 능동 시각 탐색 (Active Visual Search) 테스트
#
# [실행 방법]
#   1) 기본 GUI 모드로 사각지대 탐색 테스트 (기본: theta=+45도, r=0.32m):
#      bash scripts/active_search/00_test_scan_motion.sh
#
#   2) 특정 사각지대 각도 및 반경 지정 테스트:
#      bash scripts/active_search/00_test_scan_motion.sh --theta -45.0 --r 0.30
#
#   3) 무화면(Headless) 고속 실행:
#      bash scripts/active_search/00_test_scan_motion.sh --headless --theta 50.0
#
# [CLI 주요 옵션]
#   --headless : 창 없이 백그라운드 고속 실행 (기본값: False)
#   --usd      : 시뮬레이션 USD 스테이지 경로 (기본값: dual_cam_so101_redcube.usd)
#   --theta    : 테스트 큐브 사각지대 각도 [deg] (기본값: +45.0, 범위: -65.0 ~ +65.0)
#   --r        : 테스트 큐브 사각지대 반경 [m] (기본값: 0.320, 범위: 0.220 ~ 0.380)
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
echo ">>> [SO-101 ACTIVE SEARCH] Launching Active Visual Search test..."
echo "  • Script: ${SCRIPT_DIR}/00_test_scan_motion.py"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/00_test_scan_motion.py" "$@"


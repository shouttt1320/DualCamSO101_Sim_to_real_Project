#!/usr/bin/env bash
# =============================================================================
# [Step 05] SO-101 능동 탐색 (Active Search) 광각 사각지대 벤치마크 평가 셸 스크립트
# =============================================================================
# [실행 방법]
#   1) 기본 21개 지점(거리 3종 x 각도 7종, -45도~+45도) 헤드리스 자동 벤치마크:
#      bash scripts/active_search/05_benchmark_active_search.sh
#
#   2) GUI 화면을 보면서 천천히 평가:
#      bash scripts/active_search/05_benchmark_active_search.sh --gui
#
#   3) 극단적 사각지대(-55도 ~ +55도) 정밀 테스트:
#      bash scripts/active_search/05_benchmark_active_search.sh --theta_list -55.0 -40.0 -20.0 0.0 20.0 40.0 55.0
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
echo ">>> [SO-101 ACTIVE SEARCH BENCHMARK] Launching automated grid evaluation..."
echo "  • Script:      ${SCRIPT_DIR}/05_benchmark_active_search.py"
echo "  • Checkpoint:  ${PROJECT_ROOT}/ckpt/so101_active_search/default_clean/policy_best.pth"
echo "  • Output CSV:  ${PROJECT_ROOT}/ckpt/so101_active_search/default_clean/benchmark_report.csv"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/05_benchmark_active_search.py" "$@"


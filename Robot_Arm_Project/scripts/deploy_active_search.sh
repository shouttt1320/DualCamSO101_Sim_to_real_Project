#!/usr/bin/env bash
# ==============================================================================
# SO-101 Active Search (Sim-to-Real domain_rand) 실물 로봇 추론 원클릭 실행기
# ==============================================================================
# 사용법:
#   bash Robot_Arm_Project/scripts/deploy_active_search.sh
#   bash Robot_Arm_Project/scripts/deploy_active_search.sh --max-steps 800
#   bash Robot_Arm_Project/scripts/deploy_active_search.sh --no-preview
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${PROJECT_ROOT}"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${PYTHONPATH}:/usr/lib/python3/dist-packages"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"

echo "=============================================================================="
echo " 🤖 [SO-101 Active Search] 실물 로봇 추론 시작"
echo "  • Checkpoint : ${PROJECT_ROOT}/ckpt/so101_active_search/domain_rand/policy_best.pth"
echo "  • Cameras    : Left=/dev/video2, Right=/dev/video0 (Focus=25)"
echo "  • Control    : 50 Hz, Max 1150 steps (~23s)"
echo "  • Stop       : 프리뷰 창 [q] 또는 터미널 [Ctrl+C]"
echo "=============================================================================="

~/isaacsim/python.sh -m models.so101_dual_dinov2_act.deploy_real_robot \
  --ckpt-dir "${PROJECT_ROOT}/ckpt/so101_active_search/domain_rand" \
  --ckpt policy_best.pth \
  --left-cam 2 \
  --right-cam 0 \
  --focus 25 \
  --fps 50 \
  --max-steps 1150 \
  --start-obs \
  "$@"


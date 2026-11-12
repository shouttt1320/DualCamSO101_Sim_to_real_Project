#!/usr/bin/env bash
# =============================================================================
# [Step 05] SO-101 DINOv2-ACT Isaac Sim 폐루프 시뮬레이션 단일 추론 스크립트
#
# [실행 방법]
#   1) 기본 실행 (GUI 모드):
#      bash scripts/05_eval_sim_policy.sh
#
#   2) 특정 위치 큐브 테스트 (X, Y 좌표 지정):
#      bash scripts/05_eval_sim_policy.sh --cube_x 0.320 --cube_y -0.050
#
#   3) 특정 체크포인트 지정 및 Headless 무화면 고속 실행:
#      bash scripts/05_eval_sim_policy.sh --headless --ckpt policy_finetuned.pth
#
# [CLI 주요 옵션]
#   --headless   : 창 없이 백그라운드 고속 실행 (기본값: False, GUI 활성화)
#   --usd        : 시뮬레이션 USD 스테이지 경로 (기본값: dual_cam_so101_redcube.usd)
#   --cube_x     : 테스트 큐브 초기 X 좌표 [m] (기본값: 0.310, 범위: 0.28 ~ 0.34)
#   --cube_y     : 테스트 큐브 초기 Y 좌표 [m] (기본값: 0.000, 범위: -0.10 ~ +0.10)
#   --ckpt       : 사용할 가중치 파일명 또는 절대경로 (기본값: policy_best.pth)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="${WORKSPACE_DIR}/Robot_Arm_Project"

NV_DIR="/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle/nvidia"
ML_DIR="/home/jw/isaacsim/extsDeprecated/omni.isaac.ml_archive/pip_prebundle"

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/models:${ML_DIR}:/home/jw/isaacsim/exts/isaacsim.pip.nv/pip_prebundle:${PYTHONPATH}"
export LD_LIBRARY_PATH="${NV_DIR}/cuda_runtime/lib:${NV_DIR}/cublas/lib:${NV_DIR}/cudnn/lib:${NV_DIR}/cufft/lib:${NV_DIR}/curand/lib:${NV_DIR}/cusolver/lib:${NV_DIR}/cusparse/lib:${NV_DIR}/nccl/lib:${NV_DIR}/nvtx/lib:${LD_LIBRARY_PATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "============================================================================="
echo ">>> [SO-101 EVAL SIM POLICY] Launching closed-loop inference in Isaac Sim..."
echo "  • Script:     ${SCRIPT_DIR}/05_eval_sim_policy.py"
echo "  • Checkpoint: ${PROJECT_ROOT}/ckpt/so101_dual_dinov2_act/policy_best.pth"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/05_eval_sim_policy.py" "$@"


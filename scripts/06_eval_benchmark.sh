#!/usr/bin/env bash
# =============================================================================
# [Step 06] SO-101 DINOv2-ACT 정책 자동 벤치마크 평가 스크립트 (Grid/Range Test)
#
# [실행 방법]
#   1) 기본 4x5 격자 범위 (총 20개 지점) 벤치마크 평가:
#      bash scripts/06_eval_benchmark.sh
#
#   2) 커스텀 범위 지정 (반경 0.28~0.34m, 각도 -20~+20도):
#      bash scripts/06_eval_benchmark.sh --r_min 0.28 --r_max 0.34 --r_step 0.02 --theta_min -20 --theta_max 20 --theta_step 10
#
#   3) 무화면(Headless) 고속 실행 및 체크포인트 지정:
#      bash scripts/06_eval_benchmark.sh --headless --ckpt policy_finetuned.pth
#
# [CLI 주요 옵션]
#   --headless   : 창 없이 백그라운드 고속 평가 (기본값: False)
#   --usd        : 시뮬레이션 USD 스테이지 경로 (기본값: dual_cam_so101_redcube.usd)
#   --r_min      : 최소 테스트 반경 거리 [m] (기본값: 0.28)
#   --r_max      : 최대 테스트 반경 거리 [m] (기본값: 0.34)
#   --r_step     : 반경 거리 간격 [m] (기본값: 0.02)
#   --theta_min  : 최소 테스트 각도 [deg] (기본값: -18.0)
#   --theta_max  : 최대 테스트 각도 [deg] (기본값: +18.0)
#   --theta_step : 각도 간격 [deg] (기본값: 9.0)
#   --ckpt       : 평가할 모델 가중치 파일명 또는 경로 (기본값: policy_best.pth)
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
echo ">>> [SO-101 BENCHMARK] Launching automated policy benchmark..."
echo "  • Script:     ${SCRIPT_DIR}/06_eval_benchmark.py"
echo "  • Output CSV: ${PROJECT_ROOT}/ckpt/so101_dual_dinov2_act/benchmark_report.csv"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/06_eval_benchmark.py" "$@"


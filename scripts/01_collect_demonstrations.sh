#!/usr/bin/env bash
# =============================================================================
# [Step 01] SO-101 양안 데모 데이터셋 자동 수집 셸 스크립트
# =============================================================================
# [실행 방법]
#   1) 기본 무작위 50개 헤드리스 수집 (권장):
#      bash scripts/01_collect_demonstrations.sh --headless --num_episodes 50 --mode random
#
#   2) 조밀 격자(63개) 수집:
#      bash scripts/01_collect_demonstrations.sh --headless --mode dense
#
#   3) GUI 시뮬레이션 창을 보면서 수집:
#      bash scripts/01_collect_demonstrations.sh --num_episodes 20 --mode random
#
# [주요 옵션]
#   --headless             : 화면 렌더링 없이 초고속 백그라운드 수집 (약 10~15분)
#   --num_episodes <int>   : 수집할 목표 에피소드 수 (기본: 60)
#   --mode <random|dense>  : 큐브 배치 방식 (random: 연속 무작위, dense: 7x9 조밀 격자)
#   --random_yaw           : 큐브 Z축 회전각(+-12도) 미세 무작위 회전 적용 (기본: True)
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(dirname "$SCRIPT_DIR")"

echo "============================================================================="
echo ">>> [SO-101 DATA COLLECTION] Starting automated demonstration collection..."
echo "  • Output dir: ${WORKSPACE_DIR}/Robot_Arm_Project/data/so101_dual_eyes"
echo "============================================================================="

~/isaacsim/python.sh "${SCRIPT_DIR}/01_collect_demonstrations.py" "$@"


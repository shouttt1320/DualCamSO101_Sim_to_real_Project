#!/usr/bin/env python3
"""
[Module 0] Environment Presets & Configuration Registry
========================================================
순수 파이썬 딕셔너리로 환경 프리셋(USD 경로, 주요 Prim 경로 등)을 정의합니다.
Omniverse 의존성이 없어 SimulationApp 시작 전 ArgumentParser 단계에서도 안전하게 import 가능합니다.
"""

import os

WORKSPACE_DIR = "/home/jw/Documents/PersonalProjects/RobotArm_SimtoReal"

ENV_REGISTRY = {
    "default_clean": {
        "usd_path": os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"),
        "robot_prim": "/World/so101_dual_cam/Geometry/base_link",
        "cube_prim": "/World/RedCube",
        "box_prim": "/World/TargetBox",
        "left_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera",
        "right_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera",
        "randomize": False,
        "description": "기본 단색 테이블 및 표준 조명 환경",
    },
    "domain_rand": {
        "usd_path": os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"),
        "robot_prim": "/World/so101_dual_cam/Geometry/base_link",
        "cube_prim": "/World/RedCube",
        "box_prim": "/World/TargetBox",
        "left_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera",
        "right_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera",
        "randomize": True,
        "description": "Sim-to-Real 도메인 랜덤화 (바닥·큐브·조명·방해물·카메라 지터)",
    },
    "wood_floor_test": {
        "usd_path": os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"),
        "robot_prim": "/World/so101_dual_cam/Geometry/base_link",
        "cube_prim": "/World/RedCube",
        "box_prim": "/World/TargetBox",
        "left_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera",
        "right_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera",
        "randomize": False,
        "ckpt_env": "domain_rand",
        "table_texture": os.path.join(WORKSPACE_DIR, "assets", "textures", "floor_texture2.png"),
        "arm_color_rgb": (0.03, 0.14, 0.62),
        "description": "신규 우드 바닥(floor_texture2) + 진한 파란색 로봇팔 OOD 테스트 환경",
    },
    "real_wall_twin": {
        "usd_path": os.path.join(WORKSPACE_DIR, "dual_cam_so101_redcube.usd"),
        "robot_prim": "/World/so101_dual_cam/Geometry/base_link",
        "cube_prim": "/World/RedCube",
        "box_prim": "/World/TargetBox",
        "left_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_left_optical_frame/left_camera",
        "right_cam": "/World/so101_dual_cam/Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/camera_right_optical_frame/right_camera",
        "randomize": True,
        "room_mode": "real_wall_twin",
        "box_pos": (0.130, 0.205, 0.0),
        "table_safe_x": (0.160, 0.415),
        "table_safe_y": (-0.215, 0.215),
        "box_exclusion_dist": 0.145,
        "description": "실물 벽면+접이식 아이보리 테이블(54.5x54.5x22.5cm)+좌측 검은상자(blackbox)+천큐브(realred1~6) Digital Twin",
    },
    # 추후 신규 환경 추가 시 여기에 프리셋 추가:
    # "wood_table": {
    #     "usd_path": os.path.join(WORKSPACE_DIR, "envs/so101_wood_table.usd"),
    #     ...
    # },
}


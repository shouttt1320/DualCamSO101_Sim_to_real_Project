#!/usr/bin/env python3
"""
[Active Search] 두 체크포인트(예: v1 vs v2) 실시간 비교 검증 도구
=============================================================================
모드 1) --mode real_cam : 실물 양안 카메라(/dev/video2, /dev/video0)를 켜고
                          로봇을 움직이지 않은 채 v1/v2의 초기 예측(SEARCH vs GRASP)을 실시간 비교
모드 2) --mode sim      : Isaac Sim 환경(기본/광각 FOV/어두운 테이블 등)에서
                          v1/v2의 초기 예측 및 동작 차이를 직접 확인
"""

import argparse
import os
import pickle
import sys
import time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))
PROJECT_ROOT = os.path.join(WORKSPACE_DIR, "Robot_Arm_Project")

for p in [SCRIPT_DIR, PROJECT_ROOT, os.path.join(PROJECT_ROOT, "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from so101_dual_dinov2_act.config import (
    BACKBONE, CAMERA_NAMES, CHUNK_SIZE, DEC_LAYERS, DIM_FEEDFORWARD,
    ENC_LAYERS, HIDDEN_DIM, IMAGE_SIZE, LATENT_DIM, NHEADS, STATE_DIM,
)
from so101_dual_dinov2_act.models import build_act_model

OBS_POSE_RAD = np.array([0.0, np.radians(-60.0), np.radians(40.0), np.radians(85.0), 0.0, 0.0], dtype=np.float32)


def load_policy(ckpt_dir: str, device: torch.device):
    ckpt_path = os.path.join(ckpt_dir, "policy_best.pth")
    stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        st = pickle.load(f)
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    ql = sd["query_embed.weight"].shape[0] if "query_embed.weight" in sd else CHUNK_SIZE
    ckpt_bb = ck.get("backbone_name", "dinov2_vits14") if isinstance(ck, dict) else "dinov2_vits14"
    model = build_act_model(
        state_dim=STATE_DIM, num_queries=ql, camera_names=CAMERA_NAMES,
        backbone_name=ckpt_bb, hidden_dim=HIDDEN_DIM, dim_feedforward=DIM_FEEDFORWARD,
        enc_layers=ENC_LAYERS, dec_layers=DEC_LAYERS, nheads=NHEADS, latent_dim=LATENT_DIM,
        train_backbone=False,
    )
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return {
        "model": model,
        "qpos_mean": torch.tensor(st["qpos_mean"], dtype=torch.float32),
        "qpos_std": torch.tensor(st["qpos_std"], dtype=torch.float32),
        "act_mean": torch.tensor(st["action_mean"], dtype=torch.float32),
        "act_std": torch.tensor(st["action_std"], dtype=torch.float32),
    }


def predict_summary(pol: dict, qpos_rad: np.ndarray, rgb_l: np.ndarray, rgb_r: np.ndarray, tf, device):
    qn = ((torch.tensor(qpos_rad, dtype=torch.float32) - pol["qpos_mean"]) / pol["qpos_std"]).unsqueeze(0).to(device)
    def prep(arr):
        return tf(torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0)
    im = torch.stack([prep(rgb_l), prep(rgb_r)], dim=0).unsqueeze(0).to(device)
    with torch.inference_mode():
        ah, _, _ = pol["model"](qpos=qn, image=im)
    deg = np.degrees((ah.squeeze(0).cpu() * pol["act_std"] + pol["act_mean"]).numpy())
    pan20 = float(deg[20, 0])
    max_pan = float(np.max(np.abs(deg[:40, 0])))
    lift30 = float(deg[30, 1])
    elb30 = float(deg[30, 2])
    max_grip = float(np.max(deg[:50, 5]))
    mode = "GRASP" if max_grip > 25.0 else ("SEARCH" if (max_pan > 12.0 or lift30 < -55.0) else "MIXED")
    return {"mode": mode, "pan20": pan20, "max_pan": max_pan, "lift30": lift30, "elb30": elb30, "max_grip": max_grip}


def run_real_cam_mode(args, tf, device):
    from hardware.camera_interface import MultiCameraCapture

    pol_a = load_policy(args.ckpt_a, device)
    pol_b = load_policy(args.ckpt_b, device)

    cap = MultiCameraCapture(
        camera_dict={"cam_left": args.left_cam, "cam_right": args.right_cam},
        width=640, height=480, fps=30,
        disable_autofocus=True,
        focus_dict={"cam_left": args.focus, "cam_right": args.focus},
    )
    cap.start()
    print("\n=== [실물 카메라 v1 vs v2 실시간 예측 비교] (모터 구동 없음, [Q/ESC] 종료) ===")
    try:
        while True:
            frames = cap.read_all()
            bgr_l = frames.get("cam_left")
            bgr_r = frames.get("cam_right")
            if bgr_l is None or bgr_r is None:
                time.sleep(0.02)
                continue

            rgb_l = cv2.cvtColor(bgr_l, cv2.COLOR_BGR2RGB)
            rgb_r = cv2.cvtColor(bgr_r, cv2.COLOR_BGR2RGB)
            ra = predict_summary(pol_a, OBS_POSE_RAD, rgb_l, rgb_r, tf, device)
            rb = predict_summary(pol_b, OBS_POSE_RAD, rgb_l, rgb_r, tf, device)

            vis_l = cv2.resize(bgr_l, (480, 360))
            vis_r = cv2.resize(bgr_r, (480, 360))
            combined = np.hstack([vis_l, vis_r])
            h, w = combined.shape[:2]
            panel = np.zeros((90, w, 3), dtype=np.uint8)

            col_a = (0, 200, 0) if ra["mode"] == "SEARCH" else (0, 80, 255)
            col_b = (0, 200, 0) if rb["mode"] == "SEARCH" else (0, 80, 255)
            txt_a = f"{args.label_a}: {ra['mode']} | Pan20={ra['pan20']:+.1f} Lift30={ra['lift30']:+.1f} Elb30={ra['elb30']:+.1f} MaxGrip={ra['max_grip']:.1f}"
            txt_b = f"{args.label_b}: {rb['mode']} | Pan20={rb['pan20']:+.1f} Lift30={rb['lift30']:+.1f} Elb30={rb['elb30']:+.1f} MaxGrip={rb['max_grip']:.1f}"
            cv2.putText(panel, txt_a, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col_a, 2)
            cv2.putText(panel, txt_b, (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, col_b, 2)

            out = np.vstack([combined, panel])
            cv2.namedWindow("Compare Checkpoints (Live Real Cam)", cv2.WINDOW_NORMAL)
            cv2.imshow("Compare Checkpoints (Live Real Cam)", out)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            time.sleep(0.03)
    finally:
        cap.stop()
        cv2.destroyAllWindows()


def run_sim_mode(args, tf, device):
    from isaacsim import SimulationApp
    sim_app = SimulationApp({"headless": args.headless})
    import omni.usd, omni.timeline
    from pxr import UsdGeom, UsdShade, Gf
    from env_configs import ENV_REGISTRY
    from env_manager import EnvironmentManager

    omni.usd.get_context().open_stage(ENV_REGISTRY["real_wall_twin"]["usd_path"])
    timeline = omni.timeline.get_timeline_interface()
    timeline.play()
    for _ in range(10):
        sim_app.update()

    env = EnvironmentManager(env_name="real_wall_twin")
    env.ensure_initialized()
    pol_a = load_policy(args.ckpt_a, device)
    pol_b = load_policy(args.ckpt_b, device)

    if args.cube_out:
        env.reset_cube_pose(-0.50, -0.50, z_val=-0.20, yaw_deg=0.0)
    else:
        cx = args.r * float(np.cos(np.radians(args.theta)))
        cy = args.r * float(np.sin(np.radians(args.theta)))
        env.reset_cube_pose(cx, cy, z_val=0.015, yaw_deg=0.0)

    def set_cam_fl(fl):
        for c in (env.left_cam_prim, env.right_cam_prim):
            UsdGeom.Camera(c).GetFocalLengthAttr().Set(float(fl))

    def set_tint(rgb):
        p = omni.usd.get_context().get_stage().GetPrimAtPath("/World/Looks/RealWall_TableMat/Shader")
        if p and p.IsValid():
            inp = UsdShade.Shader(p).GetInput("diffuse_tint")
            if inp:
                inp.Set(Gf.Vec3f(*rgb))

    cases = [
        ("Baseline (fl=19.32)", 19.32, (1.0, 1.0, 1.0)),
        ("Dark/Cool Table", 19.32, (0.55, 0.58, 0.65)),
        ("Wide FOV (fl=14.0)", 14.0, (1.0, 1.0, 1.0)),
        ("Narrow FOV (fl=25.0)", 25.0, (1.0, 1.0, 1.0)),
    ]

    print(f"\n=== [시뮬레이션 비교] {args.label_a} vs {args.label_b} ===")
    for cname, fl, tint in cases:
        set_cam_fl(fl)
        set_tint(tint)
        for _ in range(25):
            env.apply_full_joints(OBS_POSE_RAD)
            sim_app.update()
        q_curr = np.array(env.robot_articulation.get_joint_positions()[:6], dtype=np.float32)
        il, ir = env.get_rgb_frames()
        ra = predict_summary(pol_a, q_curr, il, ir, tf, device)
        rb = predict_summary(pol_b, q_curr, il, ir, tf, device)
        print(
            f"{cname:<22} | {args.label_a}: {ra['mode']:<6} (Lift={ra['lift30']:+5.1f}, Grip={ra['max_grip']:4.1f}) "
            f"| {args.label_b}: {rb['mode']:<6} (Lift={rb['lift30']:+5.1f}, Grip={rb['max_grip']:4.1f})"
        )

    set_cam_fl(19.32)
    set_tint((1.0, 1.0, 1.0))
    sim_app.close()


def main():
    parser = argparse.ArgumentParser(description="SO-101 두 체크포인트 비교 도구")
    parser.add_argument("--mode", choices=["real_cam", "sim"], default="real_cam")
    parser.add_argument("--ckpt-a", type=str, default=os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", "real_wall_twin_v1"))
    parser.add_argument("--ckpt-b", type=str, default=os.path.join(PROJECT_ROOT, "ckpt", "so101_active_search", "real_wall_twin"))
    parser.add_argument("--label-a", type=str, default="v1")
    parser.add_argument("--label-b", type=str, default="v2")
    parser.add_argument("--left-cam", type=int, default=2)
    parser.add_argument("--right-cam", type=int, default=0)
    parser.add_argument("--focus", type=int, default=25)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--cube-out", action="store_true", default=True)
    parser.add_argument("--r", type=float, default=0.28)
    parser.add_argument("--theta", type=float, default=-40.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    if args.mode == "real_cam":
        run_real_cam_mode(args, tf, device)
    else:
        run_sim_mode(args, tf, device)


if __name__ == "__main__":
    main()


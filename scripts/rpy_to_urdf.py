#!/usr/bin/env python3
"""
so101 Dual Cam Mount: Isaac Sim Orient -> URDF RPY 변환 스크립트
====================================================================
사용자가 Isaac Sim Property 패널에서 맞춘 최종 값:
- 왼쪽 카메라 (Left Camera):
    Translate: [-0.0848, -0.052, 0.041]
    Orient:    [105.0, 25.0, -8.0]
- 오른쪽 카메라 (Right Camera):
    Translate: [-0.0848, -0.052, -0.005]
    Orient:    [75.0, 25.0, 8.0]

Isaac Sim Orient [X, Y, Z] (Euler XYZ 순서)를 URDF rpy [roll, pitch, yaw] (Euler ZYX 순서)로 변환합니다.
"""

from scipy.spatial.transform import Rotation as R


def orient_to_urdf(x_deg, y_deg, z_deg):
    """
    Isaac Sim Property Panel Orient [X, Y, Z] (Euler XYZ) -> URDF rpy (Euler xyz)
    Isaac Sim Property panel displays rotation in Extrinsic XYZ.
    Isaac Sim URDF Importer parses joint rpy in Intrinsic xyz.
    To make the imported link match the Property panel Orient:
        R_target = R.from_euler('XYZ', [x_deg, y_deg, z_deg], degrees=True)
        r, p, y = R_target.as_euler('xyz', degrees=False)
    """
    rot = R.from_euler("XYZ", [x_deg, y_deg, z_deg], degrees=True)
    r, p, y = rot.as_euler("xyz", degrees=False)
    return r, p, y


def urdf_to_orient(r_rad, p_rad, y_rad):
    """
    URDF rpy (Intrinsic xyz) -> Isaac Sim Property Panel Orient [X, Y, Z] (Extrinsic XYZ)
    """
    rot = R.from_euler("xyz", [r_rad, p_rad, y_rad], degrees=False)
    return [round(x, 2) for x in rot.as_euler("XYZ", degrees=True)]


if __name__ == "__main__":
    # 사용자 최종 Orient 값
    left_orient = [105.0, 25.0, -8.0]
    right_orient = [75.0, 25.0, 8.0]

    roll_l, pitch_l, yaw_l = orient_to_urdf(*left_orient)
    roll_r, pitch_r, yaw_r = orient_to_urdf(*right_orient)

    print("=" * 65)
    print("Isaac Sim 확인값 기반 URDF 최종 rpy 설정값")
    print("=" * 65)
    print(f'Left  Camera: rpy="{roll_l:.6f} {pitch_l:.6f} {yaw_l:.6f}"')
    print(f'Right Camera: rpy="{roll_r:.6f} {pitch_r:.6f} {yaw_r:.6f}"')
    print("\n[Isaac Sim Orient 복원 검증]")
    print(f"Left  input: {left_orient} -> restored: {urdf_to_orient(roll_l, pitch_l, yaw_l)}")
    print(f"Right input: {right_orient} -> restored: {urdf_to_orient(roll_r, pitch_r, yaw_r)}")
    print("=" * 65)

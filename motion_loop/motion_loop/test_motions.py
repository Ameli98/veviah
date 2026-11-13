"""Hand-made motions for checking the VRM player: raise right hand, wave, nod.

They are written as Euler angles (x, y, z radians per joint, easy to reason about one axis at a
time) and converted to the quaternion motion format at the end. Axis facts used below
(VRM normalized T-pose, y up, faces +z, left = +x):
  - right arm points along -x, so Rz(+a) lowers it and Rz(-a) raises it (left arm: opposite sign)
  - Rx(+a) on the head pitches it forward (looks down); Rz would be a roll (ear to shoulder)
  - elbows are hinges about y in T-pose; Rx(-90 deg) on the right upper arm turns that hinge so
    that bending the elbow lifts the forearm

CLI: python -m motion_loop.test_motions --out ../vrm_player/public/motions
"""
import argparse
from pathlib import Path

import numpy as np

from .rotations import euler_to_mat, from_euler, mat_to_quat, slerp
from .schema import FPS, J, JIDX
from .vrm_export import save_vrm_json

ARM_DOWN = 1.25   # rad below horizontal for the resting arms


def _ease(t, t0, t1):
    """Smoothstep 0 -> 1 between t0 and t1."""
    u = np.clip((t - t0) / (t1 - t0), 0, 1)
    return u * u * (3 - 2 * u)


def _rest(T):
    """Rest pose as Euler angles [T,J,3]."""
    e = np.zeros((T, J, 3))
    e[:, JIDX["leftUpperArm"], 2] = -ARM_DOWN
    e[:, JIDX["rightUpperArm"], 2] = ARM_DOWN
    e[:, JIDX["leftLowerArm"], 1] = -0.25    # slight elbow bend
    e[:, JIDX["rightLowerArm"], 1] = 0.25
    return e


def _times(duration_s):
    T = int(round(duration_s * FPS)) + 1
    return np.arange(T) / FPS, _rest(T)


def rest(duration_s=1.0):
    return from_euler(_times(duration_s)[1])


def raise_right_hand(duration_s=3.0):
    t, e = _times(duration_s)
    up = _ease(t, 0.3, 1.2) - _ease(t, 2.0, 2.8)                       # 0 -> 1 -> 0
    e[:, JIDX["rightUpperArm"], 2] = ARM_DOWN + up * (-1.2 - ARM_DOWN)  # ends 1.2 rad above horizontal
    e[:, JIDX["rightShoulder"], 2] = -0.25 * up
    return from_euler(e)


def wave(duration_s=4.0):
    t, e = _times(duration_s)
    up = _ease(t, 0.3, 1.0) - _ease(t, 3.0, 3.7)
    swing = 0.45 * np.sin(2 * np.pi * 2.0 * (t - 1.0)) * (_ease(t, 1.0, 1.3) - _ease(t, 2.7, 3.0))
    e[:, JIDX["rightLowerArm"], 1] = 0.25 + up * (np.pi / 2 - 0.25) + swing
    m = from_euler(e)
    # Upper arm: blend rest -> (slightly lowered, hinge turned upward). The two poses differ on more
    # than one axis, so blend the rotations themselves (slerp) instead of the Euler numbers.
    q_rest = mat_to_quat(euler_to_mat(np.array([0, 0, ARM_DOWN])))
    q_wave = mat_to_quat(euler_to_mat(np.array([0, 0, 0.35])) @ euler_to_mat(np.array([-np.pi / 2, 0, 0])))
    m[:, JIDX["rightUpperArm"]] = slerp(q_rest[None], q_wave[None], up[:, None])
    return m


def nod(duration_s=3.0):
    t, e = _times(duration_s)
    pitch = 0.30 * np.sin(np.pi * 1.5 * (t - 0.5)) ** 2 * ((t >= 0.5) & (t <= 2.5))   # 3 nods
    e[:, JIDX["head"], 0] = pitch
    e[:, JIDX["neck"], 0] = 0.4 * pitch
    return from_euler(e)


MOTIONS = {"raise_right_hand": raise_right_hand, "wave": wave, "nod": nod}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory for <name>.json (and <name>.npy)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn in MOTIONS.items():
        m = fn()
        np.save(out / f"{name}.npy", m)
        print(save_vrm_json(m, out / f"{name}.json"), f"{len(m)} frames")


if __name__ == "__main__":
    main()

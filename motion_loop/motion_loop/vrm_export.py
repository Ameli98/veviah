"""motion_loop motion -> motion JSON for the three-vrm player (vrm_player/).

JSON: {"fps", "joints": [VRM humanoid bone names], "quat_order": "xyzw", "rotations": [T][J][4],
       "hands": {"shapes": [names], "left": [T], "right": [T]}  (optional),
       "root": [T][3]  (optional: hips offset in units of the rest hips height, see schema)}.
The motion format already is local xyzw quaternions for the VRM 1.0 normalized humanoid, so this
only rounds and writes them.

CLI: python -m motion_loop.vrm_export motion.npy out.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .schema import FPS, HAND_SHAPES, J, JOINTS


def to_vrm_json(motion, fps=FPS, hands=None, root=None, eyes=None, emotion=None):
    """hands: optional uint8 [T, 2] of schema.HAND_SHAPES indices (left, right); omitted = relaxed hands.
    root: optional float [T, 3] hips offset; omitted = the hips stay at rest (standing).
    eyes: optional float [T] closed weight (keyframes.eyes_track); omitted = open, blinking.
    emotion: optional face word (face.js EMOTIONS) the player shows for the whole motion (review renders)."""
    motion = np.asarray(motion)
    assert motion.ndim == 3 and motion.shape[1:] == (J, 4), motion.shape
    out = {"fps": fps, "joints": list(JOINTS), "quat_order": "xyzw",
           "rotations": np.round(motion.astype(float), 5).tolist()}
    if hands is not None:
        hands = np.asarray(hands)
        assert hands.shape == (len(motion), 2), hands.shape
        out["hands"] = {"shapes": list(HAND_SHAPES), "left": hands[:, 0].tolist(), "right": hands[:, 1].tolist()}
    if root is not None:
        root = np.asarray(root)
        assert root.shape == (len(motion), 3), root.shape
        out["root"] = np.round(root.astype(float), 5).tolist()
    if eyes is not None:
        eyes = np.asarray(eyes, float)
        assert eyes.shape == (len(motion),), eyes.shape
        out["eyes"] = np.round(eyes, 3).tolist()
    if emotion:
        out["emotion"] = str(emotion)
    return out


def save_vrm_json(motion, path, fps=FPS, hands=None, root=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_vrm_json(motion, fps, hands, root)))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description="motion .npy [T,J,4] (+ .hands.npy, .root.npy next to it) -> player JSON")
    ap.add_argument("npy")
    ap.add_argument("out")
    a = ap.parse_args(argv)
    m = np.load(a.npy)
    hp, rp = Path(a.npy).with_suffix(".hands.npy"), Path(a.npy).with_suffix(".root.npy")
    hands = np.load(hp) if hp.exists() else None
    root = np.load(rp) if rp.exists() else None
    print(save_vrm_json(m, a.out, hands=hands, root=root), f"{len(m)} frames")


if __name__ == "__main__":
    main()

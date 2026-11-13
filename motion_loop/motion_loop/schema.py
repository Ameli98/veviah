"""Shared constants and jsonl helpers.

Motion format used everywhere: np.ndarray [T, J, 4] float32, unit quaternions (x, y, z, w), 30 fps.
J = 20 VRM humanoid bones (hips, upper body, legs), in the order of JOINTS (parents before children).
Identity = VRM normalized T-pose, standing (y up, character faces +z, character's left = +x); rotations
are local (relative to the parent bone); hips is the root, its rotation is in world axes.
Quaternions are sign-continuous along time (see rotations.continuous). The real generator must
output (or be retargeted to) this format. All rotation math lives in rotations.py.
Besides the rotations a motion has `root`, float32 [T, 3]: the hips position minus its standing rest
position, in units of the rest hips height (so it fits any model; the player multiplies by its own hips
height). Stored next to the motion as <name>.root.npy; absent = zeros (standing still).
Changed 2026-10-15 (the author: full body like Desktop Mate, standing or sitting on the floor, in place).
"""
import json
from pathlib import Path

import numpy as np

FPS = 30
JOINTS = [
    "hips",
    "spine", "chest", "upperChest", "neck", "head",
    "leftShoulder", "leftUpperArm", "leftLowerArm", "leftHand",
    "rightShoulder", "rightUpperArm", "rightLowerArm", "rightHand",
    "leftUpperLeg", "leftLowerLeg", "leftFoot",
    "rightUpperLeg", "rightLowerLeg", "rightFoot",
]
UPPER = [n for n in JOINTS if not n.endswith(("UpperLeg", "LowerLeg", "Foot")) and n != "hips"]
J = len(JOINTS)
JIDX = {n: i for i, n in enumerate(JOINTS)}

# Left/right swap permutation for mirroring.
def _mirror_name(n):
    if n.startswith("left"):
        return "right" + n[4:]
    if n.startswith("right"):
        return "left" + n[5:]
    return n
MIRROR_PERM = [JIDX[_mirror_name(n)] for n in JOINTS]

# PLACEHOLDER joint limits: largest allowed rotation angle from the T-pose, radians, per joint.
# One number per joint because it must not depend on how a rotation is split into axes.
# The upper arm can legitimately get close to 180 deg from T-pose (arm hanging = 90, across the body more).
# Legs: sitting cross-legged or in seiza needs big thigh and knee angles.
_MAX_ANGLE = {"hips": 1.8, "spine": 0.7, "chest": 0.6, "upperChest": 0.5, "neck": 1.0, "head": 1.3,
              "Shoulder": 0.6, "UpperArm": 2.95, "LowerArm": 2.9, "Hand": 1.6,
              "UpperLeg": 2.9, "LowerLeg": 3.05, "Foot": 1.4}
def _limit(name):
    if name in _MAX_ANGLE:
        return _MAX_ANGLE[name]
    for k in ("Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot"):
        if name.endswith(k):
            return _MAX_ANGLE[k]
    raise KeyError(name)
MAX_ANGLE = np.array([_limit(n) for n in JOINTS], dtype=np.float32)  # [J]


def rest_motion(T):
    """T frames of the T-pose."""
    m = np.zeros((T, J, 4), np.float32)
    m[..., 3] = 1
    return m


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_motion(workdir, rel_path):
    return np.load(Path(workdir) / rel_path)


# Hand shapes: fingers are not among the joints. A motion may come with `hands`, uint8 [T, 2]
# (column 0 = left, 1 = right), each entry an index into HAND_SHAPES; the player poses the fingers
# (vrm_player/src/player.js has the same names). Stored next to the motion as <name>.hands.npy.
HAND_SHAPES = ("relaxed", "open", "fist", "point", "peace", "thumbs_up", "flat", "loose")
# flat (2026-11-06): fingers straight and together, not spread (joined palms: "open" fanned the fingers and the
# hands showed gaps, the author). Appended, so old indices are unchanged.


def rest_hands(T):
    return np.zeros((T, 2), np.uint8)


def hands_path(motion_path):
    return Path(motion_path).with_suffix(".hands.npy")


def load_hands(workdir, rel_path, T):
    """Hand shapes for the motion at rel_path; all relaxed if it has none."""
    p = hands_path(Path(workdir) / rel_path)
    return np.load(p) if p.exists() else rest_hands(T)


def rest_root(T):
    return np.zeros((T, 3), np.float32)


def root_path(motion_path):
    return Path(motion_path).with_suffix(".root.npy")


def load_root(workdir, rel_path, T):
    """Hips offset [T, 3] for the motion at rel_path; zeros (standing still) if it has none."""
    p = root_path(Path(workdir) / rel_path)
    return np.load(p) if p.exists() else rest_root(T)

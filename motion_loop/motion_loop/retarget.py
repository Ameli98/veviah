"""SMPL -> VRM retargeting (hips, upper body and legs, see schema.JOINTS).

Why it is simple: SMPL and the VRM normalized humanoid share the same rest pose (T-pose), the
same axes (y up, faces +z, left = +x) and bone-aligned local frames, so local rotations carry over
joint by joint. Body proportions do not matter for rotations. What remains is bookkeeping:
joint mapping, dropping the root and resampling to 30 fps.

Two entry points, depending on what your text-to-motion model outputs:
  retarget_smpl_rotations(pose_aa [T,>=22,3], fps)   axis-angle SMPL poses (twist preserved)
  retarget_joint_positions(joints [T,>=22,3], fps)   3D joint positions, e.g. HumanML3D-style 22 joints
                                                     (swing-only IK: bone directions are matched,
                                                      twist about each bone is lost)
Rotations: the hips take SMPL's global orientation. Positions: the hips stay upright (yaw removed).
Neither fills the hips offset (schema root) yet: the SMPL translation would have to be scaled to the
VRM hips height (TODO when a text-to-motion model is plugged in; until then the hips stay in place).
Heads and hands: with positions there is no child joint to aim at, so they stay at rest orientation.
"""
import numpy as np

from .generators import MotionGenerator
from .rotations import aa_to_mat, continuous, mat_to_quat, normalize, resample_fps
from .schema import FPS, JIDX, JOINTS

# VRM joint -> SMPL joint whose LOCAL rotation it takes (rotation path)
SMPL_FOR = {"hips": 0, "leftUpperLeg": 1, "leftLowerLeg": 4, "leftFoot": 7,
            "rightUpperLeg": 2, "rightLowerLeg": 5, "rightFoot": 8, "spine": 3, "chest": 6, "upperChest": 9, "neck": 12, "head": 15,
            "leftShoulder": 13, "leftUpperArm": 16, "leftLowerArm": 18, "leftHand": 20,
            "rightShoulder": 14, "rightUpperArm": 17, "rightLowerArm": 19, "rightHand": 21}
SMPL_IDX = [SMPL_FOR[n] for n in JOINTS]
PARENT = {"hips": None, "leftUpperLeg": "hips", "leftLowerLeg": "leftUpperLeg", "leftFoot": "leftLowerLeg",
          "rightUpperLeg": "hips", "rightLowerLeg": "rightUpperLeg", "rightFoot": "rightLowerLeg", "spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
          "leftShoulder": "upperChest", "leftUpperArm": "leftShoulder", "leftLowerArm": "leftUpperArm",
          "leftHand": "leftLowerArm", "rightShoulder": "upperChest", "rightUpperArm": "rightShoulder",
          "rightLowerArm": "rightUpperArm", "rightHand": "rightLowerArm"}
_X, _Y = np.array([1., 0, 0]), np.array([0., 1, 0])
_TOE = np.array([0., -0.495, 0.869])                      # foot -> toes in the T-pose (skeleton.TOE)
# VRM joint -> (SMPL child joint to aim at, rest direction of that bone), position path
AIM = {"leftUpperLeg": (4, -_Y), "leftLowerLeg": (7, -_Y), "leftFoot": (10, _TOE),
       "rightUpperLeg": (5, -_Y), "rightLowerLeg": (8, -_Y), "rightFoot": (11, _TOE), "spine": (6, _Y), "chest": (9, _Y), "upperChest": (12, _Y), "neck": (15, _Y),
       "leftShoulder": (16, _X), "leftUpperArm": (18, _X), "leftLowerArm": (20, _X),
       "rightShoulder": (17, -_X), "rightUpperArm": (19, -_X), "rightLowerArm": (21, -_X)}
AIM_HAND = {"leftHand": (22, _X), "rightHand": (23, -_X)}   # only when 24 joints are given


def _gauss_smooth(x, sigma):
    if sigma <= 0:
        return x
    r = int(np.ceil(3 * sigma))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(x, [(r, r)] + [(0, 0)] * (x.ndim - 1), mode="edge")
    return sum(w * pad[i:i + len(x)] for i, w in enumerate(k))


def _finish(R_local, fps_in, fps_out, smooth_sigma):
    """R_local [T,J,3,3] at fps_in -> (quaternions [T',J,4] at fps_out, info)."""
    q = continuous(resample_fps(mat_to_quat(R_local), fps_in, fps_out))
    q = normalize(_gauss_smooth(q, smooth_sigma)).astype(np.float32)   # averaging nearby, sign-aligned quaternions
    return q, {"frames": len(q)}


# ---------------- entry points ----------------
def retarget_smpl_rotations(pose_aa, fps_in=20, fps_out=FPS, smooth_sigma=0.0):
    """pose_aa [T,>=22,3] axis-angle SMPL local rotations (joint 0 = pelvis = global orientation -> hips)."""
    pose_aa = np.asarray(pose_aa, float)
    if pose_aa.ndim == 2:                       # flat [T, 3*n]
        pose_aa = pose_aa.reshape(len(pose_aa), -1, 3)
    return _finish(aa_to_mat(pose_aa[:, SMPL_IDX]), fps_in, fps_out, smooth_sigma)


def _rot_between(a, b):
    """Minimal rotation taking unit vectors a -> b. a,b [...,3] -> [...,3,3]."""
    v, c = np.cross(a, b), np.sum(a * b, -1)
    K = np.zeros(a.shape[:-1] + (3, 3))
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    R = np.eye(3) + K + (K @ K) / (1 + c)[..., None, None].clip(1e-9)
    flip = c < -0.9999                           # antiparallel: 180 deg about any perpendicular axis
    if flip.any():
        ref = np.where(np.abs(a[..., :1]) < 0.9, _X, _Y)
        ax = np.cross(a, ref)
        ax /= np.linalg.norm(ax, axis=-1, keepdims=True)
        R = np.where(flip[..., None, None], 2 * ax[..., :, None] * ax[..., None, :] - np.eye(3), R)
    return R


def pelvis_frame(joints):
    """Upright pelvis frame [T,3,3] (columns = x,y,z in world) from the hips: keeps yaw only.
    Assumes right-handed, y-up data with SMPL joint order (1 = left hip, 2 = right hip)."""
    x = joints[:, 1] - joints[:, 2]
    x[:, 1] = 0
    x /= np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-9)
    up = np.broadcast_to(_Y, x.shape)
    z = np.cross(x, up)
    return np.stack([x, up, z], -1)


def retarget_joint_positions(joints, fps_in=20, fps_out=FPS, smooth_sigma=1.0):
    """joints [T,>=22,3] world positions, SMPL joint order, y up. Swing-only IK down the chain."""
    joints = np.asarray(joints, float)
    T = len(joints)
    P = pelvis_frame(joints)
    aim = dict(AIM)
    if joints.shape[1] >= 24:
        aim.update(AIM_HAND)
    Rg = {None: np.broadcast_to(np.eye(3), (T, 3, 3))}   # global rotation in the pelvis frame
    Rl = np.zeros((T, len(JOINTS), 3, 3))
    for name in JOINTS:                                  # parents come before children
        parent = Rg[PARENT[name]]
        if name in aim:
            child, rest = aim[name]
            d = joints[:, child] - joints[:, SMPL_FOR[name]]
            d = np.einsum("tji,tj->ti", P, d)            # world -> pelvis frame (P^T d)
            d /= np.linalg.norm(d, axis=-1, keepdims=True).clip(1e-9)
            d = np.einsum("tji,tj->ti", parent, d)       # -> parent's frame
            R = _rot_between(np.broadcast_to(rest, d.shape), d)
        else:
            R = np.broadcast_to(np.eye(3), (T, 3, 3))    # hips (upright pelvis), head / hands without a child: rest
        Rl[:, JIDX[name]] = R
        Rg[name] = parent @ R
    return _finish(Rl, fps_in, fps_out, smooth_sigma)


class SmplTextToMotionGenerator(MotionGenerator):
    """Subclass and implement `raw`; retargeting to the VRM joint format is handled here.

    raw(text, seed, duration_s) -> (kind, data, fps)
        kind = "rotations" (data [T,>=22,3] axis-angle) or "joints" (data [T,>=22,3] positions)
    """

    smooth_sigma = None   # None = per-path default

    def raw(self, instruction, seed, duration_s):
        raise NotImplementedError("wrap your text-to-motion model here")

    def generate(self, instruction, seed, duration_s):
        kind, data, fps = self.raw(instruction, seed, duration_s)
        kw = {} if self.smooth_sigma is None else {"smooth_sigma": self.smooth_sigma}
        fn = retarget_smpl_rotations if kind == "rotations" else retarget_joint_positions
        motion, _ = fn(np.asarray(data), fps, **kw)
        return motion


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Convert a saved SMPL motion (.npy) to the VRM joint format")
    ap.add_argument("input")
    ap.add_argument("--kind", choices=["rotations", "joints"], required=True)
    ap.add_argument("--fps", type=float, default=20)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fn = retarget_smpl_rotations if a.kind == "rotations" else retarget_joint_positions
    motion, info = fn(np.load(a.input), a.fps)
    np.save(a.out, motion)
    print(info, "->", a.out, motion.shape)


if __name__ == "__main__":
    main()

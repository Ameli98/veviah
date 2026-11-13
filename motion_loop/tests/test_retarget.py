"""Retarget tests. The check that matters: after retargeting, the VRM stick skeleton's bones
point in the same GLOBAL directions as the SMPL skeleton's bones for the same pose."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import retarget as rt
from motion_loop import rotations as rot
from motion_loop.render import forward_kinematics
from motion_loop.schema import J, JIDX, JOINTS

# Test-only SMPL-like T-pose skeleton (22 joints). Offsets are approximate; only the rest
# DIRECTIONS matter for this test, not the exact proportions.
PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
OFFS = np.array([
    (0, 0, 0), (.09, -.09, 0), (-.09, -.09, 0), (0, .11, 0), (0, -.4, 0), (0, -.4, 0), (0, .13, 0),
    (0, -.4, 0), (0, -.4, 0), (0, .05, 0), (0, -.05, .12), (0, -.05, .12), (0, .22, 0),
    (.04, .10, 0), (-.04, .10, 0), (0, .10, 0), (.12, 0, 0), (-.12, 0, 0), (.26, 0, 0), (-.26, 0, 0),
    (.26, 0, 0), (-.26, 0, 0)], float)
BONES = [(3, 6), (6, 9), (9, 12), (12, 15), (13, 16), (16, 18), (18, 20), (14, 17), (17, 19), (19, 21)]
VRM_BONES = [("spine", "chest"), ("chest", "upperChest"), ("upperChest", "neck"), ("neck", "head"),
             ("leftShoulder", "leftUpperArm"), ("leftUpperArm", "leftLowerArm"), ("leftLowerArm", "leftHand"),
             ("rightShoulder", "rightUpperArm"), ("rightUpperArm", "rightLowerArm"), ("rightLowerArm", "rightHand")]


def smpl_fk(pose_aa, root_yaw=0.0):
    """pose_aa [22,3] -> world joint positions [22,3] and global rotations [22,3,3]."""
    R = rot.aa_to_mat(pose_aa)
    Ry = rot.euler_to_mat(np.array([0, root_yaw, 0.0]))
    Rg, pos = [None] * 22, np.zeros((22, 3))
    for j in range(22):
        p = PARENTS[j]
        if p < 0:
            Rg[j], pos[j] = Ry @ R[j], np.zeros(3)
        else:
            Rg[j], pos[j] = Rg[p] @ R[j], pos[p] + Rg[p] @ OFFS[j]
    return pos, Rg


def random_pose(rng, mag=0.9, pelvis=True):
    """pelvis=False keeps the pelvis upright: the position path cannot see its tilt (it keeps the hips upright)."""
    pose = np.zeros((22, 3))
    for j in rt.SMPL_IDX:
        if j == 0 and not pelvis:
            continue
        v = rng.normal(size=3)
        pose[j] = v / np.linalg.norm(v) * rng.uniform(0, mag)
    return pose


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def vrm_dirs(frame):
    pos, _ = forward_kinematics(frame)
    return np.stack([unit(pos[JIDX[c]] - pos[JIDX[p]]) for p, c in VRM_BONES])


class RotationHelpers(unittest.TestCase):
    def test_quat_matrix_roundtrip_and_product(self):
        rng = np.random.default_rng(1)
        R = rot.aa_to_mat(rng.normal(size=(100, 3)))
        q = rot.mat_to_quat(R)
        self.assertTrue(np.allclose(rot.quat_to_mat(q), R, atol=1e-9))
        self.assertTrue(np.allclose(rot.quat_to_mat(rot.quat_mul(q[:50], q[50:])), R[:50] @ R[50:], atol=1e-9))
        self.assertTrue(np.allclose(rot.quat_to_mat(rot.quat_conj(q)), np.swapaxes(R, -1, -2), atol=1e-9))

    def test_euler_convention(self):
        """R = Rx @ Ry @ Rz, written out independently here."""
        a, b, c = 0.3, -0.7, 1.1
        Rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        Ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        Rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        self.assertTrue(np.allclose(rot.euler_to_mat([a, b, c]), Rx @ Ry @ Rz))
        self.assertTrue(np.allclose(rot.from_euler(np.array([[[np.pi / 2, 0, 0]]]))[0, 0], [np.sqrt(.5), 0, 0, np.sqrt(.5)], atol=1e-6))

    def test_rotvec_roundtrip_and_angle(self):
        v = np.random.default_rng(2).normal(size=(200, 3))
        v = v / np.linalg.norm(v, axis=-1, keepdims=True) * np.random.default_rng(3).uniform(0, 3.1, (200, 1))
        self.assertTrue(np.allclose(rot.rotvec(rot.from_rotvec(v)), v, atol=1e-8))
        self.assertTrue(np.allclose(rot.angle(rot.from_rotvec(v)), np.linalg.norm(v, axis=-1), atol=1e-8))
        self.assertTrue(np.allclose(rot.rotvec(-rot.from_rotvec(v)), v, atol=1e-8))      # q and -q are the same rotation

    def test_slerp_and_resample(self):
        q = rot.mat_to_quat(rot.aa_to_mat(np.random.default_rng(2).normal(size=(21, J, 3)) * .3))
        self.assertTrue(np.allclose(rot.slerp(q[0], q[1], 0.0), q[0]))
        self.assertTrue(np.allclose(np.abs(np.sum(rot.slerp(q[0], q[1], 1.0) * q[1], -1)), 1.0))
        self.assertEqual(len(rot.resample_fps(q, 20, 30)), 31)   # 1.0 s at 20 fps -> 1.0 s at 30 fps

    def test_angular_velocity_is_the_true_speed_even_at_gimbal_lock(self):
        """Spin about a fixed axis at 2 rad/s, starting from the pose where Euler y = 90 deg."""
        t = np.arange(30) / 30
        base = rot.mat_to_quat(rot.euler_to_mat([0.4, np.pi / 2, 0.7]))
        axis = np.array([0.6, 0.0, 0.8])
        q = rot.quat_mul(base[None], rot.from_rotvec(axis[None] * 2.0 * t[:, None]))[:, None]
        w = rot.angular_velocity(rot.continuous(q), 30)
        self.assertTrue(np.allclose(np.linalg.norm(w, axis=-1), 2.0, atol=1e-6))
        self.assertTrue(np.allclose(w[:, 0], axis * 2.0, atol=1e-6))

    def test_continuous_removes_sign_flips(self):
        q = rot.from_euler(np.linspace(0, 1, 20)[:, None, None] * np.array([[[2.0, 0.5, -1.0]]]))
        flipped = q.copy(); flipped[5:9] *= -1; flipped[0] *= -1
        self.assertTrue(np.allclose(rot.continuous(flipped), q, atol=1e-6))


class Retargeting(unittest.TestCase):
    def test_rotation_path_matches_smpl_bone_directions(self):
        rng = np.random.default_rng(3)
        for _ in range(20):
            pose = random_pose(rng)
            _, Rg = smpl_fk(pose)
            want = np.stack([unit(Rg[a] @ OFFS[b]) for a, b in BONES])
            m, info = rt.retarget_smpl_rotations(pose[None], fps_in=30)
            self.assertEqual(m.shape, (1, J, 4))
            self.assertTrue(np.allclose(vrm_dirs(m[0]), want, atol=1e-6))

    def test_position_path_matches_smpl_bone_directions_and_ignores_yaw(self):
        rng = np.random.default_rng(4)
        for yaw in (0.0, 1.1, -2.5):
            pose = random_pose(rng, pelvis=False)
            pos, _ = smpl_fk(pose)                                  # no yaw: reference directions
            _, Rg = smpl_fk(pose)
            want = np.stack([unit(Rg[a] @ OFFS[b]) for a, b in BONES])
            pos_yawed, _ = smpl_fk(pose, root_yaw=yaw)             # same pose, body turned
            m, _ = rt.retarget_joint_positions(pos_yawed[None], fps_in=30, smooth_sigma=0)
            self.assertTrue(np.allclose(vrm_dirs(m[0]), want, atol=1e-5), f"yaw={yaw}")

    def test_antiparallel_bone(self):
        pose = np.zeros((22, 3)); pose[16] = (0, 0, np.pi)         # left upper arm flipped 180 deg about z
        pos, _ = smpl_fk(pose)
        m, _ = rt.retarget_joint_positions(pos[None], fps_in=30, smooth_sigma=0)
        self.assertTrue(np.isfinite(m).all())
        d = vrm_dirs(m[0])[VRM_BONES.index(("leftUpperArm", "leftLowerArm"))]
        self.assertTrue(np.allclose(d, [-1, 0, 0], atol=1e-5))

    def test_time_series_shapes_and_smoothness(self):
        rng = np.random.default_rng(5)
        a, b = random_pose(rng, pelvis=False), random_pose(rng, pelvis=False)
        seq = np.stack([smpl_fk(a * (1 - t) + b * t)[0] for t in np.linspace(0, 1, 40)])   # 40 frames @20fps
        m, info = rt.retarget_joint_positions(seq, fps_in=20)
        self.assertEqual(m.shape, (int(round(39 / 20 * 30)) + 1, J, 4))
        self.assertTrue(np.allclose(np.linalg.norm(m, axis=-1), 1, atol=1e-5))
        self.assertLess(np.abs(np.diff(m, axis=0)).max(), 0.2)      # no sign flips between frames
        self.assertLess(np.linalg.norm(rot.angular_velocity(m.astype(float), 30), axis=-1).max(), 6.0)
        self.assertTrue(np.isfinite(m).all())


if __name__ == "__main__":
    unittest.main()

"""Player export + hand-made test motions. The stick-figure FK is the reference: it says where the
hands and the face point for a given motion, independent of any VRM file."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import test_motions as tm
from motion_loop.filters import check_motion
from motion_loop.render import forward_kinematics
from motion_loop.rotations import quat_to_mat
from motion_loop.schema import J, JIDX, JOINTS
from motion_loop.vrm_export import to_vrm_json


def face_dir(frame):
    """Global direction the face points (rest: +z)."""
    R = np.eye(3)
    for n in ("spine", "chest", "upperChest", "neck", "head"):
        R = R @ quat_to_mat(frame[JIDX[n]])
    return R @ np.array([0, 0, 1.0])


class Export(unittest.TestCase):
    def test_json_is_the_motion_itself(self):
        m = tm.nod()
        d = to_vrm_json(m)
        self.assertEqual(d["joints"], JOINTS)
        self.assertEqual(d["quat_order"], "xyzw")
        self.assertTrue(np.allclose(np.array(d["rotations"]), m, atol=1e-5))
        self.assertTrue(np.allclose(np.array(d["rotations"])[0, JIDX["spine"]], [0, 0, 0, 1]))   # identity = (0,0,0,1)


class HandMadeMotions(unittest.TestCase):
    def test_raise_is_the_right_hand(self):
        m = tm.raise_right_hand()
        pos, _ = forward_kinematics(m[len(m) // 2])
        rh, lh, head = (pos[JIDX[n]] for n in ("rightHand", "leftHand", "head"))
        self.assertLess(rh[0], 0)              # character's right = -x
        self.assertGreater(rh[1], head[1])     # above the head
        self.assertLess(lh[1], pos[JIDX["spine"]][1] + 0.1)   # left hand stays down

    def test_wave_forearm_up_and_swinging_sideways(self):
        m = tm.wave()
        xs = []
        for f in m[45:75]:
            pos, _ = forward_kinematics(f)
            fore = pos[JIDX["rightHand"]] - pos[JIDX["rightLowerArm"]]
            self.assertGreater(fore[1], 0.15)              # forearm points up
            self.assertLess(abs(fore[2]), 0.03)            # stays in the frontal plane
            xs.append(fore[0])
        self.assertGreater(max(xs) - min(xs), 0.15)        # swings left-right

    def test_nod_is_pitch(self):
        m = tm.nod()
        d = np.array([face_dir(f) for f in m])
        self.assertLess(d[:, 1].min(), -0.3)               # looks down
        self.assertTrue(np.allclose(d[:, 0], 0, atol=1e-6))  # no yaw/roll
        still = [j for n, j in JIDX.items() if n not in ("head", "neck")]
        self.assertTrue(np.allclose(m[:, still], tm.rest(3.0)[:, still], atol=1e-6))

    def test_start_and_end_at_rest_and_pass_the_filter(self):
        for fn in tm.MOTIONS.values():
            m = fn()
            self.assertEqual(m.shape[1:], (J, 4))
            self.assertTrue(np.allclose(m[0], m[-1], atol=1e-3))
            self.assertEqual(check_motion(m), [])


if __name__ == "__main__":
    unittest.main()

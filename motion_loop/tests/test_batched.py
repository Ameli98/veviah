"""The batched solver paths give the results of the per-frame code they replaced (2026-11-03, speed)."""
import unittest

import numpy as np

from motion_loop import contact, depenetrate as dp, keyframes as kf, refine, skeleton as sk


class BatchedTest(unittest.TestCase):
    def test_solve_arm_batch_is_solve_arm(self):
        rng = np.random.default_rng(1)
        U, F, H = (rng.normal(size=(50, 3)) for _ in range(3))
        U, F = U / np.linalg.norm(U, axis=1, keepdims=True), F / np.linalg.norm(F, axis=1, keepdims=True)
        for side in ("left", "right"):
            R, e, h = kf.solve_arm_batch(U, F, side, H)
            for i in range(len(U)):
                r1, e1, h1 = kf.solve_arm(U[i], F[i], side, H[i])
                np.testing.assert_allclose(R[i], r1, atol=1e-9); self.assertAlmostEqual(e[i], e1); np.testing.assert_allclose(h[i], h1, atol=1e-9)

    def test_leg_gap_batch_is_leg_gap(self):
        rng = np.random.default_rng(2)
        T, S = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
        for side in ("left", "right"):
            b = contact._leg_gap_batch(side, T, S)
            a = [contact._leg_gap("legs_out", side, t, s) for t, s in zip(T, S)]
            np.testing.assert_allclose(b, a, atol=1e-9)

    def test_sitting_solve_same_as_per_frame(self):
        obj = {"posture": "legs_out", "keyframes": [{"t": 0}, {"t": 1.2, "right_arm": {"touch": "knee"},
               "left_arm": {"touch": "floor_behind"}, "legs": {"right": "knee_up"}}, {"t": 2.0}]}
        out = []
        for b in (False, True):
            dp.BATCH = dp.LIFT_BATCH = contact.LEG_BATCH = b
            try:
                out.append(refine.solve(obj, 2.0, cloth=False)[0])
            finally:
                dp.BATCH = dp.LIFT_BATCH = contact.LEG_BATCH = True
        np.testing.assert_allclose(np.abs((out[0] * out[1]).sum(-1)), 1.0, atol=1e-6)   # same rotations (q or -q)


if __name__ == "__main__":
    unittest.main()

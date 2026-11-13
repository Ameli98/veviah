"""Per-model skeletons (skeleton.use_model, refine.solve(model=))."""
import unittest

import numpy as np

from motion_loop import refine, skeleton as sk
from motion_loop.schema import JIDX
from pathlib import Path

# test1 is a private model: its measurements are not in the public repository, so its checks are skipped there
HAVE = lambda m: (Path(sk.__file__).parent / "skeletons" / f"{m}.json").exists()


class PerModel(unittest.TestCase):
    def test_use_model_swaps_and_restores(self):
        self.assertEqual(sk.MODEL, sk.DEFAULT_MODEL)
        if HAVE("test1"):
            with sk.use_model("test1"):
                self.assertGreater(sk.HIPS_Y, 0.95)
        self.assertAlmostEqual(sk.HIPS_Y, 0.9081)
        self.assertEqual(sk.MODEL, sk.DEFAULT_MODEL)

    def test_sitting_touches_the_floor_on_each_model(self):
        for m in [m for m in ("test1", "AvatarSample_D", "Seed-san") if HAVE(m)]:
            motion, root, _ = refine.solve({"posture": "cross_legged", "keyframes": [{"t": 0}, {"t": 1}]}, 1.0, model=m)
            with sk.use_model(m):
                p = sk.motion_leg_points(motion, root, JIDX)
                low = min(min(p[f"{s}Knee"][-1, 1] - sk.R_KNEE, p[f"{s}Ankle"][-1, 1] - sk.R_ANKLE,
                              p[f"{s}Toes"][-1, 1] - sk.R_TOE) for s in ("left", "right"))
            self.assertLess(abs(low), 0.01, m)

    def test_wide_thighs_are_rejected_in_a_long_skirt_only(self):
        from motion_loop.filters import check_motion
        cross = refine.solve({"posture": "cross_legged", "keyframes": [{"t": 0}, {"t": 1}]}, 1.0, model="test2", cloth=False)
        seiza = refine.solve({"posture": "seiza", "keyframes": [{"t": 0}, {"t": 1}]}, 1.0, model="test2", cloth=False)
        self.assertIn("skirt_spread", check_motion(cross[0], cross[1], model="test2"))       # dress below the knees
        self.assertNotIn("skirt_spread", check_motion(seiza[0], seiza[1], model="test2"))
        short = refine.solve({"posture": "cross_legged", "keyframes": [{"t": 0}, {"t": 1}]}, 1.0, model="AvatarSample_D", cloth=False)
        self.assertNotIn("skirt_spread", check_motion(short[0], short[1], model="AvatarSample_D"))   # short skirt


if __name__ == "__main__":
    unittest.main()


class FirstSolveTest(unittest.TestCase):
    def test_first_solve_in_a_process_on_another_model_equals_the_next(self):
        # 2026-11-03: a module constant written after its on_model_change hook (contact.W_MIN) kept the default
        # model's value when the module was first imported inside skeleton.use_model
        import subprocess, sys, os
        code = ("import numpy as np, copy\nfrom motion_loop import refine\n"
                "o={'posture':'stand','keyframes':[{'t':0},{'t':1.5,'right_arm':{'touch':'cheek','with':'fingers','elbow':'together'},"
                "'left_arm':{'touch':'cheek','with':'fingers','elbow':'together'}},{'t':2.3}]}\n"
                "a=refine.solve(copy.deepcopy(o),2.3,model='test2',cloth=False)[0]\n"
                "b=refine.solve(copy.deepcopy(o),2.3,model='test2',cloth=False)[0]\n"
                "print(float(np.abs(a-b).max()))\n")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True,
                             env={**os.environ, "PYTHONPATH": root})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(float(out.stdout.strip().splitlines()[-1]), 0.0)

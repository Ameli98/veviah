import unittest

import numpy as np

from motion_loop import arm_words as aw, refine
from motion_loop.filters import check_motion
from motion_loop.keyframes import REST_ARM


class ArmWordsTest(unittest.TestCase):
    def test_sides_mirror(self):
        for hand in ("face out close", "chest other_side close", "shoulder front far", "waist center back"):
            ur, fr = aw.arm_dirs(hand, "right")
            ul, fl = aw.arm_dirs(hand, "left")
            m = np.array([-1, 1, 1])
            np.testing.assert_allclose(ur * m, ul, atol=0.02)
            np.testing.assert_allclose(fr * m, fl, atol=0.02)

    def test_words_land_where_they_say(self):
        u, f = aw.arm_dirs("above_head front far", "right")
        self.assertGreater(u[1], 0.95)                 # arm straight up
        self.assertGreater(f[1], 0.95)
        u, f = aw.arm_dirs("shoulder front far", "right")
        self.assertGreater(f[2], 0.9)                  # pointing forward
        u, f = aw.arm_dirs("shoulder wide beside", "right")
        self.assertLess(f[0], -0.9)                    # right arm out to its right (-x)

    def test_rest_and_errors(self):
        obj = aw.resolve_arm_words({"posture": "stand", "keyframes": [{"t": 0, "right_arm": "rest"}]})
        self.assertEqual(obj["keyframes"][0]["right_arm"]["upper"], list(REST_ARM["right"][0]))
        with self.assertRaises(ValueError):
            aw.arm_dirs("face out", "right")
        with self.assertRaises(ValueError):
            aw.arm_dirs("face sideways close", "right")

    def test_word_wave_passes_the_filter(self):
        kfs = [{"t": 0}, {"t": 0.5, "right_arm": {"hand": "head out beside", "palm": "forward"}, "hands": {"right": "open"}},
               {"t": 0.8, "right_arm": {"hand": "head front beside"}}, {"t": 1.1, "right_arm": {"hand": "head out beside"}},
               {"t": 1.9, "right_arm": "rest", "hands": {"right": "relaxed"}}]
        m, r, h = refine.solve({"posture": "stand", "keyframes": kfs}, 1.9)
        self.assertEqual(check_motion(m, r), [])



if __name__ == "__main__":
    unittest.main()

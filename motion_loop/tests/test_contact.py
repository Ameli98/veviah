"""Touch words (motion_loop.contact): the resolved arm puts the hand on the body surface without going in."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import refine, skeleton as sk
from motion_loop.contact import resolve_touches
from motion_loop.schema import JIDX


def hold(post, arms):
    return {"posture": post, "keyframes": [{"t": 0.0}, {"t": 0.8, **arms}, {"t": 2.0}]}


class ContactTest(unittest.TestCase):
    def test_touches_resolve_to_directions(self):
        obj = hold("stand", {"right_arm": {"touch": "chest"}})
        arm = resolve_touches(obj, 2.0)["keyframes"][1]["right_arm"]
        self.assertTrue({"upper", "fore", "palm", "wrist"} <= set(arm))
        self.assertEqual(arm["touch"], "chest")                       # the word stays in the JSON

    def test_hands_rest_on_the_surface_not_in_it(self):
        cases = [("stand", {"right_arm": {"touch": "chest"}}),
                 ("stand", {"right_arm": {"touch": "hip"}, "left_arm": {"touch": "hip"}}),
                 ("stand", {"right_arm": {"touch": "other_hand"}, "left_arm": {"touch": "other_hand"}}),
                 ("stand", {"right_arm": {"touch": "chin", "with": "fist"}}),
                 ("seiza", {"right_arm": {"touch": "thigh"}, "left_arm": {"touch": "thigh"}})]
        for post, arms in cases:
            m, r, h = refine.solve(hold(post, arms), 2.0)
            f = 45                                                    # held from 0.8 s to the 2.0 s keyframe
            self.assertLessEqual(sk.hand_body_depth(m, r, JIDX)[f], 0.0, (post, arms))
            self.assertLessEqual(sk.hands_overlap(m, r, JIDX)[f], 0.0, (post, arms))

    def test_bad_words_are_rejected(self):
        with self.assertRaises(ValueError):
            resolve_touches(hold("stand", {"right_arm": {"touch": "elbow_pit"}}), 2.0)


if __name__ == "__main__":
    unittest.main()


class ArmLegInteractionTest(unittest.TestCase):
    """2026-10-28: an elbow can rest on a knee; a hand going to a leg that moves rides with it."""

    def test_elbow_on_a_raised_knee_and_hand_riding_a_rising_leg(self):
        from motion_loop import refine, skeleton as sk
        from motion_loop.filters import check_motion
        from motion_loop.schema import JIDX
        elbow = {"posture": "legs_out", "keyframes": [{"t": 0}, {"t": 1.2, "legs": {"right": "knee_up"},
                 "right_arm": {"touch": "knee", "with": "elbow", "hand": "face center close"}}, {"t": 2.0}]}
        knee = {"posture": "legs_out", "keyframes": [{"t": 0}, {"t": 1.2, "legs": {"left": "knee_up"},
                "left_arm": {"touch": "knee", "with": "palm"}}, {"t": 2.0}]}
        for obj in (elbow, knee):
            m, r, _ = refine.solve(obj, 2.0)
            self.assertNotIn("body_clip", check_motion(m, r))
        m, r, _ = refine.solve(elbow, 2.0)
        P, _ = sk.upper_fk(m, r, JIDX)
        L = sk.motion_leg_points(m, r, JIDX)
        self.assertLess(np.linalg.norm(P["rightLowerArm"][-1] - L["rightKnee"][-1]), 0.12)   # the elbow is at the knee

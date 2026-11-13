"""Keyframe generator. Reference = stick-figure FK: after solving, the bones must point where the
keyframe said, on the correct side of the body."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import keyframes as kf
from motion_loop import skeleton as sk
from motion_loop.filters import check_motion
from motion_loop.generators import degrade, degrade_root
from motion_loop.llm import LLM, MockLLM
from motion_loop.render import forward_kinematics
from motion_loop.rotations import angular_velocity, quat_to_mat, rotvec
from motion_loop.schema import FPS, HAND_SHAPES, J, JIDX


def bone(frame, a, b):
    pos, _ = forward_kinematics(frame)
    v = pos[JIDX[b]] - pos[JIDX[a]]
    return v / np.linalg.norm(v)


def head_axes(frame):
    R = np.eye(3)
    for n in ("spine", "chest", "upperChest", "neck", "head"):
        R = R @ quat_to_mat(frame[JIDX[n]])
    return R @ np.array([0, 0, 1.0]), R @ np.array([0, 1.0, 0])     # face direction, top of head


def motion_of(keyframes, dur=2.0):
    return kf.keyframes_to_motion(*kf.parse_keyframes({"keyframes": keyframes}, dur), dur)[0]


class Fixed(LLM):
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def complete(self, prompt, images=None, max_tokens=1500):
        self.calls += 1
        return self.replies.pop(0)


class Solve(unittest.TestCase):
    def test_arm_directions_are_reproduced_on_both_sides(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            u, f = kf._unit(rng.normal(size=3)), kf._unit(rng.normal(size=3))
            if not np.deg2rad(10) < np.arccos(np.clip(u @ f, -1, 1)) < kf.MAX_ELBOW:
                continue      # nearly straight arms use the previous hinge; over-bent ones are capped
            for side in ("right", "left"):
                m = motion_of([{"t": 0}, {"t": 1, f"{side}_arm": {"upper": u.tolist(), "fore": (3 * f).tolist()}}])
                fr = m[FPS]
                self.assertTrue(np.allclose(bone(fr, f"{side}UpperArm", f"{side}LowerArm"), u, atol=1e-4), (side, u))
                self.assertTrue(np.allclose(bone(fr, f"{side}LowerArm", f"{side}Hand"), f, atol=1e-4), (side, f))
                other = "left" if side == "right" else "right"
                self.assertTrue(np.allclose(m[:, JIDX[f"{other}UpperArm"]], m[0, JIDX[f"{other}UpperArm"]], atol=1e-5))

    def test_elbow_is_a_pure_hinge(self):
        m = motion_of([{"t": 0}, {"t": 1, "right_arm": {"upper": [-1, -0.3, 0], "fore": [0, 1, 0]},
                                  "left_arm": {"upper": [0.2, -0.5, 1], "fore": [-1, 0.2, 0.3]}}])
        for side, sign in (("right", 1), ("left", -1)):
            q = m[:, JIDX[f"{side}LowerArm"]]                    # rotation axis = (x, y, z) part
            self.assertTrue(np.allclose(q[:, [0, 2]], 0, atol=1e-6))
            self.assertTrue((sign * q[:, 1] >= 0).all())          # never bends backwards

    def test_head_and_torso_signs(self):
        fr = motion_of([{"t": 0}, {"t": 1, "head": {"nod": 30}}])[FPS]
        self.assertLess(head_axes(fr)[0][1], -0.3)                 # nod + looks down
        fr = motion_of([{"t": 0}, {"t": 1, "head": {"turn": 40}}])[FPS]
        self.assertGreater(head_axes(fr)[0][0], 0.3)               # turn + faces its left (+x)
        fr = motion_of([{"t": 0}, {"t": 1, "head": {"tilt": 25}}])[FPS]
        self.assertGreater(head_axes(fr)[1][0], 0.3)               # tilt + leans to its left
        fr = motion_of([{"t": 0}, {"t": 1, "torso": {"lean": 20}}])[FPS]
        self.assertGreater(bone(fr, "spine", "neck")[2], 0.15)     # lean + goes forward
        fr = motion_of([{"t": 0}, {"t": 1, "torso": {"bend": 20}}])[FPS]
        self.assertGreater(bone(fr, "spine", "neck")[0], 0.15)     # bend + goes to its left

    def test_poses_that_broke_the_euler_format_now_pass_the_filter(self):
        rest = {"upper": [-0.3, -1, 0], "fore": [-0.3, -1, 0.3]}
        for pose in ({"upper": [0, -1, 0], "fore": [1, 0, 0]},            # was at Euler gimbal lock
                     {"upper": [-0.2, 1, 0.1], "fore": [-0.1, 1, 0]},     # needed Euler x = 3.0
                     {"upper": [-0.6, 0.8, 0], "fore": [0.9, 0.3, -0.3]}):
            m = motion_of([{"t": 0, "right_arm": rest}, {"t": 1, "right_arm": pose}, {"t": 2, "right_arm": pose},
                           {"t": 3, "right_arm": rest}], 3.0)
            # (the first pose lays the forearm across the belly, which on the real body passes through it:
            # body_clip is right there; this test is about speeds and limits)
            self.assertEqual([r for r in check_motion(m) if r != "body_clip"], [], pose)
            self.assertLess(np.linalg.norm(angular_velocity(m.astype(float), FPS), axis=-1).max(), 6.0)

    def test_limits_are_clamped_and_parts_carry_over(self):
        _, t, rows = kf.parse_keyframes({"keyframes": [{"t": 0, "head": {"nod": 500}}, {"t": 1, "head": {"turn": 5}}]}, 2)
        self.assertEqual(rows[0][0], 40)
        self.assertEqual(rows[1][0], 40)

    def test_starts_at_rest_even_if_first_keyframe_is_late(self):
        m = motion_of([{"t": 0.5, "head": {"nod": 30}}, {"t": 1.5, "head": {"nod": 0}}])
        self.assertTrue(np.allclose(m[0, JIDX["head"]], [0, 0, 0, 1]))

    def test_shrug_lifts_the_shoulder_but_keeps_the_arm_direction(self):
        arm = {"upper": [0, -0.3, 1], "fore": [0, 0, 1]}
        for side, x in (("right", -1), ("left", 1)):
            m = motion_of([{"t": 0}, {"t": 1, f"{side}_arm": arm, "shoulders": {side: 20}},
                           {"t": 2, f"{side}_arm": arm, "shoulders": {side: 0}}], 2.0)
            up, down = m[FPS], m[2 * FPS]
            elbow_joint = lambda f: forward_kinematics(f)[0][JIDX[f"{side}UpperArm"]]
            self.assertGreater(elbow_joint(up)[1] - elbow_joint(down)[1], 0.01, side)      # shoulder tip goes up
            self.assertAlmostEqual(elbow_joint(up)[0], elbow_joint(down)[0], delta=0.02)    # and stays on its side
            for f in (up, down):
                self.assertTrue(np.allclose(bone(f, f"{side}UpperArm", f"{side}LowerArm"),
                                            np.array([0, -0.3, 1]) / np.linalg.norm([0, -0.3, 1]), atol=1e-3), side)
            other = "left" if side == "right" else "right"
            self.assertTrue(np.allclose(m[:, JIDX[f"{other}Shoulder"]], [0, 0, 0, 1]))

    def test_hand_shapes_follow_the_nearer_keyframe(self):
        kfs = {"keyframes": [{"t": 0}, {"t": 1, "hands": {"right": "point"}}, {"t": 2},
                             {"t": 3, "hands": {"right": "relaxed", "left": "fist"}}]}
        _, t, rows = kf.parse_keyframes(kfs, 3.0)
        h = kf.keyframes_to_hands(t, rows, 3.0)
        point, fist = HAND_SHAPES.index("point"), HAND_SHAPES.index("fist")
        self.assertEqual(h.shape, (3 * FPS + 1, 2))                      # columns: left, right
        self.assertEqual(h[0].tolist(), [0, 0])
        self.assertEqual(h[FPS].tolist(), [0, point])
        self.assertEqual(h[2 * FPS].tolist(), [0, point])               # carried over
        self.assertEqual(h[-1].tolist(), [fist, 0])
        self.assertEqual(h[int(0.4 * FPS), 1], 0)                        # switches halfway, not at the start
        self.assertEqual(h[int(0.6 * FPS), 1], point)
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"keyframes": [{"t": 0}, {"t": 1, "hands": {"left": "claw"}}]}, 2)

    def test_palm_faces_the_named_direction(self):
        """Palm normal = the hand's local -y carried through upper arm, lower arm and hand."""
        def palm(frame, side):
            R = np.eye(3)
            for n in ("UpperArm", "LowerArm", "Hand"):
                R = R @ quat_to_mat(frame[JIDX[side + n]])
            return R @ np.array([0, -1.0, 0])
        dirs = {"forward": [0, 0, 1], "back": [0, 0, -1], "up": [0, 1, 0], "down": [0, -1, 0]}
        for side, x in (("right", -1), ("left", 1)):
            for upper, fore in (([x, -0.3, 0], [0.2 * x, 1, 0]), ([0.4 * x, -1, 0.3], [0.3 * x, 0, 1])):
                for name, want in {**dirs, "in": [-x, 0, 0], "out": [x, 0, 0]}.items():
                    arm = {"upper": upper, "fore": fore, "palm": name}
                    m = motion_of([{"t": 0}, {"t": 1, f"{side}_arm": arm}, {"t": 2, f"{side}_arm": arm}], 2.0)
                    f = np.array(fore, float) / np.linalg.norm(fore)
                    reach = np.array(want, float) - f * np.dot(want, f)       # the part a twist can reach
                    if np.linalg.norm(reach) < 0.5:
                        continue
                    got = palm(m[-1].astype(float), side)
                    cos = np.dot(got, reach / np.linalg.norm(reach))
                    self.assertGreater(cos, np.cos(np.deg2rad(25)), (side, fore, name, cos))   # a half turn stops at 160 deg
                    self.assertAlmostEqual(abs(np.dot(got, f)), 0, delta=0.05)              # a pure twist
                    self.assertEqual(check_motion(m), [], (side, name))

    def test_wrist_bends_toward_the_palm_and_natural_palm_changes_nothing(self):
        arm = {"upper": [-1, -0.3, 0], "fore": [-0.2, 1, 0]}
        base = motion_of([{"t": 0}, {"t": 1, "right_arm": arm}], 1.0)
        nat = motion_of([{"t": 0}, {"t": 1, "right_arm": {**arm, "palm": "natural"}}], 1.0)
        self.assertTrue(np.allclose(base, nat))
        self.assertTrue(np.allclose(base[-1, JIDX["rightHand"]], [0, 0, 0, 1]))
        for side, a2 in (("right", arm), ("left", {"upper": [1, -0.3, 0], "fore": [0.2, 1, 0]})):
            m = motion_of([{"t": 0}, {"t": 1, f"{side}_arm": {**a2, "wrist": 40}}], 1.0)
            Rh = quat_to_mat(m[-1, JIDX[side + "Hand"]].astype(float))
            fingers = Rh @ np.array([-1.0 if side == "right" else 1.0, 0, 0])
            self.assertLess(fingers[1], -0.5, side)                  # fingers swing toward the palm side (-y)


def leg_motion(obj, dur):
    posture, t, rows = kf.parse_keyframes(obj, dur)
    return kf.keyframes_to_motion(posture, t, rows, dur)


class Legs(unittest.TestCase):
    def test_planted_foot_stays_put_while_the_other_steps(self):
        for legs, moving in (({"right_foot": {"place": "back"}}, "right"), ({"left_foot": {"place": "forward"}}, "left"),
                             ({"left_foot": {"lift": "knee_up"}}, "left"), ({"right_foot": {"lift": "kick"}}, "right")):
            m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "legs": legs}, {"t": 2, "legs": legs}, {"t": 3}]}, 3.0)
            pts = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)
            still = "left" if moving == "right" else "right"
            a = pts[f"{still}Ankle"]
            self.assertLess(np.abs(a - a[0]).max(), 0.005, legs)                  # the standing foot does not slide
            self.assertLess(abs(a[0, 1] - sk.ANKLE_REST_Y), 0.005)                # and is on the floor
            self.assertGreater(np.abs(pts[f"{moving}Ankle"] - pts[f"{moving}Ankle"][0]).max(), 0.08, legs)
            self.assertEqual(check_motion(m, root), [], legs)

    def test_one_foot_always_on_the_floor_when_feet_swap(self):
        """Left foot up, then right foot up: the left must be down before the right leaves the floor."""
        for lift in ("lift", "knee_up", "kick"):
            m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "legs": {"left_foot": {"lift": lift}}},
                                               {"t": 2, "legs": {"left_foot": {"lift": "down"}, "right_foot": {"lift": lift}}},
                                               {"t": 3, "legs": {"left_foot": {"place": "forward"}, "right_foot": {"lift": "down"}}},
                                               {"t": 4}]}, 4.0)
            pts = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)
            low = np.minimum(pts["leftAnkle"][:, 1], pts["rightAnkle"][:, 1])
            self.assertLess(low.max(), sk.ANKLE_REST_Y + 0.01, lift)
            self.assertEqual(check_motion(m, root), [], lift)

    def test_hands_inside_the_chest_are_caught(self):
        """Forearms folded across in front of the chest pass; pulled back through it they are rejected."""
        def chest(fore_z):
            arm = lambda sx: {"upper": [0.3 * sx, -1, 0.25], "fore": [-sx, 0.4, fore_z]}
            return leg_motion({"keyframes": [{"t": 0}, {"t": 1, "right_arm": arm(-1), "left_arm": arm(1)},
                                             {"t": 2, "right_arm": arm(-1), "left_arm": arm(1)}, {"t": 3}]}, 3.0)
        m, root = chest(1.0)                        # hands a fist's width in front of the chest
        self.assertNotIn("body_clip", check_motion(m, root))
        m, root = chest(-0.2)                       # forearms pulled back through the chest
        self.assertIn("body_clip", check_motion(m, root))

    def test_hands_through_the_legs_are_caught(self):
        """knees_hugged: the rest pose (elbows on the knees) passes; forearms pulled back through the shins fail."""
        rest = {"posture": "knees_hugged", "keyframes": [{"t": 0}, {"t": 2}]}
        m, root = leg_motion(rest, 2.0)
        self.assertNotIn("body_clip", check_motion(m, root))
        arm = lambda sx: {"upper": [0.3 * sx, -0.6, 0.75], "fore": [-sx, -0.1, 0.4]}
        m, root = leg_motion({"posture": "knees_hugged", "keyframes": [{"t": 0}, {"t": 1, "right_arm": arm(-1), "left_arm": arm(1)},
                                                                    {"t": 2, "right_arm": arm(-1), "left_arm": arm(1)}, {"t": 3}]}, 3.0)
        self.assertIn("body_clip", check_motion(m, root))

    def test_hands_through_each_other_are_caught(self):
        """Both forearms pulled in to the middle (the author saw the fingers go through each other) fail; hands a
        little apart pass."""
        def hands(x):
            arm = lambda sx: {"upper": [0.2 * sx, -0.9, 0.3], "fore": [-x * sx, 0.2, 0.8], "palm": "in"}
            return leg_motion({"posture": "legs_out", "keyframes": [{"t": 0}, {"t": 0.8, "right_arm": arm(-1), "left_arm": arm(1)},
                                                                  {"t": 2, "right_arm": arm(-1), "left_arm": arm(1)}, {"t": 3}]}, 3.0)
        self.assertIn("hands_clip", check_motion(*hands(0.5)))
        self.assertNotIn("hands_clip", check_motion(*hands(0.15)))

    def test_tiptoe_raises_the_heels_and_keeps_the_toes_down(self):
        toes = {"left_foot": {"lift": "heel"}, "right_foot": {"lift": "heel"}}     # was "height": "tiptoe" (removed)
        m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "legs": toes}, {"t": 2, "legs": toes},
                                           {"t": 3, "legs": {"left_foot": {"lift": "down"}, "right_foot": {"lift": "down"}}}]}, 3.0)
        pts = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)
        for s in ("left", "right"):
            self.assertGreater(pts[f"{s}Ankle"][45, 1], sk.ANKLE_REST_Y + 0.035)
            self.assertLess(abs(pts[f"{s}Toes"][45, 1] - pts[f"{s}Toes"][0, 1]), 0.01)
        self.assertGreater(root[45, 1] * sk.HIPS_Y, 0.03)
        self.assertEqual(check_motion(m, root), [])

    def test_turn_lift_directions_and_lunges(self):
        """New leg words (2026-10-23): the body turns with its feet; side / back lifts go that way; lunges drop the hips."""
        def at(legs, i=36):
            m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "legs": legs}, {"t": 2, "legs": legs}, {"t": 3}]}, 3.0)
            self.assertEqual(check_motion(m, root), [], legs)
            return sk.motion_leg_points(m.astype(float), root.astype(float), JIDX), root[i] * sk.HIPS_Y, m[i]
        pts, hips, q = at({"turn": 90})
        # turned toward its left: the left foot is now behind (-z), the right one in front
        self.assertLess(pts["leftAnkle"][36, 2], pts["rightAnkle"][36, 2] - 0.1)
        self.assertGreater(abs(rotvec(q[JIDX["hips"]][None])[0, 1]), 1.4)
        pts, _, _ = at({"right_foot": {"lift": "kick", "dir": "side"}})
        self.assertLess(pts["rightAnkle"][36, 0], -0.35)                  # out to its right (-x)
        pts, _, _ = at({"left_foot": {"lift": "kick", "dir": "back"}})
        self.assertLess(pts["leftAnkle"][36, 2], -0.3)
        _, hips, _ = at({"right_foot": {"place": "lunge_forward"}})
        self.assertLess(hips[1], -0.12)
        _, hips, _ = at({"left_foot": {"place": "lunge_out"}, "weight": "left"})
        self.assertGreater(hips[0], 0.25)

    def test_key_points_follow_the_arm_and_the_hips(self):
        """Reward-model position features: a palm raised overhead ends above the head; squatting moves the knees forward."""
        m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "right_arm": {"upper": [0, 1, 0], "fore": [0, 1, 0]},
                                                       "legs": {"height": "squat"}}, {"t": 2}]}, 2.0)
        k = sk.key_points(m.astype(float), root.astype(float), JIDX)
        i = {n: sk.KEY_POINTS.index(n) for n in ("head", "rightPalm", "leftKnee")}
        self.assertTrue(np.allclose(k[0, i["head"]], sk.UPPER_REST["head"], atol=0.02))
        self.assertGreater(k[FPS, i["rightPalm"], 1], k[FPS, i["head"], 1] + 0.2)
        self.assertGreater(k[FPS, i["leftKnee"], 2], k[0, i["leftKnee"], 2] + 0.1)        # knees go forward
        from motion_loop.reward_model import FEAT, POS_FEAT, featurize
        self.assertEqual(featurize(m, root, pos=True)[0].shape[1], FEAT + POS_FEAT)

    def test_squat_lowers_the_hips_and_bends_the_knees_forward(self):
        m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "legs": {"height": "squat"}}, {"t": 2}]}, 2.0)
        self.assertLess(root[FPS, 1] * sk.HIPS_Y, -0.25)
        knee = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)["leftKnee"]
        self.assertGreater(knee[FPS, 2] - knee[0, 2], 0.1)                        # knee moves forward, not back
        self.assertEqual(check_motion(m, root), [])

    def test_floor_sitting_poses_touch_the_floor_and_pass_the_filter(self):
        for name in sk.SIT_NAMES:
            m, root = leg_motion({"posture": name, "keyframes": [{"t": 0}, {"t": 1, "head": {"nod": 20}}, {"t": 2}]}, 2.0)
            self.assertTrue(np.allclose(root, root[0]), name)                     # legs and hips stay put
            self.assertLess(root[0, 1] * sk.HIPS_Y, -0.4, name)
            R, hips = sk.sit_pose(name)
            p = sk.leg_points(R, hips)
            low = min(p["butt"][1], *(p[f"{s}Knee"][1] - sk.R_KNEE for s in ("left", "right")))
            if name == sk.LEDGE:                    # on a ledge only the bottom rests on it, the legs hang below
                low = p["butt"][1]
            self.assertLess(abs(low), 0.06, name)                                 # resting on the floor
            self.assertEqual(check_motion(m, root, posture=name), [], name)

    def test_leg_words_are_checked_and_one_foot_stays_down(self):
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"posture": "lying", "keyframes": [{"t": 0}, {"t": 1}]}, 2)
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"keyframes": [{"t": 0}, {"t": 1, "legs": {"height": "tall"}}]}, 2)
        _, _, rows = kf.parse_keyframes({"keyframes": [{"t": 0}, {"t": 1, "legs": {
            "left_foot": {"lift": "lift"}, "right_foot": {"lift": "kick"}}}]}, 2)
        self.assertEqual(rows[1, kf.LEG_AT + 6], 0)                               # no jumping

    def test_frozen_legs_under_moving_hips_are_rejected(self):
        m, root = leg_motion({"keyframes": [{"t": 0}, {"t": 1, "legs": {"height": "squat"}}, {"t": 2}]}, 2.0)
        self.assertIn("below_floor", check_motion(degrade(m, "freeze_legs", None), root))
        self.assertTrue(np.allclose(degrade_root(root, "mirror")[:, 0], -root[:, 0]))


class Generator(unittest.TestCase):
    def test_mock_motions_pass_the_filter(self):
        g = kf.KeyframeGenerator(MockLLM(0))
        seen = set()
        for i in range(24):
            m = g.generate(f"instruction {i}", i, 3.0)
            self.assertEqual(m.shape, (91, J, 4))
            self.assertEqual(check_motion(m, g.last_root), [], g.last_keyframes)
            seen.add(json.dumps(g.last_keyframes))
        self.assertGreater(len(seen), 2)

    def test_retries_then_fails(self):
        good = json.dumps({"keyframes": [{"t": 0}, {"t": 1, "head": {"nod": 10}}]})
        llm = Fixed(["not json", '{"keyframes": [{"t": 1}, {"t": 0.5}]}', good])
        self.assertEqual(kf.KeyframeGenerator(llm, retries=2).generate("x", 0, 2.0).shape[1:], (J, 4))
        self.assertEqual(llm.calls, 3)
        with self.assertRaises(RuntimeError):
            kf.KeyframeGenerator(Fixed(["nope", "nope"]), retries=1).generate("x", 0, 2.0)

    def test_parallel_candidates_match_serial_and_keep_keyframes(self):
        import tempfile
        from motion_loop.generators import make_candidates
        ins = [{"id": f"i{i}", "text": f"instruction {i}", "spec": {"duration_s": 2.0}} for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            out = {}
            for w in (1, 8):
                cands = make_candidates(ins, kf.KeyframeGenerator(MockLLM(0)), Path(d) / f"w{w}" / "motions",
                                        seed=3, workers=w)
                out[w] = [(c["source"], np.load(Path(d) / f"w{w}" / c["motion_path"]).tobytes()) for c in cands]
                for c in cands:
                    rec = json.loads((Path(d) / f"w{w}" / c["keyframes_path"]).read_text())
                    self.assertIn("keyframes", rec["keyframes"])
            self.assertEqual(out[1], out[8])



class JumpTest(unittest.TestCase):
    """Jumping in place (2026-11-05): the solver adds crouch / flight / landing, the filter allows floating only in a
    free-fall flight, the feet land where they took off."""

    def solve(self, jump, end=2.2, posture="stand"):
        from motion_loop import refine
        obj = {"posture": posture, "keyframes": [{"t": 0.0}, {"t": 0.3, "legs": {"jump": jump}}, {"t": end}]}
        return refine.solve(obj, end + 0.3)[:2]

    def test_jumps_pass_and_leave_the_floor(self):
        from motion_loop import filters
        for jump, h in (("small", 0.10), ("big", 0.25), ({"size": "big", "pose": "tuck"}, 0.25),
                        ({"size": "small", "count": 3}, 0.10)):
            m, root = self.solve(jump, 2.6)
            self.assertEqual(check_motion(m, root), [], jump)
            rise = root[:, 1].max() * sk.HIPS_Y
            self.assertAlmostEqual(rise, kf.JUMP_EXT + h, delta=0.02)
            self.assertTrue(filters.jump_window(m, root).any())
            self.assertLess(np.abs(root[-1] - root[0]).max(), 1e-3)  # standing at rest again

    def test_floating_without_free_fall_is_still_rejected(self):
        m, root = self.solve("big")
        r2 = root.copy()
        y = r2[:, 1] * sk.HIPS_Y
        a, b = np.where(y > kf.JUMP_EXT + 0.05)[0][[0, -1]]
        r2[a:b + 1, 1] = r2[a, 1]                                  # hold the body up in the air: no free fall
        self.assertIn("floating", check_motion(m, r2))

    def test_jump_rules(self):
        with self.assertRaises(ValueError):                        # the next keyframe before landing
            self.solve("big", end=1.0)
        with self.assertRaises(ValueError):                        # sitting
            self.solve("small", posture="seiza")
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"keyframes": [{"t": 0.3, "legs": {"jump": "small", "left_foot": {"lift": "lift"}}},
                                              {"t": 2.0}]}, 2.0)
        _, _, rows = kf.parse_keyframes({"keyframes": [{"t": 0.3, "legs": {"jump": "small"}}, {"t": 2.0}]}, 2.3)
        self.assertEqual(rows[-1, kf.JUMP_AT], 0)                  # not carried forward

if __name__ == "__main__":
    unittest.main()


class HipSwayTest(unittest.TestCase):
    def test_hip_sway_tilts_the_pelvis_and_brings_the_chest_back(self):
        """'hip' (2026-10-28): weight right -> left hip lower, hips out past the right foot, chest back over it,
        the free heel up with the ball of the foot down, and the motion passes the filter."""
        from motion_loop import refine
        obj = {"posture": "stand", "keyframes": [{"t": 0.0}, {"t": 1.2, "legs": {
            "weight": "right", "hip": "far_out", "left_foot": {"lift": "heel"}}}, {"t": 2.0}]}
        m, root, _ = refine.solve(obj, 2.0)
        self.assertEqual(check_motion(m, root), [])
        x = quat_to_mat(m[-1, JIDX["hips"]]) @ [1.0, 0, 0]
        self.assertLess(np.degrees(np.arcsin(x[1])), -10)              # the left (free) hip is lower
        P, _ = sk.upper_fk(m[-1:], root[-1:], JIDX)
        self.assertLess(P["hips"][0, 0], kf.WEIGHT_X["right"] - 0.03)  # the hips go further out than a plain weight shift
        self.assertGreater(P["upperChest"][0, 0], P["hips"][0, 0] + 0.03)   # the chest comes back
        with self.assertRaises(ValueError):                             # needs a weight foot
            kf.parse_keyframes({"posture": "stand", "keyframes": [{"t": 0.0}, {"t": 1.0, "legs": {"hip": "out"}}]}, 1.0)


    def test_hip_sway_in_degrees(self):
        """'hip' may be degrees of pelvis tilt (2026-10-29, T015): 4 tilts the pelvis 4 deg, a word equals its degrees."""
        from motion_loop import refine
        def tilt(hip):
            obj = {"posture": "stand", "keyframes": [{"t": 0.0}, {"t": 1.2, "legs": {"weight": "right", "hip": hip}},
                                                     {"t": 2.0}]}
            m, root, _ = refine.solve(obj, 2.0)
            self.assertEqual(check_motion(m, root), [])
            x = quat_to_mat(m[-1, JIDX["hips"]]) @ [1.0, 0, 0]
            return -np.degrees(np.arcsin(x[1]))
        self.assertAlmostEqual(tilt(4), 4.0, delta=0.5)
        self.assertAlmostEqual(tilt("out"), tilt(8), delta=0.1)


class HandsMeetTest(unittest.TestCase):
    def test_hands_meet_above_the_head(self):
        """hands_meet (2026-10-28): palms pressed together straight up (S011) and fingertips meeting over the head
        (U002): the hands end up close, on the midline, above the head, without the hands_clip rule firing."""
        from motion_loop import refine
        for how, at in (("palms", "above_head far"), ("fingertips", "above_head beside")):
            obj = {"posture": "stand", "keyframes": [{"t": 0.0}, {"t": 1.2, "hands_meet": {"at": at, "how": how}}, {"t": 2.0}]}
            m, root, hands = refine.solve(obj, 2.0)
            self.assertEqual(check_motion(m, root), [], how)
            P, G = sk.upper_fk(m[-1:], root[-1:], JIDX)
            tip = lambda side, sx: P[f"{side}Hand"][0] + G[f"{side}Hand"][0] @ [sx * sk.FINGERTIP, 0, 0]
            palm = lambda side, sx: P[f"{side}Hand"][0] + G[f"{side}Hand"][0] @ [sx * sk.PALM, 0, 0]
            pt = tip if how == "fingertips" else palm
            a, b = pt("left", 1.0), pt("right", -1.0)
            self.assertLess(np.linalg.norm(a - b), 0.045, how)
            self.assertLess(abs(a[0] + b[0]) / 2, 0.02, how)
            self.assertGreater(min(a[1], b[1]), P["head"][0, 1] + 0.1, how)
            self.assertEqual(list(hands[-1]), [HAND_SHAPES.index("flat" if how == "palms" else "open")] * 2)   # joined palms: flat (2026-11-06)
        with self.assertRaises(ValueError):
            refine.solve({"posture": "stand", "keyframes": [{"t": 0.0}, {"t": 1.0, "hands_meet": {"at": "chest"}}]}, 1.0)


class StarredFixesTest(unittest.TestCase):
    """2026-10-28: kneeling hip sway, knee_to_chest, wrist / forearm / elbow_on touches, the removed tiptoe word."""

    def solve(self, posture, pose, model=None):
        from motion_loop import refine
        return refine.solve({"posture": posture, "keyframes": [{"t": 0.0}, {"t": 1.5, **pose}, {"t": 2.3}]}, 2.3, model=model)

    def test_kneeling_sway_keeps_the_knees_and_caps_the_tilt(self):
        R0, h0, _ = sk.kneel_sway(0.0)
        P0 = sk.leg_points(R0, h0)
        for knees in ("default", "apart"):
            R, h, a = sk.kneel_sway(0.04, knees)
            P = sk.leg_points(R, h)
            for s in ("left", "right"):
                self.assertLess(abs(P[f"{s}Knee"][1] - P0[f"{s}Knee"][1]), 0.005)   # bones tilt a few deg off -y
        m, r, _ = self.solve("kneel", {"legs": {"weight": "left", "hip": "far_out", "knees": "apart"}})
        self.assertEqual(check_motion(m, r), [])
        tilt = np.degrees(np.arcsin((quat_to_mat(m[-1, JIDX["hips"]]) @ [1.0, 0, 0])[1]))
        self.assertLessEqual(abs(tilt), kf.HIP_KNEEL_DEG["far_out"] + 0.5)

    def test_new_touches_pass_the_filter(self):
        cases = [("legs_out", {"legs": {"right": "knee_to_chest"}, "right_arm": {"touch": "knee", "with": "palm"}}),
                 ("legs_out", {"legs": {"left": "knee_up"}, "left_arm": {"touch": "knee", "with": "forearm"}}),
                 ("legs_out", {"legs": {"right": "knee_up"}, "right_arm": {"touch": "cheek", "with": "wrist", "elbow_on": "knee"},
                               "torso": {"lean": 10}}),
                 ("stand", {"legs": {"weight": "left", "right_foot": {"lift": "knee_to_chest"}}})]
        for posture, pose in cases:
            m, r, _ = self.solve(posture, pose)
            self.assertEqual(check_motion(m, r), [], pose)

    def test_tiptoe_word_is_gone(self):
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"posture": "stand", "keyframes": [{"t": 0.0}, {"t": 1.0, "legs": {"height": "tiptoe"}}]}, 1.0)

    def test_check_motion_uses_the_given_model(self):
        m, r, _ = self.solve("legs_out", {"legs": {"left": "knee_up"}}, model="test2")
        with sk.use_model("test2"):
            inside = check_motion(m, r)
        self.assertEqual(check_motion(m, r, model="test2"), inside)


class SittingLegWordsTest(unittest.TestCase):
    """Sitting leg words (2026-10-28): each leg may change shape, the knees may open, the bottom stays on the floor."""

    def test_leg_words_keep_the_bottom_down_and_the_legs_apart(self):
        from motion_loop import refine, skeleton as sk
        from motion_loop.filters import check_legs
        from motion_loop.schema import JIDX
        for posture, legs in (("legs_out", {"left": "knee_up"}), ("cross_legged", {"right": "knee_up"}),
                              ("knees_hugged", {"knees": "wide"}), ("legs_out", {"right": "cross"}),
                              ("cross_legged", {"left": "out"})):
            obj = {"posture": posture, "keyframes": [{"t": 0, "left_arm": {"hand": "shoulder wide beside"},   # clear of the knees
                                                     "right_arm": {"hand": "shoulder wide beside"}},
                                                    {"t": 1.2, "legs": legs}, {"t": 2.0}]}
            m, r, _ = refine.solve(obj, 2.0)
            self.assertEqual([x for x in check_legs(m, r) if x != "body_clip"], [], (posture, legs))
            self.assertLessEqual(sk.legs_overlap(m, r, JIDX).max(), 0.0, (posture, legs))
        from motion_loop import skeleton as sk
        # the posture's own legs give the old pose exactly
        R0, h0 = sk.sit_pose("cross_legged")
        R1, h1 = sk.sit_pose("cross_legged", "default", "default", "default")
        np.testing.assert_allclose(h0, h1)

    def test_sitting_leg_words_need_a_sitting_posture(self):
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"posture": "seiza", "keyframes": [{"t": 0}, {"t": 1, "legs": {"left": "out"}}]}, 1.0)
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"posture": "legs_out", "keyframes": [{"t": 0}, {"t": 1, "legs": {"left": "flying"}}]}, 1.0)


class EyesTrackTest(unittest.TestCase):
    def test_eyes_closed_is_carried_forward_and_faded(self):
        from motion_loop import keyframes as kf
        obj = {"posture": "stand", "keyframes": [{"t": 0}, {"t": 0.5, "eyes": "closed"}, {"t": 1.5}, {"t": 2.0, "eyes": "open"}]}
        e = kf.eyes_track(obj, 2.5)
        self.assertEqual(len(e), 76)
        self.assertEqual(e[0], 0.0); self.assertEqual(e[45], 1.0); self.assertEqual(e[-1], 0.0)   # carried past 1.5 s
        self.assertTrue(0 < e[15] < 1)                                                      # a ramp, not a step
        self.assertIsNone(kf.eyes_track({"posture": "stand", "keyframes": [{"t": 0}, {"t": 1}]}, 1.0))
        with self.assertRaises(ValueError):
            kf.eyes_track({"keyframes": [{"t": 0}, {"t": 1, "eyes": "shut"}]}, 1.0)


class WristSideTest(unittest.TestCase):
    """wrist_side (2026-11-08): the hand bends sideways at the wrist, + = toward the little finger. A raised hand with
    the palm forward: the right hand's little finger is on its outer side, so + moves the fingertip outward."""

    def test_sideways_bend_moves_the_fingertip_toward_the_little_finger(self):
        from motion_loop import refine, skeleton as sk
        from motion_loop.filters import check_motion
        for side, sx in (("right", -1.0), ("left", 1.0)):
            tips = []
            for ws in (-25, 25):
                arm = {"upper": [sx * 0.3, -0.2, 0.93], "fore": [0, 1, 0], "palm": "forward", "wrist_side": ws}
                m, r, _ = refine.solve({"posture": "stand", "keyframes": [{"t": 0}, {"t": 1, f"{side}_arm": arm}, {"t": 2}]}, 2.0)
                self.assertEqual(check_motion(m, r), [])
                P, G = sk.upper_fk(m[30:31].astype(float), r[30:31].astype(float), JIDX)
                W = P[f"{side}Hand"][0]
                tips.append(G["upperChest"][0].T @ (W + G[f"{side}Hand"][0] @ np.array([sx * 0.15, 0, 0]) - W))
            self.assertGreater(sx * (tips[1][0] - tips[0][0]), 0.1)      # + = outward for a raised palm-forward hand
        with self.assertRaises(ValueError):
            kf.parse_keyframes({"keyframes": [{"t": 0}, {"t": 1, "right_arm": {"wrist_side": "x"}}]}, 1.0)


if __name__ == "__main__":
    unittest.main()

"""Policy option B (keyframe policy): the row <-> token <-> JSON round trip is exact, and a tiny SFT run
gives samples that parse_keyframes reads and the solver turns into motions."""
import json
import sys
import tempfile
import unittest
raise unittest.SkipTest('policy_kf is frozen: the keyframe row layout changed on 2026-11-08 (research track aborted 2026-11-05)')
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import keyframes as kf
from motion_loop import policy_kf as pk

WAVE = {"posture": "stand", "keyframes": [
    {"t": 0.0},
    {"t": 0.5, "right_arm": {"upper": [-0.3, 0.6, 0.3], "fore": [0, 1, 0.1], "palm": "forward"}, "hands": {"right": "open"},
     "legs": {"weight": "left", "height": "bent", "turn": 20, "right_foot": {"lift": "knee_up", "dir": "side"}}},
    {"t": 0.9, "right_arm": {"fore": [-0.4, 1, 0.1]}, "head": {"tilt": 10}, "shoulders": {"right": 15}},
    {"t": 1.3, "right_arm": {"fore": [0.3, 1, 0.1], "wrist": -20}},
    {"t": 2.0, "right_arm": {"upper": [-0.3, -1, 0], "fore": [-0.3, -1, 0.3], "palm": "natural"}, "hands": {"right": "relaxed"},
     "legs": {"weight": "center", "height": "straight", "turn": 0, "right_foot": {"lift": "down"}}}]}
NOD = {"posture": "seiza", "keyframes": [{"t": 0.0}, {"t": 0.4, "head": {"nod": 20}}, {"t": 0.8, "head": {"nod": -5}},
                                         {"t": 1.2, "head": {"nod": 0}, "left_arm": {"palm": "up", "wrist": 10}}]}


class PolicyKfTest(unittest.TestCase):
    def test_round_trip(self):
        for obj, dur in ((WAVE, 2.0), (NOD, 1.5)):
            p, t, rows = kf.parse_keyframes(obj, dur)
            cont, cats = pk.encode_rows(t, rows, dur)
            t2, rows2 = pk.decode_rows(p, cont, cats, dur)
            p3, t3, rows3 = kf.parse_keyframes(pk.to_keyframes_json(p, t2, rows2), dur)
            self.assertEqual(p3, p)
            np.testing.assert_allclose(t3, t, atol=1e-3)
            for at in (6, 9, 12, 15):                    # rest rows inserted by the parser are not unit length
                rows[:, at:at + 3] = kf._unit(rows[:, at:at + 3])
            np.testing.assert_allclose(rows3, rows, atol=0.06)       # JSON rounds to 0.1 deg / 0.001

    def test_gate_targets_mark_only_the_parts_that_change(self):
        p, t, rows = kf.parse_keyframes(WAVE, 2.0)
        g = pk.gate_targets(p, rows)
        names = list(pk.GROUPS)
        self.assertFalse(g[:, names.index("left_arm")].any())          # the left arm never moves
        self.assertFalse(g[:, names.index("torso")].any())
        np.testing.assert_array_equal(g[:, names.index("right_arm")], [False, True, True, True, True])
        np.testing.assert_array_equal(g[:, names.index("head")], [False, False, True, False, False])

    def test_tiny_sft_samples_parse_and_solve(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            lines = []
            for i, (obj, dur, text) in enumerate([(WAVE, 2.0, "wave hello with the right hand"),
                                                  (NOD, 1.5, "kneeling, nod twice")] * 4):
                kp = d / f"{i}.keyframes.json"
                kp.write_text(json.dumps({"instruction": text, "duration_s": dur, "keyframes": obj}))
                lines.append(json.dumps({"id": f"c{i}", "instruction_id": f"i{i}", "text": text, "keyframes_path": str(kp),
                                         "split": "val" if i >= 6 else "train"}))
            (d / "sft.jsonl").write_text("\n".join(lines))
            res = pk.train_policy(d / "sft.jsonl", d / "p.pt", epochs=3, batch=4, encoder="hash", device="cpu", log=lambda *a: None)
            self.assertEqual((res["n_train"], res["n_val"]), (6, 2))
            s = pk.KeyframePolicySampler(d / "p.pt")
            for temp in (0.0, 1.0):
                out = s.sample(["wave hello", "nod"], [2.0, 1.5], n=3, temperature=temp)
                self.assertEqual([len(o) for o in out], [3, 3])
                for objs, dur in zip(out, (2.0, 1.5)):
                    for obj in objs:
                        p, t, rows = kf.parse_keyframes(obj, dur)
                        self.assertTrue(pk.MIN_K <= len(t) <= pk.MAX_K + 1)
                        motion, root = kf.keyframes_to_motion(p, t, rows, dur)
                        self.assertTrue(np.isfinite(motion).all() and np.isfinite(root).all())


if __name__ == "__main__":
    unittest.main()

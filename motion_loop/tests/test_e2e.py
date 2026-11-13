"""End-to-end test with mock components (no API key, no GPU). Run: python -m unittest -v"""
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion_loop import generators, human_eval, pipeline, scoring
from motion_loop.schema import J, read_jsonl


class ParseTests(unittest.TestCase):
    def test_parse_scores_ok_and_bad(self):
        good = json.dumps({"candidates": {l: {"reason": "r", "match": 3, "natural": 4, "appropriate": 5,
                                              "flags": ["jitter", "bogus"]} for l in "AB"},
                           "ranking": ["B", "A"]})
        r = scoring.parse_scores("```json\n" + good + "\n```", ["A", "B"])
        self.assertEqual(r["B"]["rank"], 1)
        self.assertEqual(r["A"]["flags"], ["jitter"])  # unknown flags dropped
        self.assertIsNone(scoring.parse_scores(good, ["A", "B", "C"]))          # missing label
        self.assertIsNone(scoring.parse_scores("not json", ["A", "B"]))
        bad = json.loads(good); bad["candidates"]["A"]["match"] = 9
        self.assertIsNone(scoring.parse_scores(json.dumps(bad), ["A", "B"]))    # out of range

    def test_degrade_shapes(self):
        m = generators.MockGenerator().generate("x", 0, 2.0)
        rng = np.random.default_rng(0)
        for k in generators.DEGRADE_KINDS:
            out = generators.degrade(m, k, rng)
            self.assertEqual(out.shape[1:], (J, 4))
            self.assertTrue(np.allclose(np.linalg.norm(out, axis=-1), 1, atol=1e-5), k)
        self.assertEqual(len(generators.degrade(m, "speed_slow", rng)), 2 * len(m))

    def test_degraded_hands_keep_the_motion_length_and_swap_on_mirror(self):
        rng = np.random.default_rng(0)
        m = generators.MockGenerator().generate("wave", 0, 2.0)
        hands = np.zeros((len(m), 2), np.uint8)
        hands[:, 1] = 3                                              # right hand points
        for kind in generators.DEGRADE_KINDS:
            h = generators.degrade_hands(hands, kind)
            self.assertEqual(len(h), len(generators.degrade(m, kind, rng)), kind)
            self.assertEqual(h[0].tolist(), [3, 0] if kind == "mirror" else [0, 3], kind)


class E2E(unittest.TestCase):
    def test_full_mock_run(self):
        with tempfile.TemporaryDirectory() as d:
            # Only defects visible in the motion itself (jitter, frozen arm, out-of-limits): the mock text
            # encoder has no semantics, so text-mismatch negatives cannot be learned in this test.
            base = ["--workdir", d, "--n", "100", "--epochs", "12", "--repeats", "2", "--n-human", "10",
                    "--p-degrade", "0.5", "--p-mismatch", "0", "--degrade-kinds", "noise,freeze_arm,extreme",
                    "--margin", "2", "--d-model", "32", "--layers", "1", "--batch", "32"]
            for s in ["sample", "instruct", "candidates", "filter", "render", "score"]:
                pipeline.main([s] + base)
            w = Path(d)
            cands = read_jsonl(w / "candidates.jsonl")
            self.assertTrue(any(c["filter"] == "reject" for c in cands))     # 'extreme' gets caught
            self.assertTrue(len(list((w / "grids").glob("*.png"))) > 100)
            sm = json.loads((w / "score_metrics.json").read_text())
            self.assertGreater(sm["rank_consistency"], 0.5)

            pipeline.main(["humaneval-export"] + base)
            with open(w / "human_eval" / "labels.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            key = json.loads((w / "human_eval" / "key.json").read_text())
            src = {c["id"]: c for c in cands}
            bad = lambda c: (c["source"] != "normal") + (c["source"] == "mismatched")
            by_group = {}
            for r in rows:   # oracle "human": normal candidates rank above degraded/mismatched
                by_group.setdefault(r["group_id"], []).append(r)
            for g in by_group.values():
                order = sorted(g, key=lambda r: bad(src[key[r["group_id"]][r["label"]]]))
                for rank, r in enumerate(order, 1):
                    r["rank"] = str(rank)
            with open(w / "human_eval" / "labels.csv", "w", newline="", encoding="utf-8") as f:
                wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)

            pipeline.main(["pairs"] + base)
            human = set(json.loads((w / "human_set.json").read_text())["ids"])
            pairs = read_jsonl(w / "pairs.jsonl")
            self.assertFalse(human & {p["instruction_id"] for p in pairs})   # human set never trained on
            pipeline.main(["train"] + base)
            m = json.loads((w / "reward_model_metrics.json").read_text())
            print("reward model val pair acc:", m["best_val_pair_acc"])
            self.assertGreater(m["best_val_pair_acc"], 0.75)
            res = human_eval.score_human(w, w / "human_eval" / "labels.csv", w / "reward_model.pt")
            print("human agreement:", res)
            self.assertGreater(res["lmm_vs_human"], 0.6)
            # reward_model_vs_human is not checked: mock scorer and mock "human" are the same formula on
            # random motions with ~40 pairs, so the value means nothing here (the author, 2026-10-13).


if __name__ == "__main__":
    unittest.main()

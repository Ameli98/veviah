import unittest

from motion_loop.prefs import build_pairs


def sc(cid, natural, overall):
    return {"cand_id": cid, "instruction_id": "i", "natural": natural, "overall": overall}


class Pairs(unittest.TestCase):
    def test_two_broken_are_not_compared(self):
        scores = [sc("good", 5, 14), sc("bad1", 2, 7), sc("bad2", 1, 4)]
        pairs = {(p["winner"], p["loser"]) for p in build_pairs([], scores)}
        self.assertEqual(pairs, {("good", "bad1"), ("good", "bad2")})

    def test_rejects_lose_only_to_a_good_best(self):
        cands = [{"id": "r", "instruction_id": "i", "filter": "reject"}]
        self.assertEqual([p["kind"] for p in build_pairs(cands, [sc("a", 4, 12), sc("b", 4, 11)])],
                         ["rank", "reject"])
        self.assertEqual(build_pairs(cands, [sc("a", 2, 8), sc("b", 1, 5)]), [])


if __name__ == "__main__":
    unittest.main()


class FreePairs(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from motion_loop.reward_model import free_pool
        cands = {f"{g}_c{k}": {"id": f"{g}_c{k}", "instruction_id": g, "filter": "pass",
                               "source": "normal" if k < 2 else "mismatched"} for g in "abcd" for k in range(3)}
        cands["a_c1"]["filter"] = "reject"
        scores = {"b_c1": {"natural": 2}}
        self.pool = free_pool(cands, scores, {"a", "b", "c", "d"})
        self.rng = np.random.default_rng(0)

    def test_pool_keeps_only_good_normal_ones(self):
        self.assertEqual(self.pool, {"a": ["a_c0"], "b": ["b_c0"], "c": ["c_c0", "c_c1"], "d": ["d_c0", "d_c1"]})

    def test_hard_negatives_share_the_posture(self):
        from motion_loop.reward_model import free_mismatch_pairs
        posture = {"a": "stand", "b": "stand", "c": "seiza", "d": "seiza"}
        ps = free_mismatch_pairs(self.pool, 20, self.rng, posture, hard=1.0)
        self.assertTrue(all(posture[p["winner"][0]] == posture[p["loser"][0]] != None for p in ps))
        self.assertTrue(all(p["winner"][0] == p["instruction_id"] != p["loser"][0] for p in ps))

    def test_degrade_pairs_lose_to_their_source(self):
        from motion_loop.reward_model import free_degrade_pairs
        ps = free_degrade_pairs(self.pool, 5, self.rng)
        self.assertTrue(all(p["loser"].split("~")[0] == p["winner"] for p in ps))

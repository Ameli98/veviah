"""BatchAnthropicLLM against a fake Message Batches API (no network, no cost)."""
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

from motion_loop import llm, scoring


class FakeBatches:
    """Answers each request with reply(prompt text, times this prompt was seen)."""

    def __init__(self, reply):
        self.reply, self.created, self.store, self.asked = reply, [], {}, {}

    def create(self, requests):
        bid = f"b{len(self.created)}"
        self.created.append(len(requests))
        out = []
        for r in requests:
            text = r["params"]["messages"][0]["content"][-1]["text"]
            self.asked[text] = self.asked.get(text, 0) + 1
            msg = NS(content=[NS(type="text", text=self.reply(text, self.asked[text]))], stop_reason="end_turn",
                     usage=NS(input_tokens=1000, output_tokens=100))
            out.append(NS(custom_id=r["custom_id"], result=NS(type="succeeded", message=msg)))
        self.store[bid] = out
        return NS(id=bid)

    def retrieve(self, bid):
        return NS(processing_status="ended")

    def results(self, bid):
        return iter(self.store[bid])


def fake_client(reply):
    return NS(messages=NS(batches=FakeBatches(reply)))


def judge_reply(text, n):
    if "bad" in text and n == 1:
        return "not json"                         # first answer malformed -> scorer retries
    return json.dumps({"candidates": {l: {"reason": "", "match": 3, "natural": 3, "appropriate": 3, "flags": []}
                                      for l in "AB"}, "ranking": ["B", "A"]})


class BatchLLM(unittest.TestCase):
    def setUp(self):
        llm.USAGE.clear()

    def run_group(self, client, d, instructions):
        b = llm.BatchAnthropicLLM("claude-sonnet-5-5", log_dir=d, quiet=0.3, client=client)
        b.POLL_S = 0
        sc = scoring.LMMScorer(b)
        with ThreadPoolExecutor(8) as ex:
            return list(ex.map(lambda t: sc.score_group(t, [b"png", b"png"], [{}, {}]), instructions))

    def test_one_batch_retry_and_half_price(self):
        with tempfile.TemporaryDirectory() as d:
            c = fake_client(judge_reply)
            res = self.run_group(c, d, [f"ins {i}" for i in range(7)] + ["bad one"])
            self.assertTrue(all(r is not None and r[1]["rank"] == 1 for r in res))
            self.assertEqual(c.messages.batches.created, [8, 1])      # all at once, then the retry
            u = llm.usage_report()["claude-sonnet-5-5" + llm.BATCH]
            self.assertEqual(u["calls"], 9)
            self.assertAlmostEqual(u["usd"], 9 * (1000 * 2 + 100 * 10) / 1e6 / 2)

    def test_rerun_reuses_paid_replies(self):
        with tempfile.TemporaryDirectory() as d:
            ins = [f"ins {i}" for i in range(5)] + ["bad one"]
            self.run_group(fake_client(judge_reply), d, ins)
            c2 = fake_client(judge_reply)
            res = self.run_group(c2, d, ins)
            self.assertTrue(all(r is not None for r in res))
            self.assertEqual(c2.messages.batches.created, [])          # nothing sent again


if __name__ == "__main__":
    unittest.main()

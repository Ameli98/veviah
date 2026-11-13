"""LLM interface. AnthropicLLM is the real one; MockLLM lets the pipeline run offline."""
import base64
import hashlib
import json
import os
import re
import threading
import time
import zlib
from pathlib import Path

import numpy as np


def extract_json(text):
    """Pull the first JSON object out of a model reply (tolerates code fences / chatter)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object found")
    return json.loads(text[start:end + 1])


class LLM:
    def complete(self, prompt, images=None, max_tokens=1500):
        """images: optional list of PNG bytes. Returns the reply text."""
        raise NotImplementedError


# USD per million tokens (input, output), for the cost report only. Check current prices before trusting it.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}
BATCH = " (batch)"   # USAGE key suffix: Message Batches API calls cost half
USAGE = {}   # model (+ BATCH) -> {"calls", "input_tokens", "output_tokens", "truncated"}, summed over this process
_USAGE_LOCK = threading.Lock()


def record_usage(key, usage, stop_reason):
    with _USAGE_LOCK:
        u = USAGE.setdefault(key, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "truncated": 0})
        u["calls"] += 1
        u["input_tokens"] += usage.input_tokens
        u["output_tokens"] += usage.output_tokens
        u["truncated"] += stop_reason == "max_tokens"


def usage_report():
    """Token counts as reported by the API for every call made so far, with the cost they imply."""
    out = {}
    for key, u in USAGE.items():
        price = PRICES.get(key.removesuffix(BATCH))
        cost = None if price is None else (u["input_tokens"] * price[0] + u["output_tokens"] * price[1]) / 1e6
        if cost is not None and key.endswith(BATCH):
            cost /= 2
        out[key] = {**u, "usd": None if cost is None else round(cost, 4)}
    return out


class AnthropicLLM(LLM):
    """Real client. Needs `pip install anthropic` and ANTHROPIC_API_KEY.
    Model comes from `model`, else env MOTION_LOOP_MODEL, else the default below.
    These models think before answering and thinking counts as output tokens, so max_tokens must
    leave room for it; a reply cut off at max_tokens comes back truncated (callers retry on bad JSON)."""
    DEFAULT_MODEL = "claude-sonnet-5-5"

    def __init__(self, model=None, retries=4, client=None):
        if client is None:
            import anthropic  # imported lazily so the rest of the repo works without it
            client = anthropic.Anthropic()
        self.client = client
        self.model = model or os.environ.get("MOTION_LOOP_MODEL", self.DEFAULT_MODEL)
        self.retries = retries

    def params(self, prompt, images, max_tokens):
        content = [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                         "data": base64.b64encode(png).decode()}}
            for png in (images or [])
        ]
        content.append({"type": "text", "text": prompt})
        return {"model": self.model, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}]}

    def complete(self, prompt, images=None, max_tokens=1500):
        params = self.params(prompt, images, max_tokens)
        for attempt in range(self.retries):
            try:
                resp = self.client.messages.create(**params)
                record_usage(self.model, resp.usage, resp.stop_reason)
                return "".join(b.text for b in resp.content if b.type == "text")
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)


class BatchAnthropicLLM(AnthropicLLM):
    """Same interface, but calls go through the Message Batches API (half price, results within 24 h,
    usually much sooner). complete() blocks: requests from many caller threads are collected until no new
    one arrives for `quiet` seconds, then sent as one batch (split to stay under the size limit), so callers
    must run concurrently (pipeline --workers) or every request becomes its own batch.

    Resumable: every request gets a custom_id = hash of its content + how many times this process has asked
    the same thing (so a caller's retry after a bad reply is a new request). Submitted batches are logged in
    `log_dir/batches.jsonl` and finished replies in `log_dir/replies.jsonl`; after a crash the same run
    re-uses them instead of paying again."""
    MAX_BYTES = 150_000_000          # API limit is 256 MB per batch
    POLL_S = 30

    def __init__(self, model=None, log_dir=".", quiet=3.0, retries=3, client=None):
        super().__init__(model, retries, client)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.quiet = quiet
        self.lock = threading.Lock()
        self.pending = []            # (custom_id, params, slot)
        self.last_add = 0.0
        self.seen = {}               # content hash -> times asked
        self.replies = {r["id"]: r for r in _read(self.log_dir / "replies.jsonl")}
        self.submitted = {cid: b["batch"] for b in _read(self.log_dir / "batches.jsonl") for cid in b["ids"]}
        threading.Thread(target=self._flusher, daemon=True).start()

    def complete(self, prompt, images=None, max_tokens=1500):
        params = self.params(prompt, images, max_tokens)
        h = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:40]
        with self.lock:
            n = self.seen[h] = self.seen.get(h, 0) + 1
        for attempt in range(self.retries):
            cid = f"{h}_{n}_{attempt}"
            r = self.replies.get(cid)
            if r is None:
                slot = {"done": threading.Event()}
                with self.lock:
                    self.pending.append((cid, params, slot))
                    self.last_add = time.monotonic()
                slot["done"].wait()
                r = slot["reply"]
            if r["ok"]:
                return r["text"]
        raise RuntimeError(f"batch request failed {self.retries} times: {r.get('error')}")

    def _flusher(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                if not self.pending or time.monotonic() - self.last_add < self.quiet:
                    continue
                todo, self.pending = self.pending, []
            try:
                self._run(todo)
            except Exception as e:           # network trouble etc.: fail these requests, callers may retry
                for cid, _, slot in todo:
                    slot["reply"] = {"id": cid, "ok": False, "error": repr(e)}
                    slot["done"].set()

    def _run(self, todo):
        slots = {cid: slot for cid, _, slot in todo}
        by_batch = {}                        # batch id -> custom ids, for requests sent by an earlier run
        new, size = [[]], 0
        for cid, params, _ in todo:
            if cid in self.submitted:
                by_batch.setdefault(self.submitted[cid], []).append(cid)
                continue
            n = len(json.dumps(params))
            if new[-1] and size + n > self.MAX_BYTES:
                new.append([])
                size = 0
            new[-1].append({"custom_id": cid, "params": params})
            size += n
        for reqs in filter(None, new):
            b = self.client.messages.batches.create(requests=reqs)
            ids = [r["custom_id"] for r in reqs]
            _append(self.log_dir / "batches.jsonl", {"batch": b.id, "ids": ids})
            print(f"batch {b.id}: {len(ids)} requests", flush=True)
            by_batch[b.id] = ids
        for bid, ids in by_batch.items():
            while self.client.messages.batches.retrieve(bid).processing_status != "ended":
                time.sleep(self.POLL_S)
            want = set(ids)
            for e in self.client.messages.batches.results(bid):
                if e.custom_id not in want:
                    continue
                if e.result.type == "succeeded":
                    m = e.result.message
                    record_usage(self.model + BATCH, m.usage, m.stop_reason)
                    r = {"id": e.custom_id, "ok": True, "text": "".join(b.text for b in m.content if b.type == "text")}
                else:
                    r = {"id": e.custom_id, "ok": False, "error": e.result.type}
                if r["ok"]:
                    _append(self.log_dir / "replies.jsonl", r)
                self.replies[e.custom_id] = r
                slots[e.custom_id]["reply"] = r
                slots[e.custom_id]["done"].set()
        for cid, slot in slots.items():      # not in the results at all
            if not slot["done"].is_set():
                slot["reply"] = {"id": cid, "ok": False, "error": "missing from batch results"}
                slot["done"].set()


def _read(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()] if Path(path).exists() else []


_LOG_LOCK = threading.Lock()


def _append(path, rec):
    with _LOG_LOCK, open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# Canned keyframe replies for MockLLM (right arm; see keyframes.PROMPT for the format).
_REST_R = {"upper": [-0.3, -1, 0], "fore": [-0.3, -1, 0.3]}
_MOCK_KEYFRAMES = {
    "wave": [(0.0, {"right_arm": _REST_R}), (0.25, {"right_arm": {"upper": [-1, -0.3, 0], "fore": [-0.4, 1, 0]}}),
             (0.4, {"right_arm": {"upper": [-1, -0.3, 0], "fore": [0.3, 1, 0]}}),
             (0.55, {"right_arm": {"upper": [-1, -0.3, 0], "fore": [-0.4, 1, 0]}}),
             (0.7, {"right_arm": {"upper": [-1, -0.3, 0], "fore": [0.3, 1, 0]}}), (1.0, {"right_arm": _REST_R})],
    "nod": [(0.0, {"head": {"nod": 0}}), (0.3, {"head": {"nod": 25}}), (0.5, {"head": {"nod": 0}}),
            (0.7, {"head": {"nod": 25}}), (1.0, {"head": {"nod": 0}})],
    "point": [(0.0, {"right_arm": _REST_R}), (0.35, {"right_arm": {"upper": [-0.1, -0.3, 1], "fore": [0, 0, 1]},
                                                    "head": {"turn": -10}, "hands": {"right": "point"}}),
              (0.7, {"right_arm": {"upper": [-0.1, -0.3, 1], "fore": [0, 0, 1]}}),
              (1.0, {"right_arm": _REST_R, "head": {"turn": 0}, "hands": {"right": "relaxed"}})],
    "lean": [(0.0, {"torso": {"lean": 0}}), (0.4, {"torso": {"lean": 15, "turn": 10}, "head": {"tilt": 12}}),
             (1.0, {"torso": {"lean": 0, "turn": 0}, "head": {"tilt": 0}})],
}


class MockLLM(LLM):
    """Offline stand-in for the text-only steps (scoring uses MockScorer): writes instructions, and
    for the keyframe prompt returns one of a few canned gestures (it does not read the instruction)."""

    def __init__(self, seed=0):
        self.seed = seed

    def complete(self, prompt, images=None, max_tokens=1500):
        if "You animate a desktop companion character" in prompt:
            rng = np.random.default_rng(zlib.crc32(prompt.encode()) + self.seed)
            dur = float(re.search(r"^Duration: ([0-9.]+) s", prompt, flags=re.MULTILINE).group(1))
            name = list(_MOCK_KEYFRAMES)[rng.integers(len(_MOCK_KEYFRAMES))]
            return json.dumps({"keyframes": [{"t": round(f * dur, 3), **parts} for f, parts in _MOCK_KEYFRAMES[name]]})
        if "You write short motion instructions" not in prompt:
            raise NotImplementedError("MockLLM only supports the instruction and keyframe prompts")
        spec = dict(re.findall(r"^- ([a-z ]+): (.*)$", prompt, flags=re.MULTILINE))
        rng = np.random.default_rng(zlib.crc32(prompt.encode()) + self.seed)
        verbs = ["moves", "does", "makes", "performs"]
        v = verbs[rng.integers(len(verbs))]
        e, i, b, a = spec["emotion"], spec["intensity"], spec["body part"], spec["action type"]
        sp, sit = spec["speed"], spec["situation"]
        out = {"instructions": [
            {"style": "plain", "text": f"A {i} {a} gesture with the {b}, {sp}, looking {e}."},
            {"style": "casual", "text": f"just {v} a {sp} {e} thing with your {b} while {sit}"},
            {"style": "vague", "text": f"feeling {e} ({i}), {sit}, something with the {b}"},
        ]}
        return json.dumps(out)

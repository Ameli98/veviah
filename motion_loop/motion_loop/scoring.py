"""Step 5: LMM scores and ranks the candidates of one instruction in a single call."""
import random
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .llm import extract_json

AXES = ["match", "natural", "appropriate"]
FLAGS = ["jitter", "limb_clipping", "wrong_action", "wrong_emotion", "too_exaggerated", "frozen"]

PROMPT = """You see {k} candidate motions for a desktop companion character (it stands or sits on the floor, in place).
Each candidate is one image: a grid of 8 keyframes in time order (t is shown on each frame).{layout}
Candidates are labeled {labels} in the order the images are given.

Instruction: "{instruction}"

Score every candidate on three axes, 1 to 5:
- match: does the motion do what the instruction says (body part, action, emotion, intensity)?
- natural: is it smooth and human-like (no jitter, limb clipping, bent-back joints, stiff poses)?
- appropriate: for a desktop companion, is the size and style of the motion suitable and readable?
Anchors: 1 = completely wrong or visibly broken; 3 = roughly right with clear flaws; 5 = no visible problems.
Write one short reason BEFORE the scores. Add flags from {flags} where they apply.
Then rank all candidates from best to worst.

Return JSON only:
{{"candidates": {{"A": {{"reason": "...", "match": 1, "natural": 1, "appropriate": 1, "flags": []}}, ...}}, "ranking": ["B", "A", ...]}}"""


def build_prompt(instruction, labels, layout_note=""):
    """layout_note: one sentence from the renderer on how to read a keyframe (see Renderer.layout_note)."""
    return PROMPT.format(k=len(labels), labels=", ".join(labels), instruction=instruction,
                         flags=FLAGS, layout="\n" + layout_note if layout_note else "")


def parse_scores(text, labels):
    """Validate a model reply. Returns {label: {...}} (with 'rank') or None if malformed."""
    try:
        data = extract_json(text)
        cands, ranking = data["candidates"], data["ranking"]
        if set(cands) != set(labels) or sorted(ranking) != sorted(labels):
            return None
        out = {}
        for lab in labels:
            c = cands[lab]
            vals = {a: int(c[a]) for a in AXES}
            if any(not 1 <= v <= 5 for v in vals.values()):
                return None
            out[lab] = {**vals, "flags": [f for f in c.get("flags", []) if f in FLAGS],
                        "reason": str(c.get("reason", "")), "rank": ranking.index(lab) + 1}
        return out
    except Exception:
        return None


class Scorer:
    def score_group(self, instruction, pngs, cands):
        """pngs/cands are aligned and already shuffled. Return {index: result} or None."""
        raise NotImplementedError


class LMMScorer(Scorer):
    def __init__(self, llm, retries=2, layout_note=""):
        self.llm, self.retries, self.layout_note = llm, retries, layout_note

    def score_group(self, instruction, pngs, cands):
        labels = [chr(65 + i) for i in range(len(pngs))]
        prompt = build_prompt(instruction, labels, self.layout_note)
        for _ in range(self.retries + 1):
            res = parse_scores(self.llm.complete(prompt, images=pngs, max_tokens=2000), labels)
            if res:
                return {i: res[lab] for i, lab in enumerate(labels)}
        return None


class MockScorer(Scorer):
    """Fake judge that peeks at how each candidate was made. For offline tests only."""
    PENALTY = {"noise": ("natural", 3), "speed_slow": ("appropriate", 2), "speed_fast": ("appropriate", 2),
               "mirror": ("match", 2), "truncate": ("match", 3), "freeze_arm": ("natural", 2),
               "extreme": ("natural", 3)}

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def score_group(self, instruction, pngs, cands):
        res = {}
        for i, c in enumerate(cands):
            s = {a: 4 for a in AXES}
            if c["source"] == "mismatched":
                s["match"] = 1
            elif c["source"] == "degraded":
                axis, drop = self.PENALTY[c["degrade_kind"]]
                s[axis] -= drop
            s = {a: int(np.clip(v + self.rng.integers(-1, 2) * (self.rng.random() < .3), 1, 5)) for a, v in s.items()}
            res[i] = {**s, "flags": [], "reason": "mock"}
        order = sorted(res, key=lambda i: -sum(res[i][a] for a in AXES) + self.rng.random() * .1)
        for r, i in enumerate(order):
            res[i]["rank"] = r + 1
        return res


def overall(rec):
    return sum(rec[a] for a in AXES)


def score_groups(cands, instructions, grids_dir, scorer, repeats=1, seed=0, workers=1):
    """cands: candidate records (filter already applied; grids rendered for passes).
    Each group is shuffled per repeat to reduce position bias; repeat scores are averaged.
    workers > 1 scores groups in parallel threads (needed for a batch LLM).
    Returns (score records, metrics)."""
    from pathlib import Path
    by_ins = {}
    for c in cands:
        if c.get("filter") == "pass":
            by_ins.setdefault(c["instruction_id"], []).append(c)
    text = {i["id"]: i["text"] for i in instructions}

    def run(job):
        iid, r = job
        group = by_ins[iid]
        order = list(range(len(group)))
        random.Random(zlib.crc32(iid.encode()) + 1009 * r + seed).shuffle(order)
        shuffled = [group[j] for j in order]
        pngs = [(Path(grids_dir) / f"{c['id']}.png").read_bytes() for c in shuffled]
        res = scorer.score_group(text[iid], pngs, shuffled)
        return None if res is None else {shuffled[i]["id"]: v for i, v in res.items()}

    jobs = [(iid, r) for iid, g in by_ins.items() if len(g) >= 2 for r in range(repeats)]
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            results = dict(zip(jobs, ex.map(run, jobs)))
    else:
        results = {j: run(j) for j in jobs}
    records, agree, skipped = [], [], 0
    for iid, group in by_ins.items():
        if len(group) < 2:
            skipped += 1
            continue
        runs = [res for r in range(repeats) if (res := results[(iid, r)]) is not None]
        if not runs:
            skipped += 1
            continue
        for c in group:
            vals = [run[c["id"]] for run in runs]
            rec = {"cand_id": c["id"], "instruction_id": iid, "n_runs": len(runs),
                   **{a: float(np.mean([v[a] for v in vals])) for a in AXES},
                   "flags": sorted({f for v in vals for f in v["flags"]}),
                   "reason": vals[0]["reason"]}
            rec["overall"] = overall(rec)
            records.append(rec)
        if len(runs) >= 2:  # fraction of candidate pairs ordered the same way in run 1 vs run 2
            ids = list(runs[0])
            pairs = [(a, b) for k, a in enumerate(ids) for b in ids[k + 1:]]
            same = [(runs[0][a]["rank"] < runs[0][b]["rank"]) == (runs[1][a]["rank"] < runs[1][b]["rank"])
                    for a, b in pairs]
            agree.append(np.mean(same))
    metrics = {"groups_scored": len(by_ins) - skipped, "groups_skipped": skipped,
               "rank_consistency": float(np.mean(agree)) if agree else None}
    return records, metrics

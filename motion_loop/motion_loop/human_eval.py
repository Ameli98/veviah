"""Human validation set: export groups to label by hand, then measure agreement.

Workflow: score -> humaneval-export -> (you fill `rank` in labels.csv) -> pairs -> train
          -> humaneval-score. The exported groups are excluded from reward model training.
"""
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np

from .schema import load_motion, load_root, read_jsonl


def export_human_set(workdir, n_groups=100, seed=0):
    w = Path(workdir)
    scores = read_jsonl(w / "scores.jsonl")
    instr = {i["id"]: i for i in read_jsonl(w / "instructions.jsonl")}
    by_ins = {}
    for s in scores:
        by_ins.setdefault(s["instruction_id"], []).append(s["cand_id"])
    rng = random.Random(seed)
    tiers = {}
    for iid in sorted(by_ins):
        tiers.setdefault(instr[iid]["tier"], []).append(iid)
    for v in tiers.values():
        rng.shuffle(v)
    chosen = []
    while len(chosen) < n_groups and any(tiers.values()):  # round-robin over tiers
        for t in sorted(tiers):
            if tiers[t] and len(chosen) < n_groups:
                chosen.append(tiers[t].pop())
    out = w / "human_eval"
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    rows, key = [], {}
    for iid in chosen:
        cids = by_ins[iid][:]
        rng.shuffle(cids)  # labels carry no information about how a candidate was made
        key[iid] = {}
        for k, cid in enumerate(cids):
            label = chr(65 + k)
            key[iid][label] = cid
            img = f"images/{iid}_{label}.png"
            shutil.copy(w / "grids" / f"{cid}.png", out / img)
            rows.append({"group_id": iid, "instruction": instr[iid]["text"], "label": label,
                         "image": img, "rank": "", "flags": ""})
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "key.json").write_text(json.dumps(key, indent=1))
    (w / "human_set.json").write_text(json.dumps({"ids": chosen}))
    return chosen


def _human_pairs(workdir, csv_path):
    key = json.loads((Path(workdir) / "human_eval" / "key.json").read_text())
    ranks = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["rank"].strip():
                ranks.setdefault(r["group_id"], {})[key[r["group_id"]][r["label"]]] = float(r["rank"])
    pairs = []
    for iid, rk in ranks.items():
        ids = list(rk)
        pairs += [(iid, a, b) for a in ids for b in ids if rk[a] < rk[b]]  # a is better (rank 1 = best)
    return pairs


def score_human(workdir, csv_path, reward_model_path=None):
    """Pairwise agreement of the LMM scores (and the reward model, if given) with your rankings."""
    w = Path(workdir)
    pairs = _human_pairs(w, csv_path)
    lmm = {s["cand_id"]: s["overall"] for s in read_jsonl(w / "scores.jsonl")}
    res = {"human_pairs": len(pairs)}
    dec = [(lmm[a] > lmm[b]) for _, a, b in pairs if lmm[a] != lmm[b]]
    res["lmm_vs_human"] = float(np.mean(dec)) if dec else None
    res["lmm_ties_ignored"] = len(pairs) - len(dec)
    if reward_model_path:
        from .reward_model import RewardModelScorer
        vs = RewardModelScorer(reward_model_path)
        instr = {i["id"]: i["text"] for i in read_jsonl(w / "instructions.jsonl")}
        cands = {c["id"]: c for c in read_jsonl(w / "candidates.jsonl")}
        need = sorted({c for _, a, b in pairs for c in (a, b)})
        iid_of = {c: cands[c]["instruction_id"] for c in need}
        ms = [load_motion(w, cands[c]["motion_path"]) for c in need]
        sc = vs.score(ms, [instr[iid_of[c]] for c in need],
                      [load_root(w, cands[c]["motion_path"], len(m)) for c, m in zip(need, ms)]).sum(-1)
        sc = dict(zip(need, sc))
        res["reward_model_vs_human"] = float(np.mean([sc[a] > sc[b] for _, a, b in pairs])) if pairs else None
    return res

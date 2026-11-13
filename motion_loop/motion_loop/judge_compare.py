"""Judge comparison (Decision 7): how well does each judge agree with the author's labels?

Uses every candidate of a labelled set, including the ones the rule filter rejected, because she
labelled those too.

  python -m motion_loop.judge_compare grids  --workdir W            # 8-keyframe grids for all candidates (GPU)
  python -m motion_loop.judge_compare score  --workdir W --judge haiku [--repeats 2] [--groups N]
  python -m motion_loop.judge_compare report --workdir W [--labels Ameli]
  python -m motion_loop.judge_compare rm     --workdir W --checkpoint R/reward_model.pt   # a reward model instead

score writes W/judges/<judge>.jsonl: one line per (group, run) with the judge's rank and scores per
candidate; a run shows the group in a different shuffled order, so two runs also measure position
consistency. report compares with W/labels/<name>.json.

Pairs used in the report (the author, 2026-10-21: close normal clips were ranked by feel, so those pairs say
little): only pairs her labels order strictly.
  broken    one clip marked broken, the other not             (a judge must put the broken one lower)
  mismatch  normal vs mismatched, neither marked broken       (the motion of another instruction)
  clear     neither broken, her ranks differ by 2 or more
  close     neither broken, her ranks differ by 1             (reported, but weak evidence)
"""
import argparse
import json
import random
import zlib
from pathlib import Path

import numpy as np

from . import llm, render, scoring
from .schema import load_hands, load_motion, load_root, read_jsonl

JUDGES = {"haiku": "claude-haiku-5-5", "sonnet": "claude-sonnet-5-5"}


def make_grids(w, keyframes=8):
    from .vrm_render import VrmRenderer
    r = VrmRenderer(gpu=True)
    out = w / "grids"
    out.mkdir(exist_ok=True)
    (out / "layout.txt").write_text(r.layout_note)
    n = 0
    try:
        for c in read_jsonl(w / "candidates.jsonl"):
            p = out / f"{c['id']}.png"
            if p.exists():
                continue
            m = load_motion(w, c["motion_path"])
            g = render.render_grid(r, m, keyframes, load_hands(w, c["motion_path"], len(m)),
                                   load_root(w, c["motion_path"], len(m)))
            p.write_bytes(render.to_png(g))
            n += 1
    finally:
        r.close()
    print(f"{n} new grids in {out}")


def groups_of(w):
    by = {}
    for c in read_jsonl(w / "candidates.jsonl"):
        by.setdefault(c["instruction_id"], []).append(c)
    return by


def score(w, judge, repeats=2, n_groups=None, seed=0):
    text = {i["id"]: i["text"] for i in read_jsonl(w / "instructions.jsonl")}
    labelled = {g for f in (w / "labels").glob("*.json") for g in json.loads(f.read_text())["groups"]}
    groups = [(g, cs) for g, cs in groups_of(w).items() if g in labelled][:n_groups]
    model = JUDGES[judge]
    scorer = scoring.LMMScorer(llm.AnthropicLLM(model), layout_note=(w / "grids" / "layout.txt").read_text())
    out = w / "judges" / f"{judge}.jsonl"
    out.parent.mkdir(exist_ok=True)
    done = {(r["group"], r["run"]) for r in read_jsonl(out)}
    with open(out, "a", encoding="utf-8") as f:
        for gid, cands in groups:
            for run in range(repeats):
                if (gid, run) in done:
                    continue
                order = list(range(len(cands)))
                random.Random(zlib.crc32(gid.encode()) + 1009 * run + seed).shuffle(order)
                shuffled = [cands[j] for j in order]
                pngs = [(w / "grids" / f"{c['id']}.png").read_bytes() for c in shuffled]
                res = scorer.score_group(text[gid], pngs, shuffled)
                rec = {"group": gid, "run": run, "order": [c["id"] for c in shuffled],
                       "result": None if res is None else {shuffled[i]["id"]: v for i, v in res.items()}}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(gid, run, "ok" if res else "FAILED", flush=True)
    print(llm.usage_report())
    with open(w / "api_usage.jsonl", "a") as f:
        f.write(json.dumps({"stage": f"judge_compare {judge}", "usage_so_far": llm.usage_report()}) + "\n")


def _human_key(a):
    """Sort key for a human label: broken clips are tied last."""
    return np.inf if a["broken"] else a["rank"]


def human_pairs(w, labeller="Ameli"):
    """{(group, a, b): category} for every pair her labels order strictly (a better than b)."""
    labels = json.loads((w / "labels" / f"{labeller}.json").read_text())["groups"]
    cand = {c["id"]: c for c in read_jsonl(w / "candidates.jsonl")}
    pairs = {}
    for gid, g in labels.items():
        ids = list(g)
        for a in ids:
            for b in ids:
                ka, kb = _human_key(g[a]), _human_key(g[b])
                if not ka < kb:
                    continue
                if g[b]["broken"]:
                    cat = "broken"
                elif cand[a]["source"] == "normal" and cand[b]["source"] == "mismatched":
                    cat = "mismatch"
                else:
                    cat = "clear" if kb - ka >= 2 else "close"
                pairs[(gid, a, b)] = cat
    return pairs


def report(w, labeller="Ameli"):
    pairs = human_pairs(w, labeller)
    files = sorted((w / "judges").glob("*.jsonl"))
    # compare judges on the same groups: those every judge has scored at least once
    common = set.intersection(*({r["group"] for r in read_jsonl(f) if r["result"]} for f in files)) if files else set()
    pairs = {k: v for k, v in pairs.items() if k[0] in common}
    rows = {}
    for f in files:
        runs = [r for r in read_jsonl(f) if r["result"] and r["group"] in common]
        by_run = {}
        for r in runs:
            by_run.setdefault(r["run"], {})[r["group"]] = r["result"]
        stats = {}
        for (gid, a, b), cat in pairs.items():
            for run, res in by_run.items():
                if gid in res:
                    ok = res[gid][a]["rank"] < res[gid][b]["rank"]
                    stats.setdefault(cat, []).append(ok)
                    stats.setdefault("all", []).append(ok)
        cons = []                                        # same pair order in run 0 and run 1
        if 0 in by_run and 1 in by_run:
            for gid in set(by_run[0]) & set(by_run[1]):
                r0, r1 = by_run[0][gid], by_run[1][gid]
                ids = list(r0)
                cons += [(r0[a]["rank"] < r0[b]["rank"]) == (r1[a]["rank"] < r1[b]["rank"])
                         for k, a in enumerate(ids) for b in ids[k + 1:]]
        rows[f.stem] = {"groups": len({r["group"] for r in runs}), "runs": len(runs),
                        "failed": sum(1 for r in read_jsonl(f) if not r["result"]),
                        **{cat: (round(float(np.mean(v)), 3), len(v)) for cat, v in sorted(stats.items())},
                        "consistency": (round(float(np.mean(cons)), 3), len(cons)) if cons else None}
    counts = {}
    for cat in pairs.values():
        counts[cat] = counts.get(cat, 0) + 1
    print(f"{len(common)} groups scored by every judge; human pairs per category:", counts)
    for name, r in rows.items():
        print(name, json.dumps(r))
    return rows


def reward_model_report(w, checkpoint, labeller="Ameli"):
    """Same categories for a reward model: it ranks a pair right when it gives a higher total score.
    checkpoint may be a list: an ensemble, ranked by the mean of the models' totals."""
    from .reward_model import RewardModelScorer
    pairs = human_pairs(w, labeller)
    text = {i["id"]: i["text"] for i in read_jsonl(w / "instructions.jsonl")}
    cand = {c["id"]: c for c in read_jsonl(w / "candidates.jsonl")}
    ids = sorted({c for k in pairs for c in k[1:]})
    ms = [load_motion(w, cand[i]["motion_path"]) for i in ids]
    roots = [load_root(w, cand[i]["motion_path"], len(m)) for i, m in zip(ids, ms)]
    cks = checkpoint if isinstance(checkpoint, (list, tuple)) else [checkpoint]
    texts = [text[cand[i]["instruction_id"]] for i in ids]
    total = dict(zip(ids, np.mean([RewardModelScorer(c).score(ms, texts, roots).sum(-1) for c in cks], 0)))
    stats = {}
    for (gid, a, b), cat in pairs.items():
        for k in (cat, "all"):
            stats.setdefault(k, []).append(total[a] > total[b])
    return {cat: (round(float(np.mean(v)), 3), len(v)) for cat, v in sorted(stats.items())}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["grids", "score", "report", "rm"])
    ap.add_argument("--checkpoint", help="rm: reward_model.pt to compare with the labels")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--judge", choices=sorted(JUDGES))
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--groups", type=int, default=None, help="only the first N labelled groups")
    ap.add_argument("--labels", default="Ameli")
    a = ap.parse_args(argv)
    w = Path(a.workdir).expanduser()
    if a.cmd == "grids":
        make_grids(w)
    elif a.cmd == "score":
        score(w, a.judge, a.repeats, a.groups)
    elif a.cmd == "rm":
        print(json.dumps(reward_model_report(w, a.checkpoint, a.labels)))
    else:
        report(w, a.labels)


if __name__ == "__main__":
    main()

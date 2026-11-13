"""Merge several round workdirs into one for reward-model training.

  python -m motion_loop.merge --out ~/hsin_runs/round01 ~/hsin_runs/round0 ~/hsin_runs/round1

Ids get the source folder's name as a prefix (both rounds start at s00000), motion paths point back
into the source folders, so nothing is copied. Writes instructions / candidates / scores / pairs and a
split.json that keeps the first run's val groups as val (so results stay comparable) and trains on the rest.
"""
import argparse
import json
import os
from pathlib import Path

from . import prefs
from .schema import read_jsonl, write_jsonl


def merge(out, runs, margin=1.0):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ins, cands, scores, val = [], [], [], []
    for k, run in enumerate(map(Path, runs)):
        p = run.name + "/"
        rel = os.path.relpath(run, out)
        ins += [{**i, "id": p + i["id"]} for i in read_jsonl(run / "instructions.jsonl")]
        cands += [{**c, "id": p + c["id"], "instruction_id": p + c["instruction_id"],
                   "mismatch_from": c["mismatch_from"] and p + c["mismatch_from"],
                   "motion_path": f"{rel}/{c['motion_path']}",
                   "keyframes_path": c.get("keyframes_path") and f"{rel}/{c['keyframes_path']}"}
                  for c in read_jsonl(run / "candidates.jsonl")]
        scores += [{**s, "cand_id": p + s["cand_id"], "instruction_id": p + s["instruction_id"]}
                   for s in read_jsonl(run / "scores.jsonl")]
        if k == 0:
            val = [p + g for g in json.loads((run / "split.json").read_text())["val"]]
    pairs = prefs.build_pairs(cands, scores, margin)
    groups = sorted({p["instruction_id"] for p in pairs})
    split = {"val": val, "train": [g for g in groups if g not in set(val)], "human": []}
    write_jsonl(out / "instructions.jsonl", ins)
    write_jsonl(out / "candidates.jsonl", cands)
    write_jsonl(out / "scores.jsonl", scores)
    write_jsonl(out / "pairs.jsonl", pairs)
    (out / "split.json").write_text(json.dumps(split))
    return {"instructions": len(ins), "candidates": len(cands), "pairs": len(pairs),
            "train_groups": len(split["train"]), "val_groups": len(val)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    print(merge(Path(a.out).expanduser(), [Path(r).expanduser() for r in a.runs]))


if __name__ == "__main__":
    main()

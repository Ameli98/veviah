"""Step 6: ranked scores -> preference pairs, and train/val splits that exclude the human set."""
import random

# A candidate the judge scores this low on `natural` looks broken. Two broken candidates are not
# compared (the author, 2026-10-21: broken clips tie last, as in her labels), and a broken best candidate
# does not win against filter rejects.
BROKEN_NATURAL = 2


def broken(s):
    return s["natural"] <= BROKEN_NATURAL


def build_pairs(cands, scores, margin=1.0):
    """Pairs (winner, loser) within each instruction when the overall score differs by >= margin,
    except between two candidates that both look broken. Filter-rejected candidates additionally
    lose to the best passing candidate, if that one does not look broken."""
    by_ins = {}
    for s in scores:
        by_ins.setdefault(s["instruction_id"], []).append(s)
    rejects = {}
    for c in cands:
        if c.get("filter") == "reject":
            rejects.setdefault(c["instruction_id"], []).append(c["id"])
    pairs = []
    for iid, group in by_ins.items():
        for a in group:
            for b in group:
                if a["overall"] - b["overall"] >= margin and not (broken(a) and broken(b)):
                    pairs.append({"instruction_id": iid, "winner": a["cand_id"], "loser": b["cand_id"],
                                  "kind": "rank", "diff": a["overall"] - b["overall"]})
        best = max(group, key=lambda s: s["overall"])
        for rid in rejects.get(iid, []) if not broken(best) else []:
            pairs.append({"instruction_id": iid, "winner": best["cand_id"], "loser": rid,
                          "kind": "reject", "diff": None})
    return pairs


def split_ids(instruction_ids, human_ids=(), val_frac=0.1, seed=0):
    ids = sorted(set(instruction_ids) - set(human_ids))
    random.Random(seed).shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    return {"val": ids[:n_val], "train": ids[n_val:], "human": sorted(human_ids)}

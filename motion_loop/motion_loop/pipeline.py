"""CLI: python -m motion_loop.pipeline <stage> --workdir W [options]

Stages (each reads/writes files in the workdir, so any stage can be re-run on its own):
  sample -> instruct -> candidates -> filter -> render -> score -> humaneval-export
  -> pairs -> train -> humaneval-score        (run-all = the mock end-to-end demo)
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np

from . import dims, filters, generators, human_eval, instructions, llm, prefs, render, scoring
from .schema import load_hands, load_motion, load_root, read_jsonl, write_jsonl


def load_obj(spec):
    mod, name = spec.split(":")
    return getattr(importlib.import_module(mod), name)


def make_llm(a):
    if a.llm == "anthropic-batch":
        return llm.BatchAnthropicLLM(model=a.model, log_dir=Path(a.workdir) / "batches")
    return llm.MockLLM(a.seed) if a.llm == "mock" else llm.AnthropicLLM(model=a.model)


def workers(a):
    """Parallel callers per stage; a batch LLM needs all requests in flight at once to fill one batch."""
    return a.workers or (100_000 if a.llm == "anthropic-batch" else 1)


def make_generator(a):
    if a.generator == "mock":
        return generators.MockGenerator()
    cls = load_obj(a.generator)
    return cls(make_llm(a)) if getattr(cls, "needs_llm", False) else cls()


def make_renderer(a):
    return render.StickRenderer() if a.renderer == "stick" else load_obj(a.renderer)()


def make_scorer(a):
    return scoring.MockScorer(a.seed) if a.scorer == "mock" else scoring.LMMScorer(make_llm(a))


def stage_sample(a, w):
    specs = dims.sample_specs(a.n, a.seed)
    write_jsonl(w / "specs.jsonl", specs)
    print(f"{len(specs)} specs")


def stage_instruct(a, w):
    seeds = [l.strip() for l in open(a.seeds_file, encoding="utf-8") if l.strip()] if a.seeds_file else None
    recs, stats = instructions.generate_instructions(read_jsonl(w / "specs.jsonl"), make_llm(a),
                                                     seeds, a.seed, min(workers(a), 2000))
    write_jsonl(w / "instructions.jsonl", recs)
    print(stats)


def stage_candidates(a, w):
    instructions_ = read_jsonl(w / "instructions.jsonl")[:a.max_instructions]
    cands = generators.make_candidates(instructions_, make_generator(a),
                                       w / "motions", a.k, a.p_degrade, a.p_mismatch, a.seed,
                                       a.degrade_kinds.split(",") if a.degrade_kinds else None,
                                       max(1, min(workers(a), a.k * len(instructions_))))
    write_jsonl(w / "candidates.jsonl", cands)
    print(f"{len(cands)} candidates")


def stage_filter(a, w):
    cands = filters.filter_candidates(read_jsonl(w / "candidates.jsonl"), w)
    write_jsonl(w / "candidates.jsonl", cands)
    rej = [c for c in cands if c["filter"] == "reject"]
    print(f"{len(rej)}/{len(cands)} rejected")


def stage_render(a, w):
    r = make_renderer(a)
    (w / "grids").mkdir(exist_ok=True)
    (w / "grids" / "layout.txt").write_text(r.layout_note)   # read back by the score stage
    n = 0
    for c in read_jsonl(w / "candidates.jsonl"):
        if c["filter"] == "pass":
            m = load_motion(w, c["motion_path"])
            png = render.to_png(render.render_grid(r, m, a.keyframes, load_hands(w, c["motion_path"], len(m)),
                                                   load_root(w, c["motion_path"], len(m))))
            (w / "grids" / f"{c['id']}.png").write_bytes(png)
            n += 1
    print(f"{n} grids")


def stage_score(a, w):
    scorer = make_scorer(a)
    layout = w / "grids" / "layout.txt"
    scorer.layout_note = layout.read_text() if layout.exists() else ""
    recs, metrics = scoring.score_groups(read_jsonl(w / "candidates.jsonl"),
                                         read_jsonl(w / "instructions.jsonl"), w / "grids",
                                         scorer, a.repeats, a.seed, min(workers(a), 3000))
    write_jsonl(w / "scores.jsonl", recs)
    (w / "score_metrics.json").write_text(json.dumps(metrics, indent=1))
    print(metrics)


def stage_humaneval_export(a, w):
    ids = human_eval.export_human_set(w, a.n_human, a.seed)
    print(f"exported {len(ids)} groups to {w / 'human_eval'}; fill the `rank` column in labels.csv (1 = best)")


def stage_pairs(a, w):
    cands = read_jsonl(w / "candidates.jsonl")
    scores = read_jsonl(w / "scores.jsonl")
    pairs = prefs.build_pairs(cands, scores, a.margin)
    hs = w / "human_set.json"
    human = json.loads(hs.read_text())["ids"] if hs.exists() else []
    split = prefs.split_ids({p["instruction_id"] for p in pairs}, human, a.val_frac, a.seed)
    pairs = [p for p in pairs if p["instruction_id"] not in set(human)]
    write_jsonl(w / "pairs.jsonl", pairs)
    (w / "split.json").write_text(json.dumps(split))
    print(f"{len(pairs)} pairs; train/val/human groups = {len(split['train'])}/{len(split['val'])}/{len(split['human'])}")


def stage_train(a, w):
    from .reward_model import train_reward_model
    m = train_reward_model(w, a.epochs, a.batch, a.lr, encoder=a.encoder, seed=a.seed,
                     d=a.d_model, layers=a.layers)
    print({k: v for k, v in m.items() if k != "history"})


def stage_humaneval_score(a, w):
    csv_path = a.labels or (w / "human_eval" / "labels.csv")
    ckpt = w / "reward_model.pt"
    print(human_eval.score_human(w, csv_path, ckpt if ckpt.exists() else None))


STAGES = {"sample": stage_sample, "instruct": stage_instruct, "candidates": stage_candidates,
          "filter": stage_filter, "render": stage_render, "score": stage_score,
          "humaneval-export": stage_humaneval_export, "pairs": stage_pairs, "train": stage_train,
          "humaneval-score": stage_humaneval_score}
RUN_ALL = ["sample", "instruct", "candidates", "filter", "render", "score", "pairs", "train"]


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=[*STAGES, "run-all"])
    p.add_argument("--workdir", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n", type=int, default=3000, help="number of specs (each yields ~3 instructions)")
    p.add_argument("--k", type=int, default=4, help="candidates per instruction")
    p.add_argument("--max-instructions", type=int, default=None, help="candidates stage: use only the first N instructions")
    p.add_argument("--p-degrade", type=float, default=0.15)
    p.add_argument("--p-mismatch", type=float, default=0.15)
    p.add_argument("--degrade-kinds", default=None, help="comma list, default all: " + ",".join(generators.DEGRADE_KINDS))
    p.add_argument("--keyframes", type=int, default=8)
    p.add_argument("--repeats", type=int, default=1, help=">1 re-scores with shuffled order and averages")
    p.add_argument("--margin", type=float, default=1.0, help="min overall-score gap to form a pair")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-human", type=int, default=100)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--encoder", default="hash", choices=["hash", "minilm"])
    p.add_argument("--workers", type=int, default=None,
                   help="parallel LLM calls in instruct / candidates / score (default 1, or all at once with anthropic-batch)")
    p.add_argument("--seeds-file", default=None, help="text file, one seed instruction per line")
    p.add_argument("--labels", default=None, help="filled labels.csv for humaneval-score")
    p.add_argument("--llm", default="mock", choices=["mock", "anthropic", "anthropic-batch"],
                   help="anthropic-batch = Message Batches API: half price, waits for the batch (minutes to hours)")
    p.add_argument("--model", default=None, help="model id for --llm anthropic")
    p.add_argument("--scorer", default="mock", choices=["mock", "lmm"])
    p.add_argument("--generator", default="mock", help="'mock' or 'package.module:ClassName'")
    p.add_argument("--renderer", default="stick", help="'stick' or 'package.module:ClassName'")
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    w = Path(a.workdir)
    w.mkdir(parents=True, exist_ok=True)
    for s in (RUN_ALL if a.stage == "run-all" else [a.stage]):
        print(f"== {s}")
        STAGES[s](a, w)
        if llm.USAGE:                                    # real API calls were made: keep the bill
            rep = {"stage": s, "usage_so_far": llm.usage_report()}
            print(rep)
            with open(w / "api_usage.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(rep) + "\n")


if __name__ == "__main__":
    main()

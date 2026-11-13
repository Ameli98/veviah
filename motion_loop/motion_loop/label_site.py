"""Human labelling page: builds a static web page where people rank the candidates of each group
from videos, and reads back the answers they send.

  python -m motion_loop.label_site build  --workdir W [--name judge40]
      needs W/videos/cand_tall/<candidate id>.mp4 (motion_loop.video --per-candidate --tall --size 800) and, optionally,
      W/instructions_zh.json {instruction id: translation}.
      writes W/label_site/index.html (to publish), local.html (same page, opens from a local web
      server), W/label_site/v/<random name>.mp4 and W/label_key.json.
      Video names are random and the order inside a group is shuffled, so the page gives no hint of
      how a candidate was made. label_key.json (random name -> candidate id) stays local: never publish it.
  python -m motion_loop.label_site import --workdir W CODE [CODE ...]
      CODE = the text a labeller copied from the page's last screen (or a file containing it).
      writes W/labels/<labeller>.json: {group id: {candidate id: {"rank": 1.. or None, "broken": bool}}}.
      Broken clips (page v2) have rank None: they count as tied last and are not ranked among themselves.
  python -m motion_loop.label_site build --workdir W --name SET --groups g1,g2,... --seed 7
      a second page over some of the same groups (e.g. a re-test of one labeller): a different set name,
      so the page starts empty and keeps its own progress; groups and clips in a new random order and new
      video names (seeded by the name). Writes W/label_site_SET/ and W/label_key_SET.json; its answers
      import to W/labels/<labeller>__SET.json, next to (not over) the first ones.
"""
import argparse
import base64
import hashlib
import json
import random
import shutil
from pathlib import Path

from .schema import read_jsonl

LOCAL_SKELETON = ('<!doctype html><html><head><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">'
                  '<style>body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style></head><body>')
TEMPLATE = Path(__file__).resolve().parents[1] / "label_site" / "page.html"


def build(workdir, name, seed=0, videos="cand_tall", include_rejected=False, only=None):
    """include_rejected: also show candidates the rule filter rejected. For the human / judge validation
    set (the author, 2026-10-16): labellers can still mark them broken, the judges get those cases too, and
    comparing their labels with the filter shows whether its placeholder thresholds are too strict."""
    w = Path(workdir).expanduser()
    main_set = name == w.name                       # the workdir's own set keeps the original file names
    out = w / ("label_site" if main_set else f"label_site_{name}")
    if out.exists():
        shutil.rmtree(out)
    (out / "v").mkdir(parents=True)
    zh_path = w / "instructions_zh.json"
    zh = json.loads(zh_path.read_text(encoding="utf-8")) if zh_path.exists() else {}
    instr = {r["id"]: r for r in read_jsonl(w / "instructions.jsonl")}
    by_ins = {}
    for c in read_jsonl(w / "candidates.jsonl"):
        if (c.get("filter") == "pass" or include_rejected) and (only is None or c["instruction_id"] in only):
            by_ins.setdefault(c["instruction_id"], []).append(c["id"])
    rng = random.Random(seed)
    groups, key = [], {}
    items = list(by_ins.items())
    if not main_set:                                # a re-test also changes the order of the groups
        rng.shuffle(items)
    for iid, cids in items:
        missing = [c for c in cids if not (w / "videos" / videos / f"{c}.mp4").exists()]
        if missing:
            raise SystemExit(f"no video for {missing}; run motion_loop.video --per-candidate first")
        if len(cids) < 2:
            continue
        rng.shuffle(cids)
        vids = []
        for cid in cids:
            v = hashlib.sha256(f"{name}/{seed}/{cid}".encode()).hexdigest()[:12]
            shutil.copy(w / "videos" / videos / f"{cid}.mp4", out / "v" / f"{v}.mp4")
            key[v] = cid
            vids.append(v)
        groups.append({"id": iid, "text": instr[iid]["text"], "zh": zh.get(iid, ""), "vids": vids})
    data = {"set": name, "groups": groups}
    page = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    (out / "index.html").write_text(page, encoding="utf-8")       # for publishing: the host adds the HTML skeleton
    (out / "local.html").write_text(LOCAL_SKELETON + page + "</body></html>", encoding="utf-8")   # for a local look
    (w / ("label_key.json" if main_set else f"label_key_{name}.json")).write_text(
        json.dumps({"set": name, "groups": {g["id"]: g["vids"] for g in groups}, "videos": key}, indent=1))
    size = sum(f.stat().st_size for f in (out / "v").iterdir())
    print(f"{len(groups)} groups, {len(key)} videos, {size / 1e6:.0f} MB -> {out}")


def decode(code):
    """The page's answer text -> dict. The text is 'HSIN1.' + base64url(JSON)."""
    code = "".join(code.split())
    if not code.startswith("HSIN1."):
        raise ValueError("not an answer code (it should start with HSIN1.)")
    raw = code[6:]
    return json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8"))


def import_codes(workdir, codes):
    w = Path(workdir).expanduser()
    (w / "labels").mkdir(exist_ok=True)
    for code in codes:
        if not code.strip().startswith("HSIN1.") and Path(code).expanduser().exists():   # a file holding a code
            code = Path(code).expanduser().read_text(encoding="utf-8")
        ans = decode(code)
        variant = w / f"label_key_{ans['set']}.json"
        key = json.loads((variant if variant.exists() else w / "label_key.json").read_text())
        if ans["set"] != key["set"]:
            raise SystemExit(f"answers are for set {ans['set']!r}, this workdir is {key['set']!r}")
        out = {}
        for gid, a in ans["answers"].items():
            vids = key["groups"][gid]
            # v1 pages ranked every clip 1..n; v2 pages give broken clips rank 0 (= tied last, not ranked)
            ranked = sorted(r for r, b in zip(a["rank"], a["broken"]) if not (ans.get("v", 1) >= 2 and b))
            if ranked != list(range(1, len(ranked) + 1)) or len(a["rank"]) != len(vids):
                raise SystemExit(f"{ans['name']}: bad ranks for {gid}: {a['rank']} broken {a['broken']}")
            out[gid] = {key["videos"][v]: {"rank": r or None, "broken": bool(b)} for v, r, b in zip(vids, a["rank"], a["broken"])}
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in ans["name"]) or "anonymous"
        path = w / "labels" / (f"{safe}.json" if not variant.exists() else f"{safe}__{ans['set']}.json")
        path.write_text(json.dumps({"name": ans["name"], "seconds": ans.get("seconds"), "groups": out},
                                   ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{ans['name']}: {len(out)}/{len(key['groups'])} groups -> {path}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "import"])
    ap.add_argument("codes", nargs="*")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--name", default=None, help="set name shown in answers (default: workdir name)")
    ap.add_argument("--videos", default="cand_tall", help="folder under W/videos with <candidate id>.mp4 (motion_loop.video --per-candidate --tall)")
    ap.add_argument("--include-rejected", action="store_true", help="also show candidates the rule filter rejected")
    ap.add_argument("--groups", default=None, help="comma list of instruction ids (a re-test page)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_intermixed_args(argv)
    if a.cmd == "build":
        build(a.workdir, a.name or Path(a.workdir).expanduser().name, seed=a.seed, videos=a.videos,
              include_rejected=a.include_rejected, only=a.groups.split(",") if a.groups else None)
    else:
        import_codes(a.workdir, a.codes)


if __name__ == "__main__":
    main()

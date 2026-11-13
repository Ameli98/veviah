"""Videos for a human to watch: one mp4 per instruction, its candidates in a 2 x 2 layout,
each shown from the front and from the side at 30 fps (real-time speed).

Needs the VRM renderer (Node + vrm_player) and `pip install imageio-ffmpeg` (a bundled ffmpeg binary).

CLI: python -m motion_loop.video --workdir W [--size 400] [--only id1,id2] [--per-candidate] [--tall]
Writes W/videos/<nn>_<instruction id>.mp4 and W/videos/index.txt (file name -> instruction text).
With --per-candidate: one unlabelled mp4 per candidate, W/videos/cand/<candidate id>.mp4 (for the
labelling page, which lays the candidates out itself); existing files are kept, so it can be resumed.
With --tall: each view is the 5:6 TALL_CROP of a --size square (e.g. --size 800 -> 600 x 720 per view);
use a multiple of 40 so the crop edges fall on whole pixels and the sizes stay even for H.264.
"""
import argparse
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .schema import FPS, load_hands, load_root, read_jsonl
from .vrm_render import TALL_CROP, VrmRenderer


def write_mp4(frames, path, fps=FPS, crf=20):
    """frames: list of equally sized PIL images -> H.264 mp4."""
    import imageio_ffmpeg
    w, h = frames[0].size
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-pix_fmt", "yuv420p", "-crf", str(crf), "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {path}")


def candidate_video(renderer, motions, labels, path, hold_s=0.5):
    """motions: up to 4 triples (motion [T,J,4], hand shapes [T,2], root [T,3]). Shorter ones hold their last
    frame; the end is held for hold_s."""
    T = max(len(m) for m, _, _ in motions) + int(hold_s * FPS)
    clips = []
    for (m, hands, root), label in zip(motions, labels):
        frames = renderer.render_frames(m, range(len(m)), hands, root)
        for f in frames:
            ImageDraw.Draw(f).text((6, 4), label, fill=(180, 0, 0))
        clips.append(frames + [frames[-1]] * (T - len(frames)))
    w, h = clips[0][0].size
    cols = 2 if len(clips) > 1 else 1
    rows = (len(clips) + cols - 1) // cols
    out = []
    for t in range(T):
        im = Image.new("RGB", (cols * w, rows * h), "white")
        for i, clip in enumerate(clips):
            im.paste(clip[t], ((i % cols) * w, (i // cols) * h))
        out.append(im)
    write_mp4(out, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--size", type=int, default=400, help="pixels per view; must be even")
    ap.add_argument("--only", default=None, help="comma list of instruction ids")
    ap.add_argument("--per-candidate", action="store_true")
    ap.add_argument("--cand-dir", default="cand", help="--per-candidate output folder under W/videos")
    ap.add_argument("--crf", type=int, default=20, help="H.264 quality, lower = better and larger")
    ap.add_argument("--tall", action="store_true", help="render each view as its tall 5:6 crop")
    ap.add_argument("--cpu", action="store_true", help="render with SwiftShader instead of the GPU (~12x slower)")
    ap.add_argument("--include-rejected", action="store_true", help="also render candidates the rule filter rejected")
    a = ap.parse_args(argv)
    w = Path(a.workdir).expanduser()
    out = w / "videos"
    out.mkdir(exist_ok=True)
    by_ins = {}
    for c in read_jsonl(w / "candidates.jsonl"):
        if c.get("filter", "pass") == "pass" or a.include_rejected:
            by_ins.setdefault(c["instruction_id"], []).append(c)
    text = {r["id"]: r["text"] for r in read_jsonl(w / "instructions.jsonl")}
    only = a.only.split(",") if a.only else None
    renderer = VrmRenderer(size=a.size, crop=TALL_CROP if a.tall else None, gpu=not a.cpu)
    index = []
    try:
        renderer._start()
        print("renderer:", renderer.info.get("gl"), flush=True)
        for n, (iid, cands) in enumerate(by_ins.items(), 1):
            if only and iid not in only:
                continue
            path = out / f"{n:02d}_{iid}.mp4"
            ms = [np.load(w / c["motion_path"]) for c in cands]
            if a.per_candidate:
                (out / a.cand_dir).mkdir(exist_ok=True)
                for m, c in zip(ms, cands):
                    cp = out / a.cand_dir / f"{c['id']}.mp4"
                    if not cp.exists():
                        frames = renderer.render_frames(m, range(len(m)), load_hands(w, c["motion_path"], len(m)),
                                                        load_root(w, c["motion_path"], len(m)))
                        write_mp4(frames + [frames[-1]] * (FPS // 2), cp, crf=a.crf)
                        print(cp, flush=True)
                continue
            candidate_video(renderer, [(m, load_hands(w, c["motion_path"], len(m)), load_root(w, c["motion_path"], len(m)))
                                       for m, c in zip(ms, cands)],
                            [f"candidate {i}" for i in range(len(cands))], path)
            index.append(f"{path.name}\t{text.get(iid, '')}")
            print(path, flush=True)
    finally:
        renderer.close()
    if not a.per_candidate:      # per-candidate runs have no index; keep the one from the grid videos
        (out / "index.txt").write_text("\n".join(index) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

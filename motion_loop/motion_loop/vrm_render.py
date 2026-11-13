"""Renderer that draws the real VRM: motion_loop -> headless three-vrm player (vrm_player/).

Use in the pipeline with `--renderer motion_loop.vrm_render:VrmRenderer`.

One Node process (vrm_player/tools/render_server.mjs) keeps a headless browser with the model
loaded; each render_frames call sends the motion and the frame indices over stdin and gets PNGs
back. Each returned frame is three views next to each other: front and side (whole body, floor to raised hand),
then a close-up of head and shoulders from the front (about 2.6x larger, for shrugs and small head
moves). The side view is from the character's left, so the character faces screen-left.

Settings come from the environment because the pipeline builds the renderer without arguments:
  HSIN_VRM_MODEL  model path under vrm_player/public/ (default: the player's default model)
  HSIN_NODE       node executable (default: PATH, then the newest ~/.nvm version)
"""
import base64
import io
import json
import numpy as np
import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

from .render import Renderer
from .vrm_export import to_vrm_json

PLAYER_DIR = Path(__file__).resolve().parents[2] / "vrm_player"


def find_node():
    node = os.environ.get("HSIN_NODE") or shutil.which("node")
    if node:
        return node
    found = sorted(Path.home().glob(".nvm/versions/node/*/bin/node"),
                   key=lambda p: [int(x) for x in p.parts[-3].lstrip("v").split(".")])
    return str(found[-1]) if found else None


# Tall crop of a square view for the labelling page: middle 3/4 in width, top 9/10 in height (5:6). The
# sides and the shins are empty in almost every motion.
TALL_CROP = (0.125, 0.0, 0.75, 0.9)


class VrmRenderer(Renderer):
    draws_hands = True
    grid_cols = 2          # 8 keyframes -> 2 x 4 frames of 3 views = 1920 x 1280 at size 320
    layout_note = ("Each keyframe shows the character three times, left to right: from the front (it faces you, "
                   "so its right hand is on your left); from its left side (it faces the left edge); and a "
                   "close-up of its head and shoulders from the front, where raised hands may be out of frame.")

    def __init__(self, model=None, size=320, views=("front", "side", "close"), player_dir=PLAYER_DIR, crop=None,
                 gpu=None, springs=True, extra_colliders=True, drape=True, limb_push=True, cloth_trial=None,
                 jiggle=None):
        """crop: (x, y, w, h) fractions of each square view to keep, e.g. TALL_CROP; None = whole view.
        gpu: render on the NVIDIA GPU (see render_server.mjs --gpu); None = env HSIN_RENDER_GPU=1.
        springs: False = hair and sleeves keep their rest shape (no physics), for judge views.
        extra_colliders: bigger hair colliders around the chest (vrm_player/src/colliders.js); default on since 2026-10-23.
        drape: skirts drape over the legs when sitting (vrm_player/src/skirt.js); False = plain spring physics.
        limb_push: mesh vertices inside a forearm / hand are pushed out on the GPU (vrm_player/src/limb_push.js).
        jiggle: chest sway level 'weak' | 'medium' | 'strong' (vrm_player/src/jiggle.js); None = off (judge views).
        After the first render, self.info["gl"] names the WebGL renderer actually used."""
        self.model = model or os.environ.get("HSIN_VRM_MODEL")
        self.size, self.views, self.player_dir, self.crop = size, list(views), Path(player_dir), crop
        self.gpu = os.environ.get("HSIN_RENDER_GPU") == "1" if gpu is None else gpu
        self.springs, self.extra_colliders, self.drape, self.limb_push = springs, extra_colliders, drape, limb_push
        self.cloth_trial, self.jiggle = cloth_trial, jiggle
        self.proc = None

    def _start(self):
        node = find_node()
        if not node:
            raise RuntimeError("node not found; set HSIN_NODE")
        cmd = [node, "tools/render_server.mjs", "--size", str(self.size)]
        if self.model:
            cmd += ["--model", self.model]
        if self.crop:
            cmd += ["--crop", ",".join(str(c) for c in self.crop)]
        if self.gpu:
            cmd.append("--gpu")
        if not self.springs:
            cmd.append("--no-springs")
        if not self.extra_colliders:
            cmd.append("--model-colliders")
        if not self.drape:
            cmd.append("--skirt-physics")
        if not self.limb_push:
            cmd.append("--no-limb-push")
        if self.cloth_trial:                # skirt.js experiments, a JSON object (setClothTrial)
            cmd += ["--cloth-trial", self.cloth_trial]
        if self.jiggle:
            cmd += ["--jiggle", self.jiggle]
        self.proc = subprocess.Popen(cmd, cwd=self.player_dir, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, text=True)
        self.info = self._read()

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"render server exited (code {self.proc.poll()})")
        return json.loads(line)

    def render_frames(self, motion, idx, hands=None, root=None, eyes=None, emotion=None):
        """eyes / emotion (2026-11-10): the eyes track and the face word, shown as the app shows them (review renders;
        judge renders pass neither and are unchanged)."""
        if self.proc is None:
            self._start()
        req = {"motion": to_vrm_json(motion, hands=hands, root=root, eyes=eyes, emotion=emotion),
               "frames": [int(i) for i in idx], "views": self.views}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        res = self._read()
        if not res.get("ok"):
            raise RuntimeError(f"render failed: {res.get('error')}")
        frames = []
        for row in res["images"]:
            tiles = [Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB") for b in row]
            frame = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height))
            x = 0
            for t in tiles:
                frame.paste(t, (x, 0))
                x += t.width
            frames.append(frame)
        return frames

    def cloth_points(self, motion, idx, hands=None, root=None, cmd="cloth"):
        """World positions [N, 3] of the skirt's cloth (vrm_player/src/skirt.js drape) at each frame of idx; empty
        arrays for a model without a skirt (2026-10-29: the legs first, then the cloth, then the hands).
        cmd="body": the skin component's vertices instead (debugging the cloth)."""
        if self.proc is None:
            self._start()
        req = {"cmd": cmd, "motion": to_vrm_json(motion, hands=hands, root=root), "frames": [int(i) for i in idx]}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        res = self._read()
        if not res.get("ok"):
            raise RuntimeError(f"cloth failed: {res.get('error')}")
        return [np.asarray(p, float).reshape(-1, 3) for p in res["points"]]

    def bone_points(self, motion, idx, bones, hands=None, root=None):
        """World positions of humanoid bones as drawn: [{bone: [x, y, z] or None}] per frame of idx (incl. fingers)."""
        if self.proc is None:
            self._start()
        req = {"cmd": "bones", "bones": list(bones), "motion": to_vrm_json(motion, hands=hands, root=root), "frames": [int(i) for i in idx]}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        res = self._read()
        if not res.get("ok"):
            raise RuntimeError(f"bones failed: {res.get('error')}")
        return res["points"]

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            self.proc.stdout.close()
        self.proc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class ClothProvider:
    """refine.CLOTH_PROVIDER: one renderer per model, kept open; returns the skirt's cloth points at frames.
    paths: {model name: .vrm file} for models outside vrm_player/public/models (the MCP user's own VRM)."""

    def __init__(self, gpu=None, paths=None):
        self.gpu, self.paths, self.renderers = gpu, dict(paths or {}), {}

    def __call__(self, model, motion, root, hands, frames):
        if model not in self.renderers:
            path = str(Path(self.paths[model]).resolve()) if model in self.paths else f"models/{model}.vrm"
            self.renderers[model] = VrmRenderer(model=path, size=64, views=("front",), gpu=self.gpu)
        return self.renderers[model].cloth_points(motion, frames, hands=hands, root=root)

    def close(self):
        for r in self.renderers.values():
            r.close()
        self.renderers = {}


class HttpClothProvider:
    """The Windows app's cloth provider (2026-11-13): POST <url>/cloth to the companion, which asks the app's hidden
    player window. Returns None when that is not possible (app closed, another model, error): the solver then uses
    the SKIRT_GAP rule, as without any provider."""

    def __init__(self, url, timeout=120):
        self.url, self.timeout, self.paths, self.renderers = url.rstrip('/'), timeout, {}, {}

    def _post(self, payload):
        import urllib.request
        req = urllib.request.Request(self.url + "/cloth", data=json.dumps(payload).encode(), headers={"content-type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def __call__(self, model, motion, root, hands, frames):
        try:
            res = self._post({"model": model, "motion": to_vrm_json(motion, hands=hands, root=root), "frames": [int(i) for i in frames]})
            return [np.asarray(p, float).reshape(-1, 3) for p in res["points"]]
        except Exception as e:
            print(f"[cloth] unavailable ({e}); solving without the draped cloth", file=sys.stderr)
            return None

    def warm(self, model):
        try:
            self._post({"model": model, "motion": None, "frames": []})
            return True
        except Exception:
            return False

    def close(self):
        pass


def use_cloth_provider(gpu=None):
    """Scripts: solve with the cloth first (refine.CLOTH_PROVIDER), renderers closed at exit."""
    import atexit
    from . import refine
    if refine.CLOTH_PROVIDER is None:
        refine.CLOTH_PROVIDER = ClothProvider(gpu=gpu)
        atexit.register(refine.CLOTH_PROVIDER.close)
    return refine.CLOTH_PROVIDER

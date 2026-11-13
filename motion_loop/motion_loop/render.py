"""Step 4: keyframes -> one grid image per candidate.

StickRenderer is a dependency-light stand-in (PIL stick figure via forward kinematics) so the
pipeline runs anywhere. For real scoring, swap in a renderer that draws your VRM (three-vrm
headless, Blender, pyrender) behind the same interface.
"""
import io

import numpy as np
from PIL import Image, ImageDraw

from .rotations import quat_to_mat
from .schema import FPS, JIDX, JOINTS

# parent, rest offset (metres, y-up). REST POSE = T-pose, same as VRM normalized humanoid and SMPL,
# so the identity rotation means T-pose and rotations retarget 1:1 (see retarget.py).
# The stick figure ignores the hips offset (root) and is framed for the upper body; legs leave the image.
_SK = {
    "hips": (None, (0, 0, 0)),
    "leftUpperLeg": ("hips", (.08, -.04, 0)), "leftLowerLeg": ("leftUpperLeg", (0, -.35, 0)), "leftFoot": ("leftLowerLeg", (0, -.41, 0)),
    "rightUpperLeg": ("hips", (-.08, -.04, 0)), "rightLowerLeg": ("rightUpperLeg", (0, -.35, 0)), "rightFoot": ("rightLowerLeg", (0, -.41, 0)),
    "spine": ("hips", (0, .10, 0)), "chest": ("spine", (0, .12, 0)), "upperChest": ("chest", (0, .10, 0)),
    "neck": ("upperChest", (0, .14, 0)), "head": ("neck", (0, .10, 0)),
    "leftShoulder": ("upperChest", (.04, .10, 0)), "leftUpperArm": ("leftShoulder", (.06, 0, 0)),
    "leftLowerArm": ("leftUpperArm", (.28, 0, 0)), "leftHand": ("leftLowerArm", (.26, 0, 0)),
    "rightShoulder": ("upperChest", (-.04, .10, 0)), "rightUpperArm": ("rightShoulder", (-.06, 0, 0)),
    "rightLowerArm": ("rightUpperArm", (-.28, 0, 0)), "rightHand": ("rightLowerArm", (-.26, 0, 0)),
}


def forward_kinematics(frame):
    """frame [J,4] quaternions -> (positions [J,3], parent positions [J,3])."""
    Rg, pos, par = {}, {}, {}
    for name in JOINTS:  # JOINTS is ordered parent-before-child
        parent, off = _SK[name]
        R_local = quat_to_mat(frame[JIDX[name]])
        if parent is None:
            Rg[name], pos[name], par[name] = R_local, np.array(off, float), np.zeros(3)
        else:
            Rg[name] = Rg[parent] @ R_local
            pos[name] = pos[parent] + Rg[parent] @ np.array(off, float)
            par[name] = pos[parent]
    return (np.stack([pos[n] for n in JOINTS]), np.stack([par[n] for n in JOINTS]))


class Renderer:
    layout_note = ""   # optional sentence for the judge prompt on how to read one keyframe

    def render_frames(self, motion, idx):
        """Return a list of PIL images, one per frame index in idx."""
        raise NotImplementedError


class StickRenderer(Renderer):
    def __init__(self, size=256, scale=300, yaw_deg=25):
        self.size, self.scale = size, scale
        a = np.deg2rad(yaw_deg)
        self.view = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])

    def _px(self, p):
        q = self.view @ p
        return (self.size / 2 + q[0] * self.scale, self.size * 0.86 - q[1] * self.scale)

    def render_frames(self, motion, idx):
        frames = []
        for i in idx:
            img = Image.new("RGB", (self.size, self.size), (245, 245, 245))
            d = ImageDraw.Draw(img)
            pos, par = forward_kinematics(motion[i])
            for k, name in enumerate(JOINTS):
                color = (40, 90, 200) if name.startswith("left") else (200, 60, 60) if name.startswith("right") else (70, 70, 70)
                d.line([self._px(par[k]), self._px(pos[k])], fill=color, width=4)
            hx, hy = self._px(pos[JIDX["head"]] + np.array([0, .06, 0]))
            d.ellipse([hx - 14, hy - 14, hx + 14, hy + 14], outline=(70, 70, 70), width=3)
            frames.append(img)
        return frames


def keyframe_indices(T, k=8):
    return np.linspace(0, T - 1, k).round().astype(int)


def make_grid(frames, times, cols=4):
    w, h = frames[0].size
    rows = -(-len(frames) // cols)
    grid = Image.new("RGB", (w * cols, h * rows), (255, 255, 255))
    d = ImageDraw.Draw(grid)
    for n, (f, t) in enumerate(zip(frames, times)):
        x, y = (n % cols) * w, (n // cols) * h
        grid.paste(f, (x, y))
        d.rectangle([x, y, x + w - 1, y + h - 1], outline=(200, 200, 200))
        d.text((x + 6, y + 4), f"t={t:.2f}s", fill=(0, 0, 0))
    return grid


def render_grid(renderer, motion, k=8, hands=None, root=None):
    """hands: optional hand shapes [T, 2] (see schema.HAND_SHAPES); root: optional hips offset [T, 3].
    Only renderers with draws_hands = True (the VRM one) get them."""
    idx = keyframe_indices(len(motion), k)
    draws = getattr(renderer, "draws_hands", False)
    frames = renderer.render_frames(motion, idx, hands, root) if draws else renderer.render_frames(motion, idx)
    return make_grid(frames, idx / FPS, cols=getattr(renderer, "grid_cols", 4))


def to_png(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

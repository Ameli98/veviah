"""Step 2: candidate motions. Real generator = pretrained text-to-motion + retarget to VRM."""
import copy
import json
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import rotations as rot
from .schema import FPS, J, JIDX, JOINTS, MIRROR_PERM, hands_path, rest_hands, rest_root, root_path


class MotionGenerator:
    last_hands = None      # a generator that also chooses hand shapes sets this in generate(): uint8 [T, 2]
    last_root = None       # a generator that moves the hips sets this in generate(): float32 [T, 3]

    def generate(self, instruction, seed, duration_s):
        """Return np.ndarray [T, J, 4] (unit quaternions xyzw, 30 fps, JOINTS order)."""
        raise NotImplementedError


class TextToMotionGenerator(MotionGenerator):
    """TODO (yours): wrap a pretrained text-to-motion model, then retarget its skeleton
    (typically SMPL) to the VRM joints in JOINTS as quaternions, resample to 30 fps (see retarget.py).
    The retarget is the first real technical hurdle; check bone axes and rest poses."""

    def generate(self, instruction, seed, duration_s):
        raise NotImplementedError("plug in a real text-to-motion model + retarget here")


# Euler amplitudes (x, y, z radians) for the mock sinusoids only; not a joint-limit table.
_MOCK_AMP = {"spine": (0.6, 0.6, 0.5), "chest": (0.5, 0.5, 0.4), "upperChest": (0.4, 0.4, 0.3),
             "neck": (0.8, 1.0, 0.5), "head": (0.8, 1.2, 0.6), "Shoulder": (0.3, 0.3, 0.5),
             "UpperArm": (2.5, 2.0, 2.5), "LowerArm": (0.5, 2.6, 0.3), "Hand": (1.0, 0.5, 0.8),
             "hips": (0, 0, 0), "UpperLeg": (0, 0, 0), "LowerLeg": (0, 0, 0), "Foot": (0, 0, 0)}   # mock stands still
MOCK_AMP = np.array([next(v for k, v in _MOCK_AMP.items() if n.endswith(k) or n == k) for n in JOINTS])


class MockGenerator(MotionGenerator):
    """Smooth random sinusoids. Ignores the text; for testing only."""

    def generate(self, instruction, seed, duration_s):
        rng = np.random.default_rng(zlib.crc32(instruction.encode()) + 7919 * seed)
        T = max(10, int(duration_s * FPS))
        t = np.arange(T) / FPS
        amp = rng.uniform(0.05, 0.5, (J, 3)) * MOCK_AMP
        freq = rng.uniform(0.2, 1.0, (J, 3))
        phase = rng.uniform(0, 2 * np.pi, (J, 3))
        return rot.from_euler(amp[None] * np.sin(2 * np.pi * freq[None] * t[:, None, None] + phase[None]))


# ---------- deliberate degradations (hard negatives for the reward model) ----------
def degrade(m, kind, rng):
    m = m.copy()
    if kind == "noise":          # high-frequency jitter: a small random rotation on every joint and frame
        return rot.quat_mul(m, rot.from_rotvec(rng.normal(0, 0.03, m.shape[:2] + (3,)))).astype(np.float32)
    if kind == "speed_slow":     # 0.5x speed
        return rot.resample(m, len(m) * 2).astype(np.float32)
    if kind == "speed_fast":     # 2x speed
        return m[::2].copy()
    if kind == "mirror":         # swap left/right and reflect each rotation in the x = 0 plane:
        m = m[:, MIRROR_PERM]    # a rotation's axis keeps its x component, y and z change sign
        m[..., 1:3] *= -1
        return m
    if kind == "freeze_legs":    # legs keep the first frame's pose while the hips move: feet slide or float
        idx = [JIDX[n] for n in JOINTS if n.endswith(("UpperLeg", "LowerLeg", "Foot"))]
        m[:, idx] = m[0, idx]
        return m
    if kind == "truncate":       # cut off early
        return m[: max(10, int(len(m) * 0.4))].copy()
    if kind == "freeze_arm":     # right arm stops moving
        idx = [JIDX[n] for n in JOINTS if n.startswith("right") and not n.endswith("Shoulder")]
        m[:, idx] = m[0, idx]
        return m
    if kind == "extreme":        # 3x every rotation angle: breaks limits, should be caught by the filter
        return rot.continuous(rot.from_rotvec(3 * rot.rotvec(m.astype(float)))).astype(np.float32)
    raise ValueError(kind)


def degrade_hands(hands, kind):
    """The hand shapes [T, 2] that go with degrade(m, kind): same length change, same left/right swap."""
    if kind == "speed_slow":
        return np.repeat(hands, 2, axis=0)
    if kind == "speed_fast":
        return hands[::2].copy()
    if kind == "mirror":
        return hands[:, ::-1].copy()
    if kind == "truncate":
        return hands[: max(10, int(len(hands) * 0.4))].copy()
    return hands


def degrade_root(root, kind):
    """The hips offset [T, 3] that goes with degrade(m, kind)."""
    if kind == "speed_slow":
        idx = np.linspace(0, len(root) - 1, len(root) * 2)
        return np.stack([np.interp(idx, np.arange(len(root)), root[:, i]) for i in range(3)], -1).astype(np.float32)
    if kind == "speed_fast":
        return root[::2].copy()
    if kind == "mirror":
        return root * np.array([-1, 1, 1], np.float32)
    if kind == "truncate":
        return root[: max(10, int(len(root) * 0.4))].copy()
    return root


DEGRADE_KINDS = ["noise", "speed_slow", "speed_fast", "mirror", "truncate", "freeze_arm", "extreme", "freeze_legs"]


def make_candidates(instructions, generator, motions_dir, n=4, p_degrade=0.15, p_mismatch=0.15, seed=0,
                    kinds=None, workers=1):
    """n candidates per instruction. Slot 0 is always a normal one, so the other slots
    use rescaled probabilities to hit the overall p_degrade / p_mismatch.
    workers > 1 generates in parallel threads, each with its own shallow copy of the generator
    (the last_* attributes are per call). A generator that keeps its raw output in last_keyframes
    gets it saved as motions/<instruction id>_g<k>.keyframes.json, so motions can be re-solved later.
    An instruction whose generation fails is dropped (printed), not fatal."""
    rng = np.random.default_rng(seed)
    kinds = kinds or DEGRADE_KINDS
    motions_dir = Path(motions_dir)
    motions_dir.mkdir(parents=True, exist_ok=True)
    def gen(job):
        ins, k = job
        g = copy.copy(generator)
        g.last_hands = g.last_root = None
        g.last_keyframes = None
        s = seed * 1000 + k
        try:
            m = g.generate(ins["text"], s, ins["spec"]["duration_s"])
        except RuntimeError as e:
            print(f"{ins['id']} g{k}: {e}", flush=True)
            return None
        kf = None
        if g.last_keyframes is not None:
            kf = f"motions/{ins['id']}_g{k}.keyframes.json"
            (motions_dir.parent / kf).write_text(json.dumps(
                {"instruction": ins["text"], "seed": s, "duration_s": ins["spec"]["duration_s"],
                 "keyframes": g.last_keyframes}, ensure_ascii=False))
        return (m, g.last_hands if g.last_hands is not None else rest_hands(len(m)),
                g.last_root if g.last_root is not None else rest_root(len(m)), kf)

    jobs = [(ins, k) for ins in instructions for k in range(n)]
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            res = list(ex.map(gen, jobs))
    else:
        res = [gen(j) for j in jobs]
    base = {}
    for (ins, k), r in zip(jobs, res):
        base.setdefault(ins["id"], []).append(r)
    instructions = [ins for ins in instructions if all(r is not None for r in base[ins["id"]])]
    base = {ins["id"]: base[ins["id"]] for ins in instructions}
    ids = list(base)
    pd, pm = p_degrade * n / (n - 1), p_mismatch * n / (n - 1)
    out = []
    for ins in instructions:
        for k in range(n):
            (motion, hands, root, kf), source, kind, other = base[ins["id"]][k], "normal", None, None
            if k > 0:
                u = rng.random()
                if u < pd:
                    source, kind = "degraded", kinds[rng.integers(len(kinds))]
                    motion, hands, root = degrade(motion, kind, rng), degrade_hands(hands, kind), degrade_root(root, kind)
                elif u < pd + pm and len(ids) > 1:
                    other = ids[rng.integers(len(ids))]
                    while other == ins["id"]:
                        other = ids[rng.integers(len(ids))]
                    source, (motion, hands, root, kf) = "mismatched", base[other][0]
            cid = f"{ins['id']}_c{k}"
            np.save(motions_dir / f"{cid}.npy", motion)
            assert len(hands) == len(motion) == len(root), (cid, kind, len(hands), len(root), len(motion))
            np.save(hands_path(motions_dir / f"{cid}.npy"), hands)
            np.save(root_path(motions_dir / f"{cid}.npy"), root)
            out.append({"id": cid, "instruction_id": ins["id"], "source": source,
                        "degrade_kind": kind, "mismatch_from": other,
                        "motion_path": f"motions/{cid}.npy", "keyframes_path": kf})
    return out

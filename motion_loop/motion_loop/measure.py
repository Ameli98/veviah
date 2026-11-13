"""Numbers measured from a solved motion, as text for an LLM to check its own keyframes against
(motion_loop.refine with feedback="numbers"). An LLM misreads small details in pictures (Haiku claimed
foot taps that were not there, 2026-10-22); measured numbers cannot be misread.

measure_text(motion, root, hands) gives a table at 8 evenly spaced times (the frames a judge sees) plus
counts of repeated movements (nods, swings, foot lifts). World coordinates of the default model, metres:
x = the character's left, y = up, z = forward.
"""
import numpy as np

from . import skeleton as sk
from .schema import FPS, HAND_SHAPES, JIDX

REF = ("Reference heights in the rest pose: top of head ~1.55 m, eyes ~1.45 m, shoulders ~1.27 m, "
       "chest ~1.15 m, waist ~0.95 m, hips 0.91 m.")


def _angles(G):
    """Head-style angles in degrees from world rotations [T, 3, 3]: nod / lean (+ = forward / down),
    turn (+ = toward its left), tilt / bend (+ = top toward its left)."""
    f, u = G[:, :, 2], G[:, :, 1]
    pitch = np.degrees(np.arcsin(np.clip(-f[:, 1], -1, 1)))
    turn = np.degrees(np.arctan2(f[:, 0], f[:, 2]))
    side = np.degrees(np.arcsin(np.clip(u[:, 0], -1, 1)))
    return pitch, turn, side


def swings(x, step):
    """Number of back-and-forth movements of x larger than `step` (turning points with hysteresis)."""
    n, ext, d = 0, x[0], 0
    for v in x[1:]:
        if d >= 0 and v > ext:
            ext = v
        elif d <= 0 and v < ext:
            ext = v
        if abs(v - ext) > step and np.sign(v - ext) != d:
            n += d != 0
            d, ext = np.sign(v - ext), v
    return n + (d != 0)


def measure_text(motion, root, hands=None):
    motion, root = np.asarray(motion, float), np.asarray(root, float)
    T = len(motion)
    P, G = sk.upper_fk(motion, root, JIDX)
    kp = sk.key_points(motion, root, JIDX)
    pt = {n: kp[:, i] for i, n in enumerate(sk.KEY_POINTS)}
    h_nod, h_turn, h_tilt = _angles(G["head"])
    t_lean, t_turn, t_bend = _angles(G["upperChest"])
    hips = root * sk.HIPS_Y * 100
    lift = {s: (pt[f"{s}Ankle"][:, 1] - sk.ANKLE_REST_Y) * 100 for s in ("left", "right")}
    step = {s: np.linalg.norm((pt[f"{s}Ankle"] - pt[f"{s}Ankle"][:1])[:, [0, 2]], axis=1) * 100 for s in ("left", "right")}
    idx = np.linspace(0, T - 1, 8).round().astype(int)
    rows = ["t(s) | head nod/turn/tilt deg | torso lean/turn/bend deg | R palm x,y,z m | L palm x,y,z m | "
            "hips side/down cm | R foot lift/moved cm | L foot lift/moved cm | hands R/L"]
    for i in idx:
        hs = ("/".join(HAND_SHAPES[int(h)] for h in hands[i][::-1]) if hands is not None else "relaxed/relaxed")
        rows.append(f"{i / FPS:.2f} | {h_nod[i]:.0f}/{h_turn[i]:.0f}/{h_tilt[i]:.0f} | {t_lean[i]:.0f}/{t_turn[i]:.0f}/{t_bend[i]:.0f} | "
                    + " | ".join(",".join(f"{v:.2f}" for v in pt[f"{s}Palm"][i]) for s in ("right", "left"))
                    + f" | {hips[i, 0]:.0f}/{-hips[i, 1]:.0f} | {lift['right'][i]:.0f}/{step['right'][i]:.0f} | "
                    f"{lift['left'][i]:.0f}/{step['left'][i]:.0f} | {hs}")
    summ = [
        f"Head: nod range {h_nod.min():.0f}..{h_nod.max():.0f} deg ({swings(h_nod, 5)} nods), turn {h_turn.min():.0f}..{h_turn.max():.0f} "
        f"({swings(h_turn, 8)} turns), tilt {h_tilt.min():.0f}..{h_tilt.max():.0f} ({swings(h_tilt, 5)} tilts).",
        f"Torso: lean {t_lean.min():.0f}..{t_lean.max():.0f} deg, turn {t_turn.min():.0f}..{t_turn.max():.0f}, bend {t_bend.min():.0f}..{t_bend.max():.0f}.",
    ]
    for s, S in (("right", "R"), ("left", "L")):
        p = pt[f"{s}Palm"]
        summ.append(f"{S} palm: height {p[:, 1].min():.2f}..{p[:, 1].max():.2f} m, forward {p[:, 2].min():.2f}..{p[:, 2].max():.2f}, "
                    f"side {p[:, 0].min():.2f}..{p[:, 0].max():.2f}; swings > 4 cm: up-down {swings(p[:, 1], 0.04)}, "
                    f"side-side {swings(p[:, 0], 0.04)}, forward-back {swings(p[:, 2], 0.04)}.")
        summ.append(f"{S} foot: highest lift {lift[s].max():.0f} cm, lifts > 2 cm: {int(np.sum(np.diff((lift[s] > 2).astype(int)) == 1) + (lift[s][0] > 2))}, "
                    f"moved up to {step[s].max():.0f} cm from its start.")
    summ.append(f"Hips: side shift {hips[:, 0].min():.0f}..{hips[:, 0].max():.0f} cm (+ = its left), lowered up to {-hips[:, 1].min():.0f} cm.")
    return REF + "\n" + "\n".join(rows) + "\n" + "\n".join(summ)

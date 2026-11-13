"""Arm words in keyframes (the author, 2026-10-27: shorter keyframes for the chat LLM in the MCP harness). An arm
may say where its hand is with three words instead of two direction vectors,

    "right_arm": {"hand": "face out close", "elbow": "down", "palm": "forward"}
    "left_arm": "rest"                              (the rest pose of the current posture)

and resolve_arm_words() turns them into "upper" / "fore" for the active model before parsing, as
contact.resolve_touches() does for touch words; the solver, filters and policy format are unchanged.

hand = "<height> <across> <reach>", in the torso frame (the arm directions are relative to the torso, so the
words follow a leaning or turning torso and work the same when sitting):
  height: above_head, head, face, shoulder, chest, waist, hip, low
  across: other_side (in front of the other shoulder), center, front (in front of its own shoulder), out, wide
  reach:  back (behind the body), beside (in line with the body), close (a fist's width in front of the body),
          mid, far (arm stretched toward that point; "above_head ... far" = straight up)
The point is the palm centre ("far": the wrist, the arm stretched toward it). elbow (optional) = down, out, forward, back, up: which way the elbow bends; the default
depends on where the hand is. Heights come from the model's own skeleton (skeleton.use_model).
"""
import copy

import numpy as np

from . import keyframes as kf
from . import skeleton as sk
from .schema import JIDX, JOINTS

HEIGHTS = ("above_head", "head", "face", "shoulder", "chest", "waist", "hip", "low")
ACROSS = ("other_side", "center", "front", "out", "wide")
REACH = ("back", "beside", "close", "mid", "far")
ELBOWS = ("down", "out", "forward", "back", "up")
FAR = 0.99                     # a "far" arm is this fraction of its full length (straight arms look locked;
                               # 0.97 left the elbow bent 28 deg, T026 2026-10-29: 0.99 = ~16 deg)


def _len(a, b):
    return float(np.linalg.norm(np.asarray(sk.UPPER_REST[b]) - np.asarray(sk.UPPER_REST[a])))


def hand_point(height, across, reach, side):
    """Wrist target in rest (torso) coordinates for this model, and the shoulder joint."""
    R = {k: np.asarray(v, float) for k, v in sk.UPPER_REST.items()}
    S = R[f"{side}UpperArm"]
    L = _len(f"{side}UpperArm", f"{side}LowerArm") + _len(f"{side}LowerArm", f"{side}Hand")
    sx = 1.0 if side == "left" else -1.0
    sy, hy = S[1], R["head"][1]
    y = {"above_head": sy + 0.9 * L, "head": hy + 0.11, "face": hy + 0.02, "shoulder": sy,
         "chest": R["upperChest"][1] - 0.02, "waist": R["spine"][1] + 0.04, "hip": R["hips"][1],
         "low": R["hips"][1] - 0.13}[height]
    x = sx * {"other_side": -abs(S[0]), "center": 0.0, "front": abs(S[0]), "out": abs(S[0]) + 0.18,
              "wide": abs(S[0]) + 0.55}[across]
    z = {"back": -0.18, "beside": 0.0, "close": 0.22, "mid": 0.32, "far": 0.32}[reach]
    if reach == "far" and height == "above_head":
        z = 0.0                                          # straight up, as in hands_meet (T026: it leaned 45 deg forward)
    W = np.array([x, y, R["upperChest"][2] + z])
    if reach == "far":                                   # stretch toward the point
        d = W - S
        W = S + d / max(np.linalg.norm(d), 1e-9) * FAR * L
    return W, S


def _pole(height, across, reach, elbow, side):
    sx = 1.0 if side == "left" else -1.0
    if elbow:
        return np.array({"down": (sx * 0.3, -1, 0), "out": (sx, -0.2, 0), "forward": (sx * 0.2, -0.4, 1),
                         "back": (sx * 0.3, -0.4, -1), "up": (sx * 0.6, 1, 0)}[elbow], float)
    if height in ("above_head", "head") and across in ("out", "wide", "front"):
        return np.array([sx, 0.1, -0.3])                 # raised hand: elbow out to the side (a wave)
    if height in ("above_head", "head", "face"):
        return np.array([sx * 0.7, -0.6, 0.3])           # hand at / over the face: elbow down and out
    if across == "other_side":
        return np.array([sx * 0.3, -0.7, 0.8])           # across the body: elbow in front, low
    if reach == "back":
        return np.array([sx * 0.4, -0.5, -1.0])
    return np.array([sx * 0.8, -1.0, -0.2])              # elsewhere: elbow down, a little out and back


def arm_dirs(hand, side, elbow=None):
    words = hand.split() if isinstance(hand, str) else list(hand)
    if len(words) != 3 or words[0] not in HEIGHTS or words[1] not in ACROSS or words[2] not in REACH:
        raise ValueError(f'hand must be "<height> <across> <reach>": {hand!r}; height {HEIGHTS}, '
                         f"across {ACROSS}, reach {REACH}")
    if elbow is not None and elbow not in ELBOWS:
        raise ValueError(f"elbow must be one of {ELBOWS}: {elbow!r}")
    P, S = hand_point(*words, side)
    return _solve(P, S, side, _pole(*words, elbow, side), at_wrist=words[2] == "far")


def _solve(P, S, side, pole, at_wrist=False):
    """Two-bone IK in the torso frame: upper / fore directions that put the palm centre (at_wrist: the wrist) at P."""
    UA = _len(f"{side}UpperArm", f"{side}LowerArm")
    FA = _len(f"{side}LowerArm", f"{side}Hand")
    W = P if at_wrist else P - (P - S) / max(np.linalg.norm(P - S), 1e-9) * sk.PALM
    for _ in range(3):                                   # the point is the palm: the wrist is a hand back along the forearm
        d = W - S
        L = float(np.clip(np.linalg.norm(d), abs(UA - FA) + 1e-3, UA + FA - 1e-3))
        n = d / max(np.linalg.norm(d), 1e-9)
        a = (UA ** 2 - FA ** 2 + L ** 2) / (2 * L)
        h = np.sqrt(max(UA ** 2 - a ** 2, 0.0))
        p = pole - (pole @ n) * n
        if np.linalg.norm(p) < 1e-6:                     # pole along the arm: any perpendicular
            p = np.cross(n, [0.0, 0.0, 1.0])
        p = p / np.linalg.norm(p)
        E = S + a * n + h * p
        fore = (S + n * L - E) / FA
        if at_wrist:
            break
        W = P - fore * sk.PALM
    return (E - S) / UA, fore


# Both hands meeting (2026-10-28, pose refs S011 / U002: hands joined above the head; an "other_hand" touch went to
# where the other hand was on its way, at chest height). One keyframe field places both arms, mirror images:
#   "hands_meet": {"at": "<height> <reach>", "how": "palms" | "fingertips", "elbow": e}
# palms: the palms pressed together on the body's midline, fingers along the forearms (prayer hands, hands joined
# overhead); fingertips: the fingers point inward and a little up and touch at the tips, elbows out (a roof /
# heart over the head). Heights and reaches are the arm words' (reach close / mid / far / beside).
MEET_HOW = ("palms", "fingertips")
MEET_PALM_X = 0.013        # palm centre off the midline: palms pressed (1.6 left a gap the author saw, 2026-11-06); hands_clip needs 2 cm between bone points
PLANE_TURN = 45        # deg the joined palms' plane may turn about the lead hand (meet_on_arm)
SWIVEL_ALIGN = 60          # deg: how far a joined-palms elbow may swivel to align the fingers (_palms_aligned)
MEET_TIP_X = 0.013         # fingertip point off the midline: the hands_clip rule needs >= 2 cm between them
MEET_TIP_DIR = (1.0, 0.6, 0.0)     # finger direction (inward, up, forward) for "fingertips"
MEET_TIP_UP = {"above_head": 0.06}  # fingertips meet a hand higher (U002: the roof sat on the hair)


def meet_arms(at, how="palms", elbow=None, offset=0.0, lead=None):
    """-> {"right_arm": fields, "left_arm": fields} for a hands_meet keyframe. offset (cm): the right hand this much
    higher and the left this much lower along the palms (rubbing the hands is a few keyframes of +- offset).
    elbow may be one word or {"left": word, "right": word}. lead ("left" / "right"): that arm keeps its elbow and
    the other one swivels to lay its fingers on it (default: both meet half way). reach "back" = behind the back
    (hands clasped there, 2026-10-28). A pose read from a reference goes through "lead_arm" (meet_on_arm) instead
    (shift / tilt / raise / forward / uppers were removed 2026-11-08: lead_arm covers them)."""
    words = at.split() if isinstance(at, str) else list(at)
    if len(words) != 2 or words[0] not in HEIGHTS or words[1] not in REACH:
        raise ValueError(f'hands_meet "at" must be "<height> <reach>": {at!r}; height {HEIGHTS}, reach {REACH}')
    if how not in MEET_HOW:
        raise ValueError(f"hands_meet how must be one of {MEET_HOW}: {how!r}")
    elbows = elbow if isinstance(elbow, dict) else {"left": elbow, "right": elbow}
    for e in elbows.values():
        if e is not None and e not in ELBOWS:
            raise ValueError(f"elbow must be one of {ELBOWS}: {e!r}")
    if lead not in (None, "left", "right"):
        raise ValueError(f'lead must be "left" or "right": {lead!r}')
    out, palms = {}, {}
    for side in ("right", "left"):
        sx = 1.0 if side == "left" else -1.0
        P, S = hand_point(words[0], "center", "beside" if words[1] == "far" else words[1], side)
        if words[1] == "far":
            # arms stretched: the meeting point on the midline at that height, as far forward as the arms reach
            # (above_head far = straight up over the head, the highest point it can reach)
            reach_ = FAR * (_len(f"{side}UpperArm", f"{side}LowerArm") + _len(f"{side}LowerArm", f"{side}Hand") + sk.PALM)
            dx, dy = sx * MEET_PALM_X - S[0], P[1] - S[1]
            dz2 = reach_ ** 2 - dx ** 2 - dy ** 2
            if dz2 >= 0 and words[0] != "above_head":
                P = np.array([P[0], P[1], S[2] + np.sqrt(dz2)])
            else:
                P = np.array([P[0], S[1] + np.sqrt(max(reach_ ** 2 - dx ** 2, 0.0)), S[2]])
        pole = _pole(words[0], "out", words[1], elbows.get(side) or ("out" if words[0] in ("above_head", "head") else None), side)
        P = P + np.array([0.0, -sx * float(offset) / 100.0, 0.0])      # right (sx -1) up for offset > 0
        if how == "palms":
            # the hands lie flat against each other, parallel to the midline: a forearm slanting in would carry its
            # hand across the other (arms straight up crossed the fingers), so the wrist bends back out. n = the joined
            # palms' normal; the palm faces -sx n. The fingers point along the forearm projected onto the palms' plane;
            # both hands are aligned afterwards.
            n = np.array([1.0, 0.0, 0.0])
            P = P + sx * MEET_PALM_X * n
            u, f = _solve(P, S, side, pole)
            for _ in range(3):
                h = f - (f @ n) * n
                h = h / max(np.linalg.norm(h), 1e-9)
                u, f = _solve(P - h * sk.PALM, S, side, pole, at_wrist=True)
            palms[side] = dict(P=P, S=S, pole=pole, n=n, h=h, sx=sx)
            continue
        else:
            P = P + np.array([0.0, MEET_TIP_UP.get(words[0], 0.0), 0.0])
            h = np.array(MEET_TIP_DIR) * np.array([-sx, 1.0, 1.0])
            h = h / np.linalg.norm(h)
            W = P + np.array([sx * MEET_TIP_X, 0.0, 0.0]) - h * sk.FINGERTIP
            u, f = _solve(W, S, side, pole, at_wrist=True)
            bend = np.degrees(np.arccos(np.clip(f @ h, -1, 1)))
            fields = {"palm": "in", "wrist": round(float(np.clip(bend, *kf.WRIST_LIM)), 1)}
        out[f"{side}_arm"] = {"upper": [round(float(x), 3) for x in u], "fore": [round(float(x), 3) for x in f], **fields}
    if palms:
        out.update(_palms_aligned(palms, lead))
    return out


def meet_on_arm(lead, upper, fore, wrist=0.0, elbow=None):
    """Joined palms copied from a reference (2026-11-06, the author: copy the pose, palms and fingers pressed): the lead arm
    is placed exactly as read (upper / fore directions in the torso frame, a straight wrist unless wrist is given), and
    the other hand presses its palm on the lead palm: same plane, palm to palm, its elbow swivelled (up to SWIVEL_ALIGN)
    so its fingers lie as close to the lead fingers as the arm allows (joined hands may cross a little, as in the
    reference). The lead palm faces the other shoulder's side (where the other hand comes from)."""
    other = "right" if lead == "left" else "left"
    u = _unit_np(upper); f = _unit_np(fore)
    _, S = hand_point("chest", "center", "close", lead)
    _, So = hand_point("chest", "center", "close", other)
    UA, FA = _len(f"{lead}UpperArm", f"{lead}LowerArm"), _len(f"{lead}LowerArm", f"{lead}Hand")
    W = S + u * UA + f * FA
    h = f                                                  # straight wrist (wrist bend: about the palm axis below)
    toward = (So - W) - ((So - W) @ h) * h
    n = toward / max(np.linalg.norm(toward), 1e-9)          # the lead palm faces this way
    if wrist:
        a = np.radians(float(wrist))                       # + = toward the palm
        h = _unit_np(h * np.cos(a) + n * np.sin(a))
        n = _unit_np(n - (n @ h) * h)
    P = W + h * sk.PALM
    pole = _pole("chest", "out", "close", elbow, other)
    # the joined palms' plane may turn about the lead hand (PLANE_TURN, 2026-11-07, Grace: with the lead palm fixed
    # the other hand could only swivel its elbow and the fingers crossed in an X); the turn costs a little, so the
    # plane read from the reference is kept when the fingers already lie together
    n0, best = n, None
    turn = PLANE_TURN
    for rot in range(-turn, turn + 1, 5):
        a = np.radians(rot)
        n = _unit_np(n0 * np.cos(a) + np.cross(h, n0) * np.sin(a))
        Po = P + n * 2 * MEET_PALM_X                       # the other palm centre, on the lead palm
        for deg in range(-SWIVEL_ALIGN, SWIVEL_ALIGN + 1, 4):
            ho = h
            for _ in range(3):
                Wo = Po - ho * sk.PALM
                axis = _unit_np(Wo - So)
                uo, fo = _solve(Wo, So, other, _swivel(pole, axis, np.radians(deg)), at_wrist=True)
                ho = fo - (fo @ n) * n
                ho = ho / max(np.linalg.norm(ho), 1e-9)
            err = np.arccos(np.clip(ho @ h, -1, 1)) + 0.002 * abs(deg) + 0.004 * abs(rot)
            if best is None or err < best[0]:
                best = (err, uo, fo, ho, n)
    _, uo, fo, ho, n = best
    bend = np.degrees(np.arccos(np.clip(fo @ ho, -1, 1)))
    sgn = -1.0 if fo @ (-n) > 0 else 1.0
    r3 = lambda v: [round(float(x), 3) for x in v]
    return {f"{lead}_arm": {"upper": r3(u), "fore": r3(f), "palm": r3(n), "wrist": round(float(wrist), 1)},
            f"{other}_arm": {"upper": r3(uo), "fore": r3(fo), "palm": r3(-n), "wrist": round(float(np.clip(sgn * bend, *kf.WRIST_LIM)), 1)}}


def _unit_np(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def _swivel(v, axis, ang):
    """v turned by ang (rad) about the unit axis (Rodrigues)."""
    return v * np.cos(ang) + np.cross(axis, v) * np.sin(ang) + axis * (axis @ v) * (1 - np.cos(ang))


def _palms_aligned(palms, lead=None):
    """Joined palms: both hands' fingers along one direction (2026-11-06, the author: the fingers must lie on each other
    too). Each hand's fingers point along its forearm projected onto the palms' plane; with the hands off the midline or
    tilted the two forearms are no longer mirror images and the fingers splayed up to ~25 deg apart. The common direction
    is the mean of the two; each arm's elbow then swivels about its shoulder-wrist line (up to SWIVEL_ALIGN deg) until its
    forearm's projection points that way, and the wrist bends the rest."""
    hc = palms[lead]["h"] if lead else sum(p["h"] for p in palms.values())
    hc = hc / max(np.linalg.norm(hc), 1e-9)
    out = {}
    for side, p in palms.items():
        P, S, n = p["P"], p["S"], p["n"]
        W = P - hc * sk.PALM
        axis = (W - S) / max(np.linalg.norm(W - S), 1e-9)
        best = None
        for deg in ([0] if side == lead else range(-SWIVEL_ALIGN, SWIVEL_ALIGN + 1, 2)):
            u, f = _solve(W, S, side, _swivel(p["pole"], axis, np.radians(deg)), at_wrist=True)
            v = f - (f @ n) * n
            v = v / max(np.linalg.norm(v), 1e-9)
            err = np.arccos(np.clip(v @ hc, -1, 1)) + 0.002 * abs(deg)      # aligned first, then the least swivel
            if best is None or err < best[0]:
                best = (err, u, f)
        _, u, f = best
        bend = np.degrees(np.arccos(np.clip(f @ hc, -1, 1)))
        fields = {"palm": "in", "wrist": round(float(np.clip(-bend, *kf.WRIST_LIM)), 1)}
        out[f"{side}_arm"] = {"upper": [round(float(x), 3) for x in u], "fore": [round(float(x), 3) for x in f], **fields}
    return out


# Crossed arms (2026-11-03, the author: forearms went through each other in U030 / U041 / U017 and handmade h0036; solved
# arm by arm, each hand's target lay on the other arm). "arms_cross": {"how": "fold" | "hug", "top": "left" | "right"}
# places both arms at once:
#   fold = arms folded below the chest: the lower forearm lies across the body against it, the top one in front of it
#          and a little higher, each hand wrapping the other arm's upper arm near the elbow from outside;
#   hug  = self-hug, each hand on the front of the other shoulder, the top forearm in front of the lower one.
CROSS_HOW = ("fold", "hug", "cover")
COVER_PAST = 0.03        # cover: the wrists just past the midline under the chest ...
COVER_HAND = (0.06, 0.06, 0.012)   # ... each palm this far further across / up / off the chest front (on the other breast)
CROSS_Y = -0.07           # m below the upperChest joint: where the lower forearm lies
CROSS_PAST = 0.07         # m past the midline the lower wrist reaches (its hand goes on to the other upper arm)
CROSS_UP = 0.025          # m the top forearm lies higher than the lower one
CROSS_GAP = 0.004         # m between the two forearm surfaces
HUG_UP = 0.01             # hug: the palm this far above the other shoulder joint ...
HUG_OFF = 0.02            # ... and this far in front of the chest's surface


def _rest_front(x, y):
    """z of the torso's front surface (the filter's torso shape) at rest, at (x, y) in rest coordinates."""
    m = np.zeros((1, len(JOINTS), 4)); m[..., 3] = 1
    P, G = sk.upper_fk(m, np.zeros((1, 3)), JIDX)
    lo, hi = sk.UPPER_REST["upperChest"][2], sk.UPPER_REST["upperChest"][2] + 0.4
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        d = sk.torso_depth(np.array([[x, y, mid]]), P, G)[0][0]
        lo, hi = (mid, hi) if np.isfinite(d) and d > 0 else (lo, mid)
    return hi


def _clothed_front(x, y):
    """z of the torso's front INCLUDING clothes at rest (fit front_grid, measure.js; the 3 x 3 cells round (x, y)), or
    -inf when the fit has none (2026-11-07: hands placed on the skin-only torso lay inside the loose T-shirt)."""
    from . import contact
    g = (contact.load_fit(sk.MODEL) or {}).get("front_grid")
    if not g:
        return -np.inf
    c, i, j = g["cell"], int(np.floor(x / g["cell"])), int(np.floor(y / g["cell"]))
    zs = [g["z"].get(f"{i + a},{j + b}") for a in (-1, 0, 1) for b in (-1, 0, 1)]
    zs = [z for z in zs if z is not None and z > 0]
    return max(zs) if zs else -np.inf


def _front(x, y, skin_gap, cloth_gap):
    """z a hand / forearm should lie at in front of the torso at rest: skin_gap off the skin torso, cloth_gap off the
    clothes, whichever is further out."""
    return max(_rest_front(x, y) + skin_gap, _clothed_front(x, y) + cloth_gap)


def cross_arms(how="fold", top="left", offset=0.0):
    """-> {"right_arm": fields, "left_arm": fields} for an arms_cross keyframe. offset (cm): both hands this much lower
    along the other arm (rubbing the upper arms = keyframes alternating offsets)."""
    if how not in CROSS_HOW:
        raise ValueError(f"arms_cross how must be one of {CROSS_HOW}: {how!r}")
    if top not in ("left", "right"):
        raise ValueError(f'arms_cross top must be "left" or "right": {top!r}')
    low = "right" if top == "left" else "left"
    R = {k: np.asarray(v, float) for k, v in sk.UPPER_REST.items()}
    rf = sk.ARM_R["forearm"]
    out, segs = {}, {}
    y0 = R["upperChest"][1] + CROSS_Y
    for side in (low, top):
        sx = 1.0 if side == "left" else -1.0
        S = R[f"{side}UpperArm"]
        if how == "fold":
            # the wrist past the midline against the chest (the top one higher); the hand goes on across to the other
            # upper arm near its elbow, palm toward it
            y = y0 + (CROSS_UP if side == top else 0.0) - float(offset) / 100.0
            W = np.array([-sx * CROSS_PAST, y, _rest_front(-sx * CROSS_PAST, y) + rf + 0.003])
            h = np.array([-sx, 0.15, -0.35])
            P = None
        elif how == "cover":
            # covering the chest (2026-11-07, shy_cover, ai-pose-gallery U053): the forearms crossed under the chest,
            # each palm on the other side of the chest (on the breast), fingers out and a little up
            y = y0 + (CROSS_UP if side == top else 0.0) - float(offset) / 100.0
            W = np.array([-sx * COVER_PAST, y, _front(-sx * COVER_PAST, y, rf + 0.003, rf)])
            px, py = -sx * (COVER_PAST + COVER_HAND[0]), y + COVER_HAND[1]
            P = np.array([px, py, _front(px, py, COVER_HAND[2], 0.012)])   # on the clothes: half a hand off them
            h = P - W
        else:
            # the palm on the front-top of the other shoulder, the fingers over it (up and out); the wrist a hand back
            So = R[f"{'right' if side == 'left' else 'left'}UpperArm"]
            P = So + np.array([0.0, HUG_UP - float(offset) / 100.0, 0.0])
            P[2] = _rest_front(P[0], min(P[1], R["upperChest"][1] + 0.05)) + HUG_OFF
            h = np.array([-sx * 0.45, 0.75, -0.45])
            W = P - h / np.linalg.norm(h) * sk.PALM
        pole = np.array([sx * 0.9, -0.5, 0.6])           # elbow out, down and forward
        if how == "cover":                               # elbows down beside the body (U053; out, it held a box)
            pole = np.array([sx * 0.35, -1.0, 0.25])
        W0 = W.copy()
        for _ in range(12):
            u, f = _solve(W, S, side, pole, at_wrist=True)
            E = S + u * _len(f"{side}UpperArm", f"{side}LowerArm")
            if side == low:
                break
            # the top forearm in front of the lower one: its wrist moves forward until they clear (raising the
            # elbow instead put the forearm across the mouth in a self-hug)
            Eo, Wo = segs[low]
            uu = np.linspace(0, 1, 11)
            A, B = E + uu[:, None] * (W - E), Eo + uu[:, None] * (Wo - Eo)
            gap = np.linalg.norm(A[:, None] - B[None], axis=-1).min() - 2 * rf - CROSS_GAP
            if gap >= 0:
                break
            if how == "cover":                           # elbows down: the forearms cross at an angle. Pushing the
                # wrist forward sent the top arm straight out, raising it put the hand on the neck (2026-11-07): the
                # top elbow comes forward (its forearm passes in front of the other), the wrist moves at most 4 cm
                if pole[2] < 0.25 + 2 * 0.2:             # at most twice (12 times raised the elbow to the shoulder)
                    pole = pole + np.array([0.0, 0.0, 0.2])
                if np.linalg.norm(W - W0) < 0.04:
                    W = W + np.array([0.0, 0.0, min(-gap + 0.005, 0.01)])
                continue
            else:
                W = W + np.array([0.0, 0.0, -gap + 0.005])
        segs[side] = (E, W)
        if P is not None:                                # the hand reaches back to the shoulder from the wrist
            h = P - W
        h = h / np.linalg.norm(h)
        bend = np.degrees(np.arccos(np.clip(f @ h, -1, 1)))
        out[f"{side}_arm"] = {"upper": [round(float(v), 3) for v in u], "fore": [round(float(v), 3) for v in f],
                              "palm": "back", "wrist": round(float(np.clip(bend, *kf.WRIST_LIM)), 1)}
    return out


# Sitting rest arms on other models (2026-10-28): the fixed directions were found on the default model; on smaller or
# narrower models (test1, AvatarSample_D) the hands met in the middle of the lap (hands_clip in legs_out, w_sit, seiza).
# There the rest is a touch on the model's own thighs (shins when hugging the knees). The default model keeps the
# directions the author approved, so its data and pictures do not change.
REST_TOUCH = {"knees_hugged": "shin", "kneel": None, "stand": None}


# knees_hugged as a curled-up hug (2026-11-11, the author: the old one only had the hands resting on the knees, back
# straight, on every model): forearms crossed in front of the shins (each hand's finger pads on the OTHER shin), the
# torso leaning forward, the head bowed, shoulders forward, knees together. The body fields fill in what a keyframe of
# that posture leaves out (kf.SIT_BODY); the arms are the rest arms on every model, the default one included.
HUG_ARM = {"touch": "other_shin", "with": "fingers", "elbow": "out"}


def rest_arm(posture, side):
    """The rest arm of a posture as keyframe fields (a touch word on models other than the default one)."""
    if posture == "knees_hugged":
        return dict(HUG_ARM)
    if posture == "ledge":                            # hands on the edge beside the hips (2026-11-11)
        # palms on the thighs (2026-11-11: hands flat on the edge beside the hips cannot be reached with the torso
        # upright, the solver put them far out to the sides like props; the floor_edge region stays for a keyframe that
        # leans on a hand)
        return {"touch": "thigh", "with": "palm"}
    if posture in ("side_right", "side_left"):        # leaning on the hand away from the feet (2026-11-11, the author: both
        # hands on the thighs looked like a posed doll); the feet are on the posture's side
        prop = "left" if posture == "side_right" else "right"
        return {"touch": "floor", "with": "palm"} if side == prop else {"touch": "thigh", "with": "palm"}
    if posture == "seiza":                            # hands on the thighs on every model (2026-11-11: the default
        return {"touch": "thigh", "with": "palm"}     # model's direction rest put them near the groin)
    region = REST_TOUCH.get(posture, "thigh")
    if region:                                        # every model alike (2026-11-13: no default-model directions)
        # knees_hugged: elbows down (2026-11-11: with "elbow_on": "knee" the elbows rose above the head on test1 / test2,
        # seen on the sitting rest renders)
        return {"touch": region, "with": "palm", **({"elbow": "down"} if posture == "knees_hugged" else {})}
    if posture == "stand":
        u, f = kf.REST_ARM[side]
        return {"upper": list(u), "fore": list(f), "palm": "natural", "wrist": 0.0}
    if posture == "knees_hugged":
        u, f = kf.REST_ARM_HUG[side]
        palm, wrist = kf.REST_HAND_HUG[side]
        return {"upper": list(u), "fore": list(f), "palm": palm, "wrist": wrist}
    u, f, palm = kf.REST_SIT[posture][side]
    return {"upper": list(u), "fore": list(f), "palm": palm, "wrist": 0.0}


def resolve_arm_words(obj):
    """obj: keyframe JSON. Returns a copy whose arms given as words ("hand", or "rest") have upper / fore."""
    obj = copy.deepcopy(obj)
    posture = obj.get("posture", "stand")
    kfs = obj.get("keyframes", [])
    wall = obj.pop("wall", None)
    if wall is not None and posture == "stand":     # leaning on a wall: its body where a keyframe leaves it out
        if wall not in kf.WALL_BODY:
            raise ValueError('wall must be "left" or "right"')
        arms_given = any(f in k for k in kfs for f in ("right_arm", "left_arm", "hands_meet", "arms_cross"))
        for k in kfs:
            for field, val in kf.WALL_BODY[wall].items():
                if field == "legs":                 # the legs and the slant always stay leaning (2026-11-11, the author: a
                    k["legs"] = copy.deepcopy(val)  # touch reaction's own legs moved her off the wall)
                    continue
                if field == "torso":
                    k["torso"] = {**(k.get("torso") or {}), **copy.deepcopy(val)}
                    continue
                if field == "arms_cross":
                    # folded arms: the rest while leaning. A motion that leaves the arms alone folds throughout; one
                    # that moves them starts and ends folded wherever it says "rest" for both arms (a touch reaction
                    # from hanging arms swung them out through the wall / into the body: 6 of 17 passed)
                    both_rest = k.get("right_arm") == "rest" and k.get("left_arm") == "rest"
                    if not arms_given or both_rest:
                        k.pop("right_arm", None); k.pop("left_arm", None)
                        k["arms_cross"] = copy.deepcopy(val)
                elif field == "legs":
                    if isinstance(k.get("legs"), dict) or "legs" not in k:
                        k["legs"] = {**copy.deepcopy(val), **(k.get("legs") or {})}
                else:
                    k.setdefault(field, copy.deepcopy(val))
    for k in kfs:                                   # the posture's own body (knees_hugged: curled up), where not given
        for field, val in kf.SIT_BODY.get(posture, {}).items():
            if field == "legs":
                if isinstance(k.get("legs"), dict) or "legs" not in k:
                    k["legs"] = {**val, **(k.get("legs") or {})}
            else:
                k.setdefault(field, copy.deepcopy(val))
    if (posture in ("knees_hugged", "side_right", "side_left", "seiza", "ledge") or REST_TOUCH.get(posture, "thigh")) and kfs:
        # the motion starts from the rest pose: say so, so it is this model's rest (see REST_TOUCH)
        if float(kfs[0].get("t", 0)) > 1e-6:
            kfs.insert(0, {"t": 0.0})
        for side in ("right", "left"):
            kfs[0].setdefault(f"{side}_arm", "rest")
    for k in obj.get("keyframes", []):
        if "hands_meet" in k:
            meet = k.pop("hands_meet")
            if "right_arm" in k or "left_arm" in k:
                raise ValueError('a keyframe with "hands_meet" places both arms; leave out right_arm / left_arm')
            if not isinstance(meet, dict) or ("at" not in meet and "lead_arm" not in meet):
                raise ValueError('hands_meet must be {"at": "<height> <reach>", "how": "palms" | "fingertips"} or {"lead_arm": ...}')
            bad = set(meet) - {"at", "how", "elbow", "offset", "lead", "lead_arm"}
            if bad:
                raise ValueError(f"unknown hands_meet fields {sorted(bad)}")
            if isinstance(meet.get("lead_arm"), dict):        # the lead arm as read from a reference (meet_on_arm)
                la = meet["lead_arm"]
                k.update(meet_on_arm(meet.get("lead", "left"), la["upper"], la["fore"], la.get("wrist", 0.0), meet.get("elbow")))
            else:
                k.update(meet_arms(meet["at"], meet.get("how", "palms"), meet.get("elbow"), meet.get("offset", 0.0), meet.get("lead")))
            hands = k.setdefault("hands", {})
            for side in ("left", "right"):
                hands.setdefault(side, "flat" if meet.get("how", "palms") == "palms" else "open")   # curled fingers of joined hands go through each other; open fanned them (gaps)
        if "arms_cross" in k:
            cr = k.pop("arms_cross")
            if "right_arm" in k or "left_arm" in k or "hands_meet" in k:
                raise ValueError('a keyframe with "arms_cross" places both arms; leave out right_arm / left_arm')
            cr = cr if isinstance(cr, dict) else {"how": cr}
            k.update(cross_arms(cr.get("how", "fold"), cr.get("top", "left"), cr.get("offset", 0.0)))
            if cr.get("how") == "cover":                  # the hands lie flat on the chest (U053)
                hands = k.setdefault("hands", {})
                for side in ("left", "right"):
                    hands.setdefault(side, "flat")
        for side in ("right", "left"):
            arm = k.get(f"{side}_arm")
            if arm == "rest":
                k[f"{side}_arm"] = rest_arm(posture, side)
            elif isinstance(arm, dict) and "hand" in arm and "touch" not in arm:   # with a touch, "hand" aims the forearm
                u, f = arm_dirs(arm["hand"], side, arm.get("elbow"))
                new = {key: v for key, v in arm.items()}
                new.update(upper=[round(float(x), 3) for x in u], fore=[round(float(x), 3) for x in f])
                k[f"{side}_arm"] = new
    return obj



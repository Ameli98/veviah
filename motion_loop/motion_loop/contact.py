"""Contact words in keyframes (the author, 2026-10-23): an arm may say where the hand touches the body instead of
giving its directions,

    "right_arm": {"touch": "nape", "with": "palm", "how": "rest"}

and resolve_touches() turns those into "upper" / "fore" / "palm" for one body model, so the rest of the
pipeline (parse_keyframes, the solver, filters) is unchanged. The JSON keeps the words; the directions depend
on the model (its body fit), which is how one motion can serve any VRM.

How a touch is resolved, per keyframe:
1. solve the whole keyframe list once without the touches (the arms do not move the body) and take the
   body pose at that keyframe's time;
2. find the region's point on the body surface: a point fixed to a bone (offset from the bone's joint in the
   rest pose, from the model's fitted surface, body_fits/<model>.json written by
   vrm_player/tools/fit_capsules.mjs), carried by that bone's rotation; legs and floor use the solved leg points;
3. place the palm (or fingertips / back of the hand / fist) on it, a hand's half thickness off the surface,
   and solve the arm with two-bone IK in the torso frame (elbow out and back); a target out of reach puts the
   hand as close as the arm allows;
4. pick the palm word that best faces the surface.

Only "rest" and "hold" move the hand to the target; "rub" and "tap" are accepted and stored but resolved as
"rest" for now. The body FK is the canonical (default model) skeleton; a fit of another model only changes
the surface, until per-model skeletons are in.
"""
import copy
import json
from pathlib import Path

import numpy as np

from . import keyframes as kf
from . import skeleton as sk
from .schema import JIDX, FPS

FITS = Path(__file__).parent / "body_fits"
UA, FA = 0.2198, 0.2147                    # upper arm / forearm of the active model (skeleton.use_model)


@sk.on_model_change
def _arm_lengths():
    global UA, FA, W_MIN
    r = {n: np.asarray(sk.UPPER_REST[n]) for n in ("leftUpperArm", "leftLowerArm", "leftHand")}
    UA, FA = float(np.linalg.norm(r["leftLowerArm"] - r["leftUpperArm"])), float(np.linalg.norm(r["leftHand"] - r["leftLowerArm"]))
    W_MIN = float(np.sqrt(UA ** 2 + FA ** 2 - 2 * UA * FA * np.cos(np.pi - kf.MAX_ELBOW))) + 0.005
# W_MIN = the closest the wrist comes to the shoulder (elbow at kf.MAX_ELBOW, + 5 mm), per model, set by _arm_lengths
# (2026-11-03: a constant 0.135 here overwrote it when the module was first imported inside skeleton.use_model, so the
# first solve of a process on another model used the default model's value)
HAND_HALF = 0.015                          # half the hand's thickness
SKIRT_GAP = 0.02                           # a skirt over the leg: its cloth lies this far off the skin (skirt.js SIM_GAP + cloth)
CLOTH = None                               # {frame: cloth points [N, 3]} of the current solve (refine.solve), or None
CLOTH_LAT, CLOTH_HALF = 0.06, 0.003       # cloth points this close to the contact's normal line count; half its thickness
CLOTH_FOOT = 0.08                          # the hand's footprint on the cloth (palm to fingertips)
FINGERS_AT = 0.73                          # "with": "fingers": the finger pads, this fraction of the way to the fingertip
LEG_SIDE_BY_SIDE = 0.075                   # two hands on one leg spot: the second this far along the leg
ELBOW_POLES = {"down": (0.3, -1, 0), "out": (1, -0.2, 0), "forward": (0.2, -0.4, 1), "back": (0.3, -0.4, -1),
               "up": (0.6, 1, 0), "together": (-0.4, -0.6, 1.0)}   # "elbow" on a touch, x toward the hand's own side (arm_words._pole)
TOGETHER_JAW = ((0.15, 1.0, -0.2), (0.0, -0.035, 0.0))   # elbow "together" on the cheek: the fingers on the jaw line,
# nearly upright (hand direction, shift in the head frame as FACE_ALONG; U033: from the side-of-the-face spot the wrists
# stayed 8-11 cm off the midline and the fingers ended 2-3.6 cm off)
TOGETHER_X, TOGETHER_Z = 0.035, 0.15      # elbow "together": the wrist wanted this far off the midline and at least this
# far in front of the upperChest joint (torso frame; under the chin it lay inside the throat)
ELBOW_ON_UP = 0.10                         # elbow_on: aim the elbow this far above the surface
WRAP_ABOVE = 0.3                           # a shoulder this far above a shin it holds: the fingers wrap around it
WITH = ("palm", "back_of_hand", "fingertips", "fist", "index", "elbow", "forearm", "wrist", "fingers")   # fingers =
# the finger pads, the palm side of the fingers' middle (2026-10-29, the author on T021)   # elbow, forearm, wrist: 2026-10-28
# region -> (kind, bone or segment, t along the segment or None, direction in the rest frame (x = own side
# for sided regions, i.e. flipped for the right arm), extra standoff in metres)
HEAD_RAY = 12.0            # deg: head targets = the surface where a ray from the head's centre meets it (see _target)
REGIONS = {
    "top_of_head": ("head", "head", None, (0, 1, 0), 0.0),
    "head_front_top": ("head", "head", None, (0, 0.65, 0.76), 0.0),   # the front of the top, above the hairline: both
    # hands protecting the head, elbows forward (Cecilia head_pat K2, 2026-11-07)
    "forehead": ("head", "head", None, (0, 0.45, 0.9), 0.0),   # (0, 0.6, 0.8) while the head samples were sparse (2026-11-03 .. 11-05)
    "temple": ("head", "head", None, (1, 0.25, 0.45), 0.0),
    "head_side_top": ("head", "head", None, (0.8, 0.75, 0.25), 0.0),   # the upper side of the head above the temple (a hand in the hair there, fingers up and in over the top; at the temple itself the surface only runs up / back, so a hand lying on it cannot point in: 2026-11-08, Cecilia head_pat K4)
    "ear": ("head", "head", None, (1, 0, -0.1), 0.0),
    "cheek": ("head", "head", None, (0.5, -0.3, 0.8), 0.0),
    "chin": ("head", "head", None, (0, -0.85, 0.55), 0.0),
    "mouth": ("head", "head", None, (0, -0.45, 0.9), 0.02),
    "back_of_head": ("head", "head", None, (0, 0, -1), 0.0),
    # hair strands, by shape (2026-11-10, the author: many hair-touching gestures; measure.hairChains -> hair_chains):
    # a point on the strand (touch "at": root | mid | tip), off its outer side; a model without that kind of hair
    # falls back to HAIR_FALLBACK (no ponytail: an error)
    "side_hair": ("hair", "side", None, None, 0.0),     # the strand hanging beside the face in front of the ear (sideburn)
    "bangs": ("hair", "bangs", None, None, 0.0),        # the fringe over the forehead
    "back_hair": ("hair", "back", None, None, 0.0),     # the hair hanging down the back
    "ponytail": ("hair", "ponytail", None, None, 0.0),  # a tail growing from the back of the top of the head
    "nape": ("seg", "neck", 0.6, (0, 0, -1), 0.0),
    "neck_side": ("seg", "neck", 0.5, (1, 0, 0), 0.0),
    "shoulder": ("seg", "UpperArm", 0.1, (0, 1, 0), 0.0),
    "chest": ("seg", "chest", 0.8, (0, 0, 1), 0.0),
    "collar": ("seg", "upperChest", 0.95, (0, 0.15, 1), 0.008),   # the centre of the upper chest just below the collarbones
    # (a hand clutching the collar, shy chest 2026-11-08; "chest" is a hand's width lower; 8 mm off the surface samples:
    # on AvatarSample_D the fist's palm point sat 1-2 mm inside the filter's torso cylinder for a few frames)
    "heart": ("seg", "chest", 0.8, (0.35, 0, 0.94), 0.0),          # always the character's left
    "chest_side": ("seg", "chest", 0.85, (0.35, 0.1, 0.94), 0.0),  # the upper chest on the hand's own side (a hand pressed on its own pectoral, elbow raised; 2026-11-08, OTs-14 chest reaction)
    "belly": ("seg", "spine", 0.5, (0, 0, 1), 0.0),
    "hip": ("seg", "hips", 1.0, (1, 0, 0.5), 0.0),    # side-front of the waist (2026-10-29, the author on T015: on the side
    # the wrist sat under the shoulder, the elbow stuck out at shoulder height and the forearm pointed steeply down;
    # T020: at the top of the pelvis, so with the upper arm read from the reference the forearm angle also matches)
    "lower_back": ("seg", "spine", 0.3, (0, 0, -1), 0.0),
    "bottom": ("seg", "hips", 0.15, (0.55, -0.25, -1), 0.0),     # the buttock on the hand's own side (hands covering the bottom; 2026-11-09, shy bottom reaction)
    "waist": ("seg", "spine", 0.45, (1, 0, 0.25), 0.0),          # the side of the waist above the pelvis: a high akimbo with the elbows out (2026-11-09; hip = the top of the pelvis)
    "upper_arm": ("seg", "UpperArm", 0.5, (0, 0.3, 0.95), 0.0),
    "elbow": ("seg", "LowerArm", 0.0, (0, 0, -1), 0.0),
    "forearm": ("seg", "LowerArm", 0.5, (0, 1, 0), 0.012),   # standoff 1.2 cm (2026-11-10, the author: a hand holding the other
    # forearm sank into it; the segment radius is the skin's 75th percentile, the sleeve / hand are thicker)
    "other_hand": ("hands", None, None, None, 0.0),
    "thigh": ("leg", "thigh", 0.55, None, 0.0),
    "thigh_side": ("leg", "thigh", 0.4, (1, 0, 0.35), 0.0),   # outer side of the upper thigh: a hanging hand resting on
    # it (2026-10-29, T021: the front-top of the thigh is out of reach of a hanging arm)
    "knee": ("leg", "knee", 0.0, None, 0.0),
    "shin": ("leg", "shin", 0.3, None, 0.0),
    "ankle": ("leg", "ankle", 0.0, None, 0.0),
    "floor": ("floor", None, None, None, 0.0),
    "floor_behind": ("floor", "behind", None, None, 0.0),     # leaning back on the hands (2026-10-28)
    "floor_front": ("floor", "front", None, None, 0.0),       # in front of the knee on its side
    "floor_edge": ("floor", "edge", None, None, 0.0),         # close beside the hip, fingers forward: sitting on a ledge (2026-11-11)
}
TORSO = ("hips", "spine", "chest", "upperChest")
SIDED = {"side_hair", "temple", "head_side_top", "ear", "cheek", "neck_side", "shoulder", "hip", "waist", "bottom", "upper_arm", "elbow", "forearm", "chest_side",
         "thigh", "thigh_side", "knee", "shin", "ankle", "floor", "floor_behind", "floor_front", "floor_edge"}


HAIR_FALLBACK = {"side_hair": "temple", "bangs": "forehead", "back_hair": "back_of_head"}
HAIR_AT = {"root": 0.15, "mid": 0.5, "tip": 0.85}   # fraction of the strand's length from its root
HAIR_OFF = 0.012                                     # m: the hand rests this far outside the strand's bone line
MOUTH_TIP_POLE = (-0.3, -0.8, 0.6)
MOUTH_TIP_LEAN = 0.15
MOUTH_TIP_OFF = 0.04       # m: the fingertip this far in front of the upper lip (the author 2026-11-11: in front of the lips, within ~5 cm, not into them; at 1.5-3 cm the wrist met
# the bust on test1 / AvatarSample_D and the finger leaned 8 deg, at 4-4.5 cm every model is within 3 deg)
MOUTH_N = (0.0, -0.2, 1.0)                           # the lips' outward direction (head frame)


def hair_chains(fit):
    """The model's hair strands by kind, from measure.hairChains (rest world points root -> tail). In the head bone's
    rest frame (x her left, y up, z forward): a tail above its root is not hair (ears, ahoge); a strand growing at the
    front top over the forehead = bangs; one growing beside the face in front of the ear and hanging at least 8 cm =
    side hair (the longest per side); growing at the back high on the head = ponytail; other back strands = back hair."""
    h = np.asarray(sk.UPPER_REST["head"], float)
    out = {"bangs": [], "side_left": [], "side_right": [], "back": [], "ponytail": []}
    for c in fit.get("hair") or []:
        P = np.asarray(c["pts"], float) - h
        r, t = P[0], P[-1]
        if t[1] > r[1] or np.linalg.norm(t - r) < 0.02:
            continue
        drop = r[1] - t[1]
        if r[2] > 0.04 and abs(r[0]) < 0.08 and r[1] > 0.06:
            out["bangs"].append(P)
        elif abs(r[0]) > 0.06 and r[2] > -0.015 and drop > 0.08 and t[2] > -0.03:
            out["side_left" if r[0] > 0 else "side_right"].append(P)
        elif r[2] < -0.04 and r[1] > 0.15:
            out["ponytail"].append(P)
        elif r[2] < -0.015:
            out["back"].append(P)
    for k in ("side_left", "side_right"):            # the longest strand on each side
        out[k] = sorted(out[k], key=lambda P: P[0][1] - P[-1][1])[-1:]
    return out


def _hair_kind(region, side, fit):
    """The strands a hair region uses on this model (rest, head-bone frame), [] if it has none."""
    kind = REGIONS[region][1]
    ch = hair_chains(fit)
    if kind == "side":
        return ch["side_left" if side == "left" else "side_right"]
    return ch[kind]


def hair_region(region, side, fit):
    """region itself, or its fallback when the model has no such hair (ValueError for a missing ponytail)."""
    if REGIONS.get(region, ("",))[0] != "hair" or _hair_kind(region, side, fit):
        return region
    if region in HAIR_FALLBACK:
        return HAIR_FALLBACK[region]
    raise ValueError(f"this model has no {region}")


def _strand_point(P, frac):
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    s_ = frac * seg.sum()
    k = int(np.searchsorted(np.cumsum(seg), s_))
    k = min(k, len(seg) - 1)
    s0 = np.cumsum(seg)[k] - seg[k]
    w = (s_ - s0) / max(seg[k], 1e-9)
    return P[k] + w * (P[k + 1] - P[k])


def load_fit(model=None):
    return json.load(open(FITS / f"{model or sk.MODEL}.json"))


def _radius(seg, d):
    """Fitted surface distance from the segment axis in rest direction d (unit)."""
    keys = []
    if d[2] > 0.5: keys.append("front")
    if d[2] < -0.5: keys.append("back")
    if abs(d[0]) > 0.5: keys.append("out")
    vals = [seg[k] for k in keys if seg.get(k)] or [seg["r"]]
    return float(np.mean(vals))


AKIMBO_DOWN = 0.6          # hand on the hip: fingers forward and this much down (per 1 forward)
AKIMBO_FINGERS = {"forward": (0.0, -AKIMBO_DOWN, 1.0), "down": (0.0, -1.0, 0.3), "back": (0.0, -0.6, -1.0)}   # touch "fingers" word on the hip
UPPER_FIXED_GIVE = 3.0     # deg, for a touch with "keep": "upper" (the arms decided before the legs)
UPPER_GIVE = 35.0          # deg: how far a given upper arm may move when the contact needs it
REACH_TOL = 0.025          # m: the palm may end this far off its contact point (clothes, hand thickness)
CONTACT_W = 8.0            # per cm the palm ends off its contact point, as WEIGHT per deg
FRONT_MAX = 15.0           # deg: a real touch may move the upper arm this far off as seen from the front (see _keep_upper)
NEAR_TOL = 0.06            # m: ... or this far, when the arm as read cannot reach it on this body (see _keep_upper)
# Pose fitting (2026-10-29, the author; lower body before upper body: contact.fit_legs keeps the legs first): the reference is the ideal; on another body some part must give. Each part has a
# tolerance (deg it may be off at no cost: not visible) and a weight (how bad 1 deg past the tolerance is). Cost of a
# part = (WEIGHT * max(0, off - TOL))^2, summed over the parts; contacts, no clipping and support are hard limits.
TOL = {"hand_dir": 5.0, "torso": 3.0, "head": 5.0, "shoulder": 3.0, "upper": 2.0, "upper_depth": 15.0, "thigh": 5.0, "fore": 10.0, "shin": 10.0,
       "hand": 20.0, "foot": 20.0}
WEIGHT = {"hand_dir": 4.0, "torso": 4.0, "head": 2.0, "shoulder": 3.0, "upper": 6.0, "upper_depth": 1.5, "thigh": 3.0, "fore": 1.0, "shin": 1.0,
          "hand": 0.3, "foot": 0.3}
CLEAR_TOL = 0.015          # m: arm points a contact solution may leave inside the body (the author 2026-10-29: a little clipping is fine for now; the push-out takes the rest. 0.5 cm moved T015 / T020 forearms 5-7 deg and did not help test1 S025)
INSIDE = 0.01              # inside the tolerance a part still prefers its reading a little (ties)
DEBUG_KEEP = False


def past_cost(part, off):
    """The part of part_cost past the tolerance alone."""
    return (WEIGHT[part] * np.maximum(np.asarray(off, float) - TOL[part], 0.0)) ** 2


def part_cost(part, off):
    """off: degrees (array) a part is off its reading."""
    off = np.asarray(off, float)
    return (WEIGHT[part] * np.maximum(off - TOL[part], 0.0)) ** 2 + INSIDE * off ** 2
HAND_SIDE = 20.0           # deg: the hand may turn sideways off its forearm at most this much
HAND_TILT = 0.5            # the hand may leave its surface by at most 30 deg (sin)
HAND_GIVE = 70.0           # deg: the wrist may bend this far to reach a contact (= the back bend limit of WRIST_LIM)
HEAD_SIDE_KEEP = False     # trial (2026-11-08): _keep_upper also allows the wider sideways hand + the fingertip dip on the head (True made head_pat v5 worse, 12 -> 25 over tolerance)
HEAD_TIP_TOL = 0.02        # m: on the head the fingertip of a read arm's hand may dip this much deeper than CLEAR_TOL (the fingers follow the round head)


def _cone(v0, max_deg, step=1.0, n=24):
    """Unit directions within max_deg of v0 (rings every step deg) and their angle to v0 in degrees."""
    a = _unit(np.cross(v0, [0.0, 1.0, 0.0]) if abs(v0[1]) < 0.95 else np.cross(v0, [1.0, 0.0, 0.0]))
    b = np.cross(v0, a)
    dirs, devs = [v0], [0.0]
    for dev in np.arange(step, max_deg + 0.1, step):
        r = np.radians(dev)
        for phi in np.linspace(0, 2 * np.pi, n, endpoint=False):
            dirs.append(np.cos(r) * v0 + np.sin(r) * (np.cos(phi) * a + np.sin(phi) * b)); devs.append(dev)
    return np.array(dirs), np.array(devs)


def _keep_upper(u0, W_t, f0=None, Pp=None, reach=0.0, n=None, fixed=False, gives=None, with_cost=False, coarse=False,
                clear=None, h_pref=None, side_max=None, tip_tol=0.0):
    """fixed ("keep": "upper" on the touch): the upper arm stays within UPPER_FIXED_GIVE and the forearm and hand do what
    they can. Otherwise tries UPPER_GIVE first, then wider cones (a contact the given upper arm cannot make on this model's
    proportions moves the upper arm as little as it must)."""
    gives = gives or ((UPPER_FIXED_GIVE,) if fixed else (UPPER_GIVE, 50.0, 70.0))
    # a contact the arm as read cannot make on this body (geometry, not a limit: T021's forearm is 5 cm short of the
    # nape) first gets near enough (NEAR_TOL, the author 2026-10-29: "碰到那附近就行"), before the upper arm moves further
    # A real touch first, the upper arm moving as far as it must but no more than FRONT_MAX off as seen from the front
    # (T001: the upper arm moved in depth, invisible from the front, and the hand touched the knee); else, when only an
    # upper arm that looks different from the front would reach (geometry: T021's forearm is 5 cm short of the nape),
    # near the spot is enough (NEAR_TOL, the author 2026-10-29: "碰到那附近就行")
    fr = lambda v: np.degrees(np.arctan2(v[1], abs(v[0])))
    last = None
    for give in gives:
        res = _keep_upper_in(u0, W_t, f0, Pp, reach, n, give, coarse, clear, None, h_pref, side_max, tip_tol)
        last = res
        if res[3] and (fixed or abs(fr(res[0]) - fr(_unit(u0))) <= FRONT_MAX):
            return res[:3] if not with_cost else res
    if reach:
        for give in gives:
            res = _keep_upper_in(u0, W_t, f0, Pp, reach, n, give, coarse, clear, NEAR_TOL, h_pref, side_max, tip_tol)
            if res[3]:
                return res[:3] if not with_cost else res
    return last[:3] if not with_cost else last


def _keep_upper_in(u0, W_t, f0, Pp, reach, n, give, coarse=False, clear=None, tol=None, h_pref=None, side_max=None, tip_tol=0.0):
    """Upper arm direction near u0 (torso frame) and forearm direction, for a hand that must reach its contact.
    W_t = the wrist point the contact prefers, Pp = the contact (palm) point, reach = hand length to it; all relative to
    the shoulder. The contact comes first (2026-10-29, T001: weighing a miss against degrees left the hand 11 cm off the
    knee), the hand takes what it can (the author: the model's proportions go mostly to the far parts): an upper arm is
    usable when its forearm can put the wrist at `reach` from Pp with the wrist bent at most HAND_GIVE; among those,
    the cheapest by part_cost over upper arm (off u0), forearm (off f0, or off the one aiming at W_t) and wrist bend. The
    forearm is taken on the cone of wrists at `reach` from Pp, as near the preferred forearm as the cone allows.
    If no upper arm within UPPER_GIVE reaches, the one that misses least."""
    u0 = _unit(u0)
    if Pp is None:
        Pp, reach = W_t, 0.0
    U0, du0 = _cone(u0, give, 2.5, 16) if coarse else _cone(u0, give, 1.0, 36)
    # the palm may end up to REACH_TOL off its point (the loose clothes, the hand's thickness): three hand reaches
    tol = REACH_TOL if tol is None else tol
    offs = np.linspace(-tol, tol, 3 if tol <= REACH_TOL else 7)
    U, du = np.concatenate([U0] * len(offs)), np.concatenate([du0] * len(offs))
    rr = np.repeat(reach + offs, len(U0)) if reach else np.full(len(U), 0.0)
    E = U * UA
    D = Pp[None, :] - E
    d = np.linalg.norm(D, axis=-1)
    a = D / np.maximum(d, 1e-9)[:, None]                         # elbow -> contact
    cosq = (FA ** 2 + d ** 2 - rr ** 2) / np.maximum(2 * FA * d, 1e-9)
    reachable = np.abs(cosq) <= 1.0
    q = np.arccos(np.clip(cosq, -1, 1))                         # forearm angle off the elbow->contact line
    pref = np.array([_unit(f0)] * len(U)) if f0 is not None else (W_t[None, :] - E) / np.maximum(
        np.linalg.norm(W_t[None, :] - E, axis=-1), 1e-9)[:, None]
    # every forearm on the cone (36 around it), per upper arm
    b1 = np.cross(a, [0.0, 1.0, 0.0])
    b1 = np.where(np.linalg.norm(b1, axis=-1)[:, None] > 1e-6, b1, np.cross(a, [1.0, 0.0, 0.0]))
    b1 = b1 / np.linalg.norm(b1, axis=-1)[:, None]
    b2 = np.cross(a, b1)
    phi = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    rim = np.cos(phi)[None, :, None] * b1[:, None, :] + np.sin(phi)[None, :, None] * b2[:, None, :]
    F = np.cos(q)[:, None, None] * a[:, None, :] + np.sin(q)[:, None, None] * rim          # [U, 36, 3]
    hand = Pp[None, None, :] - (E[:, None, :] + F * FA)
    bend = np.degrees(np.arccos(np.clip((F * hand).sum(-1) / np.maximum(np.linalg.norm(hand, axis=-1), 1e-9), -1, 1)))
    ok = reachable[:, None] & ((bend <= HAND_GIVE) | (reach == 0))
    # the elbow folds at most kf.MAX_ELBOW (2026-10-29, T021: a hand on the neck asked for 166 deg; the solver clamps it
    # and the hand ended 11 cm off the neck)
    ok &= (F * U[:, None, :]).sum(-1) >= np.cos(kf.MAX_ELBOW)
    if n is not None and reach and tol <= REACH_TOL:   # the hand lies along the surface (not when only near it), not standing on it (T001: the palm stood up
        # beside the knee with the fingers pointing up)
        hu = hand / np.maximum(np.linalg.norm(hand, axis=-1), 1e-9)[..., None]
        nn = _unit(n)
        ok &= np.abs((hu * nn).sum(-1)) <= HAND_TILT
        # the wrist bends only toward / away from the palm (the palm faces the surface): the hand must stay in the plane
        # of the forearm and the surface normal, not turn sideways off the forearm
        # (on the head an arm read from a reference may turn its hand further: the sideways part becomes wrist_side,
        # 2026-11-08; before, the fingers at the temple had to point up because the hand could not leave the plane)
        side_ax = np.cross(F, nn)
        side_ax = side_ax / np.maximum(np.linalg.norm(side_ax, axis=-1), 1e-9)[..., None]
        ok &= np.abs((hu * side_ax).sum(-1)) <= np.sin(np.radians(HAND_SIDE if side_max is None else side_max))
    if clear is not None and ok.any():
        # no clipping is a hard limit here too (2026-10-29, T001: the wrist sank 2.5 cm into the thigh, the push-out
        # afterwards lifted the whole hand 7 cm off the knee): wrist, forearm and elbow must stay outside the body
        iu, iv = np.nonzero(ok)
        Ei, Fi = E[iu], F[iu, iv]
        Wi = Ei + Fi * FA
        Hi = hand[iu, iv] / np.maximum(np.linalg.norm(hand[iu, iv], axis=-1), 1e-9)[:, None]
        # the hand's own points too, except the palm on its contact (T021: the fingertips on the neck went into the
        # tilted head and the push-out moved the hand 8 cm off): fingertip and the wrist half of the hand
        pts = np.concatenate([Ei + Fi * FA * w for w in (0.0, 0.5, 0.75, 1.0)] + [Wi + Hi * (sk.FINGERTIP - sk.PALM + reach)])
        dd = clear(pts).reshape(5, -1)
        # the fingertip may lie a little deeper (tip_tol): fingers lying along the round head dip into its ellipsoid by
        # the curvature (2026-11-08: at the temple only hands pointing straight up off the head passed, the reference's
        # go in over the hair)
        depth = np.maximum(dd[:4].max(0), dd[4] - tip_tol)
        ok[iu[depth > CLEAR_TOL], iv[depth > CLEAR_TOL]] = False
        if DEBUG_KEEP:
            print("clear: before", len(iu), "after", int(ok.sum()), "min depth", float(depth.min()))
    if DEBUG_KEEP:
        print("keep_upper give", give, "reachable", int(reachable.sum()), "bend ok", int((bend <= HAND_GIVE).sum()),
              "ok", int(ok.sum()), "d range", float(d.min()), float(d.max()), "reach", reach)
    if not ok.any():
        miss = np.abs(d - FA) - reach
        i = int(np.argmin(np.maximum(miss, 0) + 0.001 * du))
        return U[i], a[i], W_t, False, np.inf
    df = np.degrees(np.arccos(np.clip((F * pref[:, None, :]).sum(-1), -1, 1)))
    # the wrist bends only toward / away from the palm, so a hand turned sideways off its forearm stood up instead
    # (T001: fingers up beside the knee): a straighter wrist is preferred too
    # the upper arm as seen from the front counts in full, its depth (forward / back) less (2026-10-29, the author: the front
    # reference first; T021: a hand on the nape out of reach of the read upper arm raised the elbow 60 deg in the front
    # view, where moving it back is hardly visible from the front)
    fr = lambda v: np.degrees(np.arctan2(v[..., 1], np.abs(v[..., 0])))
    dz = lambda v: np.degrees(np.arcsin(np.clip(v[..., 2], -1, 1)))
    cu = part_cost("upper", np.abs(fr(U) - fr(u0))) + part_cost("upper_depth", np.abs(dz(U) - dz(u0)))
    # within the tolerance a closer touch is still preferred (T001: with no cost there the solver took the far edge,
    # 3 cm off the knee)
    off_cm = (np.abs(rr - reach) * 100)[:, None] if reach else 0.0
    cost = cu[:, None] + part_cost("fore", df) + part_cost("hand", bend) + (CONTACT_W * off_cm) ** 2
    if h_pref is not None:          # the way the hand lies on a curved surface (a knee: fingers over its front)
        hu = hand / np.maximum(np.linalg.norm(hand, axis=-1), 1e-9)[..., None]
        cost = cost + part_cost("hand_dir", np.degrees(np.arccos(np.clip((hu * _unit(h_pref)).sum(-1), -1, 1))))
    cost = np.where(ok, cost, np.inf)
    i, j = np.unravel_index(np.argmin(cost), cost.shape)
    return U[i], F[i, j], W_t, True, float(cost[i, j])

def _arm_dirs(S, W, pole):
    """Two-bone IK: unit upper / fore directions reaching from S toward W, elbow bent toward `pole`."""
    d = W - S
    L = float(np.clip(np.linalg.norm(d), abs(UA - FA) + 1e-3, UA + FA - 1e-3))
    n = d / max(np.linalg.norm(d), 1e-9)
    x = (UA ** 2 - FA ** 2 + L ** 2) / (2 * L)
    h = np.sqrt(max(UA ** 2 - x ** 2, 0.0))
    p = pole - (pole @ n) * n
    p = p / max(np.linalg.norm(p), 1e-9)
    E = S + x * n + h * p
    return (E - S) / UA, (S + n * L - E) / FA


FRONT_ARM = "left"        # where the forearms cross, this one lies over / in front of the other
FRONT_RISE = 0.03          # (0.12 cleared U041 but put the top forearm across the mouth; arms_cross does it now)


def _clear_other(clear, f, side, S, Gt, P):
    """clear() plus how far points (torso frame, relative to the shoulder S) are inside the OTHER arm's forearm at frame
    f, as the arms_clip rule measures it (tapered capsule; 2026-11-03, the author: crossed arms had the forearms through each
    other). The other arm is as the last pass left it (absent in the first pass: nothing to avoid yet)."""
    o = "left" if side == "right" else "right"
    E, Wo = P[f"{o}LowerArm"][f], P[f"{o}Hand"][f]
    u = np.linspace(0.0, 1.0, 11)
    C = E[None] + u[:, None] * (Wo - E)[None]
    r = sk.ARM_R["forearm"] + u * (sk.ARM_R["hand"] - sk.ARM_R["forearm"])

    def depth(local):
        w = S[None] + local @ Gt.T
        d = np.linalg.norm(w[:, None] - C[None], axis=-1)            # [N, 11]
        return np.maximum(clear(local), (r[None] + sk.ARM_R["forearm"] - sk.ARMS_TOL - d).max(1))
    return depth


# touches that take the arm across the front of the body to the other arm: its forearm must clear the clothed front
CROSS_FRONT = ("other_forearm", "other_elbow", "other_upper_arm", "other_hand")


def _clear_cloth(clear, f, S, Gt, P, G):
    """clear() plus how far points lie behind the torso's CLOTHED front (fit front_grid: skin, clothes and the bust,
    which the skin-only torso shape leaves out), for an arm that crosses in front of the body to the other arm
    (2026-11-11, the author: the forearm of the hand holding the other arm sank ~5 cm into the bust / the shirt on every
    model while the filter said ok). Each point is looked up in the rest frame of its height band's bone; points
    behind the body's middle or beside the grid do not count."""
    g = (load_fit(sk.MODEL) or {}).get("front_grid")
    if not g:
        return clear
    c, Z = g["cell"], g["z"]
    rest = {b: np.asarray(sk.UPPER_REST[b], float) for b in sk.TORSO_BANDS}
    edges = [sk.TORSO["y0"]] + [rest[b][1] for b in sk.TORSO_BANDS[1:]] + [np.inf]

    def depth(local):
        w = S[None] + local @ Gt.T
        d = np.full(len(w), -np.inf)
        for i, b in enumerate(sk.TORSO_BANDS):
            q = (w - P[b][f]) @ G[b][f] + rest[b]
            for k in np.nonzero((q[:, 1] > edges[i]) & (q[:, 1] <= edges[i + 1]) & (q[:, 2] > 0))[0]:
                z = Z.get(f"{int(np.floor(q[k, 0] / c))},{int(np.floor(q[k, 1] / c))}")
                if z is not None and z > 0:
                    d[k] = max(d[k], z - (q[k, 2] - sk.ARM_R["forearm"]))
        return np.maximum(clear(local), d)
    return depth


SWIVEL_STEP = 10.0         # deg between the elbow swivels _swivel_clear tries
SWIVEL_MAX = 90.0          # at most this far from the given pole (2026-11-03, U019: a swivel of 150 deg took the arm
# round behind the back to the other shoulder)


def _swivel_clear(S, W, pole, clear, Gt, rise=0.03, upper=True):
    """The elbow pole (world) nearest `pole` whose elbow and forearm stay outside the body (at most CLEAR_TOL in), the
    wrist kept: the elbow turns about the shoulder-wrist line (2026-11-03, S024 / S030: hands on the shins with the fixed
    elbow pole put the forearm 2-3 cm into the knee; the push-out then moved the whole hand 4-7 cm off the shin).
    The pole as given if it is clear or no swivel is."""
    ax = _unit(W - S)
    p0 = pole - ax * (pole @ ax)
    if np.linalg.norm(p0) < 1e-6:
        return pole
    p0 = _unit(p0)
    q0 = np.cross(ax, p0)
    steps = np.arange(SWIVEL_STEP, SWIVEL_MAX + 0.1, SWIVEL_STEP)
    ang = np.radians(np.concatenate([[0.0], steps.repeat(2) * np.tile([1, -1], len(steps))]))
    poles = np.cos(ang)[:, None] * p0 + np.sin(ang)[:, None] * q0
    pts, back = [], []
    for pl in poles:
        u, fo = _arm_dirs(np.zeros(3), Gt.T @ (W - S), Gt.T @ pl)
        E = u * UA
        back.append((E[2], E[1]))
        pts.append([0.5 * E] + [E + fo * FA * w for w in (0.0, 0.5, 0.75)])
    # the upper arm's middle too (U033: an elbow swung behind the back put the upper arm through the chest)
    depth = clear(np.asarray(pts).reshape(-1, 3)).reshape(len(poles), 4)
    # upper=False: the upper arm may lie on the body (elbows together in front: on the chest / bust; the filter does
    # not check the upper arm either)
    depth = (depth if upper else depth[:, 1:]).max(1)
    if depth[0] <= CLEAR_TOL:
        return pole
    # the elbow does not swing behind the shoulder unless the given pole already has it there (U019: the hand reached
    # the other shoulder round the back)
    # nor rise above where the given pole has it (U019: the forearm went across the face to the other shoulder)
    back = np.asarray(back)
    bad = (back[:, 0] < min(back[0, 0], 0.0) - 0.02) | (back[:, 1] > max(back[0, 1], 0.0) + rise)
    depth = np.where(bad, np.inf, depth)
    okk = np.nonzero(depth <= CLEAR_TOL)[0]
    return poles[okk[0]] if len(okk) else poles[int(np.argmin(depth))]


def _palm_word(normal_t, fore_t, side):
    """Palm word (torso frame) whose direction best faces against the surface normal, off the forearm axis."""
    want = -normal_t - fore_t * (normal_t @ -fore_t)
    best, score = "natural", -2.0
    sgn = -1.0 if side == "right" else 1.0
    for word in ("forward", "back", "up", "down", "in", "out"):
        v = np.array(kf._PALM_DIR.get(word, (-sgn if word == "in" else sgn, 0, 0)), float)
        v = v - fore_t * (v @ fore_t)
        if np.linalg.norm(v) < 0.35:
            continue
        s = float(_unit(v) @ _unit(want))
        if s > score:
            best, score = word, s
    return best


def _unit(v):
    return v / max(np.linalg.norm(v), 1e-9)


# Both hands on one of these (not sided) regions: each hand goes to its own side of it, side by side, instead of one
# hand stacked on the other (2026-11-03, batch: U069 both hands behind the head aimed at one point, the arms crossed in
# front of the throat; U047 / U052 both hands on the chest stacked, hands_clip). Chin / mouth / heart keep stacking.
PAIR_SIDE = {"back_of_head": 0.75, "top_of_head": 0.7, "head_front_top": 1.2, "chest": 0.7, "belly": 0.7, "lower_back": 0.6}


def _target(region, side, f, P, G, L, fit, along=None, pair=False, crossed=False, with_=None, at=None):
    """World point on the surface and its outward normal at frame f. `along` replaces the region's fraction along its
    bone (leg_reach slides a thigh touch toward the hip); pair = the other hand touches the same region (PAIR_SIDE)."""
    kind, bone, t, d, standoff = REGIONS[region]
    t = t if along is None else along
    if region == "shoulder" and not crossed and with_ in ("fingers", "fingertips", "index"):
        # the own shoulder with the fingers: its front-top too (2026-11-03, U056 'fingers on the straps, elbows down':
        # on the top of the joint the elbow folded to kf.MAX_ELBOW and the fingers stayed 5-8 cm above)
        d = (0.0, 0.2, 1.0)
    if crossed and region == "forearm":    # the other forearm hanging at the side: held from the FRONT (2026-11-10, AK-Alpha
        # hand reaction: the forearm's 'top' of a hanging arm lies against the trunk, so the holding hand sat 7 cm inside
        # the chest and its forearm cut through the shirt on the way)
        d = (0.0, 0.3, 1.0)
    if crossed and region == "shoulder":   # the other shoulder: a hand coming across the front rests on its front-top
        # (2026-11-03, U019 / U041: on its top, a little behind the joint, the arm reached it round the back)
        d = (0.0, 1.0, 0.7)
    if pair and region in PAIR_SIDE:
        d = (d[0] + PAIR_SIDE[region] * (1.0 if side == "left" else -1.0), d[1], d[2])
    other = region.startswith("other_")
    sx = 1.0 if side == "left" else -1.0                        # own side in x
    rest = {n: np.asarray(p, float) for n, p in sk.UPPER_REST.items()}
    if kind == "hair":
        strands = _hair_kind(region, side, fit)
        frac = HAIR_AT[at or ("root" if REGIONS[region][1] == "bangs" else "mid")]   # bangs: at their root (mid covered the eye)
        q = np.mean([_strand_point(P, frac) for P in strands], axis=0)      # head-bone frame, rest
        hc = np.asarray(fit["head"]["c"], float) - np.asarray(rest["head"], float)
        if REGIONS[region][1] == "bangs":
            n_rest = np.array([0.0, 0.2, 1.0])
        elif REGIONS[region][1] == "side":
            # the hand comes from the FRONT (palm back and a little in, fingers going back into the strand): outward
            # alone put the hand behind the thin strand with the palm to the head, and from the front it looked like
            # combing the air (the author 2026-11-10: "你在撩空氣嗎")
            out = q - hc; out[1] = 0.0
            n_rest = _unit(out) * 0.35 + np.array([0.0, 0.0, 1.0])
        else:                                   # outward from the head's axis, level
            n_rest = q - hc; n_rest[1] = 0.0
        n_rest = _unit(n_rest)
        Gb, Pb = G["head"][f], P["head"][f]
        p = Pb + Gb @ (q + n_rest * HAIR_OFF); n = Gb @ n_rest
    elif kind == "head" and region == "mouth" and (fit.get("mouth") or {}).get("p"):
        # the lips as measured (the 'aa' morph's vertices, measure.mouthPoint; 2026-11-10)
        n_rest = _unit(np.array(MOUTH_N))
        p_rest = np.asarray(fit["mouth"]["p"], float) + n_rest * 0.004
        if with_ in ("index", "fingertips") and fit["mouth"].get("upper"):
            # a fingertip goes to the UPPER lip (its top edge, measure.mouthPoint `upper`; the author 2026-11-11: the shh
            # finger stands vertical with its tip on the upper lip, so it lies across both lips)
            p_rest = np.asarray(fit["mouth"]["upper"], float) + n_rest * 0.004
        elif with_ in ("index", "fingertips"):
            p_rest = p_rest + np.array([0.0, 0.008, 0.0])       # no upper lip measured: a little above the lips' middle
        Gb, Pb = G["head"][f], P["head"][f]
        p = Pb + Gb @ (p_rest - rest["head"]); n = Gb @ n_rest
    elif kind == "head":
        h = fit["head"]
        dd = _unit(np.array([d[0] * (sx if region in SIDED else 1), d[1], d[2]], float))
        pts = np.asarray(fit.get("points", {}).get("head", []), float)
        if len(pts):
            # where a ray from the head's centre along dd meets the surface (the farthest sample within HEAD_RAY deg of
            # it); was the support point (farthest along dd), which a pointed chin pulled down: the cheek target sat at
            # the corner of the jaw, the mouth on AvatarSample_D's chin (2026-11-03, batch: an index on the cheek
            # ended at the jaw)
            rel = pts - np.asarray(h["c"])
            dist = np.linalg.norm(rel, axis=1)
            cone = (rel @ dd) > dist * np.cos(np.radians(HEAD_RAY))
            p_rest = pts[cone][np.argmax(dist[cone])] if cone.any() else pts[np.argmax(rel @ dd)]
        else:
            p_rest = np.asarray(h["c"]) + dd * np.asarray(h["r"])
        n_rest = dd
        Gb, Pb = G["head"][f], P["head"][f]
        p = Pb + Gb @ (p_rest - rest["head"]); n = Gb @ n_rest
    elif kind == "seg":
        name = (("left" if sx > 0 else "right") + bone) if bone in ("UpperArm", "LowerArm") else bone
        seg = fit["segments"].get(name) or fit["segments"].get("chest")
        a, b = np.asarray(seg["a"]), np.asarray(seg["b"])
        dd = np.array([d[0] * (sx if region in SIDED else 1), d[1], d[2]], float)
        dd = _unit(dd - _unit(b - a) * (dd @ _unit(b - a)))
        c = a + t * (b - a)
        ax = _unit(b - a); L_ = np.linalg.norm(b - a)
        if name in TORSO:     # which torso bone owns a vertex varies between models: search them all, by height
            pts = np.asarray([q for bn in TORSO for q in fit.get("points", {}).get(bn, [])], float)
            near = pts[np.abs(pts[:, 1] - c[1]) < 0.03] if len(pts) else pts
        else:
            pts = np.asarray(fit.get("points", {}).get(name, []), float)
            near = pts[np.abs((pts - a) @ ax / L_ - t) < 0.15] if len(pts) else pts
        if len(near):                                  # support point near t along the bone, farthest along dd
            q = near[np.argmax((near - c) @ dd)]
            p_rest = c + dd * (((q - c) @ dd) + standoff)
        else:
            p_rest = c + dd * (_radius(seg, dd) + standoff)
        joint = name if name in rest else "upperChest"
        if region == "shoulder" and name.replace("UpperArm", "Shoulder") in rest:
            # the top of the shoulder rides on the collarbone, not on the upper arm (2026-11-03, U019: with that arm
            # raised to the ear the point turned to the back of the shoulder; the other hand reached it from behind)
            joint = name.replace("UpperArm", "Shoulder")
        Gb, Pb = G[joint][f], P[joint][f]
        p = Pb + Gb @ (p_rest - rest[joint]); n = Gb @ dd
    elif kind == "leg":
        s = "left" if sx > 0 else "right"
        hip, knee, ankle = L[f"{s}Hip"][f], L[f"{s}Knee"][f], L[f"{s}Ankle"][f]
        seg = fit["segments"][f"{s}UpperLeg" if bone == "thigh" else f"{s}LowerLeg"]
        a, b = (hip, knee) if bone == "thigh" else (knee, ankle)
        ax = _unit(b - a)
        up = np.array([0, 1.0, 0]) + 0.6 * np.array([0, 0, 1.0])
        if d is not None:                       # a given side of the leg (hips frame, x toward the leg's own side)
            up = G["hips"][f] @ np.array([d[0] * sx, d[1], d[2]], float)
        n = _unit(up - ax * (up @ ax))
        r = seg.get("front") or seg["r"]
        if bone == "shin":      # on the push-out's shin capsule (depenetrate._pushes, skeleton.leg_radius), not inside it
            # (2026-11-03, S030 / S024: the fit's front radius is ~1 cm less, the push-out lifted the hand 4-6 cm off)
            r = max(r, float(sk.leg_radius("shin", np.array([t])).max()))
        if bone == "thigh":     # the whole hand clears the clothed thigh (the wrist lies nearer the hip, where
            # loose shorts are thickest; skeleton.leg_radius, as the filter)
            r = max(r, float(sk.leg_radius("thigh", np.clip(np.linspace(t - 0.25, t + 0.1, 8), 0, 1)).max()))
        if bone == "knee":      # the top of the kneecap (2026-10-28, the author: hands on knees floated beside them): out
            # along the bisector of the bent knee, tipped up so the hand rests on top; the knee is as wide as the
            # thigh's end there, wider than the shin
            apex = -(_unit(hip - knee) + _unit(ankle - knee))
            if np.linalg.norm(apex) > 0.2:
                n = _unit(_unit(apex) + 0.6 * np.array([0, 1.0, 0]))
            r = max(r, float(sk.leg_radius("thigh", np.array([1.0])).max()))
        # A skirt covering this spot (the hem hangs below it at rest): the hand lands on the cloth, which the player's
        # cloth simulation keeps SIM_GAP (1.6 cm) off the leg (2026-10-29, the author: legs first, then the skirt, then the
        # upper body; test1's side sits: the hands sank into the skirt)
        # With the draped cloth of this frame (CLOTH, from the player, see refine.solve) the hand rests on its outermost
        # layer along n; without it, a skirt whose hem hangs below the spot at rest adds SKIRT_GAP.
        hem = (fit.get("skirt") or {}).get("hem")
        cloth_plane = None
        if bone != "ankle" and CLOTH and f in CLOTH and len(CLOTH[f]):
            # the hand lies on the cloth, not on the leg (2026-10-29, the author; measured at the palm point alone, the
            # fingers following the leg's curve went under the folds): a plane through the cloth points under the
            # hand's footprint, raised to the outermost of them, gives the spot and the surface's normal
            base = a + t * (b - a) + n * r
            rel = CLOTH[f] - base
            sn = rel @ n
            lat = np.linalg.norm(rel - sn[:, None] * n, axis=1)
            near = (lat < CLOTH_FOOT) & (sn > -0.02) & (sn < 0.2)
            if near.sum() >= 4:
                q = CLOTH[f][near]
                c = q.mean(0)
                nc = np.linalg.svd(q - c)[2][-1]
                nc = nc if nc @ n > 0 else -nc
                if nc @ n < 0.5:                 # a fold standing up: keep the leg's direction
                    nc = n
                top = c + nc * max(float(((q - c) @ nc).max()), 0.0)
                cloth_plane = (base + nc * max(float((top - base) @ nc), 0.0) + nc * CLOTH_HALF, nc)
            elif near.any():
                r += max(float(sn[near].max()), 0.0) + CLOTH_HALF
        elif hem is not None and bone != "ankle":
            hip_y = sk.HIPS_REST[1] + sk.HIP_OFFSET[1]
            rest_y = hip_y - t * sk.L_THIGH if bone == "thigh" else hip_y - sk.L_THIGH - (t * sk.L_SHIN if bone == "shin" else 0.0)
            if hem < rest_y - 0.02:
                r += SKIRT_GAP
        if bone == "ankle":     # held from the outside (2026-10-28, T014: reaching over the knee to the front of the
            # ankle took the forearm through the raised knee)
            n = _unit(G["hips"][f] @ np.array([sx, 0.3, 0.15]))
            p = ankle + np.array([0, 0.03, 0]) + n * sk.LEG_R["shin"]
        else:
            p = a + t * (b - a) + n * r
        if cloth_plane is not None:
            p, n = cloth_plane
    elif kind == "floor":
        s = "left" if sx > 0 else "right"
        hip = L[f"{s}Hip"][f]
        y0 = fit.get("floor", 0.0) or 0.0
        if bone == "behind":
            p = np.array([hip[0] + sx * 0.12, y0, hip[2] - 0.24])
        elif bone == "front":
            knee = L[f"{s}Knee"][f]
            p = np.array([knee[0] + sx * FLOOR_FRONT_OUT, y0, knee[2] + FLOOR_FRONT])
        elif bone == "edge":
            p = np.array([hip[0] + sx * 0.13, y0, hip[2] + 0.05])
        else:                                   # beside and a little behind the hip, arm slanting out (2026-10-28)
            p = np.array([hip[0] + sx * 0.22, y0, hip[2] - 0.08])
        n = np.array([0, 1.0, 0])
    else:
        raise ValueError(region)
    return p, _unit(n)


def _clear_fn(f, S, Gt, P, G, L):
    """depth(local [N,3]) -> how far arm points (torso frame, relative to the shoulder S) are inside the body at frame f,
    as the push-out measures it (depenetrate._pushes: torso, head, legs incl. the clothed thigh)."""
    from .depenetrate import _pushes

    def depth(local):
        w = S[None] + local @ Gt.T
        N = len(w)
        tile = lambda d: {k: np.repeat(v[f:f + 1], N, axis=0) for k, v in d.items()}
        push = _pushes(w, tile(P), tile(G), tile(L), sk.ARM_R["forearm"])
        return np.linalg.norm(push, axis=-1)
    return depth


# The contact points of the last resolve_touches, {(frame, side): (with, region, point [3] world)}; only a record, to
# measure afterwards how far the final hand ended from them (2026-11-03, batch run: the filter does not see a miss).
TARGETS = {}


def resolve_touches(obj, duration_s, fit=None):
    """obj: keyframe JSON ({"posture", "keyframes"}). Returns a copy whose touching arms have upper / fore /
    palm (and hand shapes for fist / index / hold) filled in for the body in `fit`."""
    fit = fit or load_fit()
    TARGETS.clear()
    obj = fit_legs(obj, duration_s, fit)
    obj = floor_lean(obj, duration_s, fit)
    obj = leg_reach(obj, duration_s, fit)
    todo = [(i, side) for i, k in enumerate(obj["keyframes"]) for side in ("right", "left")
            if isinstance(k.get(f"{side}_arm"), dict) and "touch" in k[f"{side}_arm"]]
    if not todo:
        return obj
    out = obj
    for _ in range(3):              # a target on an arm moves with that arm: re-solve with the arms just found
        out = _resolve_once(obj, out, todo, duration_s, fit)
    return out


LEG_TOUCH = ("thigh", "thigh_side", "knee", "shin", "ankle")
SUPPORT_TOL = 0.01         # m: a sitting leg's lowest point may be this far off the floor (support is a hard limit)
THIGH_SEARCH, SHIN_SEARCH = 25.0, 60.0     # deg searched around the reading


def _d3(v):
    return np.degrees(np.arctan2(v[0], v[2])), np.degrees(np.arcsin(np.clip(v[1], -1, 1)))


def _from_d3(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])


def _leg_gap(posture, side, t, s):
    """Height of this leg's lowest point above the bottom (m; < 0 = the leg would lift the bottom off the floor)."""
    t_, s_, toe = sk.sit_leg_dirs(posture, side, (t, s))
    R_up, knee = sk.thigh_frame(t_, s_)
    R = {"hips": np.eye(3), f"{side}UpperLeg": R_up, f"{side}LowerLeg": sk.rx(knee)}
    R[f"{side}Foot"] = sk.foot_local(R_up @ R[f"{side}LowerLeg"], toe)
    o = "right" if side == "left" else "left"
    R[f"{o}UpperLeg"] = R[f"{o}LowerLeg"] = R[f"{o}Foot"] = np.eye(3)
    pts = sk.leg_points(R, np.zeros(3))
    low = min(pts[f"{side}Knee"][1] - sk.R_KNEE, pts[f"{side}Ankle"][1] - sk.R_ANKLE, pts[f"{side}Toes"][1] - sk.R_TOE)
    return low - pts["butt"][1]


def _leg_gap_batch(side, t, s):
    """_leg_gap for many (thigh, shin) direction pairs [N, 3] at once (same formulas as sk.sit_leg_dirs for given
    directions, sk.thigh_frame, sk.foot_local and sk.leg_points; 2026-11-03, it was most of fit_legs)."""
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    t, s = unit(t), unit(s)
    up = np.array([0.0, 1.0, 0.0]) - s * s[:, 1:2]
    toe = np.where(s[:, 1:2] < -0.5, sk.FLAT, unit(0.85 * unit(up) + 0.5 * s))
    h = np.cross(t, s)
    X = np.array([1.0, 0.0, 0.0])
    h = np.where(np.linalg.norm(h, axis=-1, keepdims=True) < 1e-3, X - t * (t @ X)[:, None], h)
    h = unit(h)
    R_up = np.stack([h, -t, np.cross(h, -t)], axis=-1)
    k = np.arccos(np.clip((t * s).sum(-1), -1, 1))
    c, sn, z, o = np.cos(k), np.sin(k), np.zeros_like(k), np.ones_like(k)
    R_low = np.stack([np.stack([o, z, z], -1), np.stack([z, c, -sn], -1), np.stack([z, sn, c], -1)], -2)
    G = R_up @ R_low
    a = sk._unit(sk.TOE)
    b = unit(np.einsum("nji,nj->ni", G, toe))
    v, cc = np.cross(a, b), b @ a
    K = np.zeros((len(t), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -v[:, 2], v[:, 1], v[:, 2], -v[:, 0], -v[:, 1], v[:, 0]
    Rf = np.eye(3) + K + K @ K / np.maximum(1 + cc, 1e-12)[:, None, None]
    Rf = np.where((cc < -0.9999)[:, None, None], sk.rx(np.pi), Rf)
    hip = sk.HIP_OFFSET * np.array([sk.side_sign(side), 1, 1])
    knee = hip + R_up @ sk.THIGH
    ankle = knee + G @ sk.SHIN
    toes = ankle + (G @ Rf) @ sk.TOE
    low = np.minimum(np.minimum(knee[:, 1] - sk.R_KNEE, ankle[:, 1] - sk.R_ANKLE), toes[:, 1] - sk.R_TOE)
    return low - sk.BUTT[1]


LEG_BATCH = True   # _leg_candidates over all thigh directions at once; False = one at a time (the reference)


def _leg_candidates(posture, side, t0, s0):
    """Thigh / shin pairs near the reading with the foot on the floor beside the bottom (support, hard: neither lifting
    the bottom nor floating), with their part cost. Per thigh direction the shin keeps its heading and takes the
    elevation that puts the leg's lowest point on the floor."""
    T, dT = _cone(t0, THIGH_SEARCH, 3.0, 12)
    az, el0 = _d3(s0)
    out = []
    if LEG_BATCH:
        # the same bisection for every thigh direction at once
        T = np.asarray(T, float)
        sdir = lambda el: np.stack([np.sin(np.radians(az)) * np.cos(np.radians(el)), np.sin(np.radians(el)),
                                    np.cos(np.radians(az)) * np.cos(np.radians(el))], -1)
        g = lambda el: _leg_gap_batch(side, T, sdir(el))
        lo = np.full(len(T), max(el0 - SHIN_SEARCH, -89.0)); hi = np.full(len(T), min(el0 + SHIN_SEARCH, 89.0))
        ok = ~((g(lo) > SUPPORT_TOL) | (g(hi) < -SUPPORT_TOL))
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            below = g(mid) < 0
            lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
        el = 0.5 * (lo + hi)
        S = sdir(el)
        ok &= ~((np.abs(g(el)) > SUPPORT_TOL) | (np.degrees(np.arccos(np.clip((T * S).sum(-1), -1, 1))) > 165))
        for i in np.flatnonzero(ok):
            ds = abs(el[i] - el0)
            out.append((T[i], S[i], float(part_cost("thigh", dT[i]) + part_cost("shin", ds)), float(past_cost("thigh", dT[i]) + past_cost("shin", ds))))
    for t, dt in ([] if LEG_BATCH else zip(T, dT)):
        g = lambda el: _leg_gap(posture, side, t, _from_d3(az, el))
        lo, hi = max(el0 - SHIN_SEARCH, -89.0), min(el0 + SHIN_SEARCH, 89.0)
        if g(lo) > SUPPORT_TOL or g(hi) < -SUPPORT_TOL:      # the foot cannot reach the floor / always lifts the bottom
            continue
        for _ in range(20):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if g(mid) < 0 else (lo, mid)
        s = _from_d3(az, 0.5 * (lo + hi))
        if abs(g(0.5 * (lo + hi))) > SUPPORT_TOL or np.degrees(np.arccos(np.clip(t @ s, -1, 1))) > 165:
            continue
        ds = abs(0.5 * (lo + hi) - el0)
        out.append((t, s, float(part_cost("thigh", dt) + part_cost("shin", ds)), float(past_cost("thigh", dt) + past_cost("shin", ds))))
    # the lower body comes before the upper body (the author, 2026-10-29): only the legs least past their tolerances stay
    # (as a rule all within them); among those the touching arm chooses
    if out:
        least = min(o[3] for o in out)
        out = [o[:3] for o in out if o[3] <= least + 1e-6]
    return out


def fit_legs(obj, duration_s, fit):
    """Sitting legs read from a reference (thigh / shin directions) fitted to this body together with the arms that
    touch them (2026-10-29, the author: one solver for the best look, every part with a tolerance and a weight; T001: fixing
    the arms first left a foot floating, fixing the legs first moved the upper arm). Per keyframe and leg: candidates
    near the reading that keep the bottom on the floor (hard); a leg an arm touches (with upper_dir) is chosen by its own
    cost plus the arm's (upper arm, forearm, wrist), the others by their own cost."""
    posture = obj.get("posture")
    if posture not in sk.BUTT_POSTURES:
        return obj
    kfs = obj["keyframes"]
    if not any(isinstance(k.get("legs"), dict) and any(isinstance(k["legs"].get(s), dict) for s in ("left", "right"))
               for k in kfs):
        return obj
    obj = copy.deepcopy(obj)
    kfs = obj["keyframes"]
    plain = copy.deepcopy(obj)
    for k in plain["keyframes"]:
        for side in ("left", "right"):
            if isinstance(k.get(f"{side}_arm"), dict) and "touch" in k[f"{side}_arm"]:
                k.pop(f"{side}_arm")
    pst, times, rows = kf.parse_keyframes(plain, duration_s)
    motion, root = kf.keyframes_to_motion(pst, times, rows, duration_s)
    P, G = sk.upper_fk(motion, root, JIDX)
    for i, k in enumerate(kfs):
        legs = k.get("legs")
        if not isinstance(legs, dict):
            continue
        f = min(int(round(float(k["t"]) * FPS)), len(motion) - 1)
        touched = {s: any(isinstance(k.get(f"{a}_arm"), dict) and str(k[f"{a}_arm"].get("touch", "")).replace("other_", "")
                          in LEG_TOUCH for a in ("left", "right")) for s in ("left", "right")}
        # legs no arm touches first: a touched leg's arm cost needs the bottom's height, which the other leg also sets
        for tside in sorted(("left", "right"), key=lambda s_: touched[s_]):
            lg = legs.get(tside)
            if not (isinstance(lg, dict) and "thigh" in lg):
                continue
            t0, s0 = _unit(np.asarray(lg["thigh"], float)), _unit(np.asarray(lg["shin"], float))
            cands = _leg_candidates(posture, tside, t0, s0)
            if not cands:
                continue
            arms = []
            for side in ("left", "right"):
                arm = k.get(f"{side}_arm")
                if not (isinstance(arm, dict) and arm.get("upper_dir") is not None):
                    continue
                region = str(arm.get("touch", ""))
                ts = side
                if region.startswith("other_"):
                    region, ts = region[len("other_"):], ("left" if side == "right" else "right")
                if region in LEG_TOUCH and ts == tside and arm.get("with", "palm") in ("palm", "fingertips", "fingers", "fist", "index"):
                    arms.append((side, arm, region))
            o = "right" if tside == "left" else "left"
            ol = legs.get(o, "default")
            oshape = (np.asarray(ol["thigh"], float), np.asarray(ol["shin"], float)) if isinstance(ol, dict) else ol
            Gt = G["upperChest"][f]
            best = None
            for t, s, c in cands:
                if arms:
                    shapes = {tside: (t, s), o: oshape}
                    R = sk.sit_legs_R(posture, shapes["left"], shapes["right"], legs.get("knees", "default"))
                    hips = sk.sit_hips(R)
                    Lw = {n: v[None] for n, v in sk.leg_points(R, hips).items()}
                    dh = hips - (sk.HIPS_REST + root[f] * sk.HIPS_Y)
                    for side, arm, region in arms:
                        S = P[f"{side}UpperArm"][f] + dh
                        p, n = _target(region, tside, 0, P, {"hips": G["hips"][f:f + 1]}, Lw, fit)
                        reach = {"fingertips": sk.FINGERTIP, "index": sk.FINGERTIP, "fingers": FINGERS_AT * sk.FINGERTIP}.get(arm.get("with"), sk.PALM)
                        palm_pt = p + n * HAND_HALF * (2.0 if arm.get("with") == "fist" else 1.0)
                        along = palm_pt - S
                        along = _unit(along - n * (along @ n))
                        W = palm_pt - along * reach
                        f0 = None if arm.get("fore_dir") is None else np.asarray(arm["fore_dir"], float)
                        res = _keep_upper(np.asarray(arm["upper_dir"], float), Gt.T @ (W - S), f0, Gt.T @ (palm_pt - S),
                                          reach, Gt.T @ n, gives=(UPPER_GIVE,), with_cost=True, coarse=True)
                        c += res[4]
                if best is None or c < best[0]:
                    best = (c, t, s)
            lg = {**lg, "thigh": [round(float(x), 3) for x in best[1]], "shin": [round(float(x), 3) for x in best[2]]}
            legs = {**legs, tside: lg}
            k["legs"] = legs
    return obj



FLOOR_FRONT = 0.04        # m in front of the knee (was 0.12: out of reach even leaning forward 25, 2026-11-03)
FLOOR_FRONT_OUT = 0.05     # m outward of the knee (was 2 cm inward: with the knees together both hands met between the
# knees and the arms lay on the thighs, T042 / S036 hands_clip + body_clip; 2026-11-03)
FLOOR_BACK_MAX = 12.0     # deg the solver may lean the torso back for hands on the floor (the author 2026-11-03: an upright
                          # reading leaned back the full 30; past this the hand goes onto its fingertips)
FLOOR_SLACK_TIPS = 0.0     # ... on the fingertips in front of the knees (2026-11-03: with 3.5 cm they ended 4-7 cm up;
# behind the hips 3.5 cm stays right: at 0 T044 on AvatarSample_D no longer reached)
FLOOR_SLACK = 0.035       # m beyond the arm the wrist of a flat hand on the floor may lie (see _floor_reach_ok)
FLOOR_SHOULDER = -10.0    # first the shoulder of a hand that cannot reach the floor drops this far (deg, kf limit)


def _floor_flat(Gt, sx, front=False):
    """Horizontal direction the fingers of a hand on the floor point: out and a little forward beside / behind the hips;
    forward and a little out in front of the knees (2026-11-03, T042: pointing out, the wrists of two hands in front of
    the knees met in the middle and the forearms lay between the knees)."""
    return _unit(Gt @ (np.array([sx * 0.25, 0.0, 1.0]) if front else np.array([sx * 1.0, 0.0, 0.25])) * np.array([1.0, 0.0, 1.0]))


def _floor_tip_along(Gt, sx, front=False):
    """Hand direction (wrist -> fingertips) of a hand standing on its fingertips on the floor: down, a little out
    (forward in front of the knees)."""
    return _unit(_floor_flat(Gt, sx, front) * 0.5 + np.array([0.0, -1.0, 0.0]))


def _floor_reach_ok(plain, i, sides, duration_s, fit, tips=False):
    """Can each (side, region) of `sides` put its hand flat on its floor spot in keyframe i of `plain` (arms left out)?"""
    pst, times, rows = kf.parse_keyframes(plain, duration_s)
    motion, root = kf.keyframes_to_motion(pst, times, rows, duration_s)
    P, G = sk.upper_fk(motion, root, JIDX)
    L = sk.motion_leg_points(motion, root, JIDX)
    f = min(int(round(float(plain["keyframes"][i]["t"]) * FPS)), len(motion) - 1)
    for side, region in sides:
        tside = side if not region.startswith("other_") else ("left" if side == "right" else "right")
        p, _ = _target(region.replace("other_", "", 1), tside, f, P, G, L, fit)
        front = region.replace("other_", "", 1) == "floor_front"
        # the wrist of a hand lying flat on the spot (as _resolve_once places it: the palm half a hand above the floor,
        # the fingers along floor_along) within the arm + FLOOR_SLACK: calibrated 2026-11-03 against full solves
        # (T003 / T002 / T007 / T044 / T045 / U063 at several leans): the palm ends ~1 cm less than the wrist's excess
        # off the floor, so 3.5 cm keeps it within REACH_TOL
        sx = 1.0 if side == "left" else -1.0
        if tips:                                            # on the fingertips: the hand slants down to them
            W = p + np.array([0.0, 0.005, 0.0]) - _floor_tip_along(G["upperChest"][f], sx, front) * sk.FINGERTIP
        else:
            along = _floor_flat(G["upperChest"][f], sx, front)
            W = p + np.array([0.0, HAND_HALF, 0.0]) - along * sk.PALM
        if np.linalg.norm(W - P[f"{side}UpperArm"][f]) > UA + FA + (FLOOR_SLACK_TIPS if tips and front else FLOOR_SLACK):
            return False
    return True


def floor_lean(obj, duration_s, fit):
    """The torso comes down to hands on the floor (2026-11-03, the author's choice (a); the contact is a hard limit, the torso
    gives). Sitting upright, this model's shoulders are ~0.52 m up and its arm to the wrist 0.43 m: a flat hand on the
    floor beside / behind the hips was out of reach and hung 8-10 cm above it. Per keyframe whose floor touches (without
    an arm reading) cannot reach: first the touching shoulders drop to FLOOR_SHOULDER, then the torso tilts as little as
    it must (bisection): floor_behind or hands on both sides -> lean back, floor_front -> lean forward, one hand beside
    -> bend toward it, up to kf.TORSO_LIM (leaning back at most FLOOR_BACK_MAX more than written); still out of reach:
    the hands go onto their fingertips (the author 2026-11-03), with the smallest tilt that then reaches.
    Records FLOOR_LEAN[i] = (shoulder drop used, part, degrees, reached)."""
    FLOOR_LEAN.clear()
    kfs = obj["keyframes"]
    todo = {}
    for i, k in enumerate(kfs):
        for side in ("left", "right"):
            a = k.get(f"{side}_arm")
            if isinstance(a, dict) and REGIONS.get(str(a.get("touch", "")).replace("other_", "", 1), ("",))[0] == "floor" \
                    and a.get("upper_dir") is None:
                todo.setdefault(i, []).append((side, str(a["touch"])))
    if not todo:
        return obj
    obj = copy.deepcopy(obj)
    plain = copy.deepcopy(obj)
    for k in plain["keyframes"]:
        for side in ("left", "right"):
            if isinstance(k.get(f"{side}_arm"), dict) and "touch" in k[f"{side}_arm"]:
                k.pop(f"{side}_arm")
    for i, sides in todo.items():
        if _floor_reach_ok(plain, i, sides, duration_s, fit):
            continue
        for k_ in (obj, plain):                             # 1. the touching shoulders drop
            sh = dict(k_["keyframes"][i].get("shoulders") or {})
            for side, _ in sides:
                v = sh.get(side, 0)
                sh[side] = {**v, "raise": FLOOR_SHOULDER} if isinstance(v, dict) else FLOOR_SHOULDER
            k_["keyframes"][i]["shoulders"] = sh
        if _floor_reach_ok(plain, i, sides, duration_s, fit):
            FLOOR_LEAN[i] = (FLOOR_SHOULDER, None, 0.0, True)
            continue
        regs = {r.replace("other_", "", 1) for _, r in sides}
        if regs == {"floor_front"}:
            part, lim = "lean", kf.lean_max(obj.get("posture", "stand"))
        elif regs == {"floor"} and len(sides) == 1:
            part, lim = "bend", kf.TORSO_LIM["bend"][1] * (1 if sides[0][0] == "left" else -1)
        else:
            part, lim = "lean", kf.TORSO_LIM["lean"][0]
        torso = dict(plain["keyframes"][i].get("torso") or {})
        v0 = float(torso.get(part, 0.0))
        if part == "lean" and lim < 0:                     # leaning back: at most FLOOR_BACK_MAX more than written
            lim = max(lim, min(v0, v0 - FLOOR_BACK_MAX))
        at = lambda v: [kk["keyframes"][i].__setitem__("torso", {**torso, part: round(v, 1)}) for kk in (obj, plain)]
        # flat hands lean forward only as far as the spine does (kf.SPINE_LEAN); past it (the pelvis tipping, sitting) the
        # hands go onto their fingertips (2026-11-03: flat at 53-57 deg the upper arms passed their limit and the
        # forearms lay on the thighs, S027 / T042)
        lim_flat = min(lim, max(v0, kf.SPINE_LEAN)) if part == "lean" and lim > 0 else lim
        at(lim_flat)
        reached = _floor_reach_ok(plain, i, sides, duration_s, fit)
        if reached:                                         # the smallest tilt that reaches (to 0.5 deg)
            lo, hi = v0, lim_flat
            while abs(hi - lo) > 0.5:
                mid = 0.5 * (lo + hi)
                at(mid)
                lo, hi = (lo, mid) if _floor_reach_ok(plain, i, sides, duration_s, fit) else (mid, hi)
            at(hi)
        if not reached:                                     # 3. the hands go onto their fingertips
            for side, _ in sides:
                obj["keyframes"][i][f"{side}_arm"] = {**obj["keyframes"][i][f"{side}_arm"], "with": "fingertips"}
            at(lim)
            reached = _floor_reach_ok(plain, i, sides, duration_s, fit, tips=True)
            if reached:                                     # with the fingertips, the smallest tilt that reaches
                lo, hi = v0, lim
                while abs(hi - lo) > 0.5:
                    mid = 0.5 * (lo + hi)
                    at(mid)
                    lo, hi = (lo, mid) if _floor_reach_ok(plain, i, sides, duration_s, fit, tips=True) else (mid, hi)
                at(hi)
            reached = "fingertips" if reached else False
        FLOOR_LEAN[i] = (FLOOR_SHOULDER, part, float(obj["keyframes"][i]["torso"][part]), reached)
    return obj


FLOOR_LEAN = {}


LEG_SLACK = 0.02           # m: the palm point may lie this much short of the straight arm + palm
# How far a leg touch may slide toward the hip when the arm cannot reach it: (bone it slides on, from, to) as fractions
# along that bone. "Hands on the knees" lands on the thigh just above the knee, "at the ankles" on the low shin.
LEG_SLIDE = {"thigh": ("thigh", 0.55, 0.15), "thigh_side": ("thigh", 0.4, 0.15), "knee": ("thigh", 1.0, 0.4),
             "shin": ("shin", 0.3, 0.05), "ankle": ("shin", 1.0, 0.5)}
LEG_LEAN_MAX = 12.0        # deg the torso may lean forward beyond what is written (as FLOOR_BACK_MAX); past it the
# finger pads touch instead of the palm (2026-11-03, test1 T034: kneeling with a slight back arch leaned 19 deg forward)
LEG_REACH = {}             # record: keyframe -> {side: (region, along) | "lean": (deg, reached)}


def _leg_reach_ok(plain, i, sides, duration_s, fit, slid, fingers=False):
    """Can each (side, region) of `sides` reach its leg spot in keyframe i of `plain` (touching arms left out)?
    slid: {side: (region, fraction along its bone)} for touches moved by leg_reach."""
    pst, times, rows = kf.parse_keyframes(plain, duration_s)
    motion, root = kf.keyframes_to_motion(pst, times, rows, duration_s)
    P, G = sk.upper_fk(motion, root, JIDX)
    L = sk.motion_leg_points(motion, root, JIDX)
    f = min(int(round(float(plain["keyframes"][i]["t"]) * FPS)), len(motion) - 1)
    for side, region in sides:
        tside = side if not region.startswith("other_") else ("left" if side == "right" else "right")
        reg, t = slid.get(side, (region.replace("other_", "", 1), None))
        p, n = _target(reg, tside, f, P, G, L, fit, t)
        hand = FINGERS_AT * sk.FINGERTIP if fingers else sk.PALM
        S = P[f"{side}UpperArm"][f]
        palm_pt = p + n * HAND_HALF
        if np.linalg.norm(palm_pt - S) > UA + FA + hand - LEG_SLACK:
            return False
        # reachable without clipping (2026-11-03, the author on test1 S030: hands at the shins below knees pulled up to the
        # chest, the nearly straight arm ran through the knee; no elbow swivel goes round it): the wrist one hand back
        # toward the shoulder, the elbow as _resolve_once would choose it (default pole, then _swivel_clear)
        Gt, sx = G["upperChest"][f], (1.0 if side == "left" else -1.0)
        W = palm_pt - _unit(palm_pt - S) * hand
        pole = Gt @ (np.array([sx * 1.0, 0.2, 0.0]) if REGIONS[reg][1] in ("shin", "ankle") else np.array([sx * 1.0, -0.4, -0.5]))
        clear = _clear_fn(f, S, Gt, P, G, L)
        pole = _swivel_clear(S, W, pole, clear, Gt)
        u, fo = _arm_dirs(np.zeros(3), Gt.T @ (W - S), Gt.T @ pole)
        if float(clear(np.array([u * UA + fo * FA * w for w in (0.0, 0.5, 0.75)])).max()) > CLEAR_TOL:
            return False
    return True


def leg_reach(obj, duration_s, fit):
    """Hands on the legs the arm cannot reach (2026-11-03, batch: kneeling / leaning hands on the thighs, seiza hands on
    the knees, knees_hugged hands at the ankles: the arm straight, the palm 6-23 cm short; this model's shoulder to palm
    is ~0.5 m). Per keyframe whose leg touches (without an arm reading) cannot reach: the spot first slides toward the
    hip within LEG_SLIDE (the touch becomes thigh / shin with `along`), then the torso leans forward as little as it
    must (bisection) up to kf.TORSO_LIM. The contact is the hard limit; the body gives, as for the floor."""
    LEG_REACH.clear()
    kfs = obj["keyframes"]
    todo = {}
    for i, k in enumerate(kfs):
        for side in ("left", "right"):
            a = k.get(f"{side}_arm")
            if isinstance(a, dict) and str(a.get("touch", "")).replace("other_", "", 1) in LEG_TOUCH \
                    and a.get("with", "palm") not in ("elbow", "forearm") and a.get("upper_dir") is None:
                todo.setdefault(i, []).append((side, str(a["touch"])))
    if not todo:
        return obj
    obj = copy.deepcopy(obj)
    plain = copy.deepcopy(obj)
    for k in plain["keyframes"]:
        for side in ("left", "right"):
            if isinstance(k.get(f"{side}_arm"), dict) and "touch" in k[f"{side}_arm"]:
                k.pop(f"{side}_arm")
    for i, sides in todo.items():
        slid = {}
        ok = lambda: _leg_reach_ok(plain, i, sides, duration_s, fit, slid)
        if ok():
            continue
        rec = {}
        for side, region in sides:                          # 1. the spot slides toward the hip, as little as it must
            r0 = region.replace("other_", "", 1)
            bone, t0, t1 = LEG_SLIDE[r0]
            one = [(side, region)]
            reach = lambda t: _leg_reach_ok(plain, i, one, duration_s, fit, {side: (bone, t)})
            if _leg_reach_ok(plain, i, one, duration_s, fit, {}):
                continue
            if not reach(t1):
                t = t1
            else:
                lo, hi = t1, t0                             # lo reaches, hi does not
                while hi - lo > 0.02:
                    mid = 0.5 * (lo + hi)
                    lo, hi = (mid, hi) if reach(mid) else (lo, mid)
                t = lo
            slid[side] = (bone, round(t, 3))
        for side, (bone, t) in slid.items():
            a = obj["keyframes"][i][f"{side}_arm"]
            pre = "other_" if str(a["touch"]).startswith("other_") else ""
            obj["keyframes"][i][f"{side}_arm"] = {**a, "touch": pre + bone, "_along": t}
            rec[side] = (pre + bone, t)
        if not ok():                                        # 2. the torso leans forward, at most LEG_LEAN_MAX
            torso = dict(plain["keyframes"][i].get("torso") or {})
            v0 = float(torso.get("lean", 0.0))
            lim = min(float(kf.lean_max(obj.get("posture", "stand"))), v0 + LEG_LEAN_MAX)
            at = lambda v: [kk["keyframes"][i].__setitem__("torso", {**torso, "lean": round(v, 1)}) for kk in (obj, plain)]

            def least(test):                                # the smallest lean in v0..lim that passes test (to 0.5)
                at(lim)
                if not test():
                    return False
                lo, hi = v0, lim
                while hi - lo > 0.5:
                    mid = 0.5 * (lo + hi)
                    at(mid)
                    lo, hi = (lo, mid) if test() else (mid, hi)
                at(hi)
                return True
            reached = least(ok)
            if not reached:                                 # 3. the finger pads touch (palm / back_of_hand only)
                fing = lambda: _leg_reach_ok(plain, i, sides, duration_s, fit, slid, fingers=True)
                for side, _ in sides:
                    a = obj["keyframes"][i][f"{side}_arm"]
                    if a.get("with", "palm") in ("palm", "back_of_hand"):
                        obj["keyframes"][i][f"{side}_arm"] = {**a, "with": "fingers"}
                reached = "fingers" if least(fing) else False
            rec["lean"] = (float(obj["keyframes"][i]["torso"]["lean"]), reached)
        LEG_REACH[i] = rec
    return obj


WRIST_REPLACE = 50.0       # deg of wrist bend past which a non-face touch re-places the wrist (_wrist_near)
WRIST_NEAR_LOOP = False    # True = _wrist_near's original per-candidate loop (the batched one must give the same choice)
WRIST_GIVE = 45.0          # deg the hand may turn off its forearm when _wrist_near chooses a wrist
INPLANE_BEND = False       # _wrist_near scores the in-plane bend only (tried 2026-11-08: no difference on head_pat, kept off)


# _wrist_near: weight (m per deg) of the hand's angle to a wanted direction (FACE_ALONG), and a much heavier one past
# what the wrist can bend at all (BEND_HARD deg): such a wrist is clamped by the solver and the fingers end far off
# (2026-11-13, T037: chin on the finger pads with the head tilted back; the wanted direction put the wrist above the
# chin, 19 cm off. Lowering OFF_W instead broke the neck-side touches, 15-18 cm off.)
OFF_W = 0.05
BEND_HARD = 90.0      # wrist flexion 70 + sideways 45 together reach past 70 (the neck-side palm needs ~80)
BEND_HARD_W = 1.0


def _wrist_near(palm_pt, reach, S, W0, clear, Gt, pole, n, side_max=HAND_SIDE, want=None):
    """A wrist the arm can reach for a spot close to the shoulder (2026-11-03, batch: chin / mouth / cheek / ear / neck /
    shoulder touches with fist, index, fingers: the wrist one hand back along the forearm came within ~5 cm of the
    shoulder, the elbow stopped at kf.MAX_ELBOW and the hand ended 7-20 cm off). The wrist goes on the sphere of radius
    `reach` round the contact point, at least W_MIN from the shoulder S, outside the body, as near W0 as possible; the
    hand then points from it to the spot. (Tried and dropped 2026-11-03: searching elbow swivels per wrist, even with a
    joint-limit term: joint_limit / body_clip in U033 / U046 / U056 and no change for T048.)"""
    k = np.arange(400) + 0.5
    phi, z = np.pi * (1 + 5 ** 0.5) * k, 1 - 2 * k / 400
    dirs = np.stack([np.sqrt(1 - z * z) * np.cos(phi), np.sqrt(1 - z * z) * np.sin(phi), z], -1)
    C = palm_pt[None] + dirs * reach
    d = np.linalg.norm(C - S[None], axis=-1)
    ok = (d >= W_MIN) & (d <= UA + FA - 0.01)
    if not ok.any():
        return W0
    C = C[ok]
    depth = clear((C - S[None]) @ Gt)                   # the wrist outside the body (torso frame for _clear_fn)
    # the hand (wrist -> spot) within WRIST_GIVE of the forearm the IK gives for that wrist: the wrist joint bends only
    # so far (at kf.WRIST_LIM the hand of U014's index on the cheek pointed 99 deg off the cheek)
    # and in the plane of the forearm and the surface normal: the wrist bends toward the palm or the back of the hand,
    # not sideways (the palm faces the surface; the same rule as HAND_SIDE for arms read from a reference)
    if WRIST_NEAR_LOOP:                                 # the original per-candidate loop (reference for the batch)
        bend, side = [], []
        for c in C:
            fo = Gt @ _arm_dirs(np.zeros(3), Gt.T @ (c - S), Gt.T @ pole)[1]
            hd = _unit(palm_pt - c)
            bend.append(np.degrees(np.arccos(np.clip(hd @ fo, -1, 1))))
            ax = np.cross(fo, n)
            side.append(np.degrees(np.arcsin(min(abs(hd @ _unit(ax)), 1.0))) if np.linalg.norm(ax) > 1e-3 else 0.0)
        bend, side = np.array(bend), np.array(side)
    else:                                               # all candidates at once (2026-11-03, same formulas as _arm_dirs)
        d = (C - S[None]) @ Gt                          # torso frame, rows = Gt.T @ (c - S)
        dn = np.linalg.norm(d, axis=-1, keepdims=True)
        Lc = np.clip(dn, abs(UA - FA) + 1e-3, UA + FA - 1e-3)
        nn = d / np.maximum(dn, 1e-9)
        x = (UA ** 2 - FA ** 2 + Lc ** 2) / (2 * Lc)
        h = np.sqrt(np.maximum(UA ** 2 - x ** 2, 0.0))
        pl = Gt.T @ pole
        pp = pl[None] - (nn @ pl)[:, None] * nn
        pp = pp / np.maximum(np.linalg.norm(pp, axis=-1, keepdims=True), 1e-9)
        E = x * nn + h * pp
        fo = ((nn * Lc - E) / FA) @ Gt.T                # back to the world
        hd = palm_pt[None] - C
        hd = hd / np.maximum(np.linalg.norm(hd, axis=-1, keepdims=True), 1e-9)
        if want is not None:
            # a wanted hand direction (FACE_ALONG): 'sideways' = off the plane of the forearm and THAT direction, and the
            # bend is measured from the hand we want, not from the forearm (2026-11-10: the shh index wanted vertical was
            # scored 45 deg 'sideways' against the lips' forward normal and the wrist was pushed 5 cm out; the finger
            # tilted 25 deg)
            sd = np.abs((hd * np.cross(fo, want[None]) / np.maximum(np.linalg.norm(np.cross(fo, want[None]), axis=-1, keepdims=True), 1e-9)).sum(-1))
            side = np.degrees(np.arcsin(np.minimum(sd, 1.0)))
            axn = np.ones(len(C))
        else:
            ax = np.cross(fo, n[None])
            axn = np.linalg.norm(ax, axis=-1)
            axu = ax / np.maximum(axn, 1e-9)[:, None]
            sd = np.abs((hd * axu).sum(-1))
            side = np.where(axn > 1e-3, np.degrees(np.arcsin(np.minimum(sd, 1.0))), 0.0)
        if INPLANE_BEND:      # the wrist bend proper = the in-plane part (the sideways part is wrist_side on the head, 2026-11-08)
            hp = hd - (hd * axu).sum(-1)[:, None] * axu
            hp = hp / np.maximum(np.linalg.norm(hp, axis=-1, keepdims=True), 1e-9)
            bend = np.degrees(np.arccos(np.clip((hp * fo).sum(-1), -1, 1)))
        else:
            bend = np.degrees(np.arccos(np.clip((hd * fo).sum(-1), -1, 1)))
    if want is not None:                     # how far the hand we would get is from the hand we want (deg), on top
        off = np.degrees(np.arccos(np.clip((hd * want[None]).sum(-1), -1, 1)))
    else:
        off = np.zeros(len(C))
    score = np.linalg.norm(C - W0[None], axis=-1) + 10.0 * np.maximum(depth, 0.0) \
        + 0.05 * (np.maximum(bend - WRIST_GIVE, 0.0) + np.maximum(side - side_max, 0.0)) + OFF_W * off \
        + BEND_HARD_W * np.maximum(bend - BEND_HARD, 0.0)   # 1 deg over = 5 cm (T048: at
    # 1 cm the wrist bent past kf.WRIST_LIM, was clamped and the fingers ended 17 cm off the neck)
    return C[int(np.argmin(score))]


# Hand directions (wrist -> fingers) on the face, in the head's frame (x = the hand's own side outward), and a shift of
# the contact point (head frame, m) (2026-11-03, the author: following the forearm, hands on the cheeks covered the eyes,
# an index on the cheek pointed up past the eye, fists under the chin pressed up into the jaw). flat = palm / fingers,
# back = back of the hand (both laid along the surface), tip = index / fingertips (into the surface a little).
FACE_ALONG = {
    ("cheek", "flat"): ((0.35, 1.0, -0.35), (0.025, -0.02, -0.03)),  # on the side of the face (nearer the nose the fingers
    # covered the eyes), fingers up toward the temple
    ("cheek", "tip"): ((-0.45, 0.25, -0.85), (0, -0.015, 0)),     # an index poking the cheek from the front and side
    ("mouth", "tip"): ((0.0, 1.0, 0.0), (0, 0, 0)),               # shh: the index vertical, its tip on the upper lip (_target; was
    # the middle of the finger on the lips, tip 3 cm above them, until 2026-11-11) (the author 2026-11-10: "手腕以下要是鉛直的"; the forearm may slant)
    ("mouth", "flat"): ((-0.75, 0.25, -0.3), (0, 0, 0)),          # covering the mouth: the hand across it
    ("mouth", "fingers"): ((-0.5, 0.85, -0.3), (0, -0.01, 0)),    # finger pads on the lips from below, fingers up and in,
    # the wrist under the chin (shy idle 'hair_and_lips', Agent C232, 2026-11-08; across it covered the mouth; straight up
    # (-0.15, 0.95) pushed the wrist lower and the forearm 20 deg further inward)
    ("chin", "fist"): ((0.0, 0.35, -0.95), (0, -0.01, 0)),        # fists under the chin, knuckles toward the throat
    ("chin", "flat"): ((0.3, 0.2, -0.9), (0, 0, 0)),              # the chin resting on the palm
    ("chin", "back"): ((-0.95, 0.0, -0.3), (0, 0, 0)),            # the chin on the back of the hand: fingers to the other side
    ("forehead", "flat"): ((-0.95, 0.2, 0.0), (0, 0, 0)),         # across the forehead
    ("forehead", "tip"): ((-0.3, 0.3, -0.9), (0, 0, 0)),
    ("temple", "tip"): ((-0.8, 0.2, -0.6), (0, 0, 0)),
    ("temple", "flat"): ((-0.7, 0.65, -0.3), (0, 0, 0)),          # finger pads in the hair above the temple, fingers up-in
    ("head_side_top", "flat"): ((-0.75, 0.55, -0.35), (0, 0, 0)),   # fingers in over the top of the head, a little back
    ("side_hair", "flat"): ((-0.25, 0.3, -1.0), (0, 0, 0)),         # fingers back into the strand from the front, a little
    # up and in (combing / tucking it behind the ear; 2026-11-10)
    ("side_hair", "tip"): ((-0.2, 0.2, -1.0), (0, 0, 0)),
    # and up over the head, the back of the hand to the front (Cecilia head_pat K4, 2026-11-07)
    ("neck_side", "flat"): ((0.0, 0.85, -0.5), (0, 0, 0)),        # fingers up the neck toward the ear
    ("head_front_top", "flat"): ((-0.45, 0.85, -0.25), (0, 0, 0)),  # protecting the head: hands on the upper sides of the
    # head, fingers up and in over the top (Cecilia head_pat K2 redo, 2026-11-08; back over the top they stood up like a roof)
}


# The same for the torso (torso frame, x = the hand's own side outward; 2026-11-08, shy chest: a fist clutching the
# collar points its knuckles up and inward, as a hand following the forearm lay across the chest).
TORSO_ALONG = {
    ("collar", "fist"): ((-0.6, 0.75, -0.2), (0, 0, 0)),
}


def _face_along(region, with_, Gh, sx, n, Gt=None):
    """(hand direction in the world, contact shift in the world) for a hand on the face (FACE_ALONG) or, with Gt, on the
    torso (TORSO_ALONG, torso frame), or None."""
    kind = {"palm": "flat", "fingers": "flat", "back_of_hand": "back", "index": "tip", "fingertips": "tip", "fist": "fist"}.get(with_)
    e = FACE_ALONG.get((region, with_)) or FACE_ALONG.get((region, kind))     # an entry for the exact hand part first
    if e is None and Gt is not None and (TORSO_ALONG.get((region, with_)) or TORSO_ALONG.get((region, kind))):
        e = TORSO_ALONG.get((region, with_)) or TORSO_ALONG.get((region, kind)); Gh = Gt
    if e is None:
        return None
    d, sh = (np.array(v, float) * np.array([sx, 1.0, 1.0]) for v in e)
    d = Gh @ d
    if region == "mouth" and kind == "tip":
        d = _unit(np.array([0.0, 1.0, MOUTH_TIP_LEAN]))      # the shh finger stands vertical in the WORLD (the hand a
        # little forward of vertical: the pointing index curls ~8 deg toward the palm, measured on the player's bones), not along a tilted / nodding head
        # (the author 2026-11-11: "平行於正z軸(上方)"; in the head frame it leaned 11 deg with the head's nod)
    if kind in ("flat", "back") and REGIONS.get(region, ("",))[0] != "hair":   # along the surface (hair is no
        # surface: the fingers go THROUGH the strand, 2026-11-10)
        d = d - n * (d @ n)
    return _unit(d), Gh @ sh


def _resolve_once(obj, prev, todo, duration_s, fit):
    """One pass: body pose from `prev` (the touching arms as resolved by the last pass, or left out), then
    the touches of `obj` resolved against it."""
    kfs = copy.deepcopy(obj["keyframes"])
    obj = {**obj, "keyframes": kfs}
    plain = copy.deepcopy(prev)
    for i, side in todo:
        a = plain["keyframes"][i][f"{side}_arm"]
        for key in ("touch", "with", "_along"):
            a.pop(key, None)
        if "upper" not in a:                                     # first pass: no direction yet
            plain["keyframes"][i].pop(f"{side}_arm")
    posture, times, rows = kf.parse_keyframes(plain, duration_s)
    motion, root = kf.keyframes_to_motion(posture, times, rows, duration_s)
    P, G = sk.upper_fk(motion, root, JIDX)
    L = sk.motion_leg_points(motion, root, JIDX)
    placed = {}
    for i, side in todo:
        arm = kfs[i][f"{side}_arm"]
        region, with_ = arm["touch"], arm.get("with", "palm")
        if not (region in REGIONS or region.replace("other_", "", 1) in REGIONS) or with_ not in WITH:
            raise ValueError(f"bad touch {arm}")
        gone = {"how", "over", "along", "keep_upper"} & set(arm)
        if gone:
            raise ValueError(f"touch fields {sorted(gone)} are gone (2026-11-08)")
        if "hand" in arm and with_ != "elbow":
            raise ValueError('"hand" on a touch only aims the forearm of "with": "elbow"')
        f = min(int(round(float(kfs[i]["t"]) * FPS)), len(motion) - 1)
        Gt, S = G["upperChest"][f], P[f"{side}UpperArm"][f]
        sx = 1.0 if side == "left" else -1.0
        if region == "other_hand":                                 # meet in front of the chest
            mid = P["upperChest"][f] + Gt @ np.array([0.0, -0.02, 0.27])
            p, n = mid + Gt @ np.array([sx * 0.025, 0, 0]), Gt @ np.array([sx, 0, 0])   # palms ~5 cm apart (bone centres)
        else:
            crossed = region.startswith("other_")
            if crossed:
                region, tside = region[len("other_"):], ("left" if side == "right" else "right")
            else:
                tside = side
            o_arm = kfs[i].get(f"{'left' if side == 'right' else 'right'}_arm")
            pair = isinstance(o_arm, dict) and o_arm.get("touch") == arm["touch"]
            if arm.get("at") is not None and (REGIONS.get(region, ("",))[0] != "hair" or arm["at"] not in HAIR_AT):
                raise ValueError(f'touch "at" is root | mid | tip, on a hair region only: {arm}')
            region = hair_region(region, tside, fit)
            p, n = _target(region, tside, f, P, G, L, fit, arm.get("_along"), pair, crossed, with_,
                           arm.get("at") if REGIONS[region][0] == "hair" else None)
            if REGIONS[region][0] == "floor" and arm.get("upper_dir") is not None:
                # a hand on the floor read from a reference (2026-10-29, the author: the floor in the pictures is not exactly
                # flat or at our height): only flat on our floor; where on it follows the arm as read
                E = S + Gt @ _unit(np.asarray(arm["upper_dir"], float)) * UA
                fd = Gt @ _unit(np.asarray(arm.get("fore_dir") or arm["upper_dir"], float))
                end = E + fd * FA                                 # the wrist, on the floor if the arm reaches it
                if fd[1] < -1e-3 and end[1] < p[1] + HAND_HALF:
                    end = E + fd * ((p[1] + HAND_HALF - E[1]) / fd[1])
                along = _unit(Gt @ np.array([sx * 1.0, 0.0, 0.25]) * np.array([1.0, 0.0, 1.0]))   # as floor_along below
                p = np.array([end[0], p[1], end[2]]) + along * sk.PALM
        for q in placed.get(i, []):                             # the other hand is already there
            if np.linalg.norm(q - p) < 0.05:
                r1 = region.replace("other_", "", 1)
                if r1 in LEG_REGIONS:                           # on a leg: beside it along the leg (S005 / S025, both
                    s_ = tside                                  # hands on one knee: stacked fingers went through
                    a_, b_ = (L[f"{s_}Knee"][f], L[f"{s_}Ankle"][f]) if r1 in ("knee", "shin", "ankle") else \
                             (L[f"{s_}Knee"][f], L[f"{s_}Hip"][f])          # each other)
                    p = p + _unit(b_ - a_) * LEG_SIDE_BY_SIDE
                else:                                           # elsewhere: lie on top of it
                    p = p + n * 2.5 * HAND_HALF
        placed.setdefault(i, []).append(p)
        TARGETS[(f, side)] = (with_, region, p.copy())          # a record for checks (the last pass wins)
        if with_ == "elbow":                                    # the elbow rests on the spot (an elbow on a knee)
            from .arm_words import hand_point
            e_t = _unit(Gt.T @ (p + n * (sk.ARM_R["forearm"] + 0.005) - S))
            words = str(arm.get("hand", "face front close")).split()
            Hp, Sr = hand_point(*words, side)                   # the forearm points at the hand words' spot
            fo = _unit((Hp - Sr) - UA * e_t)
            new = {k: v for k, v in arm.items() if k != "hand"}
            new.update(upper=[round(float(x), 3) for x in e_t], fore=[round(float(x), 3) for x in fo])
            kfs[i][f"{side}_arm"] = new
            continue
        if with_ == "forearm":                                  # the forearm lies across the spot (2026-10-28, T005:
            # a forearm draped over a raised knee): it runs inward toward the middle of the body and level; the elbow
            # is where that line meets the upper arm's reach
            q = p + n * (sk.ARM_R["forearm"] + 0.005)
            fdir = Gt @ np.array([-sx, 0.0, 0.35])
            fdir = _unit(fdir - n * (fdir @ n))
            dq = q - S                                          # |dq - fdir*s| = UA, the elbow s behind q
            b, c = dq @ fdir, dq @ dq - UA ** 2
            disc = b * b - c
            s_ = b + np.sqrt(disc) if disc >= 0 else b
            s_ = float(np.clip(s_, 0.05, 0.8 * FA))
            E = q - fdir * s_
            new = {k: v for k, v in arm.items()}
            new.update(upper=[round(float(x), 3) for x in _unit(Gt.T @ (E - S))],
                       fore=[round(float(x), 3) for x in Gt.T @ fdir], palm="down", wrist=0.0)
            kfs[i][f"{side}_arm"] = new
            continue
        # wrist: the heel of the hand on the spot, the hand along the surface past it (T012: the cheek resting on the
        # wrist, fingers up the side of the face)
        reach = {"fingertips": sk.FINGERTIP, "index": sk.FINGERTIP, "fingers": FINGERS_AT * sk.FINGERTIP, "wrist": 0.0}.get(with_, sk.PALM)
        off = HAND_HALF * (2.0 if with_ == "fist" else 1.0) + (0.01 if REGIONS[region][0] == "head" else 0.0)
        if region == "mouth" and with_ in ("index", "fingertips"):
            off = MOUTH_TIP_OFF                 # a fingertip just in front of the lips (was 0.005 = ON them: it dipped into the face)
            # a fingertip ON the lips (the hand-half + hair standoff left it 2.8 cm in front, 2026-11-10)
        palm_pt = p + n * off
        if region == "mouth" and with_ in ("index", "fingertips"):
            nh = np.array([n[0], 0.0, n[2]])        # straight out in front of the lips, level (along the lips' normal,
            palm_pt = p + _unit(nh) * off           # which points a little down, the tip ended at the lower lip)
        if with_ == "index" and REGIONS[region][0] == "head" and (fit.get("index") or {}).get("tip"):
            # the INDEX finger on the spot, not the hand's axis (2026-11-11, the author: the shh finger sat beside the mouth):
            # the index lies ~2 cm to the thumb side of the hand bone's line (fit `index`, measure.indexFinger); the hand
            # point aimed at is moved back by that offset, turned by the hand as the last pass left it (resolve_touches
            # runs three passes, so it settles)
            it = fit["index"]["tip"]
            reach = abs(it[0])
            palm_pt = palm_pt - G[f"{side}Hand"][f] @ np.array([0.0, it[1], it[2]])
        # wrist = palm point minus the hand's length along the forearm; two passes to settle the direction
        # elbow: up and out for the top / back / sides of the head, down and a little forward for the face,
        # out and back for the body
        r0 = region.replace("other_", "", 1)
        if arm.get("elbow_on"):                                    # the elbow toward another spot (2026-10-28,
            # T012: hand on the cheek, elbow resting by the raised knee)
            eo = str(arm["elbow_on"])
            es = side if not eo.startswith("other_") else ("left" if side == "right" else "right")
            e_p, e_n = _target(eo.replace("other_", "", 1), es, f, P, G, L, fit)
            pole = e_p + e_n * ELBOW_ON_UP - S                # aimed a little above the surface: aimed at it, the
            # elbow of a shorter-armed model (0.x sample) sank 3 cm into the knee
        elif arm.get("elbow"):                                     # which way the elbow points, as in the arm words
            # (2026-10-29, T020: hand behind the head with the elbow up above the head)
            if arm["elbow"] not in ELBOW_POLES:
                raise ValueError(f"unknown elbow {arm['elbow']!r}; one of {tuple(ELBOW_POLES)}")
            pole = Gt @ (np.array(ELBOW_POLES[arm["elbow"]]) * np.array([sx, 1.0, 1.0]))
        elif r0 == "head_front_top":                               # elbows forward at face height, not out
            pole = Gt @ np.array([sx * 0.25, -0.6, 1.0])   # forward and down: hunched (2026-11-08)
        elif r0 == "side_hair":                                    # the elbow forward and out, so the forearm points back
            # toward the ear and the fingers go back into the strand from the FRONT (2026-11-10; with the elbow out / up
            # the hand stood beside the head behind the thin strand, palm to the head: "撩空氣")
            pole = Gt @ np.array([sx * 0.7, -0.3, 1.0])
        elif r0 in ("top_of_head", "back_of_head", "nape", "ear", "temple", "head_side_top", "neck_side",
                    "bangs", "back_hair", "ponytail"):
            pole = Gt @ np.array([sx * 1.0, 0.5, -0.2])
        elif r0 == "mouth" and with_ in ("index", "fingertips"):   # shh: the elbow in front of the chest so the forearm
            # comes up from below (the author 2026-11-11: the hand must stand vertical; from an elbow at the side the wrist
            # needed more sideways bend than it has)
            pole = Gt @ np.array([sx * MOUTH_TIP_POLE[0], MOUTH_TIP_POLE[1], MOUTH_TIP_POLE[2]])
        elif r0 in ("chin", "mouth", "cheek", "forehead"):
            pole = Gt @ np.array([sx * 0.3, -0.8, 0.6])             # elbow down and in front of the chest
        elif r0 in ("chest", "heart", "belly"):                    # hand on the front: elbow down at the side
            pole = Gt @ np.array([sx * 0.6, -1.0, 0.0])
        elif r0 in ("shin", "ankle"):                              # down a leg: elbow out past the knee (T014)
            pole = Gt @ np.array([sx * 1.0, 0.2, 0.0])
        elif (Gt.T @ (p - P["upperChest"][f]))[0] * sx < 0.02:    # across the body: elbow in front of the chest
            pole = Gt @ np.array([sx * 0.3, -0.6, 1.0])
        else:
            pole = Gt @ np.array([sx * 1.0, -0.4, -0.5])
        # The hand lies along the surface (its fingers would go in if it followed a forearm that points into
        # the body): the wrist sits back from the palm point along the surface, and the wrist bend turns the
        # hand from the forearm's direction to the surface.
        fore = _unit(palm_pt - S)
        # a hand flat on the floor points its fingers out to the side and a little forward (2026-10-28: following the
        # forearm, a hand below the shoulder stood on its fingertips)
        on_floor = REGIONS.get(r0, ("",))[0] == "floor"
        floor_along = _floor_flat(Gt, sx, r0 in ("floor_front", "floor_edge"))
        fixed_along = floor_along if on_floor else None
        if on_floor and with_ == "fingertips":           # standing on the fingertips (floor_lean, 2026-11-03)
            fixed_along = _floor_tip_along(Gt, sx, r0 == "floor_front")
        if r0 == "shin" and (S - p)[1] > WRAP_ABOVE:   # holding a lower leg from above (2026-10-28, S005 standing
            # with a knee pulled up): the fingers wrap around the shin from the hand's side across (following the forearm
            # they ran up the shin into the thigh). A sitting hand reaching forward (T014) keeps following its forearm:
            # wrapping there turned the forearm through the raised knee.
            ax = _unit(L[f"{tside}Ankle"][f] - L[f"{tside}Knee"][f])
            wrap = _unit(np.cross(ax, n))
            if wrap @ (Gt @ np.array([-sx, 0.0, 0.0])) < 0:
                wrap = -wrap
            fixed_along = wrap
        akimbo = None
        if "fingers" in arm and not (r0 in ("hip", "waist") and with_ == "palm"):
            raise ValueError('"fingers" (forward | down | back) is only for a palm on the hip / waist')
        if r0 in ("hip", "waist") and with_ == "palm":
            # hand on the hip (akimbo): the fingers go forward and a little down around the waist, the wrist bent back
            # and the thumb behind (2026-10-29, the author on T015: following the forearm the fingers pointed down the thigh);
            # "fingers": down / back = the fingers down the side of the hip or back around it (2026-11-08, shy idle
            # 'hair_and_lips': Agent's fingers point down-back on the hip bone)
            fw = str(arm.get("fingers", "forward"))
            if fw not in AKIMBO_FINGERS:
                raise ValueError(f'hip "fingers" must be one of {tuple(AKIMBO_FINGERS)}: {fw!r}')
            fwd = Gt @ np.array(AKIMBO_FINGERS[fw])
            # along the surface, half way between forward-down and the forearm's own direction (T020: forward-down alone
            # needed a 70 deg wrist bend, the hand stuck out in front of the belly)
            akimbo = _unit(fwd - n * (fwd @ n))
        h_pref = None
        if r0 == "knee" and with_ in ("palm", "fingers"):
            # a hand on a knee: the fingers go forward and down over its front, the wrist up over the thigh (2026-10-29,
            # T001: a hand following the forearm lay flat on the round knee, fingertips 6 cm up in the air, the wrist in
            # the thigh; the push-out then lifted the hand off)
            sh = L[f"{tside}Ankle"][f] - L[f"{tside}Knee"][f]
            h_pref = _unit(_unit(sh - n * (sh @ n)) + _unit(fore - n * (fore @ n)))
        fa = _face_along(r0, with_, G["head"][f], sx, n, Gt)
        on_head = REGIONS.get(r0, ("",))[0] in ("head", "hair")
        # on the head an arm read from a reference (upper_dir) may turn its hand sideways up to the wrist_side range further
        # off the forearm's plane when _wrist_near picks the wrist (head_pat's hands on the head; for arms without a
        # reading it moved an approved cheek touch to the collar, 2026-11-08)
        smax = HAND_SIDE + (kf.WSIDE_LIM[1] if on_head and arm.get("upper_dir") is not None else 0.0)
        kmax = smax if HEAD_SIDE_KEEP else None    # the same allowance inside _keep_upper (trial switch, 2026-11-08)
        if fa is not None and arm.get("elbow") == "together" and r0 == "cheek":
            d_, sh_ = (np.array(v, float) * np.array([sx, 1.0, 1.0]) for v in TOGETHER_JAW)
            d_ = G["head"][f] @ d_
            fa = (_unit(d_ - n * (d_ @ n)), G["head"][f] @ sh_)
        if fa is not None:                               # a preferred hand direction (see the loop below)
            palm_pt = palm_pt + fa[1]
            if h_pref is None and arm.get("upper_dir") is not None:
                h_pref = fa[0]                           # an arm read from a reference keeps it through _keep_upper's hand_dir cost (2026-11-08)
            TARGETS[(f, side)] = (with_, region, TARGETS[(f, side)][2] + fa[1])   # the record follows the shift
        if with_ == "wrist" and REGIONS.get(r0, ("",))[0] == "head":
            # the face rests on the heel of the hand at the side of the jaw and the fingers go up the side of the face
            # (following the forearm the hand lay across the eye)
            Gh = G["head"][f]
            shift = - Gh @ np.array([0.0, 0.03, 0.0]) + Gh @ np.array([sx * 0.015, 0.0, -0.01])
            palm_pt = palm_pt + shift
            TARGETS[(f, side)] = (with_, region, TARGETS[(f, side)][2] + shift)   # the record follows the shift
            up_h = Gh @ np.array([0.0, 1.0, -0.15])
            fixed_along = _unit(up_h - n * (up_h @ n))
        for _ in range(4):
            along = fore - n * (fore @ n)
            along = fixed_along if fixed_along is not None else (_unit(along) if np.linalg.norm(along) > 0.2 else fore)
            if akimbo is not None:
                along = _unit(along + akimbo)
            W = palm_pt - along * reach
            if fa is not None and reach > 0:
                # on the face the hand prefers its FACE_ALONG direction, as near as the elbow and the wrist joint allow
                # (forced, the wrist hit kf.WRIST_LIM and the hand went 6 cm off; 2026-11-03)
                W0 = palm_pt - fa[0] * reach
                if arm.get("elbow") == "together":         # wrists and elbows near the midline, the hands going up and
                    # out to the face (2026-11-03, U033: hands on the jaw, elbows together in front; with the wrist
                    # straight under the cheek the forearms stood under the shoulders)
                    w0 = Gt.T @ (W0 - P["upperChest"][f])
                    W0 = P["upperChest"][f] + Gt @ np.array([sx * TOGETHER_X, w0[1], max(w0[2], TOGETHER_Z)])
                # the wanted-direction score (2026-11-10, for the shh index / cheek pokes) not on the side of the neck:
                # it sits so close to the shoulder that no wrist gives that direction, and the score then picked a
                # wrist above the hand (2026-11-13: T048 / U046 / U071 2 -> 12-18 cm off on all four models)
                W = _wrist_near(palm_pt, reach, S, W0, _clear_fn(f, S, Gt, P, G, L), Gt, pole, n, side_max=smax,
                                want=None if r0 == "neck_side" else fa[0])
                along = _unit(palm_pt - W)
            elif reach > 0 and np.linalg.norm(W - S) < W_MIN:
                W = _wrist_near(palm_pt, reach, S, W, _clear_fn(f, S, Gt, P, G, L), Gt, pole, n, side_max=smax)
                along = _unit(palm_pt - W)
            o_t = str((kfs[i].get(f"{'left' if side == 'right' else 'right'}_arm") or {}).get("touch", "")) \
                if isinstance(kfs[i].get(f"{'left' if side == 'right' else 'right'}_arm"), dict) else ""
            both_cross = str(arm["touch"]).startswith("other_") and o_t.startswith("other_")
            over = side == FRONT_ARM                        # FRONT_ARM goes round the other forearm
            if arm.get("upper_dir") is None and not (both_cross and not over):
                # both arms crossing: only FRONT_ARM turns (round the other forearm); the other one keeps its pole
                # (turning both put U030's back arm into the chest). Earlier: not when both arms cross (other_elbow / other_shoulder ...): it ignores the other arm (U030 / U041
                # elbows and hands through each other); one crossing arm alone may swivel (U019: its forearm went
                # through the chest). Near the shoulder (back of the head, neck) the shoulder-wrist line
                # points inward and the given pole put the elbow inside the head / chest (U069, U033)
                clr = _clear_fn(f, S, Gt, P, G, L)
                rise = 0.03
                if str(arm["touch"]) in CROSS_FRONT:
                    clr = _clear_cloth(clr, f, S, Gt, P, G)
                if over:                                # crossed arms: this one goes over the other (one-sided,
                    # both avoiding each other swapped places every pass), and may lift its elbow up to FRONT_RISE (a
                    # self-hug has one forearm above the other; at 3 cm no swivel cleared U041)
                    clr = _clear_other(clr, f, side, S, Gt, P)
                    rise = FRONT_RISE
                # elbows together in front: the upper arm on the chest is meant (checking it, the elbows went back
                # under the shoulders)
                pole = _swivel_clear(S, W, pole, clr, Gt, rise, upper=arm.get("elbow") != "together")
            u, fo = _arm_dirs(np.zeros(3), Gt.T @ (W - S), Gt.T @ pole)
            fore = Gt @ fo
        if arm.get("upper_dir") is not None:
            # The upper arm as given (read from a reference), the forearm and the hand take what the model's proportions
            # leave over (2026-10-29, the author: the nearer the torso, the more a part's angle matters; 5 deg off on the upper
            # arm is too much). The upper arm may move up to UPPER_GIVE deg; TOL / WEIGHT decide which part gives;
            # the hand then turns from where the wrist really is toward the contact point.
            if h_pref is None and akimbo is not None and "fingers" in arm:
                h_pref = akimbo                    # a written "fingers" word counts for a read arm too (2026-11-08; before,
                # the hand just pointed from wherever the wrist landed to the hip, 30 deg off the reading). Only when written:
                # the default direction would move the approved T015 / T020 / T026 hands by ~30 deg
            clr_read = _clear_fn(f, S, Gt, P, G, L)
            if str(arm["touch"]) in CROSS_FRONT:
                clr_read = _clear_cloth(clr_read, f, S, Gt, P, G)
            u, fo, W_t = _keep_upper(np.asarray(arm["upper_dir"], float), Gt.T @ (W - S),
                                     None if arm.get("fore_dir") is None else np.asarray(arm["fore_dir"], float),
                                     Gt.T @ (palm_pt - S), reach, Gt.T @ n, arm.get("keep") == "upper",
                                     clear=clr_read, h_pref=None if h_pref is None else Gt.T @ h_pref,
                                     side_max=kmax, tip_tol=HEAD_TIP_TOL if on_head and HEAD_SIDE_KEEP else 0.0)
            fore = Gt @ fo
            Wa = S + Gt @ (u * UA + fo * FA)
            along = _unit(palm_pt - Wa)
        bend = np.degrees(np.arccos(np.clip(fore @ along, -1, 1)))
        if (arm.get("upper_dir") is None and fa is None and reach > 0 and fixed_along is None and akimbo is None
                and h_pref is None and bend > WRIST_REPLACE):
            # the hand would need more wrist bend than the joint has (2026-11-03, test1 U035: finger pads below the
            # collarbones on the slope of a big bust, the wrist clamped at 50 and the fingers slid 6.5 cm down): the
            # wrist goes where the bend fits, as for the face (_wrist_near)
            W = _wrist_near(palm_pt, reach, S, W, _clear_fn(f, S, Gt, P, G, L), Gt, pole, n, side_max=smax)
            along = _unit(palm_pt - W)
            u, fo = _arm_dirs(np.zeros(3), Gt.T @ (W - S), Gt.T @ pole)
            fore = Gt @ fo
            bend = np.degrees(np.arccos(np.clip(fore @ along, -1, 1)))
        # + = toward the palm (the palm faces -n): the hand bends toward the surface when it points into it more than the
        # forearm does (2026-10-29, T001: a level forearm with the knee below the wrist; the old rule 'forearm into the
        # surface -> bend back' turned the hand 34 deg up, 7 cm off the knee)
        # (the back of the hand on the surface: the palm faces away from it, so 'toward the palm' is away from the surface;
        # 2026-11-03, U048: the chin on the backs of the hands bent the hands the wrong way, 8 cm off)
        wrist_side = 0.0
        if on_head and reach > 0:
            # the part of the hand direction out of the forearm / palm-normal plane becomes a sideways wrist bend (+ =
            # toward the little finger: -sx x (fore x palm direction)); the in-plane part stays the wrist bend. Before,
            # that part was dropped and the hand pointed beside the spot (fingers up at the temple, 2026-11-08).
            # little finger = cross(fingers, palm direction) on the left hand, the opposite on the right (T-pose: fingers
            # +-x, palm -y, thumb +z on both): sx, not -sx (2026-11-10: the sign was inverted, so the sideways bend
            # pushed an index on the lips 40 deg further off vertical instead of straightening it)
            little = np.cross(fore, n if with_ == "back_of_hand" else -n) * sx
            if np.linalg.norm(little) > 1e-3:
                little = _unit(little)
                wrist_side = float(np.clip(np.degrees(np.arcsin(np.clip(along @ little, -1, 1))), *kf.WSIDE_LIM))
                along = _unit(along - little * (along @ little))
                bend = np.degrees(np.arccos(np.clip(fore @ along, -1, 1)))
        wrist = bend if (along - fore) @ (n if with_ == "back_of_hand" else -n) > 0 else -bend
        n_t = Gt.T @ n
        # the exact palm direction (against the surface; the back of the hand for back_of_hand), off the forearm axis
        want = -n_t if with_ != "back_of_hand" else n_t
        want = want - fo * (want @ fo)
        palm = [round(float(x), 3) for x in _unit(want)] if np.linalg.norm(want) > 0.2 else \
            _palm_word(n_t if with_ != "back_of_hand" else -n_t, fo, side)
        new = {k: v for k, v in arm.items()}
        new.update(upper=[round(float(x), 3) for x in u], fore=[round(float(x), 3) for x in fo], palm=palm,
                   wrist=round(float(np.clip(wrist, *kf.WRIST_LIM)), 1))
        if wrist_side:
            new["wrist_side"] = round(wrist_side, 1)
        kfs[i][f"{side}_arm"] = new
        shape = {"fist": "fist", "index": "point"}.get(with_)
        if on_floor and with_ in ("palm", "fingertips"):   # a hand pressing on the floor is flat (or on its fingertips)
            shape = "open"
        if shape:
            kfs[i].setdefault("hands", {})
            kfs[i]["hands"] = {**kfs[i]["hands"], side: shape}
    return obj


LEG_REGIONS = {"thigh": "thigh", "thigh_side": "thigh", "knee": "thigh", "shin": "shin", "ankle": "shin"}


def follow_leg_touches(motion, root, times, obj):
    """A hand going to (or staying on) a leg that moves meanwhile rides with it (2026-10-28: a resting hand was hit by
    its thigh rising to a raised knee). For every keyframe segment that ends in a touch on a thigh / knee / shin /
    ankle of a leg that moves in that segment, the wrist is interpolated in that leg segment's own frame (from where
    it is at the segment start to where it is at the end, eased like the keyframes) and the arm re-solved per frame.
    motion [T,J,4] is changed in place and returned; times = the parsed keyframe times; obj = the keyframe JSON."""
    from .depenetrate import _resolve
    from .rotations import quat_to_mat
    kfs = obj.get("keyframes", []) if isinstance(obj, dict) else []
    T = len(motion)
    hips = sk.HIPS_REST + root * sk.HIPS_Y
    Mh = quat_to_mat(motion[:, JIDX["hips"]])
    frames = {}
    for leg in ("left", "right"):
        Gu = Mh @ quat_to_mat(motion[:, JIDX[f"{leg}UpperLeg"]])
        Gl = Gu @ quat_to_mat(motion[:, JIDX[f"{leg}LowerLeg"]])
        hip = hips + np.einsum("tij,j->ti", Mh, sk.HIP_OFFSET * np.array([sk.side_sign(leg), 1, 1]))
        frames[(leg, "thigh")] = (hip, Gu)
        frames[(leg, "shin")] = (hip + np.einsum("tij,j->ti", Gu, sk.THIGH), Gl)
    kt = [float(k["t"]) for k in kfs]
    for side in ("left", "right"):
        sd = 1.0 if side == "left" else -1.0
        spec = None
        for i in range(1, len(kfs)):
            a = kfs[i].get(f"{side}_arm")
            spec = a if a is not None else spec
            if not isinstance(spec, dict) or "touch" not in spec:
                continue
            region = str(spec["touch"])
            seg = LEG_REGIONS.get(region.replace("other_", ""))
            if not seg:
                continue
            leg = side if not region.startswith("other_") else ("right" if side == "left" else "left")
            f0, f1 = min(int(round(kt[i - 1] * FPS)), T - 1), min(int(round(kt[i] * FPS)), T - 1)
            if f1 <= f0:
                continue
            O, G = frames[(leg, seg)]
            moved = np.linalg.norm(O[f1] - O[f0]) > 0.01 or np.arccos(np.clip((np.trace(G[f0].T @ G[f1]) - 1) / 2, -1, 1)) > 0.05
            if not moved:
                continue
            P, _ = sk.upper_fk(motion, root, JIDX)
            W = P[f"{side}Hand"]
            loc0, loc1 = G[f0].T @ (W[f0] - O[f0]), G[f1].T @ (W[f1] - O[f1])
            u = np.clip((np.arange(T) - f0) / (f1 - f0), 0, 1)
            u = u * u * (3 - 2 * u)
            want = O + np.einsum("tij,tj->ti", G, _around_leg(loc0, loc1, u, seg))
            pw = np.zeros((T, 3))
            pw[f0 + 1:f1] = (want - W)[f0 + 1:f1]
            P2, G2 = sk.upper_fk(motion, root, JIDX)
            _resolve(motion, side, sd, P2, G2, pw, np.zeros((T, 3)))
    _leave_moving_legs(motion, root, kt, kfs, frames)
    return motion


def _around_leg(loc0, loc1, u, seg):
    """Wrist path [T,3] in a leg segment's frame from loc0 to loc1 (eased weights u [T]), around the leg: the position
    along the bone and the direction around it are interpolated separately and the distance from the axis never drops
    below the leg's surface + a forearm radius (2026-10-28: the straight chord from a hand on the thigh to the top of
    the raised knee cut 2 cm into the thigh on the 0.x model)."""
    axis = sk.THIGH if seg == "thigh" else sk.SHIN
    L2 = float(axis @ axis)
    a0, a1 = loc0 @ axis / L2, loc1 @ axis / L2
    r0, r1 = loc0 - a0 * axis, loc1 - a1 * axis
    n0, n1 = np.linalg.norm(r0), np.linalg.norm(r1)
    d0, d1 = r0 / max(n0, 1e-9), r1 / max(n1, 1e-9)
    om = np.arccos(np.clip(d0 @ d1, -1, 1))
    uu = u[:, None]
    if om < 1e-4:
        d = np.repeat(d0[None], len(u), 0)
    else:
        d = (np.sin((1 - uu) * om) * d0 + np.sin(uu * om) * d1) / np.sin(om)
    a = a0 * (1 - u) + a1 * u
    floor = (sk.leg_radius("thigh", np.clip(a, 0, 1)) if seg == "thigh" else np.full(len(u), sk.LEG_R["shin"])) + sk.ARM_R["forearm"]
    lin = n0 * (1 - u) + n1 * u
    # clear of the surface in the middle only: at the ends the hand stays where it is (a wrist resting on the thigh is
    # nearer the axis than the floor, and clamping there jumped it outward in one frame)
    rad = lin + np.maximum(np.minimum(floor, max(n0, n1)) - lin, 0.0) * np.sin(np.pi * u)
    return a[:, None] * axis + d * rad[:, None]


LEAVE_POWER = 1.0


def _leave_moving_legs(motion, root, kt, kfs, frames):
    """A hand that rests on a leg when a segment starts (a touch, or the sitting rest pose's hands on the thighs) and
    that leg moves in the segment: the hand first rides with the leg, then blends into its own path (2026-10-28,
    S065: the thigh rising to a raised knee went 7.7 cm into the resting hand). Segments that end in a leg touch
    are follow_leg_touches' own."""
    from .depenetrate import _resolve
    T = len(motion)
    for i in range(1, len(kt)):
        f0, f1 = min(int(round(kt[i - 1] * FPS)), T - 1), min(int(round(kt[i] * FPS)), T - 1)
        if f1 <= f0:
            continue
        P, G_ = sk.upper_fk(motion, root, JIDX)
        for side in ("left", "right"):
            end = kfs[i].get(f"{side}_arm") if i < len(kfs) else None
            if isinstance(end, dict) and LEG_REGIONS.get(str(end.get("touch", "")).replace("other_", "")):
                continue
            sx = 1.0 if side == "left" else -1.0
            palm = P[f"{side}Hand"][f0] + G_[f"{side}Hand"][f0] @ np.array([sx * sk.PALM, 0, 0])
            best = None
            for (leg, seg), (O, G) in frames.items():
                axis = G[f0] @ (sk.THIGH if seg == "thigh" else sk.SHIN)
                tt = np.clip((palm - O[f0]) @ axis / (axis @ axis), 0, 1)
                r = sk.leg_radius("thigh", np.array([tt]))[0] if seg == "thigh" else sk.LEG_R["shin"]
                gap = np.linalg.norm(palm - (O[f0] + tt * axis)) - r
                if gap < 0.03 and (best is None or gap < best[0]):
                    best = (gap, leg, seg)
            if best is None:
                continue
            O, G = frames[(best[1], best[2])]
            moved = np.linalg.norm(O[f1] - O[f0]) > 0.01 or np.arccos(np.clip((np.trace(G[f0].T @ G[f1]) - 1) / 2, -1, 1)) > 0.05
            if not moved:
                continue
            W = P[f"{side}Hand"]
            loc0 = G[f0].T @ (W[f0] - O[f0])
            u = np.clip((np.arange(T) - f0) / (f1 - f0), 0, 1)
            u = u * u * (3 - 2 * u)
            carried = O + np.einsum("tij,j->ti", G, loc0)
            c = (1 - u) ** LEAVE_POWER                 # the hand lifts off soon (riding the whole way, a thigh rising
            want = carried * c[:, None] + W * (1 - c[:, None])   # to the chest carried it into the belly, 0.x model)
            pw = np.zeros((T, 3))
            pw[f0 + 1:f1] = (want - W)[f0 + 1:f1]
            _resolve(motion, side, -sx, P, G_, pw, np.zeros((T, 3)))
            P, G_ = sk.upper_fk(motion, root, JIDX)

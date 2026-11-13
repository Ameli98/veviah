"""Lenient reading of perform requests (2026-11-12, the author Q48 / Q51 b).

Open-source models (and sometimes others) make small slips in the keyframe format that have only one possible
meaning. They are fixed here before solving, and every fix is reported back as a note so the model learns the right
form (`normalize`). Errors that cannot be fixed get a hint saying exactly what to write instead (`hint`).

Fixed (each seen in the 2026-11-12 test, ~/hsin_runs/llm_test):
- an arm written as {"hand": "rest"} / {"rest": true} / {"hand": "rest", ...}  -> "rest"
- "eyes": {"closed": bool} / true / false                                      -> "closed" / "open"
- palm synonyms (front, toward_viewer, inward, outward, upward, downward, ...) -> the palm words
- hands_meet "at" with an across word ("chest center close")                   -> that word dropped
- a single keyframe                                                             -> rest at 0, it, rest again
- an unknown emotion word                                                       -> neutral (server side)
"""
import copy

from . import arm_words as aw
from .keyframes import PALMS

PALM_SYN = {"front": "forward", "toward_viewer": "forward", "viewer": "forward", "outward": "out", "inward": "in",
            "upward": "up", "downward": "down", "backward": "back", "toward_body": "back", "towards_body": "back",
            "self": "back", "face": "back", "body": "back", "neutral": "natural", "default": "natural",
            "relaxed": "natural"}
ARMS = ("right_arm", "left_arm")


def _arm(v, where, notes):
    if isinstance(v, dict):
        hand = v.get("hand")
        if hand == "rest" or (v.get("rest") is True and not v.get("touch")) or (hand is None and v.get("rest") is True):
            notes.append(f'{where}: an arm at rest is written "rest", not {{"hand": "rest"}}')
            return "rest"
        p = v.get("palm")
        if isinstance(p, str) and p not in PALMS and p.lower() in PALM_SYN:
            notes.append(f'{where}: palm "{p}" -> "{PALM_SYN[p.lower()]}"')
            v = dict(v, palm=PALM_SYN[p.lower()])
        if "rest" in v and len(v) > 1:
            v = {k: x for k, x in v.items() if k != "rest"}
    return v


def normalize(req):
    """-> (request with the slips fixed, notes). Never raises; anything it does not recognise is left as is."""
    notes = []
    req = copy.deepcopy(req)
    kfs = req.get("keyframes")
    if not isinstance(kfs, list):
        return req, notes
    for i, k in enumerate(kfs):
        if not isinstance(k, dict):
            continue
        where = f"keyframe {i}"
        for a in ARMS:
            if a in k:
                k[a] = _arm(k[a], f"{where} {a}", notes)
        e = k.get("eyes")
        if isinstance(e, dict) and "closed" in e:
            k["eyes"] = "closed" if e["closed"] else "open"
            notes.append(f'{where}: eyes is written "closed" or "open"')
        elif isinstance(e, bool):
            k["eyes"] = "closed" if e else "open"
            notes.append(f'{where}: eyes is written "closed" or "open"')
        hm = k.get("hands_meet")
        if isinstance(hm, dict) and isinstance(hm.get("at"), str):
            w = hm["at"].split()
            if len(w) == 3 and w[1] in aw.ACROSS:
                hm["at"] = f"{w[0]} {w[2]}"
                notes.append(f'{where}: hands_meet "at" is "<height> <reach>", the across word "{w[1]}" was dropped')
    if len(kfs) == 1 and isinstance(kfs[0], dict):
        k = kfs[0]
        t = max(float(k.get("t", 0) or 0), 0.8)
        rest = {"right_arm": "rest", "left_arm": "rest"}
        req["keyframes"] = [dict(rest, t=0), dict(k, t=t), dict(rest, t=t + 0.8, head={"nod": 0, "turn": 0, "tilt": 0},
                                                               torso={"lean": 0, "turn": 0, "bend": 0})]
        notes.append("only one keyframe: a motion needs at least 2 (start, ..., back to rest); rest was added before and after")
    return req, notes


HINTS = [
    ('hand must be "<height> <across> <reach>"',
     'write the hand as three words, e.g. "chest center close" (height across reach), or the whole arm as "rest"'),
    ('hands_meet "at" must be', 'hands_meet "at" is two words "<height> <reach>", e.g. "chest mid"'),
    ("unknown palm", 'palm is one of forward, back, up, down, in, out, natural'),
    ('"jump" is for standing only', 'to jump, set "posture": "stand" (she stands up first) or leave the jump out while sitting'),
    ("unknown eyes", 'eyes is "closed" or "open"'),
    ("unknown hand shape", 'hands shapes: relaxed, open, fist, point, peace, thumbs_up, flat, loose'),
    ("unknown region", 'use one of the touch regions listed in the format'),
]


def hint(msg):
    """A one-line fix for a solver error message, or ''."""
    for k, h in HINTS:
        if k in msg:
            return h
    return ""

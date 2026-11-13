"""Agent loop for one instruction: write keyframes -> solve -> rule filter -> render -> the LLM looks at
the picture and either accepts it or writes a corrected version (the author, 2026-10-22: let the LLM learn to
make the motion instead of training on lots of data).

refine(llm, renderer, instruction, duration_s, keyframes, rounds=2) starts from given keyframes (e.g. a
first draft from KeyframeGenerator) and returns the list of versions it tried. Each version keeps the
keyframes, the solved motion, filter reasons and the critique that led to the next one.
"""
import json

from . import keyframes as kf
from .schema import JIDX
from .filters import check_motion
from .llm import extract_json
from .measure import measure_text
from .render import render_grid, to_png

CRITIQUE = """{format_prompt}

---
You wrote the keyframes below for this instruction. {result}

Your keyframes:
{keyframes}
{filter_note}
Check the result against the instruction, part by part: is the right body part moving, on the right side, in the right
direction, high / far enough, with the right hand shape and palm direction, the right posture, a readable size and timing?
Is anything broken (a hand inside the body, a twisted arm, a stiff or odd pose)?

If it already does what the instruction says and looks natural, return {{"ok": true, "critique": "..."}}.
Otherwise return {{"ok": false, "critique": "<the concrete problems>", "posture": "...", "keyframes": [...]}} with
corrected keyframes in the same format. Return JSON only."""


DEPENETRATE = True       # 2026-10-23: push the arms out of the body between keyframes (motion_loop.depenetrate)


CLOTH_PROVIDER = None     # callable(model, motion, root, hands, frames) -> [points [N, 3]] (vrm_render.ClothProvider)


def solve(obj, duration_s, model=None, cloth=None):
    """Keyframes -> (motion, root, hands) for one character model (skeleton.use_model; None = the active one,
    normally the default model). The same keyframes give a different motion per model: legs, sitting height and
    contacts follow its proportions, so feet stay on the floor and hands land on its own body."""
    from . import skeleton as sk
    wall = obj.get("wall") if isinstance(obj, dict) else None
    if wall and sk.WALL is None:
        with sk.use_model(model or (None if sk.CANONICAL else sk.MODEL)):
            sk.WALL = wall_plane(wall, model)
        try:
            return solve(obj, duration_s, model, cloth)
        finally:
            sk.WALL = None
    with sk.use_model(model or (None if sk.CANONICAL else sk.MODEL)):
        from .arm_words import resolve_arm_words               # hand-position words -> arm directions
        from .contact import resolve_touches                    # touch words -> arm directions for this model
        obj0 = obj
        obj = resolve_touches(resolve_arm_words(obj), duration_s) if isinstance(obj, dict) else obj
        provider = CLOTH_PROVIDER if cloth is None else cloth
        if provider and isinstance(obj, dict):
            # The legs first, then the cloth, then the hands (2026-10-29, the author): when a hand rests on a leg the
            # model's skirt covers, the motion solved so far gives the legs; the player drapes the skirt on them and
            # the touches are solved again on the cloth (contact.CLOTH).
            from . import contact
            fit = contact.load_fit(sk.MODEL)
            fr = sorted({min(int(round(float(k["t"]) * kf.FPS)), int(round(duration_s * kf.FPS)) - 1)
                         for k in obj["keyframes"] for s in ("left", "right")
                         if isinstance(k.get(f"{s}_arm"), dict)
                         and str(k[f"{s}_arm"].get("touch", "")).replace("other_", "") in contact.LEG_REGIONS})
            if fit and fit.get("skirt") and fr:
                m1, r1, h1 = solve(obj0, duration_s, model, cloth=False)
                pts = provider(sk.MODEL, m1, r1, h1, [min(f, len(m1) - 1) for f in fr])
                if pts is None:                                  # no player to drape with: the SKIRT_GAP rule
                    return solve(obj0, duration_s, model, cloth=False)
                contact.CLOTH = dict(zip(fr, pts))
                try:
                    return solve(obj0, duration_s, model, cloth=False)
                finally:
                    contact.CLOTH = None
        linear = (obj.get("interp") if isinstance(obj, dict) else None) == "linear"
        kf.LINEAR_SEGMENTS = linear
        try:
            posture, times, rows = kf.parse_keyframes(obj, duration_s)
            motion, root = kf.keyframes_to_motion(posture, times, rows, duration_s)
            from .contact import follow_leg_touches                 # hands on a moving leg ride with it
            motion = follow_leg_touches(motion, root, times, obj)
            if DEPENETRATE:
                from .depenetrate import depenetrate, lift_paths, remove_flips, smooth_steps   # arms around the body
                motion = smooth_steps(remove_flips(depenetrate(lift_paths(motion, root, times), root)),
                                      keys=[int(round(t * kf.FPS)) for t in times])
            if linear:
                from .depenetrate import smooth_upper
                motion = smooth_upper(motion)                       # linear segments have velocity steps at the keyframes
        finally:
            kf.LINEAR_SEGMENTS = False
        return motion, root, kf.keyframes_to_hands(times, rows, duration_s)


_WALLS = {}


def wall_plane(wall, model=None):
    """(x0, s) of the wall she leans on: the outermost body point on that side in the leaning rest pose, as the desktop
    page lines her up with the window's side (stage.bodyBox: upper arms, forearms, hands, hips with their thickness)."""
    from . import skeleton as sk
    key = (sk.MODEL, sk.CANONICAL, wall)
    if key not in _WALLS:
        prev, sk.WALL = sk.WALL, (100.0 * (1.0 if wall == "left" else -1.0), 1.0 if wall == "left" else -1.0)   # far away:
        try:                                                     # no push in the rest pose itself
            m, r, _ = solve({"posture": "stand", "wall": wall, "keyframes": [{"t": 0.0}, {"t": 1.0}]}, 1.0, model)
        finally:
            sk.WALL = prev
        P, _ = sk.upper_fk(m[:1].astype(float), r[:1].astype(float), JIDX)
        L = sk.motion_leg_points(m[:1].astype(float), r[:1].astype(float), JIDX)
        s = 1.0 if wall == "left" else -1.0
        xs = [s * P[f"{sd}{j}"][0, 0] + rr for sd in ("left", "right") for j, rr in sk.WALL_PTS]
        xs += [s * L[f"{sd}Hip"][0, 0] + sk.WALL_HIP_R for sd in ("left", "right")]
        _WALLS[key] = (s * max(xs), s)
    return _WALLS[key]


IMAGE = "The image shows the result: 8 frames in time order (t on each frame).\n{layout}"
NUMBERS = ("Below are numbers measured from the motion your keyframes produced. They are exact: judge the motion "
           "from them, not from what you meant to write.\n{numbers}")


def refine(llm, renderer, instruction, duration_s, keyframes, rounds=2, retries=1, feedback="image"):
    """feedback: "image" = the LLM sees the rendered grid; "numbers" = it reads measure.measure_text."""
    fmt = kf.PROMPT.format(instruction=instruction, duration=duration_s, variation=kf.VARIATIONS[0],
                           sit_list=kf.SIT_LIST)
    versions, obj = [], keyframes
    for r in range(rounds + 1):
        motion, root, hands = solve(obj, duration_s)
        reasons = check_motion(motion, root)
        png = to_png(render_grid(renderer, motion, hands=hands, root=root))
        v = {"round": r, "keyframes": obj, "motion": motion, "root": root, "hands": hands, "filter": reasons, "png": png}
        versions.append(v)
        if r == rounds:
            break
        note = (f"\nOur rule checks rejected this motion: {', '.join(reasons)}. Fix that too.\n" if reasons else "")
        result = (IMAGE.format(layout=renderer.layout_note) if feedback == "image"
                  else NUMBERS.format(numbers=measure_text(motion, root, hands)))
        prompt = CRITIQUE.format(format_prompt=fmt, result=result, keyframes=json.dumps(obj), filter_note=note)
        new = None
        for _ in range(retries + 1):
            try:
                rep = extract_json(llm.complete(prompt, images=[png] if feedback == "image" else None, max_tokens=8000))
                if rep.get("ok") and not reasons:
                    v["critique"], v["ok"] = rep.get("critique", ""), True
                    return versions
                solve(rep, duration_s)              # must parse and solve
                v["critique"], new = rep.get("critique", ""), {"posture": rep.get("posture", "stand"),
                                                              "keyframes": rep["keyframes"]}
                break
            except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
                continue
        if new is None:
            break
        obj = new
    return versions


def pick(versions):
    """The last version that passes the rule filter (else the first)."""
    ok = [v for v in versions if not v["filter"]]
    return ok[-1] if ok else versions[0]


# ---------- a stronger critic, a cheap writer (2026-10-23) ----------
# Haiku misreads its own renders, so here a stronger model (Sonnet) only looks and writes a short list of
# problems, and the cheap model (Haiku) rewrites the keyframes from that list.
CRITIC = """You check a motion made for a desktop companion character (it stands or sits on the floor, in place)
against its instruction. The motion was written as keyframes in this format:

{format_prompt}

---
Instruction: "{instruction}"
The image shows the result: 8 frames in time order (t on each frame). {layout}
The 8 frames can miss fast movements, so here are numbers measured from the motion (exact):
{numbers}
{filter_note}
Does the motion do what the instruction says, in a natural and readable way? If yes, return {{"ok": true}}.
If not, return {{"ok": false, "problems": ["...", "..."]}} with at most 3 concrete problems, each saying which body
part, what is wrong and what it should do instead (direction, height, size, timing, hand shape, posture), in terms
the keyframe format can express. Return JSON only."""

WRITER = """{format_prompt}

---
Your earlier keyframes for this instruction:
{keyframes}

A reviewer who watched the result found these problems:
{problems}

Write corrected keyframes that fix them and keep what was already right. Return JSON only:
{{"posture": "...", "keyframes": [...]}}"""


def refine_critic(critic, writer, renderer, instruction, duration_s, keyframes, rounds=2, retries=1):
    """Same versions list as refine(); each version may carry the critic's "problems"."""
    fmt = kf.PROMPT.format(instruction=instruction, duration=duration_s, variation=kf.VARIATIONS[0],
                           sit_list=kf.SIT_LIST)
    versions, obj = [], keyframes
    for r in range(rounds + 1):
        motion, root, hands = solve(obj, duration_s)
        reasons = check_motion(motion, root)
        png = to_png(render_grid(renderer, motion, hands=hands, root=root))
        v = {"round": r, "keyframes": obj, "motion": motion, "root": root, "hands": hands, "filter": reasons, "png": png}
        versions.append(v)
        if r == rounds:
            break
        note = f"\nOur rule checks rejected this motion: {', '.join(reasons)}.\n" if reasons else ""
        rep = None
        for _ in range(retries + 1):
            try:
                rep = extract_json(critic.complete(CRITIC.format(
                    format_prompt=fmt, instruction=instruction, layout=renderer.layout_note,
                    numbers=measure_text(motion, root, hands), filter_note=note), images=[png], max_tokens=4000))
                break
            except (ValueError, json.JSONDecodeError):
                continue
        if rep is None:
            break
        if rep.get("ok") and not reasons:
            v["ok"] = True
            return versions
        problems = list(rep.get("problems") or []) + ([f"rule check failed: {', '.join(reasons)}"] if reasons else [])
        v["problems"] = problems
        new = None
        for _ in range(retries + 1):
            try:
                cand = extract_json(writer.complete(WRITER.format(
                    format_prompt=fmt, keyframes=json.dumps(obj), problems="\n".join(f"- {p}" for p in problems)),
                    max_tokens=8000))
                solve(cand, duration_s)
                new = {"posture": cand.get("posture", "stand"), "keyframes": cand["keyframes"]}
                break
            except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError):
                continue
        if new is None:
            break
        obj = new
    return versions

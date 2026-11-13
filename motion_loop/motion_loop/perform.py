"""Keyframes from the chat LLM -> motion JSON for the avatar (the MCP harness, Decision 11).

The MCP server (mcp_server/server.mjs) runs this once per `perform` call:
    python -m motion_loop.perform solve  < {"posture": ..., "keyframes": [...], "model": "<name>"}
        -> {"motion": <vrm_export JSON>, "duration": s, "problems": [filter reasons], "model": name}
    python -m motion_loop.perform describe
        -> the keyframe format as text, for the tool description (the generator prompt without its task lines)
Start-up + solve takes ~0.4 s, so no long-running process is needed.
"""
import json, os
import sys

from . import keyframes as kf
from . import skeleton as sk
from .filters import check_motion
from .refine import solve
from .vrm_export import to_vrm_json


def _no_power_throttling():
    """Windows (2026-11-12, step 4): a process without a window is put on the slow 'efficiency' path (EcoQoS) after a
    few seconds; heavy solves then ran 4x slower than in WSL (39 s vs 9 s) and stayed slow. Opt out at start (also in
    the batch workers, which import this module)."""
    if os.name != "nt":
        return
    import ctypes, ctypes.wintypes as w

    class _S(ctypes.Structure):
        _fields_ = [("Version", w.ULONG), ("ControlMask", w.ULONG), ("StateMask", w.ULONG)]
    k = ctypes.windll.kernel32
    k.GetCurrentProcess.restype = w.HANDLE
    k.SetProcessInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
    s = _S(1, 1, 0)                  # PROCESS_POWER_THROTTLING_EXECUTION_SPEED: controlled, off
    k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(s), ctypes.sizeof(s))


_no_power_throttling()


ARMS = """- "right_arm", "left_arm": where the hand is, in words, relative to the torso (so it follows a leaning or turning torso):
  {"hand": "<height> <across> <reach>", "elbow": e, "palm": p, "wrist": deg}   or just "rest" (the rest pose of the posture).
  height: above_head, head, face, shoulder, chest, waist, hip, low (thigh level).
  across: other_side (in front of the other shoulder), center, front (in front of its own shoulder), out (beside the body),
          wide (arm out to the side).
  reach: back (behind the body), beside (in line with the body), close (a fist's width in front), mid, far (arm stretched toward it; above_head far = straight up).
  Examples (right arm): wave = "head out beside" + palm forward; point at the viewer = "shoulder front far" + hands point;
  hand on chest = "chest center close"; hand to the mouth = "face center close"; arm straight up = "above_head front far";
  hand behind the back = "waist center back"; arms open = "shoulder wide beside".
  Optional: "elbow": down | out | forward | back | up (which way the elbow points; the default suits the hand position).
  "palm": which way the palm faces: forward (shown to the viewer, as in a wave or "stop"), back (toward its own body or face),
  up, down, in (toward the middle of its body), out, natural (the default).
  "wrist": deg, bending the hand at the wrist; + = toward the palm. Limits: -70..70.
  "wrist_side": deg, the hand bent sideways at the wrist, + = toward the little finger. Limits: -30..30.
  To touch the body instead: {"touch": region, "with": palm | back_of_hand | fingertips | fingers (the finger pads) | fist | index | wrist | forearm | elbow,
  "elbow_on": region}; wrist = the heel of the hand (a cheek resting on the hand), forearm = the forearm lying across the
  region (an arm draped over a raised knee); with "elbow" the elbow rests on the region and an optional "hand":
  "<height> <across> <reach>" aims the forearm (default "face front close", a chin rest); "elbow_on" (optional) puts the
  elbow toward another region while the hand touches, e.g. {"touch": "cheek", "with": "wrist", "elbow_on": "knee"};
  "elbow" (optional): down | out | forward | back | up | together = which way the elbow points while the hand touches (a hand behind
  the head with the elbow raised above it = {"touch": "back_of_head", "with": "palm", "elbow": "up"}; together = elbows
  and wrists close in front of the chest, e.g. both hands cupping the face: {"touch": "cheek", "with": "fingers", "elbow": "together"});
  "fingers" (optional, a palm on the hip / waist only): forward (default, around the waist) | down (down the side) | back;
  regions: top_of_head, head_front_top (the front of the top, hands protecting the head with the elbows forward),
  forehead, temple, head_side_top (the upper side of the head, a hand in the hair above the temple), ear, cheek, chin, mouth (the lips as measured on the model; an index upright on them = shh), back_of_head,
  hair (found on the model's own hair strands): side_hair (the strand hanging beside the face in front of the ear: tucking it
  behind the ear, twirling it), bangs (the fringe over the forehead), back_hair, ponytail (only if the model has one), each
  with an optional "at": root | mid | tip = where along the strand (default mid; bangs root); a model without side hair /
  bangs / back hair uses temple / forehead / back_of_head instead, nape, neck_side, shoulder, chest, chest_side (the upper chest on the
  hand's own side), heart, belly, hip, waist (the side of the waist above the pelvis: a high akimbo, elbows out), bottom (the buttock on
  the hand's own side, hands covering the bottom), lower_back,
  upper_arm, elbow, forearm, other_hand, thigh, thigh_side (outer side, where a hanging hand rests), knee, shin, ankle, floor (beside the hip), floor_behind (leaning back on the
  hands: lean the torso back ~10-20), floor_front (in front of the knee) (sided ones mean the hand's own side; other_<region>
  = the opposite side, e.g. "other_shoulder").
- "hands_meet": {"at": "<height> <reach>", "how": palms | fingertips, "elbow": e}   both hands together, in place of
- "arms_cross": {"how": fold | hug | cover, "top": left | right, "offset": cm}   both arms crossed, in place of right_arm / left_arm: fold = arms folded below the chest (each hand on the other arm near its elbow), hug = hugging oneself (each hand on the other shoulder), cover = covering the chest shyly (forearms crossed under it, each hand flat on the other side of the chest); top = the arm in front (default left); offset = both hands this many cm lower along the other arm (alternate it to rub the arms). Use this, not touch words, for crossed arms (touches solved one arm at a time put the forearms through each other).
  right_arm / left_arm in that keyframe: palms = pressed palm to palm on the body's middle line (prayer hands, hands joined
  over the head: "above_head far" = arms straight up), fingertips = fingers pointing in and touching at the tips, elbows
  out (a roof or heart over the head: "above_head beside"). Height and reach are the arm words' ("waist back" = clasped
  behind the back); hands open. "offset": cm, the right hand that much higher than the left (rubbing hands: alternate +-3).
"""


def describe():
    """The keyframe format for the chat LLM: the generator prompt with arms as words (arm_words, contact)."""
    text = kf.PROMPT.format(instruction="", duration=0.0, variation="", sit_list=kf.SIT_LIST)
    text = text.split("\n", 2)[2]                                 # drop the role line
    for line in ('Instruction: ""', "Duration: 0.0 s", "Variation: "):
        text = text.replace(line + "\n", "")
    a, b = text.index('- "right_arm", "left_arm"'), text.index('- "shoulders"')
    text = text[:a] + ARMS + text[b:]
    a, b = text.index("Start from the rest pose"), text.index("Repeated actions")
    text = text[:a] + ('Start from the rest pose of the posture and return to it at the end (arms "rest", head, torso and '
                       "shoulders at 0, hands relaxed, wrists 0).\n") + text[b:]
    text = text.replace("times increasing from 0 to 0.0", "times increasing from 0; the last t is the length of the motion")
    text = text.replace("If the instruction does not say, stand.", "Keep the posture the character is in unless you mean to change it.")
    return text.replace("Return JSON only: ", "Arguments `posture` and `keyframes` have this shape: ")


def known_model(name):
    """A model with a measured skeleton (skeletons/<name>.json); others fall back to the default model."""
    if not name or name == sk.DEFAULT_MODEL:
        return sk.DEFAULT_MODEL
    from pathlib import Path
    return name if (Path(sk.__file__).parent / "skeletons" / f"{name}.json").exists() else sk.DEFAULT_MODEL


def solve_request(req):
    # small format slips with one meaning are fixed and reported back (lenient.py, 2026-11-12)
    from .lenient import normalize, hint
    req, notes = normalize(req)
    try:
        out = _solve_request(req)
    except ValueError as e:
        h = hint(str(e))
        raise ValueError(f"{e}" + (f" -- fix: {h}" if h else "")) from None
    out["notes"] = notes
    return out


def _solve_request(req):
    obj = {"posture": req.get("posture", "stand"), "keyframes": req["keyframes"]}
    if req.get("wall"):                                           # leaning on a window's side (desktop app)
        obj["wall"] = req["wall"]
    if req.get("interp"):                                         # "linear": reference-read motions (sets), kf.LINEAR_SEGMENTS
        obj["interp"] = req["interp"]
    times = [float(k.get("t", 0)) for k in obj["keyframes"]]
    duration = max(times + [0.5])
    model = known_model(req.get("model"))
    # a hand on a leg under a skirt: the cloth is draped first (a headless player, started only in that case, ~5 s)
    path = req.get("model_path") if model == req.get("model") else None
    if req.get("cloth") is False:                                # e.g. the companion's idle actions: no cloth-first pass
        motion, root, hands = solve(obj, duration, model=model, cloth=False)
    elif CLOTH is not None:                                        # serve: one provider kept for the whole process
        if path:
            CLOTH.paths[model] = path
        motion, root, hands = solve(obj, duration, model=model, cloth=CLOTH)
    else:
        cloth = _cloth_provider(paths={model: path} if path else None)
        try:
            motion, root, hands = solve(obj, duration, model=model, cloth=cloth)
        finally:
            cloth.close()
    problems = check_motion(motion, root, model=model, posture=obj["posture"], wall=obj.get("wall"))
    eyes = kf.eyes_track(obj, duration)
    return {"motion": to_vrm_json(motion, hands=hands, root=root, eyes=eyes), "duration": duration,
            "problems": problems, "model": model}


# serve (2026-11-03, the author: keep the player resident): the MCP server keeps one python process and sends it one JSON
# request per line; the cloth provider's headless player (needed when a skirted model's hand touches a leg, ~5 s to
# start) is started once and kept, and "warm" starts it ahead of the first such call.
CLOTH = None


def _cloth_provider(paths=None):
    """The Windows app (env HSIN_CLOTH_URL = its companion, set by server.mjs) drapes the skirt in a hidden window;
    elsewhere (WSL) a headless Playwright player does it."""
    from .vrm_render import ClothProvider, HttpClothProvider
    url = os.environ.get("HSIN_CLOTH_URL")
    return HttpClothProvider(url) if url else ClothProvider(paths=paths)


def warm(req):
    """Start the cloth provider's player for this model now if its body fit has a skirt (else nothing)."""
    model = known_model(req.get("model"))
    from .contact import load_fit
    from .schema import JOINTS
    with sk.use_model(model):
        fit = load_fit(model)
    if not (fit or {}).get("skirt"):
        return {"warm": False, "model": model}
    path = req.get("model_path") if model == req.get("model") else None
    if path:
        CLOTH.paths[model] = path
    if hasattr(CLOTH, "warm"):                                     # the app's window: just open it
        return {"warm": CLOTH.warm(model), "model": model}
    if model not in CLOTH.renderers:
        import numpy as np
        m = np.zeros((2, len(JOINTS), 4), np.float32); m[..., 3] = 1
        CLOTH(model, m, np.zeros((2, 3), np.float32), None, [0])
    return {"warm": True, "model": model}


def serve():
    global CLOTH
    CLOTH = _cloth_provider()
    try:
        for line in sys.stdin:
            if not line.strip():
                continue
            try:
                msg = json.loads(line)
                cmd = msg.get("cmd", "solve")
                out = warm(msg["req"]) if cmd == "warm" else solve_request(msg["req"])
            except Exception as e:                               # bad keyframes: tell the LLM what was wrong
                out = {"error": f"{type(e).__name__}: {e}"}
                msg = msg if isinstance(msg, dict) else {}
            sys.stdout.write(json.dumps({"id": msg.get("id"), **out}) + "\n")
            sys.stdout.flush()
    finally:
        CLOTH.close()


def _solve_safe(req):
    try:
        return solve_request(req)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def main():
    # requests and replies are UTF-8 whatever the Windows locale (2026-11-13: in a folder named in Chinese, cp950 read
    # the path's UTF-8 bytes as Big5, a 0x5C trail byte swallowed a backslash and every request was broken JSON)
    for s in (sys.stdin, sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
    cmd = sys.argv[1] if len(sys.argv) > 1 else "solve"
    if cmd == "describe":
        print(describe())
        return
    if cmd == "serve":
        serve()
        return
    if cmd == "batch":                                           # {"reqs": [...]} -> [result or {"error"}, ...]
        # solved in parallel processes (2026-11-11, the author: the companion's touch reactions took too long after a
        # restart: 16 motions one after another); results keep the order of the requests
        reqs = json.load(sys.stdin)["reqs"]
        n = min(len(reqs), max(1, (os.cpu_count() or 2) - 2), 8)
        if n > 1:
            import multiprocessing as mp
            with mp.get_context("spawn").Pool(n) as pool:
                out = pool.map(_solve_safe, reqs)
        else:
            out = [_solve_safe(r) for r in reqs]
        json.dump(out, sys.stdout)
        return
    try:
        out = solve_request(json.load(sys.stdin))
    except Exception as e:                                       # bad keyframes: tell the LLM what was wrong
        out = {"error": f"{type(e).__name__}: {e}"}
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()

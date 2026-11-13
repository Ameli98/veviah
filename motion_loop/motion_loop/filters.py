"""Step 3: cheap rule-based pre-filter. Rejects are never sent to the LMM."""
import json
from pathlib import Path

import numpy as np

from . import skeleton as sk
from .rotations import angle, angular_velocity, rotvec
from .schema import FPS, JIDX, JOINTS, MAX_ANGLE, load_motion, load_root

# rad/s, true rotation speed of a joint (PLACEHOLDER thresholds: tune on your own motions).
# Arms get a higher limit: turning the palm while bending the elbow in a 1.5 s gesture reached
# 10-13 rad/s in ordinary LLM keyframe motions (2026-10-13), and real forearms are that fast. Upper arms
# too (2026-10-21): 6 normal 'sudden' gestures at 10-13.8 rad/s were rejected, the author marked none broken.
MAX_VEL_BODY, MAX_VEL_ARM = 10.0, 15.0
MAX_VEL = np.array([MAX_VEL_ARM if n.endswith(("UpperArm", "LowerArm", "Hand")) else MAX_VEL_BODY for n in JOINTS])
MAX_ACC = 400.0    # rad/s^2
MAX_LIMIT_FRAC = 0.02
MIN_FRAMES = 10
# Legs (PLACEHOLDERS, metres and m/s on the default model, see skeleton.py):
MAX_HIPS_SPEED = 2.5          # hips moving faster than this
WALL_TOL = 0.015              # an arm point may lie this far past the wall plane (it is the window's edge line)
FLOOR_TOL = 0.03              # knee, ankle or toes this far below the floor
PLANTED_Y = None              # an ankle this low is standing on the floor ... (_foot_heights)
MAX_SLIDE_SPEED = 0.25        # ... and must not slide faster than this
MAX_SLIDE_FRAMES = 3          # (a few frames of slip are allowed, at the start or end of a step)
MAX_KNEE_BACK = 0.15          # radians a knee may bend the wrong way (forward)
TOES_DOWN_Y = None            # toes joint this low = ball of the foot on the floor (_foot_heights)
MAX_FLOAT_FRAMES = 3          # standing, both feet off the floor for longer than this, unless it is a jump:
# Jumps (2026-11-05, the author: floating is allowed only in a jump's flight). A run of frames with both feet off the floor
# is a jump if the hips fall freely in it (vertical acceleration within JUMP_G_TOL of -g on every inner frame), it is
# at most MAX_FLIGHT_S long, both feet are down again within LAND_FRAMES after it, and each foot lands within
# LAND_SLIDE of where it took off. Around a jump (JUMP_PAD_BEFORE / AFTER seconds) the leg joints may turn up to
# MAX_VEL_JUMP and accelerate up to MAX_ACC_JUMP: pushing off and landing are fast in people too.
JUMP_G_TOL = 1.5
MAX_FLIGHT_S = 0.6
LAND_FRAMES = 2
LAND_SLIDE = 0.03
JUMP_PAD_BEFORE, JUMP_PAD_AFTER = 0.5, 0.4
MAX_VEL_JUMP = 20.0
MAX_ACC_JUMP = 1500.0
LEG_JOINTS = np.array([n.endswith(("UpperLeg", "LowerLeg", "Foot")) or n == "hips" for n in JOINTS])
MAX_CLIP_FRAMES = 3           # a hand, elbow, forearm or the chin inside the torso / head (skeleton.hand_body_depth)
# Long skirts and dresses (2026-10-29, the author: standing is fine, the trouble is sitting in a long skirt or dress; filter
# out thighs spread too wide for now): on a model whose skirt hangs below the knees (body fit `skirt.hem` under the
# knee joint: test1, test2; not AvatarSample_D's short skirt or trousers), a sitting frame whose thighs are more than
# MAX_SKIRT_SPREAD degrees apart (default model: cross_legged 95, T005 / S029 91, T014 / S025 69; T001 54, side sits
# 46, other postures <= 29) for longer than MAX_CLIP_FRAMES rejects the motion ('skirt_spread').
MAX_SKIRT_SPREAD = 60.0
# a long skirt SKINNED to the legs (test1) splits open in front and bares the folded thigh already in the side sits (46 deg;
# seen on rest renders 2026-11-11, the author: fix the motion or ban it; the skirt follows the leg bones, no motion fixes it)
MAX_SKIRT_SPREAD_SKINNED = 35.0
LONG_SKIRT = False
# A long skirt skinned to the legs (not simulated, skirt.js skinnedFor: test1) is stretched into a box between two raised
# knees that are apart (2026-11-03, step 5: test1 S022 / S024, both knees 0.72 thigh lengths up and 44 deg apart, a
# hole between them). Both knees higher than SKIRT_KNEE_UP thigh lengths above the hip joints and the thighs more than
# SKIRT_KNEES_DEG apart while sitting, for longer than MAX_CLIP_FRAMES, rejects the motion ('skirt_knees'). Fine: knees
# up together (T004 0.5 deg, knees_hugged 12), one knee up (S028), legs flat and apart (S023). The same numbers as
# skirt.js KNEE_UP / SPLIT_DEG.
SKINNED_SKIRT = False
SKIRT_KNEE_UP, SKIRT_KNEES_DEG = 0.3, 25.0


@sk.on_model_change
def _long_skirt():
    global LONG_SKIRT, SKINNED_SKIRT
    f = Path(__file__).parent / "body_fits" / f"{sk.MODEL}.json"
    fit = json.load(open(f)) if f.exists() else {}
    hem, knee = (fit.get("skirt") or {}).get("hem"), fit.get("segments", {}).get("leftUpperLeg", {}).get("b")
    LONG_SKIRT = hem is not None and knee is not None and hem < knee[1]
    SKINNED_SKIRT = LONG_SKIRT and bool((fit.get("skirt") or {}).get("skinned"))


def thigh_spread(m):
    """Degrees between the two thighs per frame (hip joint -> knee in the hips' frame), m [T,J,4]."""
    from .rotations import quat_to_mat
    d = [quat_to_mat(m[:, JIDX[f"{s}UpperLeg"]].astype(float)) @ (sk.THIGH / np.linalg.norm(sk.THIGH))
         for s in ("left", "right")]
    return np.degrees(np.arccos(np.clip((d[0] * d[1]).sum(-1), -1, 1)))


@sk.on_model_change
def _foot_heights():
    global PLANTED_Y, TOES_DOWN_Y
    PLANTED_Y = sk.ANKLE_REST_Y + 0.015
    TOES_DOWN_Y = sk.ANKLE_REST_Y + sk.TOE[1] + 0.02
# NOTE: self-collision needs mesh/capsule geometry and is not implemented here.
# NOTE: no check that elbows bend the right way: the joint-position retarget cannot know the upper-arm
# twist, so its elbows bend about an arbitrary local axis and such a check would reject all of it.


def _runs(mask):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((i, j)); i = j + 1
        else:
            i += 1
    return out


def jump_runs(up, root, pts=None, off=None):
    """Runs of frames with both feet off the floor: (the ones that are real jumps, whether any other is longer than
    MAX_FLOAT_FRAMES). See JUMP_G_TOL."""
    y = root[:, 1] * sk.HIPS_Y
    jumps, bad = [], False
    for a, b in _runs(up):
        n = b - a + 1
        ok = 3 <= n <= MAX_FLIGHT_S * FPS
        if ok:
            acc = (y[a:b - 1] - 2 * y[a + 1:b] + y[a + 2:b + 1]) * FPS * FPS     # inner frames a+1 .. b-1
            ok = len(acc) > 0 and np.abs(acc + 9.81).max() < JUMP_G_TOL
        if ok and off is not None:
            e = min(b + LAND_FRAMES, len(up) - 1)
            ok = all(not off[s_][e] for s_ in ("left", "right")) and a > 0
            if ok and pts is not None:
                ok = all(np.linalg.norm((pts[f"{s_}Ankle"][e] - pts[f"{s_}Ankle"][a - 1])[[0, 2]]) < LAND_SLIDE
                         for s_ in ("left", "right"))
        if ok:
            jumps.append((a, b))
        elif n > MAX_FLOAT_FRAMES:
            bad = True
    return jumps, bad


def jump_window(m, root):
    """Frames around real jumps (bool [T]), where the leg joints get the jump speed limits."""
    T = len(m)
    win = np.zeros(T, bool)
    if root is None or T < 3:
        return win
    pts = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)
    standing = root[:, 1] * sk.HIPS_Y > -0.3
    off = {s: (pts[f"{s}Ankle"][:, 1] > PLANTED_Y + 0.02) & (pts[f"{s}Toes"][:, 1] > TOES_DOWN_Y) for s in ("left", "right")}
    for a, b in jump_runs(standing & off["left"] & off["right"], root, pts, off)[0]:
        win[max(0, a - int(JUMP_PAD_BEFORE * FPS)):min(T, b + int(JUMP_PAD_AFTER * FPS) + 1)] = True
    return win


def check_legs(m, root, posture=None):
    """Floor contact and knees. m [T,J,4], root [T,3]. On a ledge the legs hang below the floor line."""
    reasons = []
    if len(root) > 1 and np.linalg.norm(np.diff(root, axis=0), axis=-1).max() * sk.HIPS_Y * FPS > MAX_HIPS_SPEED:
        reasons.append("hips_speed")
    pts = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)
    low = min((pts[f"{s}Knee"][:, 1] - sk.R_KNEE).min() for s in ("left", "right"))
    low = min(low, *((pts[f"{s}{p}"][:, 1] - r).min() for s in ("left", "right")
                     for p, r in (("Ankle", sk.R_ANKLE), ("Toes", sk.R_TOE))))
    if low < -FLOOR_TOL and posture != sk.LEDGE:
        reasons.append("below_floor")
    for s in ("left", "right"):
        a = pts[f"{s}Ankle"]
        if len(a) < 2:
            continue
        # standing feet only: a sitting leg that changes shape slides its foot over the floor on purpose
        stand = root[:, 1] * sk.HIPS_Y > -0.3
        down = (a[:-1, 1] < PLANTED_Y) & (a[1:, 1] < PLANTED_Y) & stand[:-1] & stand[1:]
        speed = np.linalg.norm(np.diff(a[:, [0, 2]], axis=0), axis=-1) * FPS
        slide = down & (speed > MAX_SLIDE_SPEED)
        run = best = 0
        for x in slide:
            run = run + 1 if x else 0
            best = max(best, run)
        if best > MAX_SLIDE_FRAMES:
            reasons.append("foot_slide")
            break
    standing = root[:, 1] * sk.HIPS_Y > -0.3
    # a foot is on the floor if its heel (ankle) or its ball (toes, on tiptoe) is down
    off = {s: (pts[f"{s}Ankle"][:, 1] > PLANTED_Y + 0.02) & (pts[f"{s}Toes"][:, 1] > TOES_DOWN_Y)
           for s in ("left", "right")}
    up = standing & off["left"] & off["right"]
    flights, bad = jump_runs(up, root, pts, off)
    if bad:
        reasons.append("floating")
    if sk.WALL is not None:                                   # arms through the wall she leans on
        P, _ = sk.upper_fk(m.astype(float), root.astype(float), JIDX)
        d = np.max([sk.wall_depth(P[f"{sd}{j}"], r)[0] for sd in ("left", "right") for j, r in sk.WALL_ARM_PTS], axis=0)
        if (d > WALL_TOL).sum() > MAX_CLIP_FRAMES:
            reasons.append("wall_clip")
    if (sk.hand_body_depth(m.astype(float), root.astype(float), JIDX) > 0).sum() > MAX_CLIP_FRAMES:
        reasons.append("body_clip")
    if (sk.hands_overlap(m.astype(float), root.astype(float), JIDX) > 0).sum() > MAX_CLIP_FRAMES:
        reasons.append("hands_clip")
    if (sk.forearms_overlap(m.astype(float), root.astype(float), JIDX) > 0).sum() > MAX_CLIP_FRAMES:
        reasons.append("arms_clip")
    if (sk.legs_overlap(m.astype(float), root.astype(float), JIDX) > 0).sum() > MAX_CLIP_FRAMES:
        reasons.append("legs_clip")
    if LONG_SKIRT and ((thigh_spread(m) > (MAX_SKIRT_SPREAD_SKINNED if SKINNED_SKIRT else MAX_SKIRT_SPREAD)) & ~standing).sum() > MAX_CLIP_FRAMES:
        reasons.append("skirt_spread")
    if SKINNED_SKIRT:
        L = sk.motion_leg_points(m.astype(float), root.astype(float), JIDX)
        up = np.min([(L[f"{s_}Knee"][:, 1] - L[f"{s_}Hip"][:, 1]) / sk.L_THIGH for s_ in ("left", "right")], axis=0)
        if ((up > SKIRT_KNEE_UP) & (thigh_spread(m) > SKIRT_KNEES_DEG) & ~standing).sum() > MAX_CLIP_FRAMES:
            reasons.append("skirt_knees")
    knees = np.stack([rotvec(m[:, JIDX[f"{s}LowerLeg"]].astype(float))[:, 0] for s in ("left", "right")])
    if knees.min() < -MAX_KNEE_BACK:
        reasons.append("knee_backward")
    return reasons


def check_motion(m, root=None, model=None, posture=None, wall=None):
    """m [T,J,4]; root [T,3] hips offset (None = standing still). model: the VRM the motion was solved for; the body
    shapes, floor and foot heights are that model's (2026-10-28: a check run outside skeleton.use_model judged other
    models' motions with the default model's body and reported false clipping / floor hits). None = the active one."""
    if model is not None:
        with sk.use_model(model):
            return check_motion(m, root, posture=posture, wall=wall)
    if wall and sk.WALL is None:                              # leaning on a wall: its plane for wall_clip
        from .refine import wall_plane
        sk.WALL = wall_plane(wall)
        try:
            return check_motion(m, root, posture=posture)
        finally:
            sk.WALL = None
    reasons = []
    if not np.isfinite(m).all():
        return ["nan"]
    if m.ndim != 3 or m.shape[2] != 4 or np.abs(np.linalg.norm(m, axis=-1) - 1).max() > 1e-3:
        return ["not_unit_quaternions"]
    if len(m) < MIN_FRAMES:
        reasons.append("too_short")
    a = angle(m)
    # the hips' turn about the vertical is the whole body facing another way, not a joint bending: only the tilt
    # (swing) counts (2026-11-04: legs.turn up to 180; |swing w| = sqrt(w^2 + y^2) for q = swing * twist(y))
    h = m[:, JIDX["hips"]].astype(float)
    a[:, JIDX["hips"]] = 2 * np.arccos(np.clip(np.sqrt(h[:, 3] ** 2 + h[:, 1] ** 2), 0, 1))
    if (a > MAX_ANGLE[None]).any(axis=1).mean() > MAX_LIMIT_FRAC:
        reasons.append("joint_limit")
    if len(m) > 1:
        w = angular_velocity(m.astype(float), FPS)                # [T-1, J, 3]
        win = jump_window(m, root)                               # around a jump the legs may be faster
        jw = win[:-1] | win[1:]
        vlim = np.where(jw[:, None] & LEG_JOINTS[None], MAX_VEL_JUMP, MAX_VEL[None])
        if (np.linalg.norm(w, axis=-1) > vlim).any():
            reasons.append("velocity")
        if len(w) > 1:
            acc = np.linalg.norm(np.diff(w, axis=0), axis=-1) * FPS
            alim = np.where((jw[:-1] | jw[1:])[:, None] & LEG_JOINTS[None], MAX_ACC_JUMP, MAX_ACC)
            if (acc > alim).any():
                reasons.append("acceleration")
    reasons += check_legs(m, np.zeros((len(m), 3)) if root is None else root, posture)
    return reasons


def filter_candidates(cands, workdir):
    for c in cands:
        m = load_motion(workdir, c["motion_path"])
        reasons = check_motion(m, load_root(workdir, c["motion_path"], len(m)))
        c["filter"] = "reject" if reasons else "pass"
        c["filter_reasons"] = reasons
    return cands

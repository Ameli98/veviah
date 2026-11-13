"""Keep the arms out of the body on the way between keyframes (the author, 2026-10-23). The keyframes can be
right (contact words put the hand on the surface) while the slerp between two of them passes through the
chest, the head or the legs. This pass works on the finished motion:

1. per frame and arm, the points the rule filter checks (palm, fingertip, wrist, elbow, forearm) are tested
   against the same body shapes (skeleton.TORSO / HEAD / leg capsules); a point inside gets a push vector =
   outward normal * (depth + MARGIN), i.e. back onto the surface (legs too: the filter's LEG_TOL is not used here);
2. the pushes are spread in time (each frame takes the largest push within +-SPREAD frames, then a Hann
   window smooths them), so the hand goes around the body instead of jumping off it;
3. the wrist target moves by the hand's push and the elbow pole by the elbow's push; two-bone IK in the
   torso frame gives the new arm (keyframes.solve_arm), the forearm twist and the hand joint are kept.

Repeated a few times, as one push can move another point in. Frames without a push are left untouched.
"""
import numpy as np

from . import keyframes as kf
from . import skeleton as sk
from .rotations import continuous, euler_to_mat, mat_to_quat, quat_conj, quat_mul, quat_to_mat, slerp
from .schema import JIDX

MARGIN = 0.002         # pushed points end on the surface (the author, 2026-10-23: always to the surface; 2 mm
                       # so IK round-off does not leave them a hair inside)
SPREAD = 4            # frames
SMOOTH = 6            # Hann half width, frames
UA, FA = 0.2198, 0.2147    # upper arm / forearm of the active model


@sk.on_model_change
def _arm_lengths():
    global UA, FA
    r = {n: np.asarray(sk.UPPER_REST[n]) for n in ("leftUpperArm", "leftLowerArm", "leftHand")}
    UA, FA = float(np.linalg.norm(r["leftLowerArm"] - r["leftUpperArm"])), float(np.linalg.norm(r["leftHand"] - r["leftLowerArm"]))
LIFT_MIN = 0.015       # a segment more than this inside gets an arc (lift_paths)
LIFT_STEPS = np.arange(0.02, 0.26, 0.01)


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _pushes(pts, P, G, L, ra):
    """Push vectors [T,3] that bring the points pts [T,3] (limb radius ra) out of torso, head and legs."""
    rest = {n: np.asarray(p, float) for n, p in sk.UPPER_REST.items()}
    T = len(pts)
    best = np.zeros((T, 3)); depth = np.zeros(T)
    # torso: the filter's banded elliptic cylinder (skeleton.torso_depth)
    d, n = sk.torso_depth(pts, P, G)
    d = np.where(np.isfinite(d), d, -1.0)
    take = d > depth
    best[take], depth[take] = n[take], d[take]
    # head: ellipsoid in the head's rest frame
    c, r = np.array(sk.HEAD["c"]), np.array(sk.HEAD["r"])
    h = np.einsum("tji,tj->ti", G["head"], pts - P["head"]) + rest["head"]
    d = (1 - np.sqrt((((h - c) / r) ** 2).sum(-1))) * min(r)
    n = np.einsum("tij,tj->ti", G["head"], _unit((h - c) / r ** 2))
    take = d > depth
    best[take], depth[take] = n[take], d[take]
    # legs: capsules
    for s in ("left", "right"):
        for a, b, rn in (("Hip", "Knee", "thigh"), ("Knee", "Ankle", "shin")):
            A, B = L[f"{s}{a}"], L[f"{s}{b}"]
            ab = B - A
            t = np.clip(((pts - A) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0, 1)
            v = pts - (A + t[:, None] * ab)
            d = sk.leg_radius(rn, t) + ra - np.linalg.norm(v, axis=-1)    # no LEG_TOL: to the surface
            take = d > depth
            best[take], depth[take] = _unit(v)[take], d[take]
    if sk.WALL is not None:                                   # a wall she leans on (skeleton.WALL)
        d, n = sk.wall_depth(pts, ra)
        take = d > depth
        best[take], depth[take] = n, d[take]
    return best * np.where(depth > 0, depth + MARGIN, 0.0)[:, None]


def _spread(push):
    """Largest push within +-SPREAD frames, then Hann smoothing."""
    T = len(push)
    mag = np.linalg.norm(push, axis=-1)
    out = np.zeros_like(push)
    for t in range(T):
        lo, hi = max(0, t - SPREAD), min(T, t + SPREAD + 1)
        out[t] = push[lo + int(np.argmax(mag[lo:hi]))]
    w = np.hanning(2 * SMOOTH + 3)[1:-1]; w /= w.sum()
    pad = np.pad(out, ((SMOOTH, SMOOTH), (0, 0)), mode="edge")
    sm = np.stack([np.convolve(pad[:, k], w, mode="valid") for k in range(3)], -1)
    # never weaker than the frame's own need
    keep = np.linalg.norm(sm, axis=-1) < np.linalg.norm(out, axis=-1)
    sm[keep] = out[keep]
    return sm


def _arm_ik(S, Wt, pole):
    """[T,3] arrays in the torso frame: unit upper / fore directions reaching from S toward Wt."""
    d = Wt - S
    L = np.clip(np.linalg.norm(d, axis=-1), abs(UA - FA) + 1e-3, UA + FA - 1e-3)
    n = _unit(d)
    x = (UA ** 2 - FA ** 2 + L ** 2) / (2 * L)
    h = np.sqrt(np.maximum(UA ** 2 - x ** 2, 0))
    p = _unit(pole - (pole * n).sum(-1, keepdims=True) * n)
    E = S + x[:, None] * n + h[:, None] * p
    return _unit(E - S), _unit(S + n * L[:, None] - E)


MIRROR_STRAIGHT = False  # 2026-10-29: True flipped S007 (hand pushed forward out of a swayed hip) at 15 rad/s


def _resolve(m, side, sd, P, G, pw, pe, keep_upper=False):
    """In place: re-solve the arm of `side` where the wrist moves by pw [T,3] and the elbow pole by pe (world);
    frames with no offset are left as they are. Forearm twist and the hand joint are kept. keep_upper: where the
    elbow needs no push, the upper arm stays and only the forearm turns toward the moved wrist (2026-10-29, the author:
    the nearer the torso the more an angle matters, so the far parts take the corrections; pushing a hand off the hip
    with the whole arm turned the upper arm 5 deg)."""
    S, E, W = P[f"{side}UpperArm"], P[f"{side}LowerArm"], P[f"{side}Hand"]
    todo = (np.linalg.norm(pw, axis=-1) > 0) | (np.linalg.norm(pe, axis=-1) > 0)
    if not todo.any():
        return
    Gt = G["upperChest"][todo]
    to_t = lambda v: np.einsum("tji,tj->ti", Gt, v)
    s_, e_, w_ = to_t(S[todo]), to_t(E[todo]), to_t(W[todo])
    u_old, f_old = _unit(e_ - s_), _unit(w_ - e_)
    u_new, f_new = _arm_ik(s_, w_ + to_t(pw[todo]), e_ + to_t(pe[todo]) + to_t(pw[todo]) * 0.5 - s_)
    if keep_upper:
        keep = np.linalg.norm(pe[todo], axis=-1) == 0
        u_new = np.where(keep[:, None], u_old, u_new)
        f_new = np.where(keep[:, None], _unit(w_ + to_t(pw[todo]) - e_), f_new)
    Cu, Cf = sk.arm_rest_fix(side)                      # the model's rest arm bones (skeleton.arm_rest_fix)
    if not BATCH:
        return _resolve_frames(m, side, sd, G, todo, u_old, f_old, u_new, f_new, Cu, Cf)
    # all frames at once (2026-11-03; the frames are independent, same formulas as _resolve_frames)
    idx = np.nonzero(todo)[0]
    sh, up, lo, ha = (JIDX[f"{side}{b}"] for b in ("Shoulder", "UpperArm", "LowerArm", "Hand"))
    Q = m[idx]
    # the arm's current elbow axis (column y of shoulder * upper arm, in the torso frame, ideal frame): defined even
    # for a straight arm, where the plane of the two directions is not (that fell back to +y and flipped the elbow)
    hinge = (quat_to_mat(quat_mul(Q[:, sh], Q[:, up])) @ Cu)[:, :, 1]
    u, f = u_new, f_new
    if MIRROR_STRAIGHT:     # see _resolve_frames
        sgn = 1.0 if side == "right" else -1.0
        flip = ((np.cross(u, f) * sgn * hinge).sum(-1) < 0) & ((u * f).sum(-1) > np.cos(np.deg2rad(30)))
        d = _unit(UA * u + FA * f)
        e = UA * u
        e2 = 2 * d * (e * d).sum(-1, keepdims=True) - e
        u, f = np.where(flip[:, None], _unit(e2), u), np.where(flip[:, None], _unit(UA * u + FA * f - e2), f)
    R, bend, _ = kf.solve_arm_batch(u, f, side, hinge)
    low_ideal = Cu.T @ quat_to_mat(Q[:, lo]) @ Cf
    b_old = np.minimum(np.arccos(np.clip((u_old * f_old).sum(-1), -1, 1)), kf.MAX_ELBOW)
    ey = lambda a: euler_to_mat(np.stack([np.zeros_like(a), a, np.zeros_like(a)], -1))
    tw = quat_mul(mat_to_quat(ey(-sd * b_old)), mat_to_quat(low_ideal))   # = Rx(twist)
    low_new = quat_to_mat(quat_mul(mat_to_quat(ey(sd * bend)), tw))
    m[idx, up] = quat_mul(quat_conj(Q[:, sh]), mat_to_quat(R @ Cu.T))
    m[idx, lo] = mat_to_quat(Cu @ low_new @ Cf.T)
    if KEEP_HAND:
        # the hand keeps its direction in the torso frame (see _resolve_frames)
        Gl = G["upperChest"][idx] @ quat_to_mat(m[idx, sh]) @ quat_to_mat(m[idx, up]) @ quat_to_mat(m[idx, lo])
        q = mat_to_quat(np.swapaxes(Gl, -1, -2) @ G[f"{side}Hand"][idx])
        ok = 2 * np.arccos(np.minimum(np.abs(q[:, 3]), 1.0)) <= HAND_KEEP_MAX
        m[idx[ok], ha] = q[ok]


BATCH = True   # _resolve over all frames at once; False = the frame loop it replaced (kept to check against)


def _resolve_frames(m, side, sd, G, todo, u_old, f_old, u_new, f_new, Cu, Cf):
    """The per-frame loop _resolve used before 2026-11-03 (same results; kept as the reference for BATCH)."""
    for n_, t in enumerate(np.nonzero(todo)[0]):
        # the arm's current elbow axis (column y of shoulder * upper arm, in the torso frame, ideal frame): defined even
        # for a straight arm, where the plane of the two directions is not (that fell back to +y and flipped the elbow)
        hinge = (quat_to_mat(quat_mul(m[t, JIDX[f"{side}Shoulder"]], m[t, JIDX[f"{side}UpperArm"]])) @ Cu)[:, 1]
        u, f = u_new[n_], f_new[n_]
        # a nearly straight arm whose new bend plane faces the other way would turn the upper arm half a turn
        # about itself (93 rad/s, a hand resting on the thigh in a hip sway, 2026-10-28): take the mirror
        # solution, the elbow reflected across the shoulder-wrist line (the same hand position)
        if MIRROR_STRAIGHT and np.dot(np.cross(u, f) * (1.0 if side == "right" else -1.0), hinge) < 0 and u @ f > np.cos(np.deg2rad(30)):
            d = _unit(UA * u + FA * f)
            e = UA * u
            e2 = 2 * d * (e @ d) - e
            w_ = UA * u + FA * f
            u, f = _unit(e2), _unit(w_ - e2)
        R, bend, _ = kf.solve_arm(u, f, side, hinge)
        low_ideal = Cu.T @ quat_to_mat(m[t, JIDX[f"{side}LowerArm"]]) @ Cf
        b_old = min(np.arccos(np.clip(u_old[n_] @ f_old[n_], -1, 1)), kf.MAX_ELBOW)
        tw = quat_mul(mat_to_quat(euler_to_mat(np.array([0.0, -sd * b_old, 0.0]))), mat_to_quat(low_ideal))   # = Rx(twist)
        low_new = quat_to_mat(quat_mul(mat_to_quat(euler_to_mat(np.array([0.0, sd * bend, 0.0]))), tw))
        m[t, JIDX[f"{side}UpperArm"]] = quat_mul(quat_conj(m[t, JIDX[f"{side}Shoulder"]]), mat_to_quat(R @ Cu.T))
        m[t, JIDX[f"{side}LowerArm"]] = mat_to_quat(Cu @ low_new @ Cf.T)
        if KEEP_HAND:
            # the hand keeps its direction in the torso frame (2026-10-29, T021: keeping the hand joint turned the hand
            # with the moved forearm; its fingers went into the tilted head, the next round pushed again and the hand
            # ended 8-18 cm off the neck it rested on). Within the hand joint's limit, else as before.
            ch = [f"{side}Shoulder", f"{side}UpperArm", f"{side}LowerArm"]
            Gl = G["upperChest"][t].copy()
            for n_j in ch:
                Gl = Gl @ quat_to_mat(m[t, JIDX[n_j]])
            hand = Gl.T @ G[f"{side}Hand"][t]
            q = mat_to_quat(hand)
            if 2 * np.arccos(min(abs(q[3]), 1.0)) <= HAND_KEEP_MAX:
                m[t, JIDX[f"{side}Hand"]] = q



def _arm_pushes(side, sx, P, G, L):
    """Largest push per frame [T,3] for the hand points (palm, fingertip, wrist, 3/4 of the forearm) and for
    the elbow points (elbow, 1/4 and 1/2 of the forearm) of one arm."""
    E, W = P[f"{side}LowerArm"], P[f"{side}Hand"]
    hand = [(W + np.einsum("tij,j->ti", G[f"{side}Hand"], np.array([sx * dd, 0, 0])), sk.ARM_R["hand"])
            for dd in (sk.PALM, sk.FINGERTIP)] + [(W, sk.ARM_R["forearm"]), (E + 0.75 * (W - E), sk.ARM_R["forearm"])]
    elbow = [(E, sk.ARM_R["forearm"]), (E + 0.25 * (W - E), sk.ARM_R["forearm"]), (E + 0.5 * (W - E), sk.ARM_R["forearm"])]
    pick = lambda pts: _largest([_pushes(p, P, G, L, ra) for p, ra in pts])
    return pick(hand), pick(elbow)


LIFT_BATCH = True   # _try_offsets batches; False = the one-at-a-time reference


def _try_offsets(m, root, side, sx, sd, P, G, L, fr, mid, offs):
    """lift_paths trials: for each wrist offset [T, 3] (zero outside the segment fr), the arm re-solved with it and how
    far it is still inside the body over the frames mid. Only the segment's frames are computed, and all offsets in
    one batch (2026-11-03: one by one over the whole motion this loop was most of a sitting solve). Every frame is
    independent in _resolve / upper_fk / _arm_pushes, so the results are those of trying them one at a time.
    Returns ([left inside], [trial motion, the full motion with the segment replaced])."""
    if not LIFT_BATCH:                       # one at a time over the whole motion, as before (the reference)
        lefts, trials = [], []
        for off in offs:
            t = m.copy()
            _resolve(t, side, sd, P, G, off, 0.5 * off)
            P2, G2 = sk.upper_fk(t, root, JIDX)
            q, r = _arm_pushes(side, sx, P2, G2, L)
            lefts.append(max(np.linalg.norm(q[mid], axis=-1).max(), np.linalg.norm(r[mid], axis=-1).max())); trials.append(t)
        return lefts, trials
    K, sel = len(offs), np.flatnonzero(fr)
    n = len(sel)
    tile = lambda d: {k: np.concatenate([v[sel]] * K) for k, v in d.items()}
    mm = np.concatenate([m[sel]] * K)
    rr = np.concatenate([root[sel]] * K)
    off = np.concatenate([o[sel] for o in offs])
    _resolve(mm, side, sd, tile(P), tile(G), off, 0.5 * off)
    P2, G2 = sk.upper_fk(mm, rr, JIDX)
    q, r = _arm_pushes(side, sx, P2, G2, tile(L))
    inside = np.maximum(np.linalg.norm(q, axis=-1), np.linalg.norm(r, axis=-1)).reshape(K, n)[:, mid[sel]]
    lefts, trials = [], []
    for k in range(K):
        t = m.copy()
        t[sel] = mm[k * n:(k + 1) * n]
        lefts.append(float(inside[k].max())); trials.append(t)
    return lefts, trials


def lift_paths(motion, root, times):
    """Arc the hand around the body between two keyframes (the author, 2026-10-23: add a waypoint where the way
    goes through the body; side_left hand from the thigh to the floor went 7 cm through the thigh, and per-frame
    pushes that large jumped between points and spun the elbow). For each arm and keyframe segment whose path
    is inside by more than LIFT_MIN, the wrist gets an offset dir * a * sin(pi * w), w = the segment's eased
    progress: zero (and zero speed) at both keyframes, so the keyframe poses and their timing stay, and the arm
    does not stop half-way as an extra keyframe would make it. dir = mean push over the segment, up, or both; a = the
    smallest of LIFT_STEPS that clears the middle of the segment (bump > 0.3); if none does, no arc (an arc that only
    half helps made the kneel thighs case worse). depenetrate() then cleans up the rest."""
    m = motion.astype(float).copy()
    T = len(m)
    k, w = kf._segments(np.asarray(times, float), np.arange(T) / kf.FPS)
    bump = np.sin(np.pi * w)
    for side, sx, sd in (("right", -1.0, 1.0), ("left", 1.0, -1.0)):
        P, G = sk.upper_fk(m, root, JIDX)
        L = sk.motion_leg_points(m, root, JIDX)
        pw, pe = _arm_pushes(side, sx, P, G, L)
        need = np.maximum(np.linalg.norm(pw, axis=-1), np.linalg.norm(pe, axis=-1))
        for seg in np.unique(k):
            fr = k == seg
            mid = fr & (bump > 0.3)            # the arc cannot (and need not) move the keyframes themselves,
            if not mid.any() or need[mid].max() <= LIFT_MIN:     # where a contact may rest on the surface
                continue
            ends = need[np.flatnonzero(fr)[[0, -1]]].max()
            if need[mid].max() <= ends + LIFT_MIN:
                # no deeper in the middle than at the keyframes: a hand resting on a surface (or held there), not a
                # way through the body (2026-10-29, T021: a hand held on the thigh lifted and dropped back at 14.6 rad/s)
                continue
            # directions to try: the mean push (cancels out when the hand crosses a limb), up, the two mixed, and
            # out to the side (2026-10-28: a sitting hand going from the thigh to the floor behind passes through the
            # hip; a person swings it out). Profiles: the plain arc, and one that rises earlier, for a hand
            # that starts on a surface it would go into at once.
            mean, up = _unit((pw[mid] + pe[mid]).sum(0)), np.array([0.0, 1.0, 0.0])
            out = _unit(G["upperChest"][fr].mean(0) @ np.array([sx, 0.0, 0.0]))
            early = bump * (1 - w)                  # peaks at ~1/3 of the segment, finite slope at its start
            early = early / max(early[fr].max(), 1e-9)
            late = bump * w                         # peaks at ~2/3: a hand that clips just before it arrives (2026-10-29,
            late = late / max(late[fr].max(), 1e-9)  # T021: fingertips into the tilted head on the last stretch to the neck)
            back = _unit(G["upperChest"][fr].mean(0) @ np.array([0.0, 0.0, -1.0]))
            found, best = None, (need[mid].max(), None)
            # first: the wrist on a straight line (torso frame, eased) between the two keyframe wrists, as a person
            # moves a hand from one place to another (2026-10-29, T020: slerping the arm joints from the hanging rest
            # arm to a hand on the hip swept the forearm 7 cm through the waist), alone and with the arcs below
            idx = np.flatnonzero(fr)
            i0, i1 = idx[0], min(idx[-1] + 1, T - 1)
            Gm = G["upperChest"]
            wl = lambda i: np.einsum("ji,j->i", Gm[i], P[f"{side}Hand"][i] - P[f"{side}UpperArm"][i])
            ww = w[fr][:, None]
            lin_t = wl(i0) * (1 - ww) + wl(i1) * ww                      # torso frame, relative to the shoulder
            lin = np.zeros_like(P[f"{side}Hand"])
            lin[fr] = np.einsum("tij,tj->ti", Gm[fr], lin_t) + P[f"{side}UpperArm"][fr] - P[f"{side}Hand"][fr]
            try_ = lambda offs: _try_offsets(m, root, side, sx, sd, P, G, L, fr, mid, offs)
            (left_in,), (trial,) = try_([lin])
            if left_in <= 0:
                found = trial
            elif left_in < best[0]:
                best = (left_in, trial)
            for a in ([] if found is not None else LIFT_STEPS):
                # out + up + back: a sitting hand going from the thigh to the floor behind (test1, 2026-10-28)
                offs = [np.where(fr[:, None], d * a * prof[:, None] + base, 0.0)
                        for base in (0.0, lin) for prof in (bump, early, late)
                        for d in (mean, up, _unit(mean + up), out, _unit(out + up), _unit(out + up + back))]
                lefts, trials = try_(offs)            # all of this size at once; taken in the same order as before
                for left_in, trial in zip(lefts, trials):
                    if left_in <= 0:
                        found = trial
                        break
                    if left_in < best[0]:
                        best = (left_in, trial)
                if found is not None:
                    break
            if found is None and best[1] is not None and best[0] <= LIFT_MIN:
                found = best[1]                # nearly clear: depenetrate() pushes the rest out
            if found is None:
                continue                       # no arc clears it: leave it to depenetrate()
            m = found
            P, G = sk.upper_fk(m, root, JIDX)
            pw, pe = _arm_pushes(side, sx, P, G, L)
            need = np.maximum(np.linalg.norm(pw, axis=-1), np.linalg.norm(pe, axis=-1))
    return continuous(m).astype(np.float32)


KEEP_HAND = True
HAND_KEEP_MAX = 1.5        # rad: the hand joint's own rotation may not exceed this (schema.MAX_ANGLE Hand 1.6)
KEEP_UPPER_ROUNDS = 5      # depenetrate: the first rounds move only the forearm where they can (see _resolve)


def depenetrate(motion, root, rounds=8):     # stops as soon as nothing is inside (2026-10-29: 3 rounds left
    # T021's forearm 0.08 mm into the waist, enough for body_clip)
    """motion [T,J,4] (x,y,z,w), root [T,3] -> a copy with the arms pushed out of the body where needed."""
    m = motion.astype(float).copy()
    for k in range(rounds):
        P, G = sk.upper_fk(m, root, JIDX)
        L = sk.motion_leg_points(m, root, JIDX)
        hp = _hands_pushes(m, root, G)
        changed = False
        for side, sx, sd in (("right", -1.0, 1.0), ("left", 1.0, -1.0)):
            pw, pe = _arm_pushes(side, sx, P, G, L)
            pw = _largest([pw, hp[side]])
            if not (np.abs(pw).max() > 0 or np.abs(pe).max() > 0):
                continue
            pw, pe = _spread(pw), _spread(pe)
            _resolve(m, side, sd, P, G, pw, pe, keep_upper=k < KEEP_UPPER_ROUNDS)
            changed = True
        if not changed:
            break
    return continuous(m).astype(np.float32)


CROSS_MIN = 0.02          # m the wrists must be past each other to count as crossed
CROSS_NEAR = 15          # frames: hands that cross this close in time pass in front of / behind each other


TIPS_CROSS_WRISTS = 0.06   # m: fingertips past each other count as crossed hands only with the wrists this far apart


def _hands_pushes(m, root, G):
    """Per side [T,3]: where the two hands' points come closer than skeleton.HANDS_MIN (the filter's hands_clip), each
    hand moves half the overlap (+ MARGIN) outward along the torso's x (2026-10-29, S065 on test1 / test2: both hands on
    their way to the floor behind the back met fingertip to fingertip)."""
    o = sk.hands_overlap(m, root, JIDX)
    amt = np.where(o > 0, 0.5 * o + MARGIN, 0.0)[:, None]
    x, z = G["upperChest"][:, :, 0], G["upperChest"][:, :, 2]
    # crossed hands (each on the other's side: arms crossing over the chest, U041 / U053) pass in front of / behind
    # each other instead: pushed outward along x they were pushed into each other. The hand further forward goes
    # forward. Whether crossed is measured at the wrists, in the torso frame.
    P, _ = sk.upper_fk(m, root, JIDX)
    dx = np.einsum("ti,ti->t", P["leftHand"] - P["rightHand"], x)
    dz = np.einsum("ti,ti->t", P["leftHand"] - P["rightHand"], z)
    # crossed now or within CROSS_NEAR frames (on the way to crossing, the fingers meet before the wrists pass, U030)
    cr = dx < -CROSS_MIN       # wrists really past each other (U033: both at the midline, 2 mm apart, counted as crossed)
    # or the FINGERTIPS past each other while the wrists are not (2026-11-13, S030: from the knees_hugged rest, fingers on
    # the other shin, the hands spread to the ankles with the fingers interlaced; pushed apart sideways they stayed so)
    tip = {s: P[f"{s}Hand"] + np.einsum("tij,j->ti", G[f"{s}Hand"], np.array([sx * sk.FINGERTIP, 0.0, 0.0]))
           for s, sx in (("left", 1.0), ("right", -1.0))}
    # only with the wrists well apart (stacked fists under the chin, U004 on AvatarSample_D, have 'crossed' straight-finger
    # tips and were pushed apart into the body)
    cr = cr | ((np.einsum("ti,ti->t", tip["left"] - tip["right"], x) < -CROSS_MIN) & (dx > TIPS_CROSS_WRISTS))
    crossed = np.array([cr[t:t + CROSS_NEAR + 1].any() for t in range(len(cr))])[:, None]   # now or soon (U033: crossed only on the way in, then held side by side)
    zs = np.where(dz >= 0, 1.0, -1.0)[:, None]
    # crossed: front / back, each by the whole overlap (2026-11-13, S030 on test1 / test2: half each per round left
    # under 1 cm of interlaced fingers for 4 frames)
    full = np.where(o > 0, o + MARGIN, 0.0)[:, None]
    out = {"left": np.where(crossed, full * zs * z, amt * x), "right": np.where(crossed, -full * zs * z, -amt * x)}
    # forearms through each other (the arms_clip rule) on the way into / out of crossed arms: the front arm's wrist
    # (contact.FRONT_ARM) moves forward by twice the overlap, its forearm's middle by about the overlap (2026-11-03,
    # h0036: 0.6-0.8 cm for a few frames)
    from .contact import FRONT_ARM
    fo = sk.forearms_overlap(m, root, JIDX)
    famt = np.where(fo > 0, 2.0 * fo + MARGIN, 0.0)[:, None]
    # forearms side by side (not crossed: elbows together in front, U033) move apart sideways, half each; pushed forward
    # only the front arm moved and that hand left its spot (8.5 cm off the neck on test2)
    side = np.where(crossed, 0.0, 0.5)
    out["left"] = _largest([out["left"], np.where(crossed, famt * z * (FRONT_ARM == "left"), famt * side * x)])
    out["right"] = _largest([out["right"], np.where(crossed, famt * z * (FRONT_ARM == "right"), -famt * side * x)])
    return out


def _largest(pushes):
    """Per frame, the push with the largest magnitude among several [T,3]."""
    st = np.stack(pushes)                                    # [N,T,3]
    k = np.argmax(np.linalg.norm(st, axis=-1), axis=0)
    return st[k, np.arange(st.shape[1])]


FLIP_SPEED = 30.0        # rad/s: an arm joint turning this fast for one frame is an IK flip, not a motion
FLIP_MAX = 12            # frames a flip may last before it comes back


def remove_flips(motion):
    """Undo arm flips (2026-10-28): near a straight arm the two-bone IK may turn the upper arm half a turn about its own
    axis for a few frames and back (seen at 91 / 94 rad/s with a hand reaching for an ankle). Where an arm joint jumps
    faster than FLIP_SPEED and jumps back within FLIP_MAX frames, the frames in between are slerped from the frames
    around them. Other motion is left alone."""
    m = continuous(motion.astype(float))
    for side in ("left", "right"):
        for j in (JIDX[f"{side}UpperArm"], JIDX[f"{side}LowerArm"], JIDX[f"{side}Hand"]):
            for _ in range(4):
                q = m[:, j]
                d = quat_mul(quat_conj(q[:-1]), q[1:])
                speed = 2 * np.arccos(np.clip(np.abs(d[:, 3]), 0, 1)) * kf.FPS
                jumps = np.nonzero(speed > FLIP_SPEED)[0]
                fixed = False
                for a in jumps:
                    later = jumps[(jumps > a) & (jumps <= a + FLIP_MAX)]
                    if not len(later):
                        continue
                    b = later[-1] + 1                    # frames a+1 .. b-1 are the flipped ones
                    u = (np.arange(a + 1, b) - a) / (b - a)
                    m[a + 1:b, j] = slerp(np.repeat(q[a][None], len(u), 0), np.repeat(q[b][None], len(u), 0), u[:, None])
                    fixed = True
                    break
                if not fixed:
                    break
    return continuous(m).astype(np.float32)


STEP_SPREAD = 4          # frames on each side over which a one-frame jump is spread


SPIKE_MIN, SPIKE_RATIO = 4.0, 3.0   # rad/s; a frame this many times faster than its neighbours


def smooth_steps(motion, keys=None):
    """Spread one-frame jumps (and sharp changes of speed) of the arm joints (2026-10-28): where the push-out starts or ends, the re-solved arm may
    differ from the frame before by a step (S065: the forearm's twist changed 45 deg in one frame, 25 rad/s). Any arm
    joint frame faster than the filter's limit is replaced, with STEP_SPREAD frames on each side, by a slerp across
    the window, so the step takes ~9 frames. Runs after remove_flips; other frames are left alone.
    keys: keyframe frame indices; a window stops at a keyframe so the key poses stay as solved (2026-11-08, head_pat:
    a window across the 0.4 s key pose pulled the default model's elbow in 8 cm, test2's not)."""
    from .filters import MAX_ACC, MAX_VEL
    from .rotations import angular_velocity
    m = continuous(motion.astype(float))
    T = len(m)
    for side in ("left", "right"):
        for j in (JIDX[f"{side}UpperArm"], JIDX[f"{side}LowerArm"], JIDX[f"{side}Hand"]):
            for _ in range(6 if keys else 3):
                q = m[:, j]
                d = quat_mul(quat_conj(q[:-1]), q[1:])
                speed = 2 * np.arccos(np.clip(np.abs(d[:, 3]), 0, 1)) * kf.FPS
                fast = np.nonzero(speed > MAX_VEL[j])[0]
                if not len(fast):           # also a sharp change of speed (AvatarSample_D S005: 452 > MAX_ACC)
                    w = angular_velocity(q[:, None], kf.FPS)[:, 0]
                    acc = np.linalg.norm(np.diff(w, axis=0), axis=-1) * kf.FPS
                    fast = np.nonzero(acc > MAX_ACC)[0] + 1
                if not len(fast):           # a one-frame spike under the limits (2026-10-29, S003: 13 rad/s between
                    # frames at 1-3 rad/s; within the limit, but it shows as a twitch)
                    nb = np.array([np.median(np.r_[speed[max(f - STEP_SPREAD, 0):f], speed[f + 1:f + 1 + STEP_SPREAD]])
                                   if T > 3 else 0.0 for f in range(len(speed))])
                    fast = np.nonzero((speed > SPIKE_MIN) & (speed > SPIKE_RATIO * nb))[0]
                if not len(fast):
                    break
                a = int(fast[0])
                b = int(fast[fast <= a + 2 * STEP_SPREAD][-1]) + 1      # jumps close together: one window
                inside = [k for k in keys or () if a < k < b]
                if inside:                  # fast frames on both sides of a keyframe: one window each side
                    b = min(inside)
                lo, hi = max(a - STEP_SPREAD, 0), min(b + STEP_SPREAD, T - 1)
                for k in keys or ():
                    if lo < k <= a:
                        lo = k
                    elif b <= k < hi:
                        hi = k
                u = (np.arange(lo, hi + 1) - lo) / max(hi - lo, 1)
                u = u * u * (3 - 2 * u)
                m[lo:hi + 1, j] = slerp(np.repeat(q[lo][None], len(u), 0), np.repeat(q[hi][None], len(u), 0), u[:, None])
    return continuous(m).astype(np.float32)


SMOOTH_UPPER_W = np.array([1.0, 2.0, 3.0, 2.0, 1.0])     # +-2 frames


def smooth_upper(motion):
    """Weighted moving average of the upper-body joints (spine..head, shoulders, arms, hands) over +-2 frames, for
    motions built with linear segments (obj["interp"] = "linear", 2026-11-08): rounds the velocity steps at the
    keyframes. Quaternions are made continuous, averaged and renormalised (small angles between neighbours). Legs
    and hips are left alone (planted feet)."""
    m = continuous(motion.astype(float)); out = m.copy()
    w = SMOOTH_UPPER_W / SMOOTH_UPPER_W.sum(); h = len(w) // 2
    T = len(m)
    for name, j in JIDX.items():
        if any(k in name for k in ("Leg", "Foot", "hips", "Toes")): continue
        acc = np.zeros_like(m[:, j])
        for o, wo in zip(range(-h, h + 1), w):
            idx = np.clip(np.arange(T) + o, 0, T - 1)
            acc += wo * m[idx, j]
        out[:, j] = acc / np.linalg.norm(acc, axis=-1, keepdims=True)
    return out.astype(np.float32)

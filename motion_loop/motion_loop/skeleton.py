"""Lower body: canonical skeleton, leg IK, floor sitting poses, and forward kinematics of the legs.

The upper body is written as joint rotations (see keyframes.py). The legs are not: a standing leg
has to keep its foot planted while the hips move, so legs are solved by IK from where the hips and
the feet should be. Sitting on the floor uses fixed poses (SIT_POSES) instead.

Lengths come from the default model (VRM1_Constraint_Twist_Sample, measured with
vrm_player/tools/skeleton.mjs). Other models have other proportions; the player scales the hips
offset by its own hips height, so feet stay close to, but not exactly on, the floor there.
World frame = the VRM normalized frame: y up, the character faces +z, its left = +x, floor at y = 0.
"""
import numpy as np

from .rotations import mat_to_quat, quat_to_mat

HIPS_REST = np.array([0.0, 0.9081, 0.0036])          # hips joint, standing T-pose
HIPS_Y = HIPS_REST[1]                                 # unit of the root offset (schema.root)
HIP_OFFSET = np.array([0.0772, -0.0398, -0.0037])    # hips -> left upper leg (right: x negated)
THIGH = np.array([0.0, -0.3529, -0.0074])            # upper leg -> lower leg (knee)
SHIN = np.array([0.0, -0.4147, -0.0249])             # lower leg -> foot (ankle)
TOE = np.array([0.0, -0.0631, 0.1107])               # foot -> toes
L_THIGH, L_SHIN = np.linalg.norm(THIGH), np.linalg.norm(SHIN)
ANKLE_REST_Y = 0.1007                                 # ankle height when the foot is flat on the floor
# Rough body radii for floor contact (joint centres are inside the body).
R_KNEE, R_ANKLE, R_TOE, BUTT = 0.05, 0.045, 0.02, np.array([0.0, -0.10, -0.06])

_X, _Y, _Z = np.eye(3)


def _unit(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-9)


def side_sign(side):
    return 1.0 if side == "left" else -1.0


def thigh_frame(thigh_dir, shin_dir, fallback_hinge=_X):
    """Rotation of the upper leg (in the hips frame) that points it along thigh_dir with the knee hinge
    perpendicular to both directions, and the knee flexion angle. In the T-pose the leg points along -y
    and the knee hinge is +x: Rx(+a) on the lower leg swings the shin backward (a real knee)."""
    t, s = _unit(thigh_dir), _unit(shin_dir)
    h = np.cross(t, s)
    if np.linalg.norm(h) < 1e-3:                # straight leg: the bend plane is undefined
        h = fallback_hinge - t * np.dot(fallback_hinge, t)
    h = _unit(h)
    y = -t
    R = np.stack([h, y, np.cross(h, y)], axis=1)
    knee = float(np.arccos(np.clip(np.dot(t, s), -1, 1)))
    return R, knee


def ry(a):
    """Rotation about the vertical axis (+ = toward the character's left); a may be an array -> [..., 3, 3]."""
    a = np.asarray(a, float)
    c, s_ = np.cos(a), np.sin(a)
    z, o = np.zeros_like(a), np.ones_like(a)
    return np.stack([np.stack([c, z, s_], -1), np.stack([z, o, z], -1), np.stack([-s_, z, c], -1)], -2)


def rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _swing(a, b):
    """Smallest rotation taking unit vector a to unit vector b."""
    a, b = _unit(a), _unit(b)
    v, c = np.cross(a, b), np.dot(a, b)
    if c < -0.9999:
        return rx(np.pi)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K / (1 + c)


def foot_local(R_shin_world, toe_dir_world):
    """Local foot rotation that points the toes along toe_dir_world (swing only, from the rest direction)."""
    want = R_shin_world.T @ _unit(toe_dir_world)
    return _swing(TOE, want)


FLAT = _unit(TOE)          # toe direction of a foot flat on the floor (as in the T-pose)


def leg_ik(hip_joint, ankle, side, toe_dir=FLAT, pole=_Z, iters=4):
    """Two-bone IK for one leg in world axes. Returns rotation matrices (upper leg in world axes, i.e. local
    only while the hips are not rotated, then lower leg and foot local) and the ankle actually reached
    (the target pulled in if it is out of reach). The analytic solve treats the bones as straight down
    in the T-pose; they are tilted by a few degrees, so the aim point is corrected a few times."""
    target = np.asarray(ankle, float)
    aim = target
    for _ in range(iters):
        R_up, R_low, R_foot, _ = _leg_ik_once(hip_joint, aim, side, toe_dir, pole)
        got = hip_joint + R_up @ (THIGH + R_low @ SHIN)
        if np.linalg.norm(target - got) < 1e-4:
            break
        aim = aim + (target - got)
    return R_up, R_low, R_foot, got


def _leg_ik_once(hip_joint, ankle, side, toe_dir, pole):
    d = ankle - hip_joint
    L = np.linalg.norm(d)
    L_max = (L_THIGH + L_SHIN) * 0.9995
    if L > L_max:
        d, L = d * L_max / L, L_max
    dn = d / max(L, 1e-9)
    # angle at the hip between the hip->ankle line and the thigh (law of cosines)
    a = np.arccos(np.clip((L_THIGH ** 2 + L ** 2 - L_SHIN ** 2) / (2 * L_THIGH * L), -1, 1))
    p = pole - dn * np.dot(pole, dn)            # knee goes toward the pole (forward)
    p = _unit(p) if np.linalg.norm(p) > 1e-6 else _unit(np.cross(dn, _X))
    thigh = np.cos(a) * dn + np.sin(a) * p
    knee_pos = hip_joint + thigh * L_THIGH
    shin = _unit(hip_joint + d - knee_pos)
    R_up, knee = thigh_frame(thigh, shin)
    R_low = rx(knee)
    R_foot = foot_local(R_up @ R_low, toe_dir)
    return R_up, R_low, R_foot, hip_joint + d


def hip_joint(hips_pos, side, R_hips=np.eye(3)):
    off = HIP_OFFSET * np.array([side_sign(side), 1, 1])
    return hips_pos + R_hips @ off


# ---------------- sitting on the floor ----------------
# Per pose, for the LEFT leg: the thigh direction, the shin direction and the toe direction, in the hips frame (x = its left, y = up, z = forward). The right
# leg is the mirror image unless given. Directions only, like the arm keyframes: the solver turns them
# into rotations and the hips are lowered until the body touches the floor.
SIT_POSES = {
    # the right shin lies on top of and in front of the left one (they must not pass through each other)
    "cross_legged": dict(left=((0.75, -0.12, 0.65), (-0.97, -0.05, 0.15), (-0.3, -0.2, 0.9)),
                         right=((-0.75, -0.12, 0.65), (0.88, 0.1, 0.46), (0.3, -0.2, 0.9))),
    # W-sit / wariza: knees together in front with the thighs flat on the floor, shins out and back, feet
    # beside the hips with the tops of the feet on the floor; the bottom sits on the floor between the feet
    # (knee ~128 deg: at ~150 the 0.x model's knees tore, the author 2026-10-16)
    "w_sit":        dict(left=((-0.05, -0.3, 0.95), (0.8, 0.02, -0.6), (0.75, -0.05, -0.65))),
    # sitting back on the heels. The knee is bent ~135 deg, not the anatomical ~170: deeper folds tear the
    # skinned knee on the 0.x sample model (the author saw it at 156 and 145 deg and chose 135, 2026-10-16)
    "seiza":        dict(left=((0.06, -0.68, 0.73), (0.0, 0.03, -1.0), (0.0, 0.05, -1.0))),
    # both legs to the character's right: the left shin lies across in front of the right knee, the right
    # shin folds back beside the right hip (so the legs do not pass through each other); side_left is
    # the mirror image
    "side_right":   dict(left=((0.15, -0.15, 0.97), (-0.95, 0.0, 0.3), (-0.95, -0.1, 0.3)),
                         right=((-0.6, -0.15, 0.78), (-0.3, 0.03, -0.95), (-0.3, 0.0, -0.95))),
    # knees up, feet flat in front: this model's shins are longer than its thighs, so with the bottom on
    # the floor the shins slope forward
    "knees_hugged": dict(left=((0.1, 0.9, 0.42), (0.0, -0.72, 0.69), FLAT)),
    "legs_out":     dict(left=((0.08, -0.05, 1.0), (0.08, -0.05, 1.0), (0.0, 0.85, 0.5))),
    "kneel":        dict(left=((0.05, -1.0, 0.05), (0.0, -0.05, -1.0), (0.0, 0.0, -1.0))),
    # sitting on a ledge (a window's top edge in the desktop app, 2026-11-11): the bottom on the edge, thighs forward
    # and a little down, shins hanging below it, toes relaxed down; the floor is the edge, the legs go below it
    "ledge":        dict(left=((0.04, -0.1, 0.99), (0.06, -1.0, 0.06), (0.0, -0.6, 0.8))),   # knees together, feet a little apart
}
LEDGE = "ledge"
# A wall the body leans on (desktop app, a window's side; 2026-11-11): (x0, s) in world metres, s = +1 the wall is on
# the +x side (her left), -1 on her right; arm points may not pass it (depenetrate pushes them back, the filter's
# wall_clip). Set by refine.solve while solving a request with "wall"; None otherwise.
WALL = None
# the body points the desktop page lines up with a window's side (stage.js bodyBox): joint, radius
WALL_PTS = (("UpperArm", 0.05),)                  # the shoulders (the outer edge of the upper arm's root)
WALL_ARM_PTS = (("UpperArm", 0.05), ("LowerArm", 0.045), ("Hand", 0.04))   # what may not pass the wall
WALL_HIP_R = 0.1


def wall_depth(pts, r):
    """How far points [T,3] with radius r are past the wall (> 0 = through it), and the push direction."""
    x0, s = WALL
    return s * (pts[:, 0] - x0) + r, np.array([-s, 0.0, 0.0])


def _mirror(v):
    return np.array([-v[0], v[1], v[2]], float)


SIT_POSES["side_left"] = dict(left=tuple(_mirror(v) for v in SIT_POSES["side_right"]["right"]),
                              right=tuple(_mirror(v) for v in SIT_POSES["side_right"]["left"]))
SIT_NAMES = tuple(k for k in SIT_POSES if not k.startswith("_"))


# Sitting leg words (2026-10-28, pose references group C): each leg of a posture that sits on its bottom may take
# another shape, and the knees may open, e.g. one knee up and the other leg out. Shapes are the legs of the existing
# postures (directions for the LEFT leg; the right one is the mirror image, except 'cross', whose right shin lies on
# top as in cross_legged). 'default' = the posture's own leg.
SIT_LEGS = ("default", "out", "knee_up", "cross", "tucked", "knee_to_chest")
# knee_to_chest = knees_hugged's leg: thigh steep, foot right in front of the bottom (2026-10-28, S019 / S025)
_SIT_LEG_FROM = {"out": "legs_out", "knee_up": "_knee_up", "cross": "cross_legged", "tucked": "w_sit",
                 "knee_to_chest": "knees_hugged"}
# knee_up is looser than knees_hugged's legs (2026-10-28, the author: the raised knee looked squeezed against the chest,
# the foot right at the bottom): thigh ~50 deg up, foot ~0.5 m in front of the hips, sole flat on the floor
SIT_POSES["_knee_up"] = dict(left=((0.1, 0.762, 0.64), (0.0, -0.669, 0.743), FLAT))
SIT_KNEES = ("default", "together", "normal", "apart", "wide")
LONE_CROSS = tuple(_unit(np.array(v)) for v in ((0.96, -0.12, 0.25), (-0.92, -0.05, 0.4), (-0.3, -0.2, 0.9)))
_KNEES_DEG = {"default": 0.0, "together": -6.0, "normal": 0.0, "apart": 18.0, "wide": 35.0}
BUTT_POSTURES = ("cross_legged", "w_sit", "side_right", "side_left", "knees_hugged", "legs_out")   # not seiza / kneel


def _shape_of(posture, shape):
    """The leg shape a word means in this posture ('default' = the posture's own leg)."""
    if not isinstance(shape, str) and shape is not None:
        return None
    if shape not in (None, "default"):
        return shape
    return {v: k for k, v in _SIT_LEG_FROM.items()}.get(posture)


def sit_leg_dirs(posture, side, shape="default", other="default"):
    """(thigh, shin, toe) directions in the hips frame for one leg; `other` = the other leg's word. shape may also be
    (thigh, shin) directions (read from a reference, 2026-10-29): the sole lies flat when the shin stands up, else
    the toes point up as in legs_out."""
    if not isinstance(shape, str) and shape is not None:
        t, sh = _unit(np.asarray(shape[0], float)), _unit(np.asarray(shape[1], float))
        if sh[1] < -0.5:
            toe = FLAT
        else:                    # toes up as in legs_out, turned with the leg's heading (straight ahead, the foot of a
            # leg turned out broke the foot's joint limit, T001)
            up = np.array([0.0, 1.0, 0.0]) - sh * sh[1]
            toe = _unit(0.85 * _unit(up) + 0.5 * sh)
        return t, sh, np.asarray(toe, float)
    spec = SIT_POSES[posture if shape in (None, "default") else _SIT_LEG_FROM[shape]]
    dirs = spec.get(side) or tuple(_mirror(v) for v in spec["left"])
    # cross_legged's shins reach across under / over each other; with the other leg not crossed a shin would
    # lie on that leg's foot or thigh, so a lone crossed leg is folded closer: thigh more to the side, foot at the
    # middle in front of the hips (clear of an out / knee_up / tucked leg, legs_overlap -0.05 / -0.15 / -0.02)
    if _shape_of(posture, shape) == "cross" and (not isinstance(other, str) or _shape_of(posture, other) != "cross"):
        dirs = LONE_CROSS if side == "left" else tuple(_mirror(v) for v in LONE_CROSS)
    return tuple(np.asarray(v, float) for v in dirs)


def _rot(axis, a):
    axis = _unit(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def spread_leg(dirs, side, deg):
    """Open (deg > 0) or close a sitting leg. A straight leg swings outward about the vertical. A bent leg first turns
    outward about the vertical (the foot moves out too, as when sitting with the knees apart) and then rolls its knee
    out about the hip-ankle line; the foot keeps its tilt and only turns with the leg, so a flat sole stays flat
    (2026-10-28: turning about the hip-ankle line alone kept both feet together in the middle, toes crossed)."""
    if not deg:
        return dirs
    t, sh, toe = dirs
    sg = side_sign(side)
    bent = np.degrees(np.arccos(np.clip(np.dot(_unit(t), _unit(sh)), -1, 1))) > 30
    def outward(axis, probe, a):
        R = _rot(axis, np.radians(a))
        return R if ((R @ probe)[0] - probe[0]) * sg * np.sign(a) >= 0 else _rot(axis, -np.radians(a))
    knee, ankle = t * L_THIGH, t * L_THIGH + sh * L_SHIN
    if not bent:
        R = outward(_Y, ankle, deg)
        return tuple(R @ v for v in dirs)
    Ry = outward(_Y, ankle, 0.7 * deg)
    t, sh, toe = (Ry @ v for v in (t, sh, toe))
    knee, ankle = t * L_THIGH, t * L_THIGH + sh * L_SHIN
    Rr = outward(_unit(ankle), knee, 0.6 * deg)
    return Rr @ t, Rr @ sh, toe


def sit_legs_R(posture, left="default", right="default", knees="default"):
    """Local rotations {joint: [3,3]} for hips (identity) and both legs."""
    R = {"hips": np.eye(3)}
    for side, shape, other in (("left", left, right), ("right", right, left)):
        t, s, toe = spread_leg(sit_leg_dirs(posture, side, shape, other), side, _KNEES_DEG[knees or "default"])
        R_up, knee = thigh_frame(t, s)
        R_low = rx(knee)
        R[f"{side}UpperLeg"], R[f"{side}LowerLeg"] = R_up, R_low
        R[f"{side}Foot"] = foot_local(R_up @ R_low, toe)
    return R


def sit_hips(R, name=None):
    """Hips position for sitting leg rotations R: above the rest position in x / z, lowered until the lowest body
    point touches the floor. On a ledge only the bottom and the thighs' undersides rest on it (the shins hang)."""
    pts = leg_points(R, np.zeros(3))
    if name == LEDGE:
        low = min(pts["butt"][1], *(0.5 * (pts[f"{s}Hip"][1] + pts[f"{s}Knee"][1]) - R_KNEE for s in ("left", "right")))
        return np.array([HIPS_REST[0], -low, HIPS_REST[2]])
    low = min(pts["butt"][1], *(pts[f"{s}Knee"][1] - R_KNEE for s in ("left", "right")),
              *(pts[f"{s}Ankle"][1] - R_ANKLE for s in ("left", "right")),
              *(pts[f"{s}Toes"][1] - R_TOE for s in ("left", "right")))
    # A foot under the bottom (seiza) carries it: the bottom rests on the heel instead of the floor.
    for side in ("left", "right"):
        a = pts[f"{side}Ankle"]
        if np.hypot(*(a - pts["butt"])[[0, 2]]) < 0.12:
            low = min(low, pts["butt"][1] - (a[1] + R_ANKLE + 0.03))
    return np.array([HIPS_REST[0], -low, HIPS_REST[2]])


# Kneeling with the hips swayed to one side (2026-10-28, pose T040): the knees stay where they are on the floor and the
# thighs keep their length, so for a sideways hips shift dx the hips height and the pelvis tilt follow (two thighs,
# two unknowns); the shins and feet keep their world directions. Returns (R, hips position, pelvis tilt rad).
# knees: SIT_KNEES word; each knee (and its shin) moves out by KNEEL_KNEES on the floor (pose group D: knees apart).
KNEEL_KNEES = {"default": 0.0, "together": -0.025, "normal": 0.0, "apart": 0.07, "wide": 0.14}


def kneel_sway(dx, knees="default"):
    R0 = sit_legs_R("kneel")
    h0 = sit_hips(R0)
    spread = KNEEL_KNEES[knees or "default"]
    if not dx and not spread:
        return R0, h0, 0.0
    P0 = leg_points(R0, h0)
    K = {s: P0[f"{s}Knee"] + np.array([side_sign(s) * spread, 0.0, 0.0]) for s in ("left", "right")}
    shin = {s: _unit(P0[f"{s}Ankle"] - K[s]) for s in K}
    toe = {s: _unit(P0[f"{s}Toes"] - P0[f"{s}Ankle"]) for s in K}
    L = float(np.linalg.norm(THIGH))
    rz = lambda a: np.array([[np.cos(a), -np.sin(a), 0.0], [np.sin(a), np.cos(a), 0.0], [0.0, 0.0, 1.0]])
    def err(x):
        hips = np.array([h0[0] + dx, x[0], h0[2]])
        return np.array([np.linalg.norm(K[s] - (hips + rz(x[1]) @ (HIP_OFFSET * [side_sign(s), 1, 1]))) - L for s in K])
    x = np.array([h0[1], 0.0])
    for _ in range(30):                                       # Newton on (hips height, tilt)
        e = err(x)
        if np.abs(e).max() < 1e-6:
            break
        Jm = np.stack([(err(x + d) - e) / 1e-5 for d in (np.array([1e-5, 0]), np.array([0, 1e-5]))], 1)
        x = x - np.linalg.solve(Jm, e)
    hips, Rh = np.array([h0[0] + dx, x[0], h0[2]]), rz(x[1])
    R = {"hips": Rh}
    for s in K:
        hip = hips + Rh @ (HIP_OFFSET * [side_sign(s), 1, 1])
        R_up, knee = thigh_frame(Rh.T @ _unit(K[s] - hip), Rh.T @ shin[s])
        R_low = rx(knee)
        R[f"{s}UpperLeg"], R[f"{s}LowerLeg"] = R_up, R_low
        R[f"{s}Foot"] = foot_local(R_up @ R_low, Rh.T @ toe[s])
    return R, hips, float(x[1])


def sit_pose(name, left="default", right="default", knees="default"):
    """-> (local rotation matrices {joint: [3,3]} for hips and both legs, hips position [3] in world)."""
    R = sit_legs_R(name, left, right, knees)
    return R, sit_hips(R, name)


# ---------------- forward kinematics ----------------
def leg_points(R, hips_pos):
    """R: {joint: [3,3] local rotation} for hips and legs (or [..., 3, 3] arrays). Returns world
    positions of butt, hip joints, knees, ankles and toes."""
    Rh = R["hips"]
    out = {"butt": hips_pos + Rh @ BUTT}
    for side in ("left", "right"):
        sg = np.array([side_sign(side), 1, 1])
        hip = hips_pos + Rh @ (HIP_OFFSET * sg)
        G = Rh @ R[f"{side}UpperLeg"]
        knee = hip + G @ THIGH
        G = G @ R[f"{side}LowerLeg"]
        ankle = knee + G @ SHIN
        G = G @ R[f"{side}Foot"]
        out[f"{side}Hip"], out[f"{side}Knee"], out[f"{side}Ankle"], out[f"{side}Toes"] = hip, knee, ankle, ankle + G @ TOE
    return out


def motion_leg_points(motion, root, jidx):
    """Per-frame world positions [T, 3] of knees, ankles and toes for a motion [T, J, 4] with root [T, 3]."""
    hips_pos = HIPS_REST + root * HIPS_Y
    M = {n: quat_to_mat(motion[:, jidx[n]]) for n in
         ("hips", "leftUpperLeg", "leftLowerLeg", "leftFoot", "rightUpperLeg", "rightLowerLeg", "rightFoot")}
    out = {}
    for side in ("left", "right"):
        sg = np.array([side_sign(side), 1, 1])
        hip = hips_pos + np.einsum("tij,j->ti", M["hips"], HIP_OFFSET * sg)
        G = M["hips"] @ M[f"{side}UpperLeg"]
        knee = hip + np.einsum("tij,j->ti", G, THIGH)
        G = G @ M[f"{side}LowerLeg"]
        ankle = knee + np.einsum("tij,j->ti", G, SHIN)
        G = G @ M[f"{side}Foot"]
        out[f"{side}Hip"], out[f"{side}Knee"], out[f"{side}Ankle"] = hip, knee, ankle
        out[f"{side}Toes"] = ankle + np.einsum("tij,j->ti", G, TOE)
    return out


def to_quats(R):
    return {k: mat_to_quat(v) for k, v in R.items()}


# ---------------- upper body: hands against the torso and head ----------------
# Rest positions of the upper-body joints (default model, vrm_player/tools/skeleton.mjs) and the body
# surface around them (vrm_player/tools/body_shape.mjs: torso ~0.10 m half-width, front z +0.12, back
# z -0.09 between hips and shoulders; head ~0.10 x 0.12 x 0.11 around (0, 1.49, -0.03)). The torso is an
# elliptic cylinder carried by the upperChest bone, the head an ellipsoid carried by the head bone, so
# both follow the spine when the character leans or turns. Hands that go INTO them are clipping
# (the author saw hands sink into the chest, 2026-10-21); touching the surface is fine.
UPPER_REST = {
    "hips": (0.0, 0.9081, 0.0036), "spine": (0.0, 0.9601, 0.0161), "chest": (0.0, 1.0732, 0.0191),
    "upperChest": (0.0, 1.1808, 0.0045), "neck": (0.0, 1.3131, -0.0339), "head": (0.0, 1.3863, -0.0247),
    "leftShoulder": (0.0224, 1.2865, -0.0255), "leftUpperArm": (0.1086, 1.2742, -0.0255),
    "leftLowerArm": (0.3284, 1.2742, -0.0255), "leftHand": (0.5431, 1.2742, -0.0258),
}
for _n in ("Shoulder", "UpperArm", "LowerArm", "Hand"):
    _p = UPPER_REST["left" + _n]
    UPPER_REST["right" + _n] = (-_p[0], _p[1], _p[2])
UPPER_PARENT = {"spine": "hips", "chest": "spine", "upperChest": "chest", "neck": "upperChest", "head": "neck",
                "leftShoulder": "upperChest", "leftUpperArm": "leftShoulder", "leftLowerArm": "leftUpperArm",
                "leftHand": "leftLowerArm", "rightShoulder": "upperChest", "rightUpperArm": "rightShoulder",
                "rightLowerArm": "rightUpperArm", "rightHand": "rightLowerArm"}
PALM, FINGERTIP = 0.07, 0.15                     # along the hand bone from the wrist
# in rest coordinates; the front is set a little ahead of the measured 0.12 m because the loose shirt
# stands off the chest (a hand resting 1 cm "outside" the measured surface was visibly sunk in it)
TORSO = dict(y0=0.86, y1=1.30, cz=0.025, hx=0.10, hz=0.115)
CHIN = np.array([0.0, -0.06, 0.08])             # from the head joint, in the head's frame
CHIN_FLOOR = 1.22                               # rest-coordinate height of the collarbones
HEAD = dict(c=(0.0, 1.49, -0.03), r=(0.10, 0.12, 0.11))
# legs as capsules along the bones (vrm_player/tools/leg_shape.mjs, skin only, median radius per slice:
# upper thigh ~0.065, knee ~0.045-0.05, mid shin ~0.045, ankle ~0.03). The arm points are bone centres, so
# the arm's own thickness counts too: forearm ~0.03, hand half-thickness ~0.015. Contact is fine (the
# knees_hugged rest pose rests the arms on the shins); LEG_TOL of overlap is allowed for soft tissue.
# the author, 2026-10-23: in a knees_hugged recoil the forearms went through the knees.
LEG_R = dict(thigh=0.055, shin=0.045)


def _leg_profile(model="VRM1_Constraint_Twist_Sample", pct=75):
    """Leg thickness along the bone (2026-10-23 with the clothes: kneeling hands went into the loose shorts; since
    2026-10-29 the fit holds the skin only and the hands press the clothes on the GPU, limb_push.js). From the surface samples of
    body_fits/<model>.json (vrm_player/tools/fit_capsules.mjs): per 0.1 of the bone, the pct-th percentile
    distance from the axis (thigh also takes the hips' samples on its side, where the shorts hang from);
    never thinner than the skin LEG_R. Returns {name: (t [N], r [N])}, interpolated in leg_radius."""
    import json
    from pathlib import Path
    f = Path(__file__).parent / "body_fits" / f"{model}.json"
    if not f.exists():
        return {}
    fit = json.load(open(f))
    out = {}
    # thigh only: the shin's samples include the calf (7 cm behind the bone, 3.4 cm in front), and a round
    # capsule that thick would push hands resting on the front of the shin off it
    for name, seg, src in (("thigh", "rightUpperLeg", ("hips", "rightUpperLeg")),):
        a, b = np.array(fit["segments"][seg]["a"]), np.array(fit["segments"][seg]["b"])
        P = np.concatenate([np.array(fit["points"][k]) for k in src])
        P = P[P[:, 0] < a[0] / 2]                          # the right leg's side of the body
        ab = b - a
        t = (P - a) @ ab / (ab @ ab)
        r = np.linalg.norm(P - a - t[:, None] * ab, axis=1)
        ts, rs = [], []
        for lo in np.arange(0.0, 1.0, 0.1):
            m = (t >= lo) & (t < lo + 0.1) & (r < 0.15)
            if m.sum() > 3:
                ts.append(lo + 0.05); rs.append(max(float(np.percentile(r[m], pct)), LEG_R[name]))
        out[name] = (np.array(ts), np.array(rs))
    return out


LEG_PROFILE = _leg_profile()


def leg_radius(name, t):
    """Leg radius (metres, the skin) at fraction t [..] along the thigh / shin."""
    if name not in LEG_PROFILE:
        return np.full(np.shape(t), LEG_R[name])
    ts, rs = LEG_PROFILE[name]
    return np.interp(t, ts, rs)
ARM_R = dict(forearm=0.03, hand=0.015)
LEG_TOL = 0.01


def _seg_depth(pt, a, b, r):
    """How far pt [T,3] is inside the capsule a-b [T,3] of radius r (<= 0: outside). r may be a function of
    the fraction t along a-b (a leg's profile)."""
    ab = b - a
    t = np.clip(((pt - a) * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9), 0.0, 1.0)
    return (r(t) if callable(r) else r) - np.linalg.norm(pt - (a + t[:, None] * ab), axis=-1)


def upper_fk(motion, root, jidx):
    """World positions P[n] [T, 3] and rotations G[n] [T, 3, 3] of the upper-body joints."""
    rest = {n: np.asarray(p, float) for n, p in UPPER_REST.items()}
    G = {"hips": quat_to_mat(motion[:, jidx["hips"]])}
    P = {"hips": rest["hips"] + root * HIPS_Y}
    for n, par in UPPER_PARENT.items():                    # parents come first in this dict
        P[n] = P[par] + np.einsum("tij,j->ti", G[par], rest[n] - rest[par])
        G[n] = G[par] @ quat_to_mat(motion[:, jidx[n]])
    return P, G


KEY_POINTS = ("head", "leftLowerArm", "rightLowerArm", "leftPalm", "rightPalm",
              "leftKnee", "rightKnee", "leftAnkle", "rightAnkle")


def key_points(motion, root, jidx):
    """World positions [T, len(KEY_POINTS), 3] of head, elbows, palm centres, knees and ankles."""
    P, G = upper_fk(motion, root, jidx)
    for side, sx in (("left", 1.0), ("right", -1.0)):
        P[f"{side}Palm"] = P[f"{side}Hand"] + np.einsum("tij,j->ti", G[f"{side}Hand"], np.array([sx * PALM, 0.0, 0.0]))
    P.update(motion_leg_points(motion, root, jidx))
    return np.stack([P[n] for n in KEY_POINTS], 1)


HANDS_MIN = 0.02       # palm / finger points of the two hands closer than this (bone centres; a hand is ~3 cm thick)


ARMS_TOL = 0.01        # the two forearms (capsules of ARM_R["forearm"]) may overlap this much (soft flesh, as LEG_TOL)


def forearms_overlap(motion, root, jidx, n=11):
    """Per frame, how far (metres) the two forearms are inside each other beyond ARMS_TOL (> 0 = clipping). The forearm =
    elbow to wrist, its radius tapering from ARM_R["forearm"] at the elbow to ARM_R["hand"] at the wrist (2026-11-03,
    the author: crossed arms U030 / U041 had the forearms through each other, axes 0.2-0.8 cm apart; hands_clip checks the
    hands' points only. With a constant radius, palms pressed together / hands clasped counted: the wrists are 3-4 cm
    apart, which hands_clip already judges)."""
    P, _ = upper_fk(motion, root, jidx)
    u = np.linspace(0.0, 1.0, n)
    r = ARM_R["forearm"] + u * (ARM_R["hand"] - ARM_R["forearm"])
    A = P["leftLowerArm"][:, None] + u[None, :, None] * (P["leftHand"] - P["leftLowerArm"])[:, None]
    B = P["rightLowerArm"][:, None] + u[None, :, None] * (P["rightHand"] - P["rightLowerArm"])[:, None]
    d = np.linalg.norm(A[:, :, None] - B[:, None, :], axis=-1)              # [T, n, n]
    return (r[:, None] + r[None, :] - ARMS_TOL - d).max(axis=(1, 2))


def hands_overlap(motion, root, jidx):
    """Per frame, how far (metres) the two hands' points (wrist, palm centre, fingertip) are inside each other:
    HANDS_MIN minus the smallest left-right distance; > 0 = the hands pass through each other (the author saw
    interlaced fingers going through each other, 2026-10-23). Palms pressed together stay ~3 cm apart."""
    P, G = upper_fk(motion, root, jidx)
    pts = {}
    for side, sx in (("left", 1.0), ("right", -1.0)):
        pts[side] = [P[f"{side}Hand"] + np.einsum("tij,j->ti", G[f"{side}Hand"], np.array([sx * d, 0.0, 0.0]))
                     for d in (0.0, PALM, 0.11, FINGERTIP)]
    dist = np.min([np.linalg.norm(a - b, axis=-1) for a in pts["left"] for b in pts["right"]], axis=0)
    return HANDS_MIN - dist


LEGS_TOL = 0.02          # legs may press into each other this much (cross_legged's stacked shins: 0.5 cm)


def legs_gap(R, hips):
    """legs_overlap for one pose: R {joint: [3,3]} for hips and legs, hips position [3]."""
    P = leg_points(R, hips)
    u = np.linspace(0, 1, 12)[:, None]
    worst = -np.inf
    for a, b, ra in (("Hip", "Knee", LEG_R["thigh"]), ("Knee", "Ankle", LEG_R["shin"])):
        for c, d, rb in (("Hip", "Knee", LEG_R["thigh"]), ("Knee", "Ankle", LEG_R["shin"])):
            A = P[f"left{a}"] + u * (P[f"left{b}"] - P[f"left{a}"])
            B = P[f"right{c}"] + u * (P[f"right{d}"] - P[f"right{c}"])
            if a == "Hip" and c == "Hip":
                A, B = A[4:], B[4:]
            worst = max(worst, ra + rb - np.linalg.norm(A[:, None] - B[None], axis=-1).min() - LEGS_TOL)
    return worst


def legs_overlap(motion, root, jidx):
    """Per frame, how far (metres) the two legs (thigh and shin capsules, LEG_R) are inside each other beyond
    LEGS_TOL; > 0 = they pass through each other (2026-10-28: sitting legs can now be combined freely)."""
    L = motion_leg_points(motion, root, jidx)
    u = np.linspace(0, 1, 12)
    def seg(side, a, b):
        return L[f"{side}{a}"][:, None] + u[None, :, None] * (L[f"{side}{b}"] - L[f"{side}{a}"])[:, None]
    worst = np.full(len(motion), -np.inf)
    for a, b, ra in (("Hip", "Knee", LEG_R["thigh"]), ("Knee", "Ankle", LEG_R["shin"])):
        for c, d, rb in (("Hip", "Knee", LEG_R["thigh"]), ("Knee", "Ankle", LEG_R["shin"])):
            A, B = seg("left", a, b), seg("right", c, d)
            if a == "Hip" and c == "Hip":                 # the thighs meet at the crotch: skip their top third
                A, B = A[:, 4:], B[:, 4:]
            dist = np.linalg.norm(A[:, :, None] - B[:, None], axis=-1).min(axis=(1, 2))
            worst = np.maximum(worst, ra + rb - dist - LEGS_TOL)
    return worst


TORSO_BANDS = ("hips", "spine", "chest", "upperChest")


def torso_depth(pts, P, G):
    """How far points pts [T,3] are inside the torso, and the outward normal (world) [T,3]. The elliptic cylinder TORSO
    is cut into height bands at the spine joints, each carried by its own bone (2026-10-28: one cylinder carried by
    upperChest swept its lower end forward over the lap when the torso leaned back, so hands resting on the thighs of a
    reclining character counted as inside the belly). Rest coordinates; depth < 0 = outside."""
    rest = {n: np.asarray(p, float) for n, p in UPPER_REST.items()}
    T = len(pts)
    hx = TORSO["hx"]
    edges = [TORSO["y0"]] + [rest[b][1] for b in TORSO_BANDS[1:]] + [TORSO["y1"]]
    depth, normal = np.full(T, -np.inf), np.zeros((T, 3))
    for i, b in enumerate(TORSO_BANDS):
        hz, cz = TORSO.get("bands", {}).get(b, (TORSO["hz"], TORSO["cz"]))   # per-band depth (see _skin_torso)
        q = np.einsum("tji,tj->ti", G[b], pts - P[b]) + rest[b]
        e = np.sqrt((q[:, 0] / hx) ** 2 + ((q[:, 2] - cz) / hz) ** 2)
        lo, hi = max(edges[i], TORSO["y0"]), min(edges[i + 1], TORSO["y1"])
        if i == len(TORSO_BANDS) - 1:
            hi = TORSO["y1"]
        d = np.where((q[:, 1] > lo) & (q[:, 1] <= hi), (1 - e) * min(hx, hz), -np.inf)
        n = np.stack([q[:, 0] / hx ** 2, np.zeros(T), (q[:, 2] - cz) / hz ** 2], -1)
        n = np.einsum("tij,tj->ti", G[b], n / np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9))
        take = d > depth
        depth[take], normal[take] = d[take], n[take]
    return depth, normal


def arm_rest_fix(side):
    """(Cu, Cf) for one arm of the active model (2026-10-28). The solver works as if the T-pose arm lay along x
    (solve_arm: upper arm along +-x, elbow a pure y hinge), but a model's rest bones may tilt: the 0.x sample's upper arm
    by 2.7 deg and its forearm by 4.7 deg forward, so re-solving an arm with zero change moved its wrist 2.6 cm. Cu turns
    +-x onto the real rest upper arm, Cf onto the real rest forearm; rotations are converted with
    U_real = U Cu^T, L_real = Cu L Cf^T, H_real = Cf H (arm_to_real / arm_to_ideal). Below 0.5 deg: identity."""
    r = {k: np.asarray(v, float) for k, v in UPPER_REST.items()}
    x = np.array([1.0 if side == "left" else -1.0, 0.0, 0.0])
    out = []
    for a, b in ((f"{side}UpperArm", f"{side}LowerArm"), (f"{side}LowerArm", f"{side}Hand")):
        v = _unit(r[b] - r[a])
        out.append(np.eye(3) if np.degrees(np.arccos(np.clip(v @ x, -1, 1))) < 0.5 else _swing(x, v))
    return tuple(out)


def arm_to_real(side, U, L, H):
    """Ideal-frame rotation matrices (upper relative to the torso, lower, hand) -> the model's own; [..., 3, 3]."""
    Cu, Cf = arm_rest_fix(side)
    return U @ Cu.T, Cu @ L @ Cf.T, Cf @ H


def arm_to_ideal(side, U, L, H):
    Cu, Cf = arm_rest_fix(side)
    return U @ Cu, Cu.T @ L @ Cf, Cf.T @ H


HEAD_TOL = 0.001      # m: a hand point this little inside the head ellipsoid is not a clip (2026-11-08: a fingertip
# 0.5 mm inside for a whole hold, invisible; the push-out's own margin is 2 mm)


def hand_body_depth(motion, root, jidx):
    """Per frame, how far (metres) the deepest checked point is inside the body; <= 0 means outside.
    Points: palm centre and fingertip (vs torso and head), elbow and mid forearm (vs torso), the forearm
    and hand as thick limbs (vs the leg capsules), chin (vs torso). motion [T,J,4], root [T,3]."""
    T = len(motion)
    rest = {n: np.asarray(p, float) for n, p in UPPER_REST.items()}
    P, G = upper_fk(motion, root, jidx)
    def in_torso(pt):
        return torso_depth(pt, P, G)[0]

    def in_head(pt):
        h = np.einsum("tji,tj->ti", G["head"], pt - P["head"]) + rest["head"]
        return (1 - np.sqrt((((h - np.array(HEAD["c"])) / np.array(HEAD["r"])) ** 2).sum(-1))) * min(HEAD["r"]) - HEAD_TOL

    L = motion_leg_points(motion, root, jidx)
    legs = [(L[f"{s}{a}"], L[f"{s}{b}"], (lambda t, n=n: leg_radius(n, t))) for s in ("left", "right")
            for a, b, n in (("Hip", "Knee", "thigh"), ("Knee", "Ankle", "shin"))]
    depth = np.full(T, -np.inf)
    for side, sx in (("left", 1.0), ("right", -1.0)):
        hand = [P[f"{side}Hand"] + np.einsum("tij,j->ti", G[f"{side}Hand"], np.array([sx * d, 0.0, 0.0]))
                for d in (PALM, FINGERTIP)]
        arm = [P[f"{side}LowerArm"], 0.5 * (P[f"{side}LowerArm"] + P[f"{side}Hand"])]    # elbow, mid forearm
        for pt in hand + arm:
            depth = np.maximum(depth, in_torso(pt))
        e, w = P[f"{side}LowerArm"], P[f"{side}Hand"]
        limb = [(e + f * (w - e), ARM_R["forearm"]) for f in (0.0, 0.25, 0.5, 0.75, 1.0)] + [(pt, ARM_R["hand"]) for pt in hand]
        for pt, ra in limb:
            for a, b, r in legs:
                depth = np.maximum(depth, _seg_depth(pt, a, b, lambda t, r=r: r(t) + ra) - LEG_TOL)
        for pt in hand:
            depth = np.maximum(depth, in_head(pt))
    # the chin sinking into the chest (head dropped too far). A hard nod tucks the chin into the throat,
    # which is inside the cylinder's top but natural, so only count it below the collarbones.
    chin = P["head"] + np.einsum("tij,j->ti", G["head"], CHIN)
    cq = np.einsum("tji,tj->ti", G["upperChest"], chin - P["upperChest"]) + rest["upperChest"]
    return np.maximum(depth, np.where(cq[:, 1] < CHIN_FLOOR, in_torso(chin), -np.inf))


# ---------------- per-model skeletons ----------------
# the author, 2026-10-24: motions must work on any VRM model. The keyframe format is model-free (directions, words,
# contact words), so a motion is solved per model: use_model(name) swaps the bone lengths and body shapes above
# for that model's, and refine.solve(obj, dur, model=name) solves the keyframes inside it (legs IK with its leg
# lengths, sitting hips height from its own FK, contacts on its own surface, the filter's body shapes). The
# stored data and the policy stay on the default model. Skeletons: skeletons/<model>.json (rest positions of the
# normalized bones, vrm_player/tools/skeleton.mjs); body fits: body_fits/<model>.json (tools/fit_capsules.mjs).
# Modules that derive constants from these register a hook with on_model_change.
from contextlib import contextmanager as _contextmanager
from pathlib import Path as _Path
import json as _json

DEFAULT_MODEL = "VRM1_Constraint_Twist_Sample"
MODEL = DEFAULT_MODEL
CANONICAL = True     # the built-in defaults are active (not a measured model, even the default one; 2026-11-13)
_HOOKS = []
_DEFAULTS = dict(HIPS_REST=HIPS_REST, HIP_OFFSET=HIP_OFFSET, THIGH=THIGH, SHIN=SHIN, TOE=TOE, ANKLE_REST_Y=ANKLE_REST_Y,
                 UPPER_REST=dict(UPPER_REST), R=(R_KNEE, R_ANKLE, R_TOE), BUTT=BUTT, TORSO=dict(TORSO), CHIN=CHIN,
                 CHIN_FLOOR=CHIN_FLOOR, HEAD=dict(HEAD), LEG_R=dict(LEG_R), ARM_R=dict(ARM_R), PALM=(PALM, FINGERTIP),
                 LEG_PROFILE=LEG_PROFILE)


def on_model_change(fn):
    """Register fn() to be called after the active model changes (and once now)."""
    _HOOKS.append(fn)
    fn()
    return fn


def models():
    return sorted(p.stem for p in (_Path(__file__).parent / "skeletons").glob("*.json"))


def _fit(name):
    f = _Path(__file__).parent / "body_fits" / f"{name}.json"
    return _json.load(open(f)) if f.exists() else None


def _hand_lengths(fit, name):
    """(palm centre, fingertip) distance from the wrist along the hand, from the model's own surface samples: the
    fingertip = the farthest finger sample, the palm centre = the default model's ratio of the palm samples' reach
    (0.07 m at 0.071). None without finger samples."""
    if not fit:
        return None
    pts = fit.get("points", {})
    f = _Path(__file__).parent / "skeletons" / f"{name}.json"
    skl = _json.load(open(f)) if f.exists() else None
    if not skl or "leftHand" not in skl:
        return None
    h = np.asarray(skl["leftHand"], float)
    fing = [np.asarray(v, float) for k, v in pts.items() if k.startswith("left") and
            any(f in k for f in ("Index", "Middle", "Ring", "Little")) and len(v)]
    palm = np.asarray(pts.get("leftHand", []), float)
    if not fing or not len(palm):
        return None
    tip = float(max(((f - h)[:, 0]).max() for f in fing))
    knuckle = float(((palm - h)[:, 0]).max())
    return knuckle * HAND_PALM_RATIO, tip


HAND_PALM_RATIO = 0.07 / 0.071


# The torso cylinder (TORSO) was set by hand on the default model with its loose shirt; since 2026-10-29 the body is
# measured from the skin only (vrm_player/src/measure.js), so the cylinder is scaled from the skin fit: its width
# matched the default's skin waist side radius (0.103 m), its depth had the shirt in it (chest front radius 0.127 m).
# The width comes from the waist (spine) side radius: a model without upperChest (Seed-san) has a chest segment up to
# the neck that takes in the shoulders (0.15 m), which spread its arms past their limits.
TORSO_REF_OUT, TORSO_REF_FRONT = 0.103, 0.127


def _skin_torso(T0, fit, s):
    seg = fit["segments"] if fit else {}
    w, c = seg.get("spine") or seg.get("chest"), seg.get("chest")
    wx = (w.get("out") or w["r"]) / TORSO_REF_OUT if w else s
    wz = (c.get("front") or c["r"]) / TORSO_REF_FRONT if c else s
    T = dict(y0=T0["y0"] * s, y1=T0["y1"] * s, cz=T0["cz"] * s, hx=T0["hx"] * wx, hz=T0["hz"] * wz)
    # Per-band depth (2026-11-10): one ellipse for the whole trunk put the upper chest's front at the bust's depth, ~6 cm
    # in front of the real collar on every VRoid model (the wrist of an index on the lips counted as 3.6 cm inside, so
    # the contact solver moved it forward and the finger tilted 25 deg; the author: "你不是已經知道嘴唇在哪了"). Each band
    # takes its own bone's front / back radii from the skin fit (hz = their mean, cz = the axis z + their difference);
    # the width stays the shared hx. A band without radii keeps the shared values.
    bands = {}
    for b in TORSO_BANDS:
        g = seg.get(b)
        if not g or not g.get("front"):            # no skin in front (VRoid deletes it under clothes: test2's bust): shared
            bands[b] = (T["hz"], T["cz"]); continue
        front = g["front"]; back = g.get("back") or front
        az = 0.5 * (g["a"][2] + g["b"][2])
        bands[b] = (0.5 * (front + back), az + 0.5 * (front - back))
    T["bands"] = bands
    return T


def set_model(name=None):
    """Make `name` (a file in skeletons/) the active model; None restores the built-in defaults (the canonical skeleton
    the research data, judge and policy use). Every named model, the default sample included, is read from its measured
    files (2026-11-13, the author: no special case by model name; the product always passes a name)."""
    global MODEL, HIPS_REST, HIPS_Y, HIP_OFFSET, THIGH, SHIN, TOE, L_THIGH, L_SHIN, ANKLE_REST_Y, FLAT, UPPER_REST
    global R_KNEE, R_ANKLE, R_TOE, BUTT, TORSO, CHIN, CHIN_FLOOR, HEAD, LEG_R, ARM_R, PALM, FINGERTIP, LEG_PROFILE, CANONICAL
    D = _DEFAULTS
    CANONICAL = not name
    if not name:
        name = DEFAULT_MODEL
        HIPS_REST, HIP_OFFSET, THIGH, SHIN, TOE = D["HIPS_REST"], D["HIP_OFFSET"], D["THIGH"], D["SHIN"], D["TOE"]
        ANKLE_REST_Y, UPPER_REST = D["ANKLE_REST_Y"], dict(D["UPPER_REST"])
        (R_KNEE, R_ANKLE, R_TOE), BUTT = D["R"], D["BUTT"]
        TORSO, CHIN, CHIN_FLOOR, HEAD = _skin_torso(D["TORSO"], _fit(name), 1.0), D["CHIN"], D["CHIN_FLOOR"], dict(D["HEAD"])
        LEG_R, ARM_R, (PALM, FINGERTIP), LEG_PROFILE = dict(D["LEG_R"]), dict(D["ARM_R"]), D["PALM"], D["LEG_PROFILE"]
    else:
        sk = {k: np.asarray(v, float) for k, v in _json.load(open(_Path(__file__).parent / "skeletons" / f"{name}.json")).items()}
        if "chest" not in sk:                            # optional VRM bones, folded as the player does
            sk["chest"] = sk["spine"]
        if "upperChest" not in sk:
            sk["upperChest"] = sk["chest"]
        for side in ("left", "right"):                   # no toe bones: a toe in front of the ankle, near the floor
            if f"{side}Toes" not in sk:
                fy = sk[f"{side}Foot"][1]
                sk[f"{side}Toes"] = sk[f"{side}Foot"] + np.array([0.0, -0.5 * fy, 0.85 * fy])
        HIPS_REST = sk["hips"]
        HIP_OFFSET = sk["leftUpperLeg"] - sk["hips"]
        THIGH = sk["leftLowerLeg"] - sk["leftUpperLeg"]
        SHIN = sk["leftFoot"] - sk["leftLowerLeg"]
        TOE = sk["leftToes"] - sk["leftFoot"]
        ANKLE_REST_Y = float(sk["leftFoot"][1])
        UPPER_REST = {n: tuple(sk[n]) for n in D["UPPER_REST"] if n in sk}
        for n in D["UPPER_REST"]:                         # missing shoulder / neck: at the parent's position
            if n not in UPPER_REST:
                UPPER_REST[n] = UPPER_REST.get(UPPER_PARENT.get(n, "hips"), tuple(HIPS_REST))
        s = HIPS_REST[1] / D["HIPS_REST"][1]              # body size relative to the default model
        R_KNEE, R_ANKLE, R_TOE = (r * s for r in D["R"])
        BUTT = D["BUTT"] * s
        fit = _fit(name)
        TORSO = _skin_torso(D["TORSO"], fit, s)
        CHIN, CHIN_FLOOR = D["CHIN"] * s, D["CHIN_FLOOR"] * s
        HEAD = dict(c=tuple(fit["head"]["c"]), r=tuple(fit["head"]["r"])) if fit and fit.get("head") else \
            dict(c=tuple(np.array(D["HEAD"]["c"]) * s), r=tuple(np.array(D["HEAD"]["r"]) * s))
        LEG_R = {k: v * s for k, v in D["LEG_R"].items()}
        ARM_R = {k: v * s for k, v in D["ARM_R"].items()}
        PALM, FINGERTIP = (v * s for v in D["PALM"])
        hand = _hand_lengths(fit, name)     # measured from the model's own hand (2026-10-29, the author: hands differ;
        if hand:                            # by body height Seed-san's fingers came out 3.4 cm short, test1's 2 cm long)
            PALM, FINGERTIP = hand
        LEG_PROFILE = _leg_profile(name) if fit else {}
    HIPS_Y = HIPS_REST[1]
    L_THIGH, L_SHIN = np.linalg.norm(THIGH), np.linalg.norm(SHIN)
    FLAT = _unit(TOE)
    MODEL = name
    for fn in _HOOKS:
        fn()


TORSO = _skin_torso(_DEFAULTS["TORSO"], _fit(DEFAULT_MODEL), 1.0)   # the default model's skin torso from the start


@_contextmanager
def use_model(name):
    """with use_model("test2"): ... solves / checks for that model; restores the previous model afterwards."""
    prev = None if CANONICAL else MODEL
    set_model(name)
    try:
        yield
    finally:
        set_model(prev)

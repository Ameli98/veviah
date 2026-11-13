"""Candidate source 2: an LLM writes a few keyframes, we turn them into a motion.

Use in the pipeline with `--generator motion_loop.keyframes:KeyframeGenerator` (`--llm mock|anthropic`).

The LLM does not write joint angles (sign conventions are easy to get wrong). It writes, per keyframe:
  head / torso   three plain angles in degrees each
  each arm       two directions: where the upper arm points and where the forearm points,
                 in the torso's frame (x = character's left, y = up, z = forward)
keyframes_to_motion() then solves the joints:
  - upper arm: points along `upper`, and is twisted so that the elbow hinge is perpendicular to
    both directions (a real elbow only bends one way); see solve_arm()
  - lower arm: one hinge angle = the angle between the two directions
  - head and torso angles are spread over neck/head and spine/chest/upperChest
  - shoulders: one raise angle each (a shrug); the upper arm is counter-rotated so that the arm
    directions stay relative to the torso
  - hands: one named finger shape per hand (schema.HAND_SHAPES), switched halfway between keyframes;
    returned separately by keyframes_to_hands()
  - wrists: per arm, an optional palm direction word and a wrist bend angle; solve_palm() turns the
    palm direction into a twist about the forearm, shared between the lower arm and the hand joint
  - posture (once per motion, the author 2026-10-15): "stand" or one way of sitting on the floor
    (skeleton.SIT_NAMES). Sitting legs are a fixed pose. Standing legs are words, not numbers: weight
    side, hips height, stance width, and per foot a place and a lift; legs_frames() turns them into hips
    and ankle targets and solves the legs by IK every frame, so a planted foot does not slide
Arms are solved at the keyframes; in between, joint rotations are blended (slerp) with a smoothstep,
so the motion pauses at each keyframe.
"""
import json

import numpy as np

from .dims import SIT_DESC
from .generators import MotionGenerator
from .llm import extract_json
from . import schema
from . import skeleton as sk
from .rotations import continuous, euler_to_mat, from_euler, mat_to_quat, quat_conj, quat_mul, quat_to_mat, slerp
from .schema import FPS, HAND_SHAPES, J, JIDX, JOINTS

PROMPT_MARK = "You animate a desktop companion character"
PROMPT = PROMPT_MARK + """ by writing keyframes for its whole body. It stays in place: it either
stands or sits on the floor, and keeps that posture for the whole motion.

Instruction: "{instruction}"
Duration: {duration:.1f} s
Variation: {variation}

First choose "posture": "stand", or one way of sitting on the floor: {sit_list}.
If the instruction does not say, stand.

Coordinates are the character's own: x = its left, y = up, z = forward (toward the viewer).
Each keyframe has a time `t` in seconds and any of these parts (a part you leave out keeps its previous value):
- "head": {{"nod": deg, "turn": deg, "tilt": deg}}   nod + = looks down; turn + = toward its left; tilt + = the head leans toward its LEFT (left ear toward the left shoulder). Limits: nod -30..40, turn -70..70, tilt -30..30.
- "torso": {{"lean": deg, "turn": deg, "bend": deg}}   lean + = forward; turn + = toward its left; bend + = the torso leans toward its LEFT side. Limits: lean -30 (back) .. 25 (sitting on the floor up to 60: past 25 the hips tip forward too), turn 30 and bend 20 either way (torso turn only twists the waist; to turn the whole body use the standing legs "turn").
- "right_arm", "left_arm": {{"upper": [x, y, z], "fore": [x, y, z]}}   directions (any length) of the upper arm (shoulder to elbow) and the forearm (elbow to hand), relative to the torso.
  Examples for the RIGHT arm: hanging down = upper [-0.3, -1, 0], fore [-0.3, -1, 0.3]; hand raised in a wave = upper [-1, -0.3, 0], fore [-0.2, 1, 0];
  pointing forward = upper [0, -0.3, 1], fore [0, 0, 1]; hand on chest = upper [-0.4, -1, 0.3], fore [1, 0.4, 0.3];
  sitting, hands resting on the thighs: see the rest pose of each sitting posture below. For the LEFT arm, flip the sign of x.
  Optional, inside the same arm object:
  "palm": which way the palm faces, one of: forward (shown to the viewer, as in a wave or "stop"), back (toward its own body or face; the viewer sees the back of the hand), up, down, in (toward the middle of its body), out (away from the middle), natural (the default: no particular twist).
  "wrist": deg, bending the hand at the wrist; + = toward the palm side, - = toward the back of the hand. Limits: -70..70.
  "wrist_side": deg, the hand bent sideways at the wrist, + = toward the little finger. Limits: -45..45.
- "shoulders": {{"left": deg, "right": deg}}   how far each shoulder is raised toward the ear (a shrug or a tense hunch). Limits: -10..25; a clear shrug is about 20.
  Or per shoulder {{"raise": deg, "forward": deg}}: forward + = the shoulder rolls forward (rounded, hunched), - = pulled back (chest out, hands clasped behind). Limits -15..20.
- "eyes": closed | open   the eyes closed from this keyframe on, until a keyframe says open (default open; blinking pauses while closed).
- "hands": {{"left": shape, "right": shape}}   finger shape, one of: relaxed (loosely curled, the default), open (flat, fingers spread), fist, point (index finger out), peace (V sign), thumbs_up, flat (fingers straight and together), loose (fingers lightly bent and a little spread, e.g. hands raised in surprise).
- "legs" when sitting on the bottom (cross_legged, w_sit, side_right, side_left, knees_hugged, legs_out; not seiza / kneel):
  {{"left": shape, "right": shape, "knees": k}}   shape: out (stretched forward), knee_up (knee raised, foot flat), knee_to_chest (knee pulled up close to the chest, foot by the bottom), cross
  (folded across in front), tucked (folded back beside the hip), default (the posture's own); knees: together, normal,
  apart, wide. A leg part you leave out keeps its shape; e.g. legs_out with "left": "knee_up" = one knee raised.
- "legs" when kneeling (kneel): {{"knees": k, "weight": left | right, "hip": out | far_out}}   knees as above; weight + hip = the hips
  sway to that side with the knees staying where they are (as the standing hip sway below).
- "legs" when standing: {{"weight": w, "height": h, "stance": s, "knees": normal | in, "hip": hp, "turn": deg, "left_foot": {{"place": p, "lift": l, "dir": d, "toes": t}}, "right_foot": {{...}}}}
  weight: center (default), left, right = which foot carries the body.  height: straight (default), bent (knees soft), crouch, squat.
  stance: normal (default), together, wide.  place: home (default, under the hip), forward, back, out (a small step; the foot steps there and stays until moved back),
  lunge_forward, lunge_out (a big step forward or to the side; the hips drop between the feet, put the weight on that foot for a side lunge),
  forward_out (a small step diagonally forward and out), cross (the foot in front of the other one, at the middle: a model's crossed step; usually the free foot, often with "lift": "heel").
  lift: down (default), lift (a little off the floor), knee_up (knee raised), knee_to_chest (knee pulled up high, e.g. hugged), kick (leg swung), heel (heel raised, ball of the foot stays down; counts as on the floor; both feet heel = up on the toes). dir: forward (default), side, back = which way the lifted leg goes.
  toes: forward (default), out (turned out ~35 deg, the knee follows), in (pigeon-toed), or degrees -30..60 (+ = out).
  Only one foot may be off the floor; lifting one foot puts the weight on the other (except in a jump).
  "jump": small | big, or {{"size": small | big, "count": 1..3, "pose": default | tuck}} = a jump straight up in place starting
  at this keyframe (both feet down): crouch, take-off, flight (small ~10 cm, big ~25 cm), landing, standing up again, all added
  by the solver; count > 1 = bouncing jumps; tuck = knees pulled up in the air. It takes small <<JUMP_SMALL>> s, big <<JUMP_BIG>> s per jump
  (count 2 / 3: add <<JUMP_REP_SMALL>> / <<JUMP_REP_BIG>> s per extra jump); the next keyframe must come after it. Arms not given in the
  jump keyframe swing with the jump. Standing only.
  "hip": level (default), out, far_out, or degrees of pelvis tilt 0..20 (out = 8, far_out = 14) = hip sway / contrapposto: the hip juts out over the weight foot, the pelvis drops toward the free leg (its knee bends and turns in),
  the shoulders tilt the other way (S-curve). Needs weight left or right; often with the free foot's "lift": "heel".
  "turn": degrees, -180..180 (+ = toward its left): the whole body turns in place, the feet step or pivot with it (180 = its back to the viewer). Default 0 (facing the viewer). To turn round use this; torso "turn" only twists the waist.

Rules: 3 to 8 keyframes, times increasing from 0 to {duration:.1f}. Start from the rest pose of the posture and return to it at the end
(standing: arms hanging, legs at their defaults; sitting: the hands rest on the thighs, which sit differently in each posture:
<<REST_SIT>>
for side_right / side_left the hand away from the feet props on the floor beside the hip and the other rests on the thigh,
the torso leaning onto the propping hand (a side-sit keyframe without "torso" / "head" gets that lean; write "rest" for those arms);
for seiza both hands rest on the thighs (write "rest");
for knees_hugged the arms are wrapped round the shins, forearms crossed in front of them (write "rest"), the torso leans
forward 20, the head is bowed 15 and the shoulders come forward: a knees_hugged keyframe without "torso" / "head" /
"shoulders" gets these, so write them only to change them;
head, torso and shoulders at 0 otherwise, hands relaxed, wrists 0, palms natural unless given above).
Repeated actions (waving, nodding, bouncing) need one keyframe per extreme. The character has nothing around it (no chair, desk or objects).
Hands and arms never go into the body: a hand on the chest rests on its surface, and hands held in front of the chest stay about a fist's width away from it.
The two hands never pass through each other: hands brought together touch palm to palm or stay a little apart.

Return JSON only: {{"posture": "...", "keyframes": [{{"t": 0.0, ...}}, ...]}}"""

VARIATIONS = ["the most typical way to do it", "smaller and more subtle", "larger and more expressive",
              "a different but valid interpretation"]

REST_ARM = {"right": ([-0.315, -0.949, 0.0], [-0.305, -0.919, 0.247]),     # same pose as test_motions._rest
            "left": ([0.315, -0.949, 0.0], [0.305, -0.919, 0.247])}
HEAD_LIM = {"nod": (-30, 40), "turn": (-70, 70), "tilt": (-30, 30)}
TORSO_LIM = {"lean": (-30, 60), "turn": (-30, 30), "bend": (-20, 20)}   # lean 50 only sitting on the floor, see lean_max   # lean back 30: reclining on the hands (S065;
# the default model managed 34, but at 34 the hands going from the thighs to the floor behind passed through the
# torso / met behind the back on test1 and the 0.x sample; 30 passes on every model)
# Leaning forward further than the spine does (2026-11-03, floor_front: w_sit / seiza with the hands on the floor in front
# of the knees; "lean 25" spread over spine / chest / upperChest tilted the hips -> shoulder line only ~10 deg and moved
# the shoulders 9 cm): sitting on the floor (PELVIS_POSTURES) lean goes up to 60 (seiza hands on the floor: 50 left them 9-16 cm short) and what is past SPINE_LEAN tips the
# pelvis forward over the thighs (the legs keep their place). Standing and kneeling stay at 25.
SPINE_LEAN = 25.0
# the body of a sitting posture where a keyframe leaves it out (arm_words.resolve_arm_words; 2026-11-11)
SIT_BODY = {"knees_hugged": {"torso": {"lean": 20}, "head": {"nod": 15, "tilt": 0, "turn": 0},
                             "shoulders": {"left": {"raise": 6, "forward": 10}, "right": {"raise": 6, "forward": 10}},
                             "legs": {"knees": "together"}},
            # side sits lean onto the propping hand (away from the feet), the head tilts back up a little
            "ledge": {"torso": {"lean": 6}},
            "side_right": {"torso": {"bend": 7}, "head": {"nod": 0, "tilt": -5, "turn": 0}},      # + = toward her left (the prop)
            "side_left": {"torso": {"bend": -7}, "head": {"nod": 0, "tilt": 5, "turn": 0}}}
# Leaning on a wall (a window's side in the desktop app, 2026-11-11): standing, the shoulder on the wall's side against
# it, the weight on the other foot so the body slants into the wall, the head resting toward it, arms folded. `wall` in
# the request = the side of HER body the wall is on; these fill in what a standing keyframe leaves out.
WALL_BODY = {
    # 2026-11-11, the author: the first version leaned AWAY (weight on the far foot, hip pushed out). The weight goes onto
    # the wall: the hips over the wall-side foot, the torso bent on toward the wall so feet -> hips -> shoulder slant
    # into it and the shoulder / upper arm lie on it; the far foot crossed in front on its toes; head nearly upright.
    # Arms folded (the author 2026-11-11: hanging arms lost the lean she liked); the wall plane comes from the upper arms
    # and hips only (skeleton.WALL_PTS), forearms / hands are pushed back from it
    # 2026-11-11 (the author: the upper body needs something on the wall): bend 15 (was 9) so the wall-side shoulder and
    # upper arm are the outermost point and lie on the wall; the head stays nearly upright. SIGNS (measured by FK and on
    # render pixels the same day): bend + and tilt + are toward her LEFT (+x); the prompt said so until 2026-11-10, when
    # Claude 'corrected' it to RIGHT from a misread render; rounds 1-4 of the lean therefore bent AWAY from the wall.
    "right": {"torso": {"bend": -15}, "head": {"nod": 2, "tilt": 9, "turn": 0}, "shoulders": {"right": 0, "left": 0},
              "legs": {"weight": "right", "left_foot": {"place": "cross", "lift": "heel"}}, "arms_cross": {"how": "fold", "top": "left"}},
    "left": {"torso": {"bend": 15}, "head": {"nod": 2, "tilt": -9, "turn": 0}, "shoulders": {"left": 0, "right": 0},
             "legs": {"weight": "left", "right_foot": {"place": "cross", "lift": "heel"}}, "arms_cross": {"how": "fold", "top": "right"}},
}
PELVIS_POSTURES = ("cross_legged", "w_sit", "side_right", "side_left", "knees_hugged", "legs_out", "seiza", "ledge")


def lean_max(posture):
    return TORSO_LIM["lean"][1] if posture in PELVIS_POSTURES else SPINE_LEAN


SHOULDER_LIM = (-10, 25)
SHOULDER_FWD_LIM = (-15, 20)
WRIST_LIM = (-70, 70)      # back of the hand / palm side; a hand flat on the floor needs ~70 back (2026-10-28)
WSIDE_LIM = (-45, 45)      # wrist sideways (ulnar / radial deviation), + = toward the little finger (2026-11-08; 30 -> 45 on 2026-11-10: an index upright on the lips needs ~45 with the forearm slanting up from the side)
# Palm words -> direction in the torso frame. "in"/"out" depend on the side: see solve_palm().
PALMS = ("natural", "forward", "back", "up", "down", "in", "out")
_PALM_DIR = {"forward": (0, 0, 1), "back": (0, 0, -1), "up": (0, 1, 0), "down": (0, -1, 0)}
# Forearm twist about the lower arm's local x, 0 = as in the T-pose (palm down, a real forearm's middle
# position). A real forearm turns about 90 deg each way and the shoulder does the rest; here the upper
# arm is already fixed by the arm directions, so the twist is allowed further and split between the
# lower arm (up to TWIST_LOWER) and the hand joint (the rest).
MAX_TWIST, TWIST_LOWER = np.deg2rad(160), np.deg2rad(100)
MAX_ELBOW = np.deg2rad(145)

# Standing legs, all words. Distances are for the default model, in metres.
POSTURES = ("stand",) + sk.SIT_NAMES
WEIGHTS = ("center", "left", "right")
HEIGHTS = ("straight", "bent", "crouch", "squat")
STANCES = ("normal", "together", "wide")
PLACES = ("home", "forward", "back", "out", "lunge_forward", "lunge_out", "cross", "forward_out")
LIFTS = ("down", "lift", "knee_up", "kick", "heel", "knee_to_chest")   # heel: the heel up, the ball of the foot stays on the floor
LIFT_DIRS = ("forward", "side", "back")       # which way a lifted leg goes
# Which way the toes point (2026-10-29, T015: the free foot turned out ~45 deg); the whole leg turns with the foot.
FOOT_TURNS = ("forward", "out", "in")
FOOT_TURN_DEG = {"forward": 0.0, "out": 35.0, "in": -20.0}
TOE_LIM = (-30.0, 60.0)
WEIGHT_X = {"center": 0.0, "left": 0.10, "right": -0.10}
HEIGHT_DROP = {"straight": 0.0, "bent": 0.07, "crouch": 0.16, "squat": 0.30}   # hips down
HEIGHT_BACK = {"straight": 0.0, "bent": 0.0, "crouch": -0.04, "squat": -0.10}   # hips back (balance)
# On tiptoe the planted heels rise this much and the foot tilts so the ball of the foot stays on the floor
# (the author, 2026-10-23: "rise onto your toes" had no word, so Haiku lifted one foot instead).
TIPTOE_UP = 0.045
STANCE_X = {"together": 0.05, "normal": 0.10, "wide": 0.22}                     # ankle |x|
PLACE_OFF = {"home": (0.0, 0.0), "forward": (0.0, 0.30), "back": (0.0, -0.25), "out": (0.16, 0.0),  # (outward, forward)
             "lunge_forward": (0.0, 0.55), "lunge_out": (0.36, 0.0),
             "cross": (-0.10, 0.10), "forward_out": (0.13, 0.30)}   # forward_out: a step diagonally (2026-10-29, T020)   # in front, at the middle (2026-10-29, S001: the free foot crossed in front)   # a big step: the hips drop between the feet
LUNGE_DROP = 0.14
# Hip sway / contrapposto (2026-10-28, pose group A): "hip" tilts the pelvis down toward the free leg, pushes the
# hips further out over the weight foot, turns the free knee in, and tilts the torso back the other way so the
# shoulder line slopes opposite to the pelvis (an S-curve). Needs weight left or right.
HIPS = ("level", "out", "far_out")
HIP_ROLL = {"level": 0.0, "out": 8.0, "far_out": 14.0}          # pelvis tilt, degrees
HIP_OUT_X = {"level": 0.0, "out": 0.03, "far_out": 0.05}        # extra hips shift over the weight foot, m
HIP_FOOT_OUT = {"level": 0.0, "out": 0.04, "far_out": 0.06}   # the weight foot stands this much further out, m
# (the author chose this over pushing the hips less: the weight leg slanted ~10 deg, now ~6)
HIP_ROLL_MAX = 20         # degrees: a free foot placed far out may tilt the pelvis this far (_stand_targets)
HIP_COUNTER = 0.5          # the shoulder line tilts this fraction of the pelvis tilt the other way
# Kneeling (skeleton.kneel_sway): the hips shift sideways until the pelvis tilts HIP_KNEEL_DEG or the shift reaches
# HIP_KNEEL_X. With the knees apart a small shift already tilts the pelvis a lot (10 cm -> 28 deg, the upper body
# fell over), with the knees together 10 cm only tilts it 4.
HIP_KNEEL_X = {"level": 0.0, "out": 0.06, "far_out": 0.10}
HIP_KNEEL_DEG = {"level": 0.0, "out": 5.0, "far_out": 9.0}
HIP_KNEE_IN = 20.0         # the free knee turns in this much (degrees) at far_out, scaled by the tilt
# Lifted foot, per direction: ankle offset from its floor position (up, outward, forward) and toe
# direction (outward, up, forward). Side and back added 2026-10-23 (HumanML3D has side / back kicks and
# leg lifts; the author agreed to fill the leg vocabulary from its texts).
LIFT_OFF = {
    "forward": {"down": (0.0, 0.0, 0.0, sk.FLAT), "lift": (0.10, 0.0, 0.02, sk.FLAT),
                "knee_up": (0.30, 0.0, 0.12, (0.0, -0.8, 0.6)), "kick": (0.32, 0.0, 0.50, (0.0, 0.45, 0.9)),
                # the knee pulled up to the chest (S005, held with the hands): the foot high under the knee
                "knee_to_chest": (0.56, 0.0, 0.14, (0.0, -0.85, 0.5))},
    "side":    {"down": (0.0, 0.0, 0.0, sk.FLAT), "lift": (0.08, 0.08, 0.0, sk.FLAT),
                "knee_up": (0.30, 0.18, 0.10, (0.3, -0.7, 0.6)), "kick": (0.28, 0.40, 0.0, (0.5, 0.2, 0.8))},
    "back":    {"down": (0.0, 0.0, 0.0, sk.FLAT), "lift": (0.16, 0.0, -0.12, (0.0, -0.6, -0.8)),
                "knee_up": (0.32, 0.0, -0.16, (0.0, -0.2, -1.0)), "kick": (0.26, 0.0, -0.42, (0.0, -0.5, -0.85))},
}


@sk.on_model_change
def _foot_constants():
    """Constants that depend on the active model's foot (skeleton.use_model)."""
    global TIPTOE_DIR, FOOT_Z
    _s = (TIPTOE_UP - sk.TOE[1]) / np.linalg.norm(sk.TOE)
    TIPTOE_DIR = np.array([0.0, -_s, np.sqrt(1 - _s * _s)])
    FOOT_Z = float((sk.HIPS_REST + sk.HIP_OFFSET + sk.THIGH + sk.SHIN)[2])   # ankle z at rest
    for dir_, lift in (("forward", "down"), ("forward", "lift"), ("side", "down"), ("side", "lift"), ("back", "down")):
        LIFT_OFF[dir_][lift] = LIFT_OFF[dir_][lift][:3] + (sk.FLAT,)      # the flat-foot entries


TURN_LIM = (-180, 180)     # whole body (was +-90; 2026-11-04 the author: Claude may turn her round), degrees, + = toward its left; the feet turn with it
PIVOT = 0.06               # a planted foot that moves less than this pivots on the floor instead of stepping
STEP_ARC = 0.07            # a planted foot that moves to a new place is lifted this high on the way
REACH = 0.997              # planted legs are at most this fraction of their full length: knee ~9 deg (0.99 was 16 deg, the author's references stand straight; 2026-10-29: 2 of 265 round0 standing motions then pass the 10 rad/s leg limit by <= 0.6)
# Row layout: one keyframe = one row of N_PARAMS floats (parse_keyframes fills it, keyframes_to_motion reads it).
# Blocks in order, sides always RIGHT then LEFT (2026-11-08: derived from this table instead of hand-set numbers;
# policy_kf's checkpoints assume the old layout and are frozen):
#   HEAD nod, turn, tilt | TORSO lean, turn, bend | ARM right upper xyz, right fore xyz, left upper, left fore |
#   SHR shoulder raise | HAND hand shape index into HAND_SHAPES | PALM index into PALMS (-1 = PALMV) | WRIST bend deg |
#   LEG weight, height, stance, left place, left lift, right place, right lift | TURN body turn deg, lift dir left, right |
#   SIT sitting legs: indices into skeleton.SIT_LEGS (left, right; -1 = LEGDIR) and SIT_KNEES (0 = the posture's own) |
#   HIP hip sway, degrees of pelvis tilt (words out / far_out = HIP_ROLL) | TOE toes turned out deg (left, right) |
#   PALMV exact palm directions (torso frame, right xyz, left xyz): touch words give the palm as a vector |
#   SHF shoulders forward (+) / back deg | LEGDIR sitting legs as directions: thigh xyz, shin xyz, left then right |
#   JUMP size index into JUMPS (0 = none), count, pose, then per arm (right, left) 1 = not written at this keyframe |
#   KNEES standing knees: 0 normal, 1 in (knock-kneed, feet apart) | WSIDE wrist sideways bend deg (+ = toward the
#   little finger; 2026-11-08: a hand leaning off its forearm, as a hand swung out with the palm forward)
_LAYOUT = (("HEAD", 3), ("TORSO", 3), ("ARM", 12), ("SHR", 2), ("HAND", 2), ("PALM", 2), ("WRIST", 2), ("LEG", 7),
           ("TURN", 3), ("SIT", 3), ("HIP", 1), ("TOE", 2), ("PALMV", 6), ("SHF", 2), ("LEGDIR", 12), ("JUMP", 5), ("KNEES", 1),
           ("WSIDE", 2))
COL = {}
for _name, _width in _LAYOUT:
    COL[_name] = sum(w for _, w in _LAYOUT[:len(COL)])
N_PARAMS = sum(w for _, w in _LAYOUT)
HEAD_AT, TORSO_AT, ARM_AT, SHR_AT, HAND_AT, PALM_AT, WRIST_AT, WSIDE_AT = (COL[n] for n in ("HEAD", "TORSO", "ARM", "SHR", "HAND", "PALM", "WRIST", "WSIDE"))
LEG_AT, TURN_AT, SIT_AT, HIP_AT, TOE_AT, PALMV_AT, SHF_AT, LEGDIR_AT, JUMP_AT, KNEES_AT = \
    (COL[n] for n in ("LEG", "TURN", "SIT", "HIP", "TOE", "PALMV", "SHF", "LEGDIR", "JUMP", "KNEES"))
LEG_COLS = [c for n, w in _LAYOUT if n in ("LEG", "TURN", "SIT", "HIP", "TOE", "LEGDIR", "JUMP", "KNEES") for c in range(COL[n], COL[n] + w)]
KNEES_IN_DROP = 0.02       # m the hips go down so the knees can bend and come together
KNEES_IN_DEG = 55.0        # deg each knee turns in (the IK pole)
STAND_KNEES = ("normal", "in")
SIT_LIST = ", ".join(f'"{k}" ({SIT_DESC[k]})' for k in sk.SIT_NAMES)


# Sitting on the floor, hanging arms would go through the floor: the rest pose has the hands on the thighs, one
# pose per posture (upper, fore, palm word). Until 2026-10-23 one shared pose left the palms 5-29 cm above the
# thighs (the author saw it in seiza). Found by ~/hsin_runs/agent_exp/sit_rest_search.py (palm ~1 cm above its own
# thigh, no clipping); kneel = hands hanging beside the thighs (variant D of kneel_rest.py, the author's pick: the
# arms are too short to reach the upright thighs, and closer to the hips the hands went into the body).
REST_SIT = {
    # cross_legged hands moved toward the knees (2026-10-23): the old ones put the left fingertips into the shorts,
    # which flare up when sitting (~/hsin_runs/agent_exp/cross_rest_cloth.py)
    "cross_legged": {"right": ([-0.54, -0.834, -0.113], [-0.079, -0.577, 0.813], "natural"), "left": ([0.539, -0.835, -0.109], [0.081, -0.58, 0.81], "natural")},
    "w_sit": {"right": ([-0.162, -0.983, 0.089], [0.326, -0.617, 0.717], "down"), "left": ([0.162, -0.983, 0.089], [-0.326, -0.617, 0.717], "down")},
    # seiza wrists 1 cm higher (2026-10-23, clothed thigh check: fingertips were 0.1 cm into the shorts)
    "seiza": {"right": ([-0.016, -0.956, 0.294], [0.106, -0.841, 0.531], "natural"), "left": ([0.016, -0.956, 0.294], [-0.106, -0.841, 0.531], "natural")},
    "side_right": {"right": ([-0.529, -0.845, 0.082], [0.223, -0.691, 0.687], "down"), "left": ([0.324, -0.919, 0.224], [-0.406, -0.602, 0.688], "natural")},
    "side_left": {"right": ([-0.324, -0.919, 0.224], [0.406, -0.602, 0.688], "natural"), "left": ([0.529, -0.845, 0.082], [-0.223, -0.691, 0.687], "down")},
    "legs_out": {"right": ([-0.263, -0.963, -0.059], [0.389, -0.481, 0.786], "natural"), "left": ([0.263, -0.963, -0.059], [-0.389, -0.481, 0.786], "natural")},
    "ledge": {"right": ([-0.263, -0.963, -0.059], [0.389, -0.481, 0.786], "natural"), "left": ([0.263, -0.963, -0.059], [-0.389, -0.481, 0.786], "natural")},   # only a fallback row: the ledge rest is a touch (arm_words.rest_arm)
    "kneel": {"right": ([-0.264, -0.964, 0.031], [-0.2, -0.971, 0.127], "in"), "left": ([0.264, -0.964, 0.031], [0.2, -0.971, 0.127], "in")},
}


def _rest_sit_text():
    f = lambda v: "[" + ", ".join(f"{x:.2f}" for x in v) + "]"
    lines = []
    for post, arms in REST_SIT.items():
        if post in ("side_right", "side_left", "seiza", "ledge"):      # touch rests since 2026-11-11 (described in words)
            continue
        parts = [f"{side} upper {f(u)}, fore {f(fo)}" + ("" if palm == "natural" else f", palm {palm}")
                 for side, (u, fo, palm) in arms.items()]
        lines.append(f"  {post}: " + "; ".join(parts))
    return "\n".join(lines)


PROMPT = PROMPT.replace("<<REST_SIT>>", _rest_sit_text())


# Hugging the knees: arms around the knees from outside, not crossed, each palm on its own shin. Found by a
# search with skeleton.hand_body_depth / hands_overlap (2026-10-23). Earlier poses: upper arms down beside the
# knees put the forearms through the shins; crossed forearms meet near the elbows with this model's short
# arms, so one hand had to hang 8 cm lower (the author: the left hand looked odd).
REST_ARM_HUG = {"right": ([-0.418, -0.357, 0.835], [0.284, -0.524, 0.803]),     # elbows beside the knees, each palm on
                "left": ([0.418, -0.357, 0.835], [-0.284, -0.524, 0.803])}       # its own shin below the knee (the author, 2026-10-23)


REST_HAND_HUG = {"right": ("back", 15.0), "left": ("back", 15.0)}   # palm word, wrist bend (deg)


def _rest_row(posture="stand"):
    arm = REST_ARM if posture == "stand" else REST_ARM_HUG if posture == "knees_hugged" else \
        {side: REST_SIT[posture][side][:2] for side in ("right", "left")}
    row = np.zeros(N_PARAMS)
    for i, side in enumerate(("right", "left")):
        row[ARM_AT + 6 * i:ARM_AT + 6 * i + 3], row[ARM_AT + 6 * i + 3:ARM_AT + 6 * i + 6] = arm[side]
        if posture == "knees_hugged":
            row[PALM_AT + i], row[WRIST_AT + i] = PALMS.index(REST_HAND_HUG[side][0]), REST_HAND_HUG[side][1]
        elif posture in REST_SIT:
            row[PALM_AT + i] = PALMS.index(REST_SIT[posture][side][2])
    return row


def _unit(v):
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _word(v, options, what):
    if v not in options:
        raise ValueError(f"unknown {what} {v!r}")
    return options.index(v)


MAX_KEYFRAMES = 64        # was 16 = policy_kf.MAX_K (12 before 2026-10-23); hand-built motions read from a clip every 0.1 s need ~40 (2026-11-08); the policy keeps its own MAX_K


def parse_keyframes(obj, duration_s):
    """Validate an LLM reply. Returns (posture, times [K], rows [K, N_PARAMS]) with missing parts filled forward
    (the row layout is _LAYOUT above)."""
    posture = obj.get("posture", "stand") if isinstance(obj, dict) else "stand"
    if posture not in POSTURES:
        raise ValueError(f"unknown posture {posture!r}")
    kfs = obj.get("keyframes") if isinstance(obj, dict) else None
    if not isinstance(kfs, list) or not 2 <= len(kfs) <= MAX_KEYFRAMES:
        raise ValueError(f"need 2..{MAX_KEYFRAMES} keyframes")
    cur = _rest_row(posture)
    times, rows = [], []
    for kf in kfs:
        t = float(kf["t"])
        if times and t <= times[-1]:
            raise ValueError("keyframe times must increase")
        cur = cur.copy()
        for part, lims, at in (("head", HEAD_LIM, HEAD_AT), ("torso", TORSO_LIM, TORSO_AT)):
            for i, (name, (lo, hi)) in enumerate(lims.items()):
                if name in kf.get(part, {}):
                    if part == "torso" and name == "lean":
                        hi = lean_max(posture)
                    cur[at + i] = np.clip(float(kf[part][name]), lo, hi)
        for i, part in enumerate(("right_arm", "left_arm")):
            if "palm" in kf.get(part, {}):
                pv = kf[part]["palm"]
                if isinstance(pv, (list, tuple)) and len(pv) == 3:
                    cur[PALM_AT + i] = -1
                    cur[PALMV_AT + 3 * i:PALMV_AT + 3 * i + 3] = np.asarray(pv, float)
                elif pv not in PALMS:
                    raise ValueError(f"unknown palm direction {pv!r}")
                else:
                    cur[PALM_AT + i] = PALMS.index(pv)
            if "wrist" in kf.get(part, {}):
                cur[WRIST_AT + i] = np.clip(float(kf[part]["wrist"]), *WRIST_LIM)
            if "wrist_side" in kf.get(part, {}):
                cur[WSIDE_AT + i] = np.clip(float(kf[part]["wrist_side"]), *WSIDE_LIM)
        for part, at in (("right_arm", ARM_AT), ("left_arm", ARM_AT + 6)):
            for i, name in enumerate(("upper", "fore")):
                if name in kf.get(part, {}):
                    v = np.asarray(kf[part][name], float)
                    if v.shape != (3,) or not np.isfinite(v).all() or np.linalg.norm(v) < 1e-3:
                        raise ValueError(f"bad direction for {part}.{name}")
                    cur[at + 3 * i: at + 3 * i + 3] = _unit(v)
        for i, name in enumerate(("right", "left")):
            if name in kf.get("shoulders", {}):
                sv = kf["shoulders"][name]          # {"raise": deg, "forward": deg}; a bare number = {"raise": n}
                if isinstance(sv, dict):
                    if "raise" in sv:
                        cur[SHR_AT + i] = np.clip(float(sv["raise"]), *SHOULDER_LIM)
                    if "forward" in sv:
                        cur[SHF_AT + i] = np.clip(float(sv["forward"]), *SHOULDER_FWD_LIM)
                else:
                    cur[SHR_AT + i] = np.clip(float(sv), *SHOULDER_LIM)
        for i, name in enumerate(("right", "left")):
            if name in kf.get("hands", {}):
                if kf["hands"][name] not in HAND_SHAPES:
                    raise ValueError(f"unknown hand shape {kf['hands'][name]!r}")
                cur[HAND_AT + i] = HAND_SHAPES.index(kf["hands"][name])
        legs = kf.get("legs", {})
        if legs.get("height") == "tiptoe":
            raise ValueError('"tiptoe" is gone: give the feet that stay down "lift": "heel" (both = up on the toes)')
        for i, (name, options) in enumerate((("weight", WEIGHTS), ("height", HEIGHTS), ("stance", STANCES))):
            if name in legs:
                cur[LEG_AT + i] = _word(legs[name], options, name)
        for i, side in enumerate(("left", "right")):
            foot = legs.get(f"{side}_foot", {})
            if "place" in foot:
                cur[LEG_AT + 3 + 2 * i] = _word(foot["place"], PLACES, "foot place")
            if "lift" in foot:
                cur[LEG_AT + 4 + 2 * i] = _word(foot["lift"], LIFTS, "foot lift")
            if "dir" in foot:
                cur[TURN_AT + 1 + i] = _word(foot["dir"], LIFT_DIRS, "lift direction")
            if "toes" in foot:          # a word or degrees (+ = out); stored as degrees (2026-10-29, T029: 35 was 20 too many)
                tv = foot["toes"]
                cur[TOE_AT + i] = np.clip(float(tv), *TOE_LIM) if isinstance(tv, (int, float)) else \
                    FOOT_TURN_DEG[FOOT_TURNS[_word(tv, FOOT_TURNS, "toes")]]
        if "turn" in legs:
            cur[TURN_AT] = np.clip(float(legs["turn"]), *TURN_LIM)
        if "hip" in legs:                    # a word or degrees of pelvis tilt (0..HIP_ROLL_MAX); stored as degrees
            hv = legs["hip"]
            if isinstance(hv, (int, float)) and not isinstance(hv, bool):
                cur[HIP_AT] = float(np.clip(hv, 0.0, HIP_ROLL_MAX))
            else:
                cur[HIP_AT] = HIP_ROLL[HIPS[_word(hv, HIPS, "hip")]]
        if posture == "stand" and "knees" in legs:           # standing knees together, feet apart
            cur[KNEES_AT] = _word(legs["knees"], STAND_KNEES, "standing knees")
        sit_words = {k: legs[k] for k in ("left", "right", "knees") if k in legs and not (posture == "stand" and k == "knees")}
        if sit_words:                        # sitting legs (skeleton.SIT_LEGS / SIT_KNEES)
            if posture == "kneel" and set(sit_words) == {"knees"}:
                pass                         # kneeling: only the knees apart / together (skeleton.kneel_sway)
            elif posture not in sk.BUTT_POSTURES:
                raise ValueError(f"sitting leg words need a posture sitting on its bottom: {sk.BUTT_POSTURES}")
            for i, side in enumerate(("left", "right")):
                if side in sit_words:
                    sw = sit_words[side]
                    if isinstance(sw, dict):           # {"thigh": [x, y, z], "shin": [x, y, z]}
                        cur[SIT_AT + i] = -1
                        for j, part in enumerate(("thigh", "shin")):
                            v = np.asarray(sw[part], float)
                            if v.shape != (3,) or np.linalg.norm(v) < 1e-3:
                                raise ValueError(f"bad {side} {part} direction")
                            cur[LEGDIR_AT + 6 * i + 3 * j:LEGDIR_AT + 6 * i + 3 * j + 3] = _unit(v)
                    else:
                        cur[SIT_AT + i] = _word(sw, sk.SIT_LEGS, f"{side} sitting leg")
            if "knees" in sit_words:
                cur[SIT_AT + 2] = _word(sit_words["knees"], sk.SIT_KNEES, "knees")
        cur[JUMP_AT:JUMP_AT + 5] = 0
        if "jump" in legs:
            jv = legs["jump"]
            jv = {"size": jv} if isinstance(jv, str) else jv
            if not isinstance(jv, dict) or jv.get("size", "small") not in JUMPS[1:]:
                raise ValueError('"jump": "small" | "big", or {"size": ..., "count": 1..3, "pose": "default" | "tuck"}')
            if posture != "stand":
                raise ValueError('"jump" is for standing only (it cannot jump while sitting or kneeling)')
            if jv.get("pose", "default") not in JUMP_POSES:
                raise ValueError(f'jump "pose" must be one of {JUMP_POSES}')
            cur[JUMP_AT] = JUMPS.index(jv.get("size", "small"))
            cur[JUMP_AT + 1] = int(np.clip(int(jv.get("count", 1)), 1, JUMP_COUNT_MAX))
            cur[JUMP_AT + 2] = JUMP_POSES.index(jv.get("pose", "default"))
            cur[JUMP_AT + 3] = "right_arm" not in kf
            cur[JUMP_AT + 4] = "left_arm" not in kf
            if any(LIFTS[int(cur[LEG_AT + 4 + 2 * i])] != "down" for i in range(2)):
                raise ValueError('a "jump" keyframe needs both feet down (it takes off from both feet)')
        heel = LIFTS.index("heel")
        if cur[LEG_AT + 4] not in (0, heel) and cur[LEG_AT + 6] not in (0, heel):   # no jumping: one foot stays down
            cur[LEG_AT + 6] = 0
        if cur[HIP_AT] and posture not in ("stand", "kneel"):
            raise ValueError('"hip" (hip sway) is for standing or kneeling')
        if cur[HIP_AT] and WEIGHTS[int(cur[LEG_AT])] == "center":
            raise ValueError('"hip" needs "weight": "left" or "right" (the hip goes out over that side)')
        times.append(t)
        rows.append(cur)
    if times[0] > 1e-6:                      # the motion always starts from rest
        times.insert(0, 0.0)
        rows.insert(0, _rest_row(posture))
    if times[-1] > duration_s + 0.5:
        raise ValueError("keyframes run past the duration")
    for i, r in enumerate(rows):            # a jump takes a fixed time; nothing else may be keyed until it has landed
        if r[JUMP_AT]:
            end = times[i] + jump_length(r)
            if i + 1 < len(times) and times[i + 1] < end - 1e-6:
                raise ValueError(f"the jump at t={times[i]:g} lasts {jump_length(r):.2f} s (until t={end:.2f}): "
                                 f"the next keyframe must be at t >= {end:.2f}")
            if end > duration_s + 0.5:
                raise ValueError(f"the jump at t={times[i]:g} lasts until t={end:.2f}, past the duration")
    return posture, np.array(times), np.array(rows)


LINEAR_SEGMENTS = False    # True: segments blend linearly (keyframes read from a clip every 0.1 s; a smoothstep per segment
# stops the motion at every keyframe = a 10 Hz stutter, 2026-11-08); set by refine.solve from obj["interp"] = "linear"


def _segments(times, t):
    """For each time in t: index k of the keyframe before it and the eased weight w toward k+1."""
    k = np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2)
    u = np.clip((t - times[k]) / (times[k + 1] - times[k]), 0, 1)
    return k, (u if LINEAR_SEGMENTS else u * u * (3 - 2 * u))


def solve_arm(upper, fore, side, prev_hinge):
    """One keyframe of one arm. upper/fore: unit directions in the torso frame.
    Returns (R_upper [3,3] local rotation of the upper arm, elbow angle, hinge axis).

    Rest (T-pose): the right arm points along -x and its elbow hinge is +y. So the rotation must
    send -x to `upper` and +y to the hinge = the normal of the plane holding both directions.
    When the arm is nearly straight that plane is undefined, so we fade to the previous keyframe's hinge."""
    sgn = -1.0 if side == "right" else 1.0                      # rest direction of the arm = sgn * x
    cross = np.cross(upper, fore) * -sgn                         # right: upper x fore, left: fore x upper
    n = np.linalg.norm(cross)
    prev = prev_hinge - upper * np.dot(prev_hinge, upper)        # keep it perpendicular to the arm
    prev = _unit(prev) if np.linalg.norm(prev) > 1e-6 else _unit(np.cross(upper, [0.0, 0.0, 1.0]))
    w = np.clip((n - 0.02) / 0.10, 0, 1)      # below ~7 deg of bend the plane is too noisy to trust
    w = w * w * (3 - 2 * w)
    hinge = _unit(w * cross / max(n, 1e-9) + (1 - w) * prev)
    ex = sgn * upper                                             # image of local +x
    R = np.stack([ex, hinge, np.cross(ex, hinge)], axis=1)
    elbow = min(np.arccos(np.clip(np.dot(upper, fore), -1, 1)), MAX_ELBOW)
    return R, elbow, hinge


def solve_arm_batch(upper, fore, side, prev_hinge):
    """solve_arm for many frames at once: upper / fore / prev_hinge [N, 3] -> (R [N, 3, 3], elbow [N], hinge [N, 3]).
    Same formulas, as numpy over the frames (2026-11-03: depenetrate._resolve called solve_arm frame by frame and was
    ~90 % of a sitting solve)."""
    unit = lambda v: v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)
    sgn = -1.0 if side == "right" else 1.0
    cross = np.cross(upper, fore) * -sgn
    n = np.linalg.norm(cross, axis=-1)
    prev = prev_hinge - upper * (prev_hinge * upper).sum(-1, keepdims=True)
    alt = unit(np.cross(upper, np.array([0.0, 0.0, 1.0])))
    prev = np.where(np.linalg.norm(prev, axis=-1, keepdims=True) > 1e-6, unit(prev), alt)
    w = np.clip((n - 0.02) / 0.10, 0, 1)
    w = (w * w * (3 - 2 * w))[:, None]
    hinge = unit(w * cross / np.maximum(n, 1e-9)[:, None] + (1 - w) * prev)
    ex = sgn * upper
    R = np.stack([ex, hinge, np.cross(ex, hinge)], axis=-1)
    elbow = np.minimum(np.arccos(np.clip((upper * fore).sum(-1), -1, 1)), MAX_ELBOW)
    return R, elbow, hinge


TWIST_RATE = 8.0    # rad/s: the fastest the whole palm twist may turn between two keyframes to reach the exact palm on
                    # another branch (a floor hand from a palm on the thigh: 250 deg in 1.5 s); faster = stay, clamped


def solve_palm(R_upper, elbow, side, palm, prev_twist, dt=None):
    """Twist about the forearm (radians, about the lower arm's local x) that turns the palm toward the
    named direction. In the T-pose the palm faces down (-y) and the forearm lies along x.
    If the wanted direction is nearly along the forearm no twist can reach it: keep the previous twist.
    dt: seconds since the previous keyframe (None = no limit on the turn)."""
    sgn = -1.0 if side == "right" else 1.0
    if isinstance(palm, str):
        if palm == "natural":
            return 0.0
        want = np.array(_PALM_DIR[palm] if palm in _PALM_DIR else (-sgn if palm == "in" else sgn, 0, 0), float)
    else:                                       # an exact direction (from a touch word)
        want = np.asarray(palm, float)
    C = R_upper @ euler_to_mat(np.array([0.0, -sgn * elbow, 0.0]))      # torso -> forearm frame
    axis, now = C @ np.array([1.0, 0, 0]), C @ np.array([0, -1.0, 0])
    want = want - axis * np.dot(want, axis)
    if np.linalg.norm(want) < 0.35:
        return prev_twist
    twist = np.arctan2(np.dot(axis, np.cross(now, want)), np.dot(now, want))
    # The branch nearest the previous keyframe's twist, THEN the clamp (2026-11-10): clamping each keyframe on its own
    # sent a hand needing +170 then +190 (= -170) to +160 then -160, a 320 deg spin in 0.1 s (the cool hair sweep, where
    # the touch's palm vector is carried into the arm's way down). A hand pronated to its limit now stays there.
    # Branches (2026-11-10): clamping the raw angle on its own sent a hand needing +170 then +190 (= -170) to +160 then
    # -160, a 320 deg spin in 0.1 s (the cool hair sweep, where the touch's palm vector is carried into the arm's way
    # down); now that hand stays pronated at its limit, the palm a little off. A slow turn the long way round is still
    # allowed when it gives the exact palm (a floor hand reached from a palm on the thigh: 250 deg over 1.5 s).
    # Each branch clamped to the range; of those the hand can turn to in time (TWIST_RATE), the one whose palm is
    # nearest the wanted direction (least clamping); if none is slow enough, the smallest turn.
    cands = twist + 2 * np.pi * np.array([-1.0, 0.0, 1.0])
    clamped = np.clip(cands, -MAX_TWIST, MAX_TWIST)
    err, jump = np.abs(cands - clamped), np.abs(clamped - prev_twist)
    ok = jump <= TWIST_RATE * dt if dt is not None else np.ones(3, bool)
    i = np.argmin(np.where(ok, err, np.inf)) if ok.any() else np.argmin(jump)
    return float(clamped[i])


def rz(a):
    """Rotation about z (+ = the top toward its right, the left hip up)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def hip_deg(row):
    """Pelvis tilt of a hip sway in degrees."""
    return float(row[HIP_AT])


def hip_amount(table, row):
    """A hip-sway quantity given per word (level / out / far_out), at this keyframe's tilt (linear in the degrees)."""
    return float(np.interp(hip_deg(row), [HIP_ROLL[h] for h in HIPS], [table[h] for h in HIPS],
                           right=table["far_out"] * hip_deg(row) / HIP_ROLL["far_out"]))


def hip_roll(row):
    """Pelvis tilt of one keyframe (radians): + = the left hip up (weight on the left foot)."""
    a = np.deg2rad(hip_deg(row))
    return a * np.sign(WEIGHT_X[WEIGHTS[int(row[LEG_AT])]])


def foot_turn(row, side):
    """Toe direction of one standing foot about the vertical (radians, + = toward its left)."""
    return sk.side_sign(side) * np.deg2rad(row[TOE_AT + (side == "right")])


def kneel_dx(row):
    """Kneeling hip sway of one keyframe: sideways hips shift (m), + = toward its left (see HIP_KNEEL_DEG)."""
    knees = sk.SIT_KNEES[int(row[SIT_AT + 2])]
    sg = np.sign(WEIGHT_X[WEIGHTS[int(row[LEG_AT])]])
    hi, cap = hip_amount(HIP_KNEEL_X, row), np.deg2rad(hip_amount(HIP_KNEEL_DEG, row))
    if not hi or abs(sk.kneel_sway(hi, knees)[2]) <= cap:
        return sg * hi
    lo = 0.0
    for _ in range(20):                       # bisection: the shift where the tilt reaches the cap
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if abs(sk.kneel_sway(mid, knees)[2]) < cap else (lo, mid)
    return sg * lo


def pelvis_rolls(posture, rows):
    """Pelvis tilt per keyframe (radians, + = left hip up) of a hip sway; 0 for other postures."""
    if posture == "stand":
        return np.array([_stand_targets(r)[3] for r in rows])
    if posture == "kneel":
        return np.array([sk.kneel_sway(kneel_dx(r), sk.SIT_KNEES[int(r[SIT_AT + 2])])[2] for r in rows])
    return np.zeros(len(rows))


def _stand_targets(row):
    """One keyframe's leg words -> hips position, and per foot (ankle target, toe direction, planted?),
    and the body turn (radians). Everything is laid out facing forward, then turned about the vertical
    axis through the point between the feet."""
    weight, height, stance = WEIGHTS[int(row[LEG_AT])], HEIGHTS[int(row[LEG_AT + 1])], STANCES[int(row[LEG_AT + 2])]
    feet, floor = {}, {}
    for i, side in enumerate(("left", "right")):
        place, lift = PLACES[int(row[LEG_AT + 3 + 2 * i])], LIFTS[int(row[LEG_AT + 4 + 2 * i])]
        sg = sk.side_sign(side)
        out, fwd = PLACE_OFF[place]
        if WEIGHT_X[weight] * sg > 0:            # the weight foot of a hip sway stands a little further out
            out += hip_amount(HIP_FOOT_OUT, row)
        floor[side] = np.array([sg * (STANCE_X[stance] + out), sk.ANKLE_REST_Y, FOOT_Z + fwd])
        up, aside, ahead, toe = LIFT_OFF[LIFT_DIRS[int(row[TURN_AT + 1 + i])]].get(lift, LIFT_OFF["forward"].get(lift, LIFT_OFF["forward"]["down"]))
        toe = np.array([sg * toe[0], toe[1], toe[2]], float)
        if lift == "heel":
            up, aside, ahead, toe = TIPTOE_UP, 0.0, 0.0, TIPTOE_DIR
        toe = sk.ry(foot_turn(row, side)) @ toe
        feet[side] = (floor[side] + np.array([sg * aside, up, ahead]), toe, lift in ("down", "heel"))
    planted = [s for s in feet if feet[s][2]]
    hips = sk.HIPS_REST.copy()
    if len(planted) == 1:
        hips[0] = 0.85 * floor[planted[0]][0]
    else:       # a wide step to one side (lunge_out): the hips go well over the foot that carries the weight
        w = {"left": "left", "right": "right"}.get(weight)
        hips[0] = WEIGHT_X[weight] if w is None else np.sign(WEIGHT_X[weight]) * max(abs(WEIGHT_X[weight]), 0.75 * abs(floor[w][0]))
        hips[0] += np.sign(WEIGHT_X[weight]) * hip_amount(HIP_OUT_X, row)
    hips[1] -= HEIGHT_DROP[height] + KNEES_IN_DROP * row[KNEES_AT]
    if all(LIFTS[int(row[LEG_AT + 4 + 2 * i])] == "heel" for i, s in enumerate(("left", "right")) if s in planted):
        hips[1] += TIPTOE_UP            # every foot that is down is on its toes: the body rises (as "tiptoe" did)
    if any(PLACES[int(row[LEG_AT + 3 + 2 * i])].startswith("lunge") for i, s in enumerate(("left", "right")) if s in planted):
        hips[1] -= LUNGE_DROP           # the reach limit alone leaves the legs nearly straight
    wfoot = {"left": "left", "right": "right"}.get(weight)
    over = [wfoot] if wfoot in planted else planted     # the hips stand over the weight foot (2026-10-29, T020: a free
    # foot stepped forward on its toes pulled the hips 8 cm forward)
    hips[2] += HEIGHT_BACK[height] + 0.5 * np.mean([floor[s][2] + 0.0324 for s in over])
    # Lower the hips until every planted foot is within reach (with the knee a little bent: a straight
    # knee is a singular point of the IK, where tiny hip moves swing the knee fast). Done per keyframe,
    # so the interpolated hips path stays smooth.
    L_max = (sk.L_THIGH + sk.L_SHIN) * REACH
    roll = hip_roll(row)

    def reach_y(hips, s, roll):             # highest hips y at which foot s is within reach
        hj = sk.hip_joint(hips, s, rz(roll))
        flat = np.sum((feet[s][0] - hj)[[0, 2]] ** 2)
        return feet[s][0][1] + np.sqrt(max(L_max ** 2 - flat, 0.0)) - (hj - hips)[1], flat <= L_max ** 2

    w = {"left": "left", "right": "right"}.get(weight)
    if w and len(planted) == 2:
        # Weight on one foot (2026-10-29, T015): the weight leg sets the height and stays straight, as in a real
        # contrapposto; lowering the hips for a free foot placed far out had bent the weight knee 44 deg with the
        # free leg straight. A free foot out of reach first drops the free hip (a larger pelvis tilt, only with a
        # hip word, up to HIP_ROLL_MAX), then comes in toward its hip.
        free = "right" if w == "left" else "left"
        top, sg = hips[1], np.sign(roll) if roll else 0.0
        cands = [roll] + ([sg * np.deg2rad(a) for a in range(int(abs(np.degrees(roll))) + 1, HIP_ROLL_MAX + 1)] if sg else [])
        for roll in cands:
            hips[1] = min(top, reach_y(hips, w, roll)[0])
            if reach_y(hips, free, roll)[0] >= hips[1]:
                break
        hj = sk.hip_joint(hips, free, rz(roll))
        a = feet[free][0].copy()
        for _ in range(30):                     # pull the free foot in until it is within reach
            if np.linalg.norm(a - hj) <= L_max:
                break
            a[[0, 2]] = hj[[0, 2]] + 0.95 * (a - hj)[[0, 2]]
        feet[free] = (a, feet[free][1], feet[free][2])
    else:
        for s in planted:
            hips[1] = min(hips[1], reach_y(hips, s, roll)[0])
    yaw = np.deg2rad(row[TURN_AT])
    if yaw:
        R = sk.ry(yaw)
        c = np.array([0.0, 0.0, FOOT_Z])          # turn about the point between the feet at rest
        hips = c + R @ (hips - c)
        feet = {s: (c + R @ (a - c), R @ t, p) for s, (a, t, p) in feet.items()}
    return hips, feet, yaw, roll


def _leg_dip(R, hips, side):
    """How far (m) one sitting leg's lowest point (knee, ankle or toes, minus its radius) is under the floor, and
    that point."""
    P = sk.leg_points(R, hips)
    best, at = 0.0, None
    for p, r in (("Knee", sk.R_KNEE), ("Ankle", sk.R_ANKLE), ("Toes", sk.R_TOE)):
        d = r - P[f"{side}{p}"][1]
        if d > best:
            best, at = d, P[f"{side}{p}"]
    return best, at


TURN_STEP = 90.0           # deg: a standing turn bigger than this between two keyframes (after the pivots below) is stepped round in parts
# (60 made the knees 10.5 rad/s; one 135 step crossed the legs)
# Turning on the balls of the feet (2026-11-05, the author: the steps of a 180 turn were too busy, 2 parts x 2 feet in
# 1.5 s): in a segment where both feet are down at both ends, both heels lift and the body turns up to PIVOT_DEG about
# the feet, which stay where they are (only the toes turn), in the first PIVOT_FRAC of the segment; then one step per
# foot (up to TURN_STEP); what is left after that (up to PIVOT_DEG more) is pivoted again in the last PIVOT_END_FRAC
# with the heels coming down at the end. 180 = pivot 45 + step 90 + pivot 45. A turn of at most PIVOT_DEG is all
# pivot (heels down at the end, the feet do not step).
PIVOT_DEG, PIVOT_FRAC, PIVOT_END_FRAC, PIVOT_SMALL_FRAC = 45.0, 0.25, 0.2, 0.7
TURN_LEAD = True


def _pivot_turns(times, rows):
    """Insert the pivot keyframes (see PIVOT_DEG). Returns times, rows and per keyframe whether its feet stay where
    the previous keyframe left them (hold)."""
    heel = LIFTS.index("heel")
    down = lambda r: all(LIFTS[int(r[LEG_AT + 4 + 2 * i])] in ("down", "heel") for i in range(2))
    t2, r2, hold = [times[0]], [rows[0]], [False]
    for i in range(len(times) - 1):
        a, b = rows[i], rows[i + 1]
        d = b[TURN_AT] - a[TURN_AT]
        if abs(d) < 2.0 or not (down(a) and down(b)):
            t2.append(times[i + 1]); r2.append(b); hold.append(False); continue
        small, sg, dt = abs(d) <= PIVOT_DEG, np.sign(d), times[i + 1] - times[i]
        piv = a.copy()
        piv[TURN_AT] = a[TURN_AT] + (d if small else sg * PIVOT_DEG)
        piv[LEG_AT + 4] = piv[LEG_AT + 6] = heel
        t2.append(times[i] + dt * (PIVOT_SMALL_FRAC if small else PIVOT_FRAC)); r2.append(piv); hold.append(True)
        rest = abs(d) - PIVOT_DEG
        if not small and rest > TURN_STEP:     # step TURN_STEP, then pivot the rest on the balls
            st = b.copy()
            st[TURN_AT] = a[TURN_AT] + sg * (PIVOT_DEG + TURN_STEP)
            st[LEG_AT + 4] = st[LEG_AT + 6] = heel
            t2.append(times[i + 1] - dt * PIVOT_END_FRAC); r2.append(st); hold.append(False)
            t2.append(times[i + 1]); r2.append(b); hold.append(True)
        else:
            t2.append(times[i + 1]); r2.append(b); hold.append(small)
    return np.array(t2), np.array(r2), np.array(hold)


def _split_turns(times, rows):
    """Keyframes for the legs only, with every turn of more than TURN_STEP split into equal parts in time (the
    other leg words change at the last part)."""
    t2, r2 = [times[0]], [rows[0]]
    for i in range(len(times) - 1):
        d = rows[i + 1, TURN_AT] - rows[i, TURN_AT]
        n = int(np.ceil(abs(d) / TURN_STEP - 1e-9)) if abs(d) > TURN_STEP else 1
        for j in range(1, n + 1):
            r = rows[i + 1]
            if j < n:
                r = rows[i].copy()
                r[TURN_AT] = rows[i, TURN_AT] + d * j / n
            t2.append(times[i] + (times[i + 1] - times[i]) * j / n)
            r2.append(r)
    return np.array(t2), np.array(r2)


# Jumping in place (2026-11-05, the author: standing only, straight up, no turn; Claude writes only "jump" and the solver
# adds the phases): crouch, push off (the heels come up, the body reaches its take-off speed), flight (the hips follow
# free fall exactly, so the filter can tell a jump from floating: filters.check_legs), landing on the balls of the feet
# (absorbed in the knees), stand up. Height is set by physics: take-off speed sqrt(2 g H). count > 1 = bouncing:
# the landing goes straight into the next push. The arms swing back in the crouch and forward on the push unless the
# jump keyframe gives them. The feet land where they took off.
JUMPS = ("none", "small", "big")
JUMP_POSES = ("default", "tuck")
JUMP_COUNT_MAX = 3
G = 9.81
JUMP_EXT = 0.04            # m: the hips at take-off above standing (legs straight, heels up)
JUMP = {   # H flight height (m), crouch / land depth (m), phase times (s), foot lift in the air (default / tuck), lean, arm swing
    "small": dict(H=0.10, crouch=0.08, land=0.06, t_crouch=0.25, t_push=0.12, t_land=0.15, t_stand=0.22,
                  bend=0.03, tuck=0.12, lean=10.0, back=-35.0, fwd=40.0),
    "big": dict(H=0.25, crouch=0.15, land=0.12, t_crouch=0.32, t_push=0.15, t_land=0.18, t_stand=0.28,
                bend=0.05, tuck=0.22, lean=16.0, back=-45.0, fwd=70.0),
}
JUMP_TOE_DEG = 55.0        # toes pointed down in the air (degrees below horizontal)


def _jump_params(row):
    J = JUMP[JUMPS[int(row[JUMP_AT])]]
    v0 = np.sqrt(2 * G * J["H"])
    return J, v0, 2 * v0 / G, int(row[JUMP_AT + 1]) or 1


def jump_length(row):
    """Seconds from the jump keyframe until standing again."""
    J, _, t_air, n = _jump_params(row)
    return J["t_crouch"] + n * (J["t_push"] + t_air + J["t_land"]) + J["t_stand"]


def _jump_text():
    r = np.zeros(N_PARAMS)
    out = {}
    for size in ("small", "big"):
        r[JUMP_AT], r[JUMP_AT + 1] = JUMPS.index(size), 1
        one = jump_length(r); r[JUMP_AT + 1] = 2
        out[size] = (one, jump_length(r) - one)
    return out


_JT = _jump_text()
PROMPT = (PROMPT.replace("<<JUMP_SMALL>>", f"{_JT['small'][0]:.2f}").replace("<<JUMP_BIG>>", f"{_JT['big'][0]:.2f}")
          .replace("<<JUMP_REP_SMALL>>", f"{_JT['small'][1]:.2f}").replace("<<JUMP_REP_BIG>>", f"{_JT['big'][1]:.2f}"))


def _hermite(s, p0, v0, p1, v1, T):
    s2, s3 = s * s, s * s * s
    return (2 * s3 - 3 * s2 + 1) * p0 + (s3 - 2 * s2 + s) * T * v0 + (-2 * s3 + 3 * s2) * p1 + (s3 - s2) * T * v1


def jump_profile(times, rows, T):
    """Per frame: hips dy (m), heel up 0..1, extra ankle lift in the air (m), toes-down weight in the air 0..1,
    crouch 0..1 (for the lean), lean (deg), arm swing per arm (deg, + = forward; right, left). Zero outside jumps."""
    t = np.arange(T) / FPS
    out = {k: np.zeros(T) for k in ("dy", "heel", "lift", "point", "lean")}
    out["swing"] = np.zeros((T, 2))
    smooth = lambda x: x * x * (3 - 2 * x)
    for i, r in enumerate(rows):
        if not r[JUMP_AT]:
            continue
        J, v0, ta, n = _jump_params(r)
        bump = J["tuck"] if int(r[JUMP_AT + 2]) == JUMP_POSES.index("tuck") else J["bend"]
        tau = t - times[i]
        # phases as (start, end, kind, depth before)
        ph, c = [], 0.0
        ph.append((c, c + J["t_crouch"], "crouch")); c += J["t_crouch"]
        for j in range(n):
            ph.append((c, c + J["t_push"], "push", J["crouch"] if j == 0 else J["land"])); c += J["t_push"]
            ph.append((c, c + ta, "air")); c += ta
            ph.append((c, c + J["t_land"], "land")); c += J["t_land"]
        ph.append((c, c + J["t_stand"], "stand")); c += J["t_stand"]
        sw_keys = [(0.0, 0.0), (J["t_crouch"], J["back"])]
        for (a, b, kind, *x) in ph:
            m = (tau >= a) & (tau < b) if kind != "stand" else (tau >= a) & (tau <= b)
            if not m.any():
                pass
            s = np.clip((tau[m] - a) / (b - a), 0, 1)
            if kind == "crouch":
                out["dy"][m] = -J["crouch"] * smooth(s)
            elif kind == "push":
                out["dy"][m] = _hermite(s, -x[0], 0.0, JUMP_EXT, v0, b - a)
                out["heel"][m] = smooth(np.clip(2 * s - 1, 0, 1))
                sw_keys.append((b, J["fwd"]))
            elif kind == "air":
                u = tau[m] - a
                y = JUMP_EXT + v0 * u - 0.5 * G * u * u
                out["dy"][m] = y
                out["heel"][m] = 1.0
                out["lift"][m] = (y - JUMP_EXT) + bump * np.sin(np.pi * u / ta)
                out["point"][m] = np.sin(np.pi * u / ta)
                sw_keys.append((b, 0.4 * J["fwd"]))
            elif kind == "land":
                out["dy"][m] = _hermite(s, JUMP_EXT, -v0, -J["land"], 0.0, b - a)
                out["heel"][m] = 1 - smooth(np.clip(s / 0.35, 0, 1))
                last = b >= c - J["t_stand"] - 1e-9
                sw_keys.append((b, 0.15 * J["fwd"] if last else 0.6 * J["back"]))
            else:
                out["dy"][m] = -J["land"] * (1 - smooth(s))
                sw_keys.append((b, 0.0))
        depth = np.where(tau < J["t_crouch"], J["crouch"], J["land"])
        win = (tau >= 0) & (tau <= c)
        out["lean"][win] += J["lean"] * np.clip(-out["dy"][win] / depth[win], 0, 1)
        kt = np.array([k for k, _ in sw_keys]); kv = np.array([v for _, v in sw_keys])
        kk = np.clip(np.searchsorted(kt, tau, side="right") - 1, 0, len(kt) - 2)
        u = smooth(np.clip((tau - kt[kk]) / np.maximum(kt[kk + 1] - kt[kk], 1e-6), 0, 1))
        sw = np.where(win, kv[kk] * (1 - u) + kv[kk + 1] * u, 0.0)
        for j in range(2):
            if r[JUMP_AT + 3 + j]:
                out["swing"][:, j] += sw
    return out


def _jump_holds(times, rows):
    """The legs stand still under a jump: a copy of the jump keyframe's legs where it has landed."""
    t2, r2 = list(times), [r.copy() for r in rows]
    for i in range(len(times) - 1, -1, -1):
        if rows[i][JUMP_AT]:
            end = times[i] + jump_length(rows[i])
            if i + 1 >= len(times) or times[i + 1] > end + 1e-6:
                h = rows[i].copy(); h[JUMP_AT:JUMP_AT + 5] = 0
                t2.insert(i + 1, end); r2.insert(i + 1, h)
    return np.array(t2), np.array(r2)


_LEGS_CACHE = {}
_LEGS_CACHE_MAX = 16


def legs_frames(posture, times, rows, k, w):
    """Lower body for every frame: {joint: quaternions [T,4]} for hips and legs, and root [T,3].
    Cached on the leg columns of the rows (LEG_COLS), the timing and the active model (2026-11-03, Claude Desktop
    test: contact.resolve_touches solves the same legs in every pass, and the standing leg IK per frame was most of
    a 3-4 s solve)."""
    key = (posture, sk.MODEL, sk.CANONICAL, np.asarray(times).tobytes(), np.ascontiguousarray(rows[:, LEG_COLS]).tobytes(),
           np.asarray(k).tobytes(), np.asarray(w).tobytes())
    hit = _LEGS_CACHE.get(key)
    if hit is None:
        hit = _legs_frames(posture, times, rows, k, w)
        if posture == "stand" and hit[2] and not _pivot_ok(*hit[:2]):
            # this body cannot pivot this turn cleanly (test1's wide legs crossed on the one big step of a 180,
            # AvatarSample_D's short legs went over the knee speed limit): step it round as before
            hit = _legs_frames(posture, times, rows, k, w, pivot=False)
        hit = hit[:2]
        if len(_LEGS_CACHE) >= _LEGS_CACHE_MAX:
            _LEGS_CACHE.pop(next(iter(_LEGS_CACHE)))
        _LEGS_CACHE[key] = hit
    q, root = hit
    return {n: v.copy() for n, v in q.items()}, root.copy()


PIVOT_MAX_VEL = 9.8          # rad/s, a little under the filter's leg limit (filters.MAX_VEL 10)


def _pivot_ok(q, root):
    """A pivoted turn is kept only if the legs do not pass through each other (as the filter's legs_clip) and no leg
    joint turns faster than PIVOT_MAX_VEL."""
    names = list(q)
    m = np.stack([q[n] for n in names], 1).astype(float)
    if (sk.legs_overlap(m, root.astype(float), {n: i for i, n in enumerate(names)}) > 0).sum() > 3:
        return False
    for n in names:
        if n == "hips":
            continue
        d = np.abs(np.sum(q[n][1:] * q[n][:-1], -1)).clip(0, 1)
        if len(d) and (2 * np.arccos(d) * FPS).max() > PIVOT_MAX_VEL:
            return False
    return True


def _legs_frames(posture, times, rows, k, w, pivot=True):
    """Returns q, root and whether a turn was pivoted (standing only)."""
    T = len(k)
    names = ["hips"] + [f"{s}{b}" for s in ("left", "right") for b in ("UpperLeg", "LowerLeg", "Foot")]
    if posture == "kneel" and (rows[:, HIP_AT].any() or rows[:, SIT_AT + 2].any()):   # kneeling sway / knees apart
        poses = [sk.kneel_sway(kneel_dx(r), sk.SIT_KNEES[int(r[SIT_AT + 2])]) for r in rows]
        qk = {n: continuous(np.array([mat_to_quat(P[0][n]) for P in poses])) for n in names}
        q = {n: continuous(slerp(qk[n][k], qk[n][k + 1], w[:, None])) for n in names}
        hips_k = np.array([P[1] for P in poses])
        hips = hips_k[k] * (1 - w[:, None]) + hips_k[k + 1] * w[:, None]
        return q, ((hips - sk.HIPS_REST) / sk.HIPS_Y).astype(np.float32), False
    if posture != "stand":
        S = rows[:, SIT_AT:SIT_AT + 3].astype(int)
        if not S.any():                                      # the posture's own legs throughout
            R, hips = sk.sit_pose(posture)
            q = {n: np.repeat(mat_to_quat(R[n])[None], T, 0) for n in names}
            return q, np.repeat(((hips - sk.HIPS_REST) / sk.HIPS_Y)[None], T, 0).astype(np.float32), False
        # leg words change between keyframes: slerp each leg joint, then lower the hips per frame onto the floor
        def shape(r, i):
            a = int(r[SIT_AT + i])
            return sk.SIT_LEGS[a] if a >= 0 else (r[LEGDIR_AT + 6 * i:LEGDIR_AT + 6 * i + 3], r[LEGDIR_AT + 6 * i + 3:LEGDIR_AT + 6 * i + 6])
        Rk = [sk.sit_legs_R(posture, shape(r, 0), shape(r, 1), sk.SIT_KNEES[c]) for r, c in zip(rows, S[:, 2])]
        qk = {n: continuous(np.array([mat_to_quat(R[n]) for R in Rk])) for n in names}
        q = {n: slerp(qk[n][k], qk[n][k + 1], w[:, None]) for n in names}
        # The bottom stays on the floor: the hips move between the keyframe poses' heights, and a leg that would
        # sweep through the floor on the way is lifted over it (its thigh turned up about the hip).
        hips_k = np.array([sk.sit_hips(R) for R in Rk])
        hips = hips_k[k] * (1 - w[:, None]) + hips_k[k + 1] * w[:, None]
        for f in range(T):
            Rf = {n: quat_to_mat(q[n][f]) for n in names}
            for side in ("left", "right"):
                for _ in range(6):
                    lift, pt = _leg_dip(Rf, hips[f], side)
                    if lift <= 1e-4:
                        break
                    v = pt - sk.hip_joint(hips[f], side)
                    axis = np.cross(v, [0.0, 1.0, 0.0])             # turning v about it raises the point
                    if np.linalg.norm(axis) < 1e-6:
                        break
                    Rf[f"{side}UpperLeg"] = sk._rot(axis, 1.2 * lift / max(np.linalg.norm(v), 0.05)) @ Rf[f"{side}UpperLeg"]
            # legs that pass through each other on the way: the upper leg (the one lying on top) rises over the other
            for _ in range(10):
                gap = sk.legs_gap(Rf, hips[f]) + 0.005          # stop 5 mm clear
                if gap <= 0:
                    break
                P = sk.leg_points(Rf, hips[f])
                top = max(("left", "right"), key=lambda sd: P[f"{sd}Knee"][1] + P[f"{sd}Ankle"][1])
                v = P[f"{top}Ankle"] - P[f"{top}Hip"]
                axis = np.cross(v, [0.0, 1.0, 0.0])
                if np.linalg.norm(axis) < 1e-6:
                    break
                Rf[f"{top}UpperLeg"] = sk._rot(axis, 1.2 * gap / max(np.linalg.norm(v), 0.05)) @ Rf[f"{top}UpperLeg"]
            for side in ("left", "right"):
                q[f"{side}UpperLeg"][f] = mat_to_quat(Rf[f"{side}UpperLeg"])
        q = {n: continuous(v) for n, v in q.items()}
        return q, ((hips - sk.HIPS_REST) / sk.HIPS_Y).astype(np.float32), False
    jump = None
    if rows[:, JUMP_AT].any():
        jump = jump_profile(times, rows, T)
        times, rows = _jump_holds(times, rows)
        k, w = _segments(times, np.arange(T) / FPS)
    hold = np.zeros(len(times), bool)
    if pivot and len(times) > 1 and PIVOT_DEG > 0 and np.abs(np.diff(rows[:, TURN_AT])).max() >= 2.0:
        times, rows, hold = _pivot_turns(times, rows)
        k, w = _segments(times, np.arange(T) / FPS)
    if len(times) > 1 and np.abs(np.diff(rows[:, TURN_AT])).max() > TURN_STEP:
        # a big turn is stepped round in parts of at most TURN_STEP, each foot stepping in turn (in one go the two
        # feet slid straight across to each other's places and passed through each other; 2026-11-04)
        n0 = len(times)
        times2, rows2 = _split_turns(times, rows)
        if len(times2) != n0:                  # the split keyframes do not hold their feet
            idx = np.searchsorted(times2, times)
            h2 = np.zeros(len(times2), bool); h2[idx] = hold; hold = h2
        times, rows = times2, rows2
        k, w = _segments(times, np.arange(T) / FPS)
    tg = [_stand_targets(r) for r in rows]
    for i in range(1, len(tg)):
        if hold[i]:                            # pivoting: the feet (and the hips over them) stay where they were
            h, f, y, r = tg[i]                 # (letting the ankle swing round the ball of the foot made the knees
            hp, fp, _, _ = tg[i - 1]           # turn at 13-16 rad/s, 2026-11-05)
            h = h.copy(); h[[0, 2]] = hp[[0, 2]]
            f = {sd: (np.array([fp[sd][0][0], a[1], fp[sd][0][2]]), t, p) for sd, (a, t, p) in f.items()}
            tg[i] = (h, f, y, r)
    hips_k = np.array([h for h, *_ in tg])
    hips = hips_k[k] * (1 - w[:, None]) + hips_k[k + 1] * w[:, None]
    yaw = np.array([y for _, _, y, _ in tg])
    yaw = yaw[k] * (1 - w) + yaw[k + 1] * w
    turn = {s: np.array([foot_turn(r, s) for r in rows]) for s in ("left", "right")}
    turn = {s: v[k] * (1 - w) + v[k + 1] * w for s, v in turn.items()}
    roll_k = np.array([t[3] for t in tg])
    roll = roll_k[k] * (1 - w) + roll_k[k + 1] * w
    kin = rows[:, KNEES_AT][k] * (1 - w) + rows[:, KNEES_AT][k + 1] * w      # knees in (0..1)
    if jump is not None:                       # jumping: hips up / down, heels up, feet off the floor, toes pointed
        hips[:, 1] += jump["dy"]
        fwd = np.stack([np.sin(yaw), np.zeros(T), np.cos(yaw)], -1)
        hips += 0.25 * np.minimum(jump["dy"], 0)[:, None] * fwd          # the hips go back a little as they go down
    R = {n: np.empty((T, 3, 3)) for n in names}
    R["hips"][:] = sk.ry(yaw) @ np.array([rz(a) for a in roll])
    # One foot must stay on the floor all the time, also between keyframes. When one foot comes down and
    # the other goes up in the same segment, blending both at once left both feet in the air for ~1.7 s
    # (the author saw floating feet, 2026-10-21). So within a segment the foot that comes down (or steps
    # first) moves in the first half and the foot that goes up (or steps second) in the second half.
    ease = lambda x: x * x * (3 - 2 * x)
    u = np.clip((np.arange(T) / FPS - times[k]) / (times[k + 1] - times[k]), 0, 1)
    first, second = ease(np.clip(2 * u, 0, 1)), ease(np.clip(2 * u - 1, 0, 1))
    a0, t0, p0, ev = {}, {}, {}, {}
    for side in ("left", "right"):
        a0[side] = np.array([f[side][0] for _, f, *_ in tg])
        t0[side] = np.array([sk._unit(f[side][1]) for _, f, *_ in tg])
        p0[side] = np.array([f[side][2] for _, f, *_ in tg])
        down0, down1 = p0[side][k], p0[side][k + 1]
        ev[side] = {"rising": down0 & ~down1, "falling": ~down0 & down1,
                    "moved": down0 & down1 & (np.linalg.norm(a0[side][k] - a0[side][k + 1], axis=-1) > PIVOT)}
    ankles, toes = {}, {}
    for side, other in (("left", "right"), ("right", "left")):
        e, o = ev[side], ev[other]
        # which foot steps first when both step: the one on the side the body turns to (a turn to the left opens with
        # the left foot; with the left always first a 180 turn to the right crossed test1's legs, 2026-11-05), else left
        dturn = (rows[k + 1, TURN_AT] - rows[k, TURN_AT]) if TURN_LEAD else np.zeros(T)
        lead = np.where(dturn < -1e-6, "right", "left")
        step_first = o["rising"] | (o["moved"] & (lead == side))
        ws = np.where(e["rising"], second, np.where(e["falling"], first,
             np.where(e["moved"], np.where(step_first, first, np.where(o["falling"] | o["moved"], second, w)), w)))
        a = a0[side][k] * (1 - ws[:, None]) + a0[side][k + 1] * ws[:, None]
        # a foot that stays down but changes place steps: lift it on the way
        a[:, 1] += np.where(e["moved"], STEP_ARC * 4 * ws * (1 - ws), 0.0)
        ankles[side] = a
        toes[side] = t0[side][k] * (1 - ws[:, None]) + t0[side][k + 1] * ws[:, None]
    if jump is not None:
        a_tip = np.arcsin(-TIPTOE_DIR[1])
        for side in ("left", "right"):
            ankles[side][:, 1] += jump["heel"] * TIPTOE_UP + jump["lift"]
            tv = toes[side]
            hz = np.linalg.norm(tv[:, [0, 2]], axis=-1, keepdims=True)
            hd = np.where(hz > 1e-6, tv / np.maximum(hz, 1e-9), fwd); hd[:, 1] = 0
            hd /= np.linalg.norm(hd, axis=-1, keepdims=True)
            base = np.arctan2(-tv[:, 1], hz[:, 0])
            pitch = base * (1 - jump["heel"]) + a_tip * jump["heel"] + jump["point"] * (np.deg2rad(JUMP_TOE_DEG) - a_tip)
            toes[side] = hd * np.cos(pitch)[:, None] + np.array([0.0, -1.0, 0.0]) * np.sin(pitch)[:, None]
    for side in ("left", "right"):
        for f in range(T):
            Rh = R["hips"][f]
            hj = sk.hip_joint(hips[f], side, Rh)
            pole = sk.ry(yaw[f] + turn[side][f])[:, 2]          # the knee points the way its toes do
            if kin[f] > 0:                          # knees together, feet apart (KNEES_AT)
                pole = sk.ry(yaw[f] + turn[side][f] - sk.side_sign(side) * np.deg2rad(KNEES_IN_DEG) * kin[f])[:, 2]
            if roll[f] * sk.side_sign(side) < 0:    # the free leg of a hip sway: its knee turns in
                knee_in = np.deg2rad(HIP_KNEE_IN) * min(1.0, abs(roll[f]) / np.deg2rad(HIP_ROLL["far_out"]))
                pole = sk.ry(yaw[f] + turn[side][f] - sk.side_sign(side) * knee_in)[:, 2]
            U, L, F, _ = sk.leg_ik(hj, ankles[side][f], side, toes[side][f], pole=pole)
            # leg_ik works in world axes; the upper leg's local rotation is relative to the (turned) hips
            R[f"{side}UpperLeg"][f], R[f"{side}LowerLeg"][f], R[f"{side}Foot"][f] = Rh.T @ U, L, F
    q = {n: mat_to_quat(R[n]) for n in names}
    return q, ((hips - sk.HIPS_REST) / sk.HIPS_Y).astype(np.float32), bool(hold.any())



def _arm_path(R, q, k, w, side):
    """Upper-arm rotations per frame between the keyframe rotations R [K,3,3] (quaternions q). Plain slerp, except
    in a segment where slerp would pass beyond the joint limit (2026-10-28, kneeling hand to the back of the head:
    the shortest rotation between the two elbow orientations swung the arm back behind the body, 3.1 rad from the
    T-pose): there the arm direction moves along the great circle between the two directions and the twist about
    the arm turns evenly (swing-twist), so the arm goes up past the side instead."""
    out = slerp(q[k], q[k + 1], w[:, None])
    lim = schema.MAX_ANGLE[JIDX[f"{side}UpperArm"]]
    ang = 2 * np.arccos(np.clip(np.abs(out[:, 3]), 0, 1))
    a = np.array([1.0 if side == "left" else -1.0, 0.0, 0.0])        # rest direction of the arm
    def split(M):
        d = M @ a
        S = sk._swing(a, d)
        T_ = S.T @ M                                                 # rotation about a
        return d, np.arctan2(T_[2, 1], T_[1, 1])
    for seg in np.unique(k[ang > lim]):
        f = np.nonzero(k == seg)[0]
        (d0, t0), (d1, t1) = split(R[seg]), split(R[seg + 1])
        dt = (t1 - t0 + np.pi) % (2 * np.pi) - np.pi
        om = np.arccos(np.clip(d0 @ d1, -1, 1))
        # the twist may turn either way round; the shorter way can add up with the swing to almost a half turn
        # (2026-11-13, S026: the side-sit floor prop -> hands behind the head reached 3.09 rad): take the way whose
        # largest angle from the T-pose is smaller
        best = None
        for dtw in (dt, dt - np.sign(dt) * 2 * np.pi if dt else dt):
            qs = []
            for i in f:
                u = w[i]
                d = d0 if om < 1e-6 else (np.sin((1 - u) * om) * d0 + np.sin(u * om) * d1) / np.sin(om)
                tw = t0 + u * dtw
                Rt = np.array([[1, 0, 0], [0, np.cos(tw), -np.sin(tw)], [0, np.sin(tw), np.cos(tw)]])
                qs.append(mat_to_quat(sk._swing(a, d) @ Rt))
            qs = np.asarray(qs)
            peak = float((2 * np.arccos(np.clip(np.abs(qs[:, 3]), 0, 1))).max())
            if best is None or peak < best[0]:
                best = (peak, qs)
        out[f] = best[1]
    return continuous(out)


def keyframes_to_motion(posture, times, rows, duration_s):
    """(posture, times [K], rows [K, N_PARAMS]) -> (motion [T,J,4] quaternions at 30 fps, root [T,3])."""
    T = max(10, int(round(duration_s * FPS)) + 1)
    k, w = _segments(times, np.arange(T) / FPS)
    k_legs, w_legs = k, w
    e = np.zeros((T, J, 3))                     # Euler angles for the single-axis joints
    d = np.deg2rad(rows[k, HEAD_AT:TORSO_AT + 3] * (1 - w[:, None]) + rows[k + 1, HEAD_AT:TORSO_AT + 3] * w[:, None])
    # Euler axes: x = pitch (+ down/forward), y = yaw (+ toward its left), z = roll (+ toward its LEFT: measured 2026-11-11, bend +15 moves the head to +x)
    head = np.stack([d[:, 0], d[:, 1], -d[:, 2]], -1)
    jump = jump_profile(times, rows, T) if posture == "stand" and rows[:, JUMP_AT].any() else None
    if jump is not None:
        d[:, 3] += np.deg2rad(jump["lean"])
    pelvis = np.maximum(d[:, 3] - np.deg2rad(SPINE_LEAN), 0.0)       # see PELVIS_POSTURES
    torso = np.stack([d[:, 3] - pelvis, d[:, 4], -d[:, 5]], -1)
    e[:, JIDX["neck"]], e[:, JIDX["head"]] = 0.35 * head, 0.65 * head
    for name in ("spine", "chest", "upperChest"):
        e[:, JIDX[name]] = torso / 3
    if posture in ("stand", "kneel"):
        # hip sway: the lower back follows the tilted pelvis (toward the free leg), the chest bends back and tilts
        # the shoulders the other way, an S-curve (spread over all three joints the torso leaned as one line)
        roll_k = pelvis_rolls(posture, rows)
        roll = roll_k[k_legs] * (1 - w_legs) + roll_k[k_legs + 1] * w_legs
        for name in ("chest", "upperChest"):
            e[:, JIDX[name], 2] -= (1 + HIP_COUNTER) * roll / 2
        e[:, JIDX["neck"], 2] += HIP_COUNTER * roll        # the head stays upright (its own tilt words add on top)
    # Shoulder raise: the right arm points along -x, so Rz(-a) lifts it; the left arm is the mirror image.
    sh = np.deg2rad(rows[k, SHR_AT:SHR_AT + 2] * (1 - w[:, None]) + rows[k + 1, SHR_AT:SHR_AT + 2] * w[:, None])
    e[:, JIDX["rightShoulder"], 2], e[:, JIDX["leftShoulder"], 2] = -sh[:, 0], sh[:, 1]
    # Shoulder forward / back (the collarbone turning about the vertical): Ry(+a) moves the right arm's -x toward +z
    sf = np.deg2rad(rows[k, SHF_AT:SHF_AT + 2] * (1 - w[:, None]) + rows[k + 1, SHF_AT:SHF_AT + 2] * w[:, None])
    e[:, JIDX["rightShoulder"], 1], e[:, JIDX["leftShoulder"], 1] = sf[:, 0], -sf[:, 1]
    upper_q, lower_q, hand_q = {}, {}, {}
    for side, at in (("right", ARM_AT), ("left", ARM_AT + 6)):
        # Solve the arm at the keyframes only, then blend the joint rotations (slerp) in between.
        # Blending the two directions instead lets the elbow plane flip mid-way, which spins the arm.
        upper, fore = rows[:, at:at + 3], rows[:, at + 3:at + 6]
        first = np.cross(upper[0], fore[0]) * (1.0 if side == "right" else -1.0)
        hinge = _unit(first) if np.linalg.norm(first) > 1e-3 else np.array([0.0, 1.0, 0.0])
        R, elbow = np.empty((len(times), 3, 3)), np.empty(len(times))
        for i in range(len(times)):
            R[i], elbow[i], hinge = solve_arm(upper[i], fore[i], side, hinge)
        q = mat_to_quat(R)
        upper_q[side] = _arm_path(R, q, k, w, side)
        if jump is not None and jump["swing"][:, 0 if side == "right" else 1].any():   # the arm swings with the jump
            a = -np.deg2rad(jump["swing"][:, 0 if side == "right" else 1]) / 2          # Rx(-a): a hanging arm goes forward
            qs = np.stack([np.sin(a), np.zeros(T), np.zeros(T), np.cos(a)], -1)
            upper_q[side] = quat_mul(qs, upper_q[side])
        bend = elbow[k] * (1 - w) + elbow[k + 1] * w
        # Wrist: the palm twist is split between the lower arm (most of it, as in a real forearm) and
        # the hand joint. The hand also takes the wrist bend.
        col = 0 if side == "right" else 1
        twist, tw = np.empty(len(times)), 0.0
        for i in range(len(times)):
            pc = rows[i, PALM_AT + col]
            palm = rows[i, PALMV_AT + 3 * col:PALMV_AT + 3 * col + 3] if pc < 0 else PALMS[int(pc)]
            twist[i] = tw = solve_palm(R[i], elbow[i], side, palm, tw, dt=float(times[i] - times[i - 1]) if i else None)
        t_low = np.clip(twist, -TWIST_LOWER, TWIST_LOWER)
        t_hand = twist - t_low
        mix = lambda v: v[k] * (1 - w) + v[k + 1] * w
        sd = 1.0 if side == "right" else -1.0                             # sign of elbow and wrist flexion
        zero = np.zeros(T)
        eq = lambda x, y, z: mat_to_quat(euler_to_mat(np.stack([x, y, z], -1)))
        lower_q[side] = quat_mul(eq(zero, sd * bend, zero), eq(mix(t_low), zero, zero))    # hinge, then twist
        # hand joint: the rest of the palm twist (about the forearm), the wrist bend (about z), then the sideways bend
        # (about the palm normal y; + = toward the little finger: for the right hand, pointing -x with the thumb at +z,
        # that is a turn about -y)
        hand_q[side] = quat_mul(quat_mul(eq(mix(t_hand), zero, zero), eq(zero, zero, sd * np.deg2rad(mix(rows[:, WRIST_AT + col])))),
                                eq(zero, -sd * np.deg2rad(mix(rows[:, WSIDE_AT + col])), zero))
    m = from_euler(e)
    legs, root = legs_frames(posture, times, rows, k_legs, w_legs)
    for name, q in legs.items():
        m[:, JIDX[name]] = q
    if pelvis.any():                             # the pelvis tips forward about the hips; the thighs keep their place
        qp = np.zeros((T, 4)); qp[:, 0], qp[:, 3] = np.sin(pelvis / 2), np.cos(pelvis / 2)
        m[:, JIDX["hips"]] = quat_mul(m[:, JIDX["hips"]], qp)
        for side in ("left", "right"):
            m[:, JIDX[f"{side}UpperLeg"]] = quat_mul(quat_conj(qp), m[:, JIDX[f"{side}UpperLeg"]])
    for side, q in upper_q.items():
        # the model's own rest arm bones (skeleton.arm_rest_fix; identity on models whose T-pose arms lie along x)
        U, Lw, Hd = sk.arm_to_real(side, quat_to_mat(q), quat_to_mat(lower_q[side]), quat_to_mat(hand_q[side]))
        # The arm directions are relative to the torso, so undo the shoulder's rotation on the upper arm:
        # a shrug then lifts the shoulder joint without also swinging the whole arm outward.
        m[:, JIDX[f"{side}UpperArm"]] = quat_mul(quat_conj(m[:, JIDX[f"{side}Shoulder"]]), mat_to_quat(U))
        m[:, JIDX[f"{side}LowerArm"]] = mat_to_quat(Lw)
        m[:, JIDX[f"{side}Hand"]] = mat_to_quat(Hd)
    return continuous(m).astype(np.float32), root


def keyframes_to_hands(times, rows, duration_s):
    """(times [K], rows [K, N_PARAMS]) -> hand shapes uint8 [T,2] (LEFT, right, the motion format's order): each
    frame takes the shape of the nearer keyframe, so a shape is reached together with its pose."""
    T = max(10, int(round(duration_s * FPS)) + 1)
    k, w = _segments(times, np.arange(T) / FPS)
    return rows[np.where(w >= 0.5, k + 1, k)][:, [HAND_AT + 1, HAND_AT]].astype(np.uint8)


EYES = ("open", "closed")
EYES_FADE = 0.15           # s: the lids close / open over this (2026-11-07, the author: head_pat with the eyes closed)


def eyes_track(obj, duration_s):
    """Optional per-keyframe "eyes": closed | open (carried forward, default open) -> float [T] closed weight 0..1
    with EYES_FADE ramps, or None when no keyframe mentions the eyes. Independent of the solver (no row column)."""
    kfs = obj.get("keyframes") if isinstance(obj, dict) else None
    if not isinstance(kfs, list) or not any(isinstance(k, dict) and "eyes" in k for k in kfs):
        return None
    T = max(10, int(round(duration_s * FPS)) + 1)
    raw = np.zeros(T); state = 0.0
    times = [float(k.get("t", 0)) for k in kfs]
    for k, t in zip(kfs, times):
        if "eyes" in k:
            if k["eyes"] not in EYES:
                raise ValueError(f"unknown eyes {k['eyes']!r}; one of {EYES}")
            state = 1.0 if k["eyes"] == "closed" else 0.0
        raw[min(int(round(t * FPS)), T - 1):] = state          # (a keyframe AT the end still counts: an "open" on the
        # last keyframe fell past the last frame and the eyes stayed shut after the motion, 2026-11-11)
    n = max(1, int(round(EYES_FADE * FPS)))                   # box filter = linear ramp over EYES_FADE
    pad = np.concatenate([np.full(n // 2, raw[0]), raw, np.full(n - n // 2 - 1, raw[-1])])
    out = np.convolve(pad, np.ones(n) / n, mode="valid")
    if raw[-1] == 0.0:                                        # ends open: fully open on the last frame (the held pose)
        out[-1] = 0.0
    return out


class KeyframeGenerator(MotionGenerator):
    needs_llm = True       # the pipeline passes its --llm to the constructor

    def __init__(self, llm, retries=2):
        self.llm, self.retries = llm, retries
        self.last_keyframes = None     # the parsed reply of the last call, for inspection

    def generate(self, instruction, seed, duration_s):
        prompt = PROMPT.format(instruction=instruction, duration=duration_s,
                               variation=VARIATIONS[seed % len(VARIATIONS)], sit_list=SIT_LIST)
        err = None
        for _ in range(self.retries + 1):
            try:
                obj = extract_json(self.llm.complete(prompt, max_tokens=6000))
                posture, times, rows = parse_keyframes(obj, duration_s)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                err = e
                continue
            self.last_keyframes = obj
            self.last_hands = keyframes_to_hands(times, rows, duration_s)
            motion, self.last_root = keyframes_to_motion(posture, times, rows, duration_s)
            return motion
        raise RuntimeError(f"no valid keyframes for {instruction!r}: {err}")

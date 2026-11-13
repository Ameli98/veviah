"""Dimension table and spec sampling (round-0 design, section 2)."""
import random

AXES = {
    "emotion": ["happy", "sad", "angry", "surprised", "shy", "bored", "smug",
                "confused", "scared", "calm", "disgusted", "excited"],
    "intensity": ["very slight", "moderate", "strong"],
    "speed": ["slow", "normal", "fast", "sudden"],
    "situation": ["greeting", "saying goodbye", "listening to the user", "thinking",
                  "being praised", "being interrupted",
                  "keeping the user company while they watch a video or play a game"],
    "duration": ["1-2 s", "3-5 s", "6-10 s"],
}
DURATION_S = {"1-2 s": 1.5, "3-5 s": 4.0, "6-10 s": 8.0}

# Whole body, in place (the author, 2026-10-15): the character stands, or sits on the floor in one of these
# ways, for the whole motion (no standing up or sitting down inside one motion yet).
SIT_DESC = {"cross_legged": "cross-legged", "w_sit": "in a W-sit / wariza (knees together in front, feet out beside the hips)",
            "seiza": "kneeling and sitting back on the heels (seiza)", "side_right": "with both legs folded to its right side",
            "side_left": "with both legs folded to its left side",
            "knees_hugged": "hugging the knees", "legs_out": "with the legs stretched out in front",
            "kneel": "kneeling upright", "ledge": "on a ledge (a window's top edge), the legs hanging down"}
P_STAND = 0.6              # the rest: sitting, each way equally likely

# action type -> body parts that make sense for it (rejects nonsense combinations)
ALLOWED_BODY = {
    "gesture": ["one hand", "both hands", "upper body"],
    "head": ["head only", "head and shoulders", "upper body"],
    "posture": ["upper body", "whole body"],
    "legs": ["legs", "whole body"],                 # weight shift, a step, crouching, lifting a foot (standing only)
    "reaction": ["head only", "head and shoulders", "one hand", "both hands", "upper body", "whole body"],
    "idle": ["head only", "head and shoulders", "upper body", "whole body"],
    "dance": ["whole body"],                        # a short dance in place: kept for later, weight 0 for now
}
ACTION_WEIGHTS = {"gesture": 0.25, "head": 0.15, "posture": 0.15, "legs": 0.15, "reaction": 0.15, "idle": 0.15,
                  "dance": 0.0}
STAND_ONLY = {"legs", "dance"}

TIER_WEIGHTS = {"T1": 0.45, "T2": 0.20, "T3": 0.20, "T4": 0.10, "T5": 0.05}
TIER_HINT = {
    "T1": "a single simple action",
    "T2": "two actions happening at the same time",
    "T3": "a short sequence (first ..., then ...)",
    "T4": "vague or colloquial, describing a feeling rather than a movement",
    "T5": "OUT OF SCOPE on purpose: the instruction asks for {out_of_scope}",
}
OUT_OF_SCOPE = ["walking across the room", "dancing", "jumping", "wiggling every finger",
                "making an exaggerated facial expression", "standing up from the floor and spinning"]


def posture_text(spec):
    p = spec.get("posture", "stand")
    return "standing" if p == "stand" else f"sitting on the floor, {SIT_DESC[p]}"


def sample_specs(n, seed=0):
    rng = random.Random(seed)
    tiers, weights = zip(*TIER_WEIGHTS.items())
    specs = []
    for i in range(n):
        tier = rng.choices(tiers, weights)[0]
        posture = "stand" if rng.random() < P_STAND else rng.choice(list(SIT_DESC))
        acts = [a for a, w in ACTION_WEIGHTS.items() if w > 0 and (posture == "stand" or a not in STAND_ONLY)]
        action = rng.choices(acts, [ACTION_WEIGHTS[a] for a in acts])[0]
        spec = {
            "id": f"s{i:05d}", "tier": tier, "posture": posture, "action_type": action,
            "body_part": rng.choice(ALLOWED_BODY[action]),
        }
        for axis, values in AXES.items():
            spec[axis] = rng.choice(values)
        spec["duration_s"] = DURATION_S[spec["duration"]]
        if tier == "T5":
            spec["out_of_scope"] = rng.choice(OUT_OF_SCOPE)
        specs.append(spec)
    return specs


def tier_hint(spec):
    return TIER_HINT[spec["tier"]].format(out_of_scope=spec.get("out_of_scope", ""))

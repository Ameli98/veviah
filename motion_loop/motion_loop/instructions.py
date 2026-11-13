"""Step 1: LMM writes diverse instructions from sampled specs."""
import random
import re
from concurrent.futures import ThreadPoolExecutor

from .dims import posture_text, tier_hint
from .llm import extract_json

# PLACEHOLDER seeds: replace with ~20 of your own, in the voice you want.
DEFAULT_SEEDS = [
    "Nod slowly twice to show you agree.",
    "Lean in a little and tilt your head, curious about what the user just said.",
    "Shrug with both shoulders and look a bit sheepish.",
    "Wave one hand quickly to say hi.",
    "Rest your chin on your hand and gaze off, bored.",
    "Flinch back suddenly, startled, then relax.",
    "Look pleased and stand up straighter after being praised.",
    "Shift your weight onto one leg and cross your arms, waiting.",
    "Take a small step back, startled, with both hands raised.",
    "Sitting cross-legged on the floor, rock slowly from side to side.",
    "Bounce lightly on your knees a few times, excited.",
    "Shake your head gently, disappointed but not angry.",
    "Raise one hand to point at the screen while looking excited.",
    "Stretch both arms overhead and then slump back, relaxed.",
    "Glance away and fidget, embarrassed.",
    "Cross your arms and lift your chin, smug.",
]

PROMPT = """You write short motion instructions for a desktop companion character. It moves its whole body in
place: it either stands or sits on the floor, and keeps that posture for the whole motion (head, shoulders, arms,
wrists, torso; standing, also the legs: shifting its weight, a small step, bending the knees, lifting a foot).
Hands can make simple shapes only: open, fist, pointing, peace sign, thumbs up. No walking away, no jumping, no
standing up or sitting down, no individual finger movement, no face, no voice.

Spec:
- posture: {posture}
- emotion: {emotion}
- intensity: {intensity}
- body part: {body_part}
- action type: {action_type}
- speed: {speed}
- situation: {situation}
- duration: {duration}
- difficulty: {tier_hint}

Write 3 instructions for this spec, each in a different style:
1. plain description  2. casual / colloquial  3. vague or emotional
Rules: one sentence each; describe body movement only: show the emotion through posture and gesture, and do not
mention facial expressions, eyes, mouth, or anything the character says; the character has nothing around it, so do not
mention a desk, chair, screen or any other object to touch or hold; all of this unless the difficulty says it is out of scope;
make the posture clear (when sitting, say how it sits); do not copy the examples below.

Examples:
{examples}

Return JSON only: {{"instructions": [{{"style": "plain", "text": "..."}}, {{"style": "casual", "text": "..."}}, {{"style": "vague", "text": "..."}}]}}"""


def build_prompt(spec, seeds, rng):
    ex = "\n".join(f"- {s}" for s in rng.sample(seeds, k=min(3, len(seeds))))
    return PROMPT.format(
        posture=posture_text(spec), emotion=spec["emotion"], intensity=spec["intensity"], body_part=spec["body_part"],
        action_type=spec["action_type"], speed=spec["speed"], situation=spec["situation"],
        duration=spec["duration"], tier_hint=tier_hint(spec), examples=ex)


def parse_response(text):
    items = extract_json(text)["instructions"]
    out = []
    for it in items:
        t = str(it["text"]).strip()
        if t:
            out.append({"style": str(it.get("style", "")), "text": t})
    if not out:
        raise ValueError("empty instruction list")
    return out


def _tokens(t):
    return frozenset(re.findall(r"[a-z']+", t.lower()))


def dedupe(records, threshold=0.8):
    """Drop exact and near-duplicate texts (token Jaccard), bucketed by (emotion, action type).
    Swap in embedding similarity here later if lexical dedupe is not enough."""
    kept, buckets, dropped = [], {}, 0
    for r in records:
        key = (r["spec"]["emotion"], r["spec"]["action_type"])
        toks = _tokens(r["text"])
        dup = any(len(toks & o) / max(1, len(toks | o)) >= threshold for o in buckets.get(key, []))
        if dup:
            dropped += 1
            continue
        buckets.setdefault(key, []).append(toks)
        kept.append(r)
    return kept, dropped


def generate_instructions(specs, llm, seeds=None, seed=0, workers=1, max_retries=3):
    seeds = seeds or DEFAULT_SEEDS
    rng = random.Random(seed)
    prompts = [(s, build_prompt(s, seeds, rng)) for s in specs]  # built up front: deterministic

    def work(item):
        spec, prompt = item
        for _ in range(max_retries):
            try:
                return spec, parse_response(llm.complete(prompt))
            except Exception:
                continue
        return spec, None

    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(work, prompts))
    else:
        results = [work(p) for p in prompts]

    records, failed = [], 0
    for spec, items in results:
        if items is None:
            failed += 1
            continue
        for k, it in enumerate(items):
            records.append({"id": f"{spec['id']}_{k}", "spec_id": spec["id"], "tier": spec["tier"],
                            "style": it["style"], "text": it["text"], "spec": spec})
    records, dropped = dedupe(records)
    return records, {"specs": len(specs), "failed_specs": failed,
                     "instructions": len(records), "deduped": dropped}

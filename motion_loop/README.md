# motion_loop — round-0 data pipeline for the motion reward model

Implements `../docs/round0_design.md`: sample specs -> LMM writes instructions -> candidate motions
(incl. deliberate bad ones) -> rule filter -> render keyframe grids -> LMM scores/ranks ->
preference pairs -> train the reward model (on joint sequences) -> check against your labels.

**Not in this repo:** round 1+ (using the reward model to improve the generator: best-of-n, DPO, RL).

## Quickstart (offline demo, no API key, CPU)

```
pip install -r requirements.txt
python -m unittest discover -s tests -v          # full mock end-to-end, ~1 min
python -m motion_loop.pipeline run-all --workdir runs/demo --n 100 --epochs 12 \
    --p-degrade 0.5 --p-mismatch 0 --degrade-kinds noise,freeze_arm,extreme \
    --margin 2 --d-model 32 --layers 1
```

Every stage writes files into `--workdir` (jsonl + motions/ + grids/), so any stage can be re-run alone:
`sample instruct candidates filter render score humaneval-export pairs train humaneval-score`.

## Going real: what to plug in

| Piece | Flag | You provide |
|---|---|---|
| LMM for instructions | `--llm anthropic [--model ID]` | `ANTHROPIC_API_KEY`; model via `--model` or env `MOTION_LOOP_MODEL` |
| LMM judge | `--scorer lmm` | same key; it needs a model that accepts images |
| Generator | `--generator pkg.mod:Class` | subclass `retarget.SmplTextToMotionGenerator` and implement `raw(text, seed, duration_s) -> (kind, data, fps)`; the SMPL->VRM retarget is done for you (below). Or subclass `MotionGenerator` and return `[T,13,4]` unit quaternions (x, y, z, w) at 30 fps yourself |
| Generator (LLM keyframes) | `--generator motion_loop.keyframes:KeyframeGenerator` | uses the same `--llm`; with `--llm mock` it returns canned gestures that ignore the text |
| Renderer | `--renderer motion_loop.vrm_render:VrmRenderer` | real VRM through the headless player in `../vrm_player` (needs Node and `npm install` there; model via env `HSIN_VRM_MODEL`). Or subclass `Renderer`: `render_frames(motion, idx) -> [PIL.Image]` |
| Text encoder for reward model | `--encoder minilm` | `pip install sentence-transformers` (default `hash` has no semantics) |
| Seed instructions | `--seeds-file f.txt` | ~20 of your own, one per line (built-in ones are placeholders) |

## SMPL -> VRM retarget (`motion_loop/retarget.py`)

Zero rotation = VRM normalized T-pose, the same rest pose and axes as SMPL, so local rotations carry
over joint by joint (13 upper-body bones; root and legs dropped, output is relative to an upright pelvis).

| Your model outputs | Call | Notes |
|---|---|---|
| SMPL axis-angle `[T,>=22,3]` | `retarget_smpl_rotations(pose, fps)` | twist preserved |
| 3D joint positions `[T,>=22,3]` (HumanML3D-style, SMPL joint order, y up) | `retarget_joint_positions(joints, fps)` | swing-only IK: bone directions match, twist about each bone is lost; head and hands stay at rest orientation |

Both resample to 30 fps (slerp) and return `(motion, info)`; motion is `[T,13,4]` unit quaternions (x, y, z, w), see `rotations.py`.
Convert a saved array: `python -m motion_loop.retarget motion.npy --kind joints --fps 20 --out out.npy`.

Check before trusting it on real data:
- Data must be right-handed, y-up, SMPL joint order, character's left = +x when facing +z. Mirrored data comes out mirrored.
- Your VRM must be T-pose normalized (VRM 1.0 normalized humanoid). If the renderer's rest pose differs, add per-joint rest offsets.
- Fingers, face and legs are not retargeted.

## Human validation set (do not skip)

1. `humaneval-export --n-human 100` -> `workdir/human_eval/` (images + `labels.csv`)
2. Fill the `rank` column (1 = best) for each group.
3. `pairs` (excludes those groups from training) -> `train` -> `humaneval-score`.
   It reports how often the LMM and the reward model agree with your rankings.

## Status: what was actually run

Verified (mock components, CPU torch, one-core sandbox; 11 tests): every stage end to end, the
human-set exclusion, JSON parsing/validation of judge replies, all degradations, and the reward model
learning (val pair accuracy ~0.9+) on defects visible in the motion (jitter, frozen arm, limit breaks).

Retarget verified on a synthetic SMPL-like skeleton (bone directions match after retargeting, yaw ignored,
180-degree bone flips; mutation checks confirmed the tests fail when left/right or the pelvis frame is wrong).
NOT verified: real SMPL model files / real text-to-motion output, so exact rest-pose offsets (e.g. the shoulder bones) are unchecked.

**Never run:** `AnthropicLLM` / `LMMScorer` against a real API, any real text-to-motion model,
full-size GPU training, `SentenceTextEncoder`. Expect small fixes when you plug these in.

Known limits and assumptions to check:
- Joint limits (`schema.MAX_ANGLE`, one max angle from T-pose per joint) and filter thresholds (`filters.py`) are placeholders.
- `mirror` swaps left/right joints and negates quaternion y and z (reflection in the x = 0 plane).
- Self-collision is not checked. Duplicate removal is lexical, not embedding-based.
- With the mock text encoder the reward model cannot learn "motion does not match the text"
  (text-mismatch negatives); that needs a real encoder and real judged data.
- Default reward model size (d=128, 3 layers) was not tuned; the tests use a tiny one.

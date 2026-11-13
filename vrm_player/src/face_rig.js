// Complete a model's expressions (2026-10-27, the author: hand-make the faces a model lacks). Every model gets
//   happy angry sad relaxed surprised shy blink blinkLeft blinkRight aa ih ou ee oh
// as VRM expressions, so FaceController (face.js) and lipsync can use the same names on any model:
//   1. an expression that exists under another name is aliased (case: 'Surprised' on the 0.x sample; joy / fun /
//      sorrow; a i u e o; blink_l ...);
//   2. a missing one is built from the face's morph targets by a recipe for the naming families seen in VRM models:
//      VRoid (Fcl_*), ARKit 52 (eyeBlinkLeft ...), plain English words (Seed-san: face_happy, eye_close, lip_a,
//      cheek ...) and MMD Japanese (まばたき, 笑い, あ ...). The first recipe whose morphs all exist wins.
//   3. no recipe fits (e.g. morphs without names): a mix of the model's own expressions where one is defined
//      (MIXES), else the expression stays missing (listed in the result).
// 'shy' has no VRM preset; it is always made here (worried brows, soft eyes, a small smile, a blush morph if any).
// completeExpressions(vrm, { force: true }) ignores the presets and builds everything from morphs (to check recipes).
import { VRMExpression, VRMExpressionMorphTargetBind } from '@pixiv/three-vrm';

export const FACE_NAMES = ['happy', 'angry', 'sad', 'relaxed', 'surprised', 'shy',
  'blink', 'blinkLeft', 'blinkRight', 'aa', 'ih', 'ou', 'ee', 'oh'];

const ALIASES = {
  happy: ['joy', 'smile', 'fun_happy'], angry: ['anger', 'mad'], sad: ['sorrow', 'sadness'], relaxed: ['fun', 'calm'],
  surprised: ['surprise', 'shock'], blink: ['blink_both', 'eyeclose'], blinkLeft: ['blink_l', 'blinkl', 'winkleft'],
  blinkRight: ['blink_r', 'blinkr', 'winkright'], aa: ['a'], ih: ['i'], ou: ['u'], ee: ['e'], oh: ['o'], shy: ['blush', 'shy'],
};

// name -> alternatives; each alternative = [[morph name, weight], ...] (names are matched normalised)
const L = (n, w = 1) => [[`${n}left`, w], [`${n}right`, w]];          // ARKit left / right pair
const RECIPES = {
  happy: [[['Fcl_ALL_Joy', 1]], [['face_happy', 1]], [['eye_smile', 1], ['lip_up', 0.6]],
    [...L('mouthSmile'), ...L('eyeSquint', 0.6), ...L('cheekSquint', 0.5)], [['笑い', 1]]],
  angry: [[['Fcl_ALL_Angry', 1]], [['face_angry', 1]], [['eye_angry', 1], ['eye_brow_down', 1]],
    [...L('browDown'), ...L('eyeSquint', 0.3), ...L('mouthFrown', 0.4)], [['怒り', 1]]],
  sad: [[['Fcl_ALL_Sorrow', 1]], [['face_sad', 1]], [['eye_brow_up', 1], ['lip_down', 0.6]],
    [['browInnerUp', 1], ...L('mouthFrown', 0.7), ...L('eyeLookDown', 0.2)], [['困る', 1]]],
  relaxed: [[['Fcl_ALL_Fun', 1]], [['face_relax', 1]], [['eye_smile', 0.6]],
    [...L('mouthSmile', 0.4), ...L('eyeSquint', 0.3)], [['なごみ', 1]]],
  surprised: [[['Fcl_ALL_Surprised', 1]], [['face_surprise', 1]], [['eye_open', 1], ['eye_brow_up', 1], ['mouse_open', 0.3]],
    [...L('eyeWide'), ['browInnerUp', 0.8], ...L('browOuterUp', 0.8), ['jawOpen', 0.3]], [['びっくり', 1]]],
  // stronger (2026-11-11, the author agreed): brows clearly worried, eyes lowered half way, mouth closed (no smile), a blush
  // morph when the model has one (the VRoid samples have none: their blush is a texture)
  shy: [[['Fcl_BRW_Sorrow', 1], ['Fcl_EYE_Close', 0.35], ['Fcl_EYE_Sorrow', 0.25], ['cheek', 1]],
    [['Fcl_BRW_Sorrow', 1], ['Fcl_EYE_Close', 0.35], ['Fcl_EYE_Sorrow', 0.25]], [['Fcl_BRW_Sorrow', 1], ['Fcl_EYE_Close', 0.35]],
    [['cheek', 1], ['eye_smile', 0.3], ['eye_brow_up', 0.8]], [['eye_smile', 0.3], ['eye_brow_up', 0.8]],
    [['browInnerUp', 0.9], ...L('eyeBlink', 0.3), ...L('eyeLookDown', 0.4)], [['照れ', 1], ['困る', 0.5]], [['照れ', 1]]],
  blink: [[['Fcl_EYE_Close', 1]], [['eye_close', 1]], [['blink_L', 1], ['blink_R', 1]], [...L('eyeBlink')], [['まばたき', 1]]],
  blinkLeft: [[['Fcl_EYE_Close_L', 1]], [['blink_L', 1]], [['eyeBlinkLeft', 1]], [['ウィンク', 1]]],
  blinkRight: [[['Fcl_EYE_Close_R', 1]], [['blink_R', 1]], [['eyeBlinkRight', 1]], [['ウィンク右', 1]]],
  aa: [[['Fcl_MTH_A', 1]], [['lip_a', 1]], [['jawOpen', 0.6], ['mouthFunnel', 0.1]], [['あ', 1]]],
  ih: [[['Fcl_MTH_I', 1]], [['lip_i', 1]], [['jawOpen', 0.2], ...L('mouthStretch', 0.5)], [['い', 1]]],
  ou: [[['Fcl_MTH_U', 1]], [['lip_u', 1]], [['mouthPucker', 0.8], ['jawOpen', 0.1]], [['う', 1]]],
  ee: [[['Fcl_MTH_E', 1]], [['lip_e', 1]], [['jawOpen', 0.25], ...L('mouthStretch', 0.7)], [['え', 1]]],
  oh: [[['Fcl_MTH_O', 1]], [['lip_o', 1]], [['mouthFunnel', 0.7], ['jawOpen', 0.4]], [['お', 1]]],
};
// Last resort, for faces whose morphs have no usable names (the 0.x sample's are numbers): mix the model's own
// expressions (their morph binds, scaled).
const MIXES = { shy: [['sad', 0.7], ['blink', 0.3]] };
const BLINK_SAFE = new Set(['happy', 'relaxed', 'shy']);   // (shy lowers the eyes half way itself)     // eyes already narrowed: blinking on top looks broken

// 'Face_Blendshape.Fcl_EYE_Close_L' -> 'fcleyeclosel'; 'eyeBlinkLeft' / 'EyeBlink_L' -> 'eyeblinkl'
export function norm(name) {
  let s = String(name).split('.').pop().toLowerCase().replace(/[^\p{L}\p{N}]/gu, '');
  return s.replace(/left$/, 'l').replace(/right$/, 'r');
}

function morphIndex(vrm) {
  const idx = new Map();                        // normalised morph name -> [{ mesh, index }]
  vrm.scene.traverse((o) => {
    if (!o.isMesh || !o.morphTargetDictionary) return;
    for (const [name, i] of Object.entries(o.morphTargetDictionary)) {
      const k = norm(name);
      if (!idx.has(k)) idx.set(k, []);
      idx.get(k).push({ mesh: o, index: i });
    }
  });
  return idx;
}

function makeExpression(name, binds) {
  const e = new VRMExpression(name);
  for (const b of binds) e.addBind(b);
  e.overrideBlink = BLINK_SAFE.has(name) ? 'blend' : 'none';
  return e;
}

export function completeExpressions(vrm, { force = false } = {}) {
  const em = vrm.expressionManager;
  const out = { aliased: {}, built: {}, missing: [] };
  if (!em) { out.missing = [...FACE_NAMES]; return out; }
  const morphs = morphIndex(vrm);
  const byNorm = new Map(em.expressions.map((e) => [norm(e.expressionName), e]));
  for (const name of FACE_NAMES) {
    const have = em.getExpression(name);
    if (have && !force) continue;
    if (have && force) em.unregisterExpression(have);
    // 1. the same expression under another name
    const src = force ? null : [name, ...(ALIASES[name] ?? [])].map((a) => byNorm.get(norm(a))).find((e) => e && e._binds?.length);
    if (src) {
      const e = makeExpression(name, src._binds);
      e.overrideMouth = src.overrideMouth; e.isBinary = src.isBinary;
      em.registerExpression(e);
      out.aliased[name] = src.expressionName;
      continue;
    }
    // 2. a recipe of morph targets
    const recipe = (RECIPES[name] ?? []).find((alt) => alt.every(([m]) => morphs.has(norm(m))));
    if (recipe) {
      const binds = recipe.flatMap(([m, w]) => morphs.get(norm(m)).map(({ mesh, index }) =>
        new VRMExpressionMorphTargetBind({ primitives: [mesh], index, weight: w })));
      em.registerExpression(makeExpression(name, binds));
      out.built[name] = recipe.map(([m, w]) => `${m}*${w}`).join(' + ');
      continue;
    }
    // 3. a mix of the model's own expressions
    const mix = MIXES[name]?.map(([n, w]) => [em.getExpression(n), w]);
    if (mix && mix.every(([e]) => e)) {
      const binds = mix.flatMap(([e, w]) => (e._binds ?? []).filter((b) => b.primitives && b.index !== undefined)
        .map((b) => new VRMExpressionMorphTargetBind({ primitives: b.primitives, index: b.index, weight: b.weight * w })));
      if (binds.length) {
        em.registerExpression(makeExpression(name, binds));
        out.built[name] = MIXES[name].map(([n, w]) => `${n}*${w}`).join(' + ');
        continue;
      }
    }
    out.missing.push(name);
  }
  return out;
}

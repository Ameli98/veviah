import * as THREE from 'three';
import { VRMExpression, VRMExpressionMorphTargetBind } from '@pixiv/three-vrm';
// Face from an emotion word, with blinking (2026-10-27). Uses the VRM's own preset expressions (VRM 1.0 names;
// three-vrm maps 0.x presets joy / sorrow / fun / angry to happy / sad / relaxed / angry), so it works on any model.
// The mouth follows the voice (setMouth, lipsync.js; 2026-11-05): presets that open the mouth too (VRoid's happy) are
// always shown through a twin whose mouth stays nearly closed (TALK_OPEN), and no preset blocks the mouth shapes.
//   const face = new FaceController(vrm); face.setEmotion('happy'); ... each frame: face.update(dt) BEFORE vrm.update(dt)

// emotion word -> { preset: weight }. The words are the TTS reference labels (an earlier voice project: neutral, happy, angry,
// sad, shy, surprised), so one word sets face and voice. VRM has no 'shy' preset: face_rig.js makes one; a model
// where it could not be made gets a soft relaxed face instead.
// How much an expression moves the upper face (brows, eyes; the mouth region = where 'aa' moves is left out), relative
// to how much a blink moves it (the face's own size): per vertex the bind-weighted displacement, the 95th percentile
// of the moved vertices. Used to strengthen presets that are faint on a model (2026-11-11, the author: test2's angry / sad,
// AvatarSample_D's faces were hard to read).
export function upperStrength(vrm, name) {
  const em = vrm.expressionManager, e = em?.getExpression(name), A = aaDisplacement(em?.getExpression('aa'));
  if (!e) return 0;
  const vals = [];
  for (const b of e._binds ?? []) for (const g of new Set((b.primitives ?? []).map((m) => m.geometry))) {
    const d = g.morphAttributes?.position?.[b.index];
    if (!d) continue;
    const a = A.get(g);
    let maxA = 0;
    if (a) for (let i = 0; i < a.d.count; i++) maxA = Math.max(maxA, Math.hypot(a.d.getX(i), a.d.getY(i), a.d.getZ(i)));
    for (let i = 0; i < d.count; i++) {
      if (a && maxA > 0 && Math.hypot(a.d.getX(i), a.d.getY(i), a.d.getZ(i)) > MASK_FRAC * maxA) continue;   // mouth
      const v = Math.hypot(d.getX(i), d.getY(i), d.getZ(i)) * (b.weight ?? 1);
      if (v > 1e-5) vals.push(v);
    }
  }
  if (!vals.length) return 0;
  vals.sort((x, y) => x - y);
  return vals[Math.floor(0.95 * (vals.length - 1))];
}
export const EMOTIONS = {
  neutral: {}, happy: { happy: 1 }, angry: { angry: 1 }, sad: { sad: 1 }, shy: { shy: 1 }, surprised: { surprised: 1 },
};
const PRESETS = ['happy', 'relaxed', 'sad', 'angry', 'surprised', 'shy'];
const EASE = 0.25;            // s to reach a new expression
const BLINK = 0.14;           // s for one blink
const MOUTH = ['aa', 'ih', 'ou', 'ee', 'oh'];
// While speaking (2026-11-06): a preset that opens the mouth (VRoid's happy / surprised / sad are ONE morph each,
// Fcl_ALL_*, eyes and mouth together) would hold the mouth wide open over the lip sync (the author's screenshot on test1).
// At load each such preset gets a talking twin (`<preset>Talk`): a copy of its morph targets whose displacement is
// scaled down only where the 'aa' shape moves the face (the mouth region, mask = |aa displacement| / (MASK_FRAC x its
// largest), capped at 1), by s = TALK_OPEN / the preset's mouth opening, so its mouth opens at most like aa TALK_OPEN;
// eyes and brows are untouched. While a voice plays the face crossfades from the preset to its twin.
// (Before: the whole preset dropped to 0.75 and the lip sync shrank, which also weakened the smiling eyes.)
const TALK_OPEN = 0.3;
const MASK_FRAC = 0.15;

// How far a preset opens the mouth, by shape (2026-11-05): its morph displacement projected onto the 'aa' shape's own
// displacement per vertex (sum p.a / sum a.a): ~1 = it opens the mouth as aa does. Measured: VRoid models happy
// 0.8-1.7, surprised 0.8-1.6, sad 0.5-0.65, relaxed / angry / shy ~0; Seed-san all ~0. Sharing a morph target is not the
// test (VRoid's happy has its own morph), nor the size of the displacement (every preset moves the mouth corners).
const OPENS = 0.3;
function aaDisplacement(aa) {
  const A = new Map();                         // geometry -> {d, w}
  for (const b of aa?._binds ?? []) for (const m of b.primitives ?? []) {
    const d = m.geometry?.morphAttributes?.position?.[b.index];
    if (d) A.set(m.geometry, { d, w: b.weight ?? 1 });
  }
  return A;
}
function mouthOpening(expr, A) {
  let num = 0, den = 0;
  for (const [g, { d, w }] of A) {
    const own = [];
    for (const b of expr?._binds ?? []) for (const m of b.primitives ?? []) {
      const q = m.geometry === g ? g.morphAttributes?.position?.[b.index] : null;
      if (q) own.push([q, b.weight ?? 1]);
    }
    for (let i = 0; i < d.count; i++) {
      const ax = d.getX(i) * w, ay = d.getY(i) * w, az = d.getZ(i) * w, l2 = ax * ax + ay * ay + az * az;
      if (l2 < 1e-8) continue;
      let px = 0, py = 0, pz = 0;
      for (const [q, k] of own) { px += q.getX(i) * k; py += q.getY(i) * k; pz += q.getZ(i) * k; }
      num += px * ax + py * ay + pz * az; den += l2;
    }
  }
  return den > 0 ? num / den : 0;
}
// the talking twin of a preset (see TALK_OPEN); registered as `<name>Talk`, returns its name
function talkTwin(vrm, name, expr, A, open) {
  const s = Math.min(1, TALK_OPEN / open), made = new Map(), binds = [];
  // A face is several primitives (face, eye white, lashes, ...), each its own geometry, sharing one vertex buffer
  // (test1 / test2) or not (the default model: 7 geometries with their own vertex counts). Every geometry of a bind
  // gets its own twin morph (the first version changed only the first primitive: the twin lost its smiling eyes).
  const attrOf = new Map();                      // source morph attribute -> twin attribute
  for (const b of expr._binds ?? []) {
    if (!b.primitives || b.index === undefined) continue;
    const geos = [...new Set(b.primitives.map((m) => m.geometry))];
    if (!geos.some((g) => A.get(g))) { binds.push(new VRMExpressionMorphTargetBind({ primitives: b.primitives, index: b.index, weight: b.weight })); continue; }
    for (const g of geos) {
      const src = g.morphAttributes?.position?.[b.index];
      if (!src) continue;
      // the mask comes from aa's displacement on THIS geometry (the default model's face is 7 geometries with their
      // own vertex counts; one mask for all put garbage on the eyes); a geometry aa does not move keeps its shape
      const a = A.get(g);
      if (!attrOf.has(src)) {
        let maxA = 0;
        if (a) for (let i = 0; i < a.d.count; i++) maxA = Math.max(maxA, Math.hypot(a.d.getX(i), a.d.getY(i), a.d.getZ(i)));
        const out = new Float32Array(src.count * 3);
        for (let i = 0; i < src.count; i++) {
          const ad = a && maxA > 0 ? Math.hypot(a.d.getX(i), a.d.getY(i), a.d.getZ(i)) : 0;
          const f = 1 - (1 - s) * Math.min(1, ad / (MASK_FRAC * maxA || 1));
          out[3 * i] = src.getX(i) * f; out[3 * i + 1] = src.getY(i) * f; out[3 * i + 2] = src.getZ(i) * f;
        }
        attrOf.set(src, new THREE.BufferAttribute(out, 3));
      }
      const key = `${g.uuid}:${b.index}`;
      if (!made.has(key)) {
        g.morphAttributes.position.push(attrOf.get(src));
        const nrm = g.morphAttributes.normal;      // keep the arrays the same length
        if (nrm) nrm.push(nrm[b.index]);
        made.set(key, g.morphAttributes.position.length - 1);
        vrm.scene.traverse((o) => { if (o.isMesh && o.geometry === g) o.updateMorphTargets(); });
      }
      // one bind per geometry: primitives may hold different numbers of morph targets (the default model's do), so the
      // twin's index differs between them
      binds.push(new VRMExpressionMorphTargetBind({ primitives: b.primitives.filter((m) => m.geometry === g), index: made.get(key), weight: b.weight }));
    }
  }
  const e = new VRMExpression(`${name}Talk`);
  e.overrideMouth = 'none'; e.overrideBlink = expr.overrideBlink; e.overrideLookAt = expr.overrideLookAt;
  for (const bd of binds) e.addBind(bd);
  vrm.expressionManager.registerExpression(e);
  return e.expressionName;
}

export class FaceController {
  constructor(vrm) {
    this.em = vrm.expressionManager;
    this.target = {};
    this.cur = Object.fromEntries(PRESETS.map((p) => [p, 0]));
    this.nextBlink = 1 + Math.random() * 3;
    this.blinkT = -1;
    this.eyes = 0;               // eyes closed by the motion (player.eyesClosed), 0..1; no blinks start while closed
    this.blinks = true;          // false = no random blinks (deterministic batch renders, main.js)
    this.mouth = Object.fromEntries(MOUTH.map((m) => [m, 0]));
    // no preset may block the mouth shapes; presets that open the mouth get a talking twin (TALK_OPEN)
    // ... nor the eyes: a motion's "eyes": closed must close them under a shy / relaxed face too (2026-11-10, the author saw
    // Sakura's wink missing: the presets' overrideBlink blocked the blink shape); natural blinking under those faces is
    // already damped in update()
    for (const p of PRESETS) { const e = this.em?.getExpression(p); if (e) { e.overrideMouth = 'none'; e.overrideBlink = 'none'; } }
    const A = aaDisplacement(this.em?.getExpression('aa'));
    this.twin = {};              // preset -> its talking twin's name
    this.opening = {};
    for (const p of PRESETS) {
      const e = this.em?.getExpression(p);
      if (!e) continue;
      const open = (this.opening[p] = mouthOpening(e, A));
      if (open > OPENS) this.twin[p] = this.em.getExpression(`${p}Talk`) ? `${p}Talk` : talkTwin(vrm, p, e, A, open);
    }
  }

  // the lip sync's weights for this frame ({aa, ih, ou, ee, oh}, all 0 = closed); speaking = a voice is playing
  setMouth(w, speaking = false) {
    for (const m of MOUTH) this.mouth[m] = w?.[m] ?? 0;
    this.speaking = speaking;
  }

  // name: an EMOTIONS word (unknown words = neutral); scale: 0..1 of its full strength
  setEmotion(name, scale = 1) {
    let e = EMOTIONS[String(name ?? '').toLowerCase()] ?? {};
    if (e.shy && !this.em?.getExpression('shy')) e = { relaxed: 0.6 };
    this.target = Object.fromEntries(Object.entries(e).map(([k, v]) => [k, v * scale]));
  }

  update(dt) {
    if (!this.em) return;
    const k = Math.min(1, dt / EASE);
    for (const p of PRESETS) {
      this.cur[p] += ((this.target[p] ?? 0) - this.cur[p]) * k;
      // always the twin (2026-11-06, the author: every emotion's mouth as small as while speaking; test1's happy held it
      // wide open), so the lip sync and the emotions never fight over the mouth
      if (this.twin[p]) { this.em.setValue(p, 0); this.em.setValue(this.twin[p], this.cur[p]); }
      else this.em.setValue(p, this.cur[p]);
    }
    for (const m of MOUTH) if (this.em.getExpression(m)) this.em.setValue(m, this.mouth[m]);
    // blink every 2-6 s; not while the eyes are already closed / squinted by a happy, relaxed or shy face
    this.nextBlink -= dt;
    if (this.blinks && this.nextBlink <= 0 && this.blinkT < 0 && this.eyes < 0.5) { this.blinkT = 0; this.nextBlink = 2 + Math.random() * 4; }
    let b = 0;
    if (this.blinkT >= 0) {
      this.blinkT += dt;
      b = Math.sin(Math.PI * Math.min(this.blinkT / BLINK, 1));
      if (this.blinkT >= BLINK) this.blinkT = -1;
    }
    this.em.setValue('blink', Math.max(b * (1 - Math.max(this.cur.happy, this.cur.relaxed, this.cur.shy)), this.eyes));
  }

  setEyes(closed) { this.eyes = Math.max(0, Math.min(1, closed || 0)); }
}

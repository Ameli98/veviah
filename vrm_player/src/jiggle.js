// Soft-tissue jiggle (2026-11-04, the author: the chest should sway with the body's movement, also while the window is
// dragged; 'off' / 'weak' / 'medium' / 'strong' in the desktop app, weak by default; off for judge renders).
//
// Chest bones are found by STRUCTURE, never by name: a non-humanoid bone that is a direct child of the chest
// (upperChest, else chest), lies in front of it and to one side, has a mirror partner on the other side, and has
// skin vertices weighted to it (or to its children). VRoid models have such bones (J_Sec_*_Bust1); some carry stiff
// spring bones (stiffness 0.75, gravity 0), others none. Their spring joints are removed and this controller drives
// the bones instead, so every model sways alike.
//
// Each bone's tip is a point mass on a damped spring (natural frequency HZ, damping ratio per level) attached to where
// the tip would be at rest; the lag of the mass behind its anchor shifts the bone. The anchor follows the body in the
// world, so steps, turns, nods and sitting down all shake it. Dragging the window: shift(d) moves the masses (and
// every world-space spring tail of hair / skirts, springTailsShift) by -d, which is what moving the body by +d does.
// Hips (butt) jiggle is not done: no model has bones there (a shader method later).
import * as THREE from 'three';

// gain 1 = the physical lag of the mass (anchor acceleration / (2 pi HZ)^2); peak lag at gain 1 on test2: step + knee
// lift 0.9 cm, bounce on the heels 2.0, turn 180 1.0, stretch / sitting sway < 0.1, wave 0 (arms hardly move the chest).
// maxCm = soft limit of the shift. zeta 2026-11-05: 0.30 / 0.22 / 0.15 -> 0.20 / 0.15 / 0.10 (the author: bounce a little more) -> 0.28 / 0.21 / 0.14 (amplitude fine, damping a little higher) -> 0.40 / 0.30 / 0.20 (more) -> 0.45 for every level (a reference game clip, measured: ~3 Hz, one small overshoot, settled within ~0.4 s; the levels differ only in size)
export const LEVELS = {
  off:    null,
  weak:   { gain: 1.0, zeta: 0.45, maxCm: 1.0, shape: 0.35 },
  medium: { gain: 1.6, zeta: 0.45, maxCm: 1.6, shape: 0.65 },
  strong: { gain: 2.5, zeta: 0.45, maxCm: 2.5, shape: 1.0 },
};
const HZ = 3.0;                  // natural frequency of the mass
const SUB = 1 / 240;             // integration step (s)
const JUMP = 0.12;               // m: an anchor moving more than this in one frame (a new model / motion switch / teleport) resets

const _v = new THREE.Vector3(), _a = new THREE.Vector3(), _u = new THREE.Vector3(), _t = new THREE.Vector3();
const _rel = new THREE.Vector3();

// bones (and their children) that have at least `min` skin vertices weighted > 0.2
function weightedCount(vrm, bone) {
  const set = new Set();
  bone.traverse((b) => set.add(b));
  let n = 0;
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const idx = new Set(o.skeleton.bones.map((b, i) => (set.has(b) ? i : -1)).filter((i) => i >= 0));
    if (!idx.size) return;
    const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
    const used = o.geometry.index ? new Set(o.geometry.index.array) : null;
    for (let i = 0; i < si.count; i++) {
      if (used && !used.has(i)) continue;
      for (let k = 0; k < 4; k++) if (sw.getComponent(i, k) > 0.2 && idx.has(si.getComponent(i, k))) { n++; break; }
    }
  });
  return n;
}

export function findChestBones(vrm) {
  const H = vrm.humanoid;
  const human = new Set(Object.keys(H.humanBones).map((n) => H.getRawBoneNode(n)).filter(Boolean));
  const chest = H.getRawBoneNode('upperChest') ?? H.getRawBoneNode('chest');
  if (!chest) return [];
  vrm.scene.updateMatrixWorld(true);
  const toModel = vrm.scene.matrixWorld.clone().invert();
  const at = (o) => o.getWorldPosition(new THREE.Vector3()).applyMatrix4(toModel);
  const c = at(chest);
  const fz = vrm.meta?.metaVersion === '0' ? -1 : 1;          // 0.x models face -z in their own space
  const cand = chest.children.filter((b) => b.isBone && !human.has(b)).map((b) => ({ b, p: at(b).sub(c).multiply(new THREE.Vector3(fz, 1, fz)) }))
    .filter(({ p }) => p.z > 0.02 && Math.abs(p.x) > 0.015 && Math.abs(p.x) < 0.2 && Math.abs(p.y) < 0.2)
    // the bone itself points forward (to its child): a breast does, a strap or ribbon hangs down (Seed-san's
    // backpack ropes sit at the same place in front of the chest)
    .filter(({ b }) => {
      const k = b.children.find((x) => x.isBone);
      if (!k) return false;
      const d = at(k).sub(at(b)).multiply(new THREE.Vector3(fz, 1, fz));
      return d.length() > 0.01 && d.z > 0.7 * d.length();
    });
  const out = [];
  for (const L of cand) {
    if (L.p.x <= 0) continue;
    const R = cand.find((r) => r.p.x < 0 && Math.abs(r.p.x + L.p.x) < 0.02 && Math.abs(r.p.y - L.p.y) < 0.02 && Math.abs(r.p.z - L.p.z) < 0.02);
    if (!R) continue;
    if (weightedCount(vrm, L.b) < 10 || weightedCount(vrm, R.b) < 10) continue;
    out.push(L.b, R.b);
  }
  return out;
}

// hair / skirt spring tails: moving the body by d = moving them by -d (window drag). VRoid sets a `center` node (the
// hips / root) on every spring so that moving the whole body does not swing the hair; those tails live in the
// center's space, so d is turned into it (direction and scale). (2026-11-05, the author saw no hair / skirt sway: the
// first version skipped every spring with a center, i.e. all of test1's and test2's.)
const _inv = new THREE.Matrix4(), _o = new THREE.Vector3(), _dc = new THREE.Vector3();
export function springTailsShift(vrm, d) {
  const m = vrm.springBoneManager;
  if (!m) return;
  const byCenter = new Map();
  for (const j of m.joints) {
    if (!j._currentTail || !j._prevTail) continue;
    let dc = byCenter.get(j.center ?? null);
    if (!dc) {
      if (j.center) {
        _inv.copy(j.center.matrixWorld).invert();
        dc = _dc.copy(d).applyMatrix4(_inv).sub(_o.set(0, 0, 0).applyMatrix4(_inv)).clone();
      } else dc = d.clone();
      byCenter.set(j.center ?? null, dc);
    }
    j._currentTail.sub(dc); j._prevTail.sub(dc);
  }
}

// Hair (any spring) and vertical body moves (2026-11-05, jumping): three-vrm springs feel their own gravityPower (VRoid
// 0-1 against a real 9.8), so a body falling freely in a jump left its hair floating up and the landing flung the twin
// tails out sideways. The tails are carried along FOLLOW_Y of the hips' vertical move each frame (as if the hair felt
// most of the real gravity / inertia), before vrm.update. Horizontal moves keep their full lag. st = {} per model.
export const FOLLOW_Y = 0.7;
const _fy = new THREE.Vector3();
export function followVertical(vrm, st, k = FOLLOW_Y) {
  const h = vrm.humanoid?.getNormalizedBoneNode('hips');
  if (!h) return;
  const y = h.position.y;
  if (st.y !== undefined && Math.abs(y - st.y) < 0.2) springTailsShift(vrm, _fy.set(0, -k * (y - st.y), 0));
  st.y = y;
}

// Air damping for the model's own springs (hair, sleeves, skirts): VRoid's skirts / sleeves have dragForce 0.05 and
// swing back and forth like a spring after a turn or a drag (test2's skirt swung back 83 deg past rest); cloth and
// hair in air settle without that (2026-11-05, the author: like a skirt in the wind, when the wind stops it does not
// oscillate). Measured with tools/drag_springs_check.mjs (test2, 10 cm in 0.2 s): at 0.5 the skirt does not swing
// back past rest, hair 1-6 deg (0.7: none, but the hair then follows more stiffly).
export const MIN_DRAG = 0.5;
export function dampSprings(vrm, min = MIN_DRAG) {
  for (const j of vrm.springBoneManager?.joints ?? []) if (j.settings.dragForce < min) j.settings.dragForce = min;
}

// One damped spring mass on an anchor that moves with the body (substeps, linear anchor path over the frame); returns
// the lag x gain, softly limited to maxCm (tanh), in s.out (world). Shared by the chest bones and the buttocks.
function step(s, A, dt, { gain, zeta, maxCm }) {
  const w = 2 * Math.PI * HZ;
  if (!s.x || A.distanceTo(s.prevA) > JUMP) { s.x = A.clone(); s.v.set(0, 0, 0); s.prevA = A.clone(); }
  const vA = _v.subVectors(A, s.prevA).divideScalar(dt);
  const n = Math.max(1, Math.ceil(dt / SUB)), h = dt / n;
  for (let i = 1; i <= n; i++) {
    _t.copy(s.prevA).lerp(A, i / n);
    const acc = _u.subVectors(_t, s.x).multiplyScalar(w * w).addScaledVector(_rel.subVectors(vA, s.v), 2 * zeta * w);
    s.v.addScaledVector(acc, h);
    s.x.addScaledVector(s.v, h);
  }
  s.prevA.copy(A);
  const lag = (s.out ??= new THREE.Vector3()).subVectors(s.x, A).multiplyScalar(gain);
  const len = lag.length(), max = maxCm / 100;
  s.cm = len * 100;                       // for tests (before the limit)
  if (len > 1e-6) lag.multiplyScalar(max * Math.tanh(len / max) / len);
  return lag;
}

// Buttocks (2026-11-05, the author): no bones, so the lag of a mass per side is a displacement in the vertex shader
// (limb_push.js: hsinButt uniforms, per-vertex weights by shape). Anchor = the buttock's centre carried by the hips.
// No sway with the bottom on the floor (its own weight rests on it; the sitting skirt is a precomputed drape): faded
// out by the hips' height over the floor, BUTT_FLOOR fractions of the rest height (kneeling upright 0.49 keeps it,
// seiza 0.37 on its heels and the floor sits do not). BUTT_SCALE = its share of the level's limit (less loose than the chest).
// Moving back, the bottom would come out through loose clothes over it: limb_push.js pushes them back where it reaches
// them (hsinButtBack = the body's backward, world). (A 0.6 cm cap on moving back did this job before the clothes
// stopped following, 2026-11-05.)
const BUTT_FLOOR = [0.40, 0.47], BUTT_SCALE = 0.8;
// Each buttock also follows its own thigh (2026-11-05, from a reference clip of a game's walk: the cheek on the side
// of the leg going back lifts and bulges, the other drops; most of what shows is this, not inertia): from the
// thigh's forward angle against its rest (hips frame), extension e -> lift / bulge up to EXT_* at EXT_DEG, flexion f
// -> lift / bulge down to FLEX_* at FLEX_DEG (capped at SHAPE_CAP x those), times the level's `shape`; the target goes
// through the same damped spring, so a footfall bounces once. In metres, strong.
// In place the legs hardly ever go back (stepping / marching flex them ~20 deg forward), so the cheek of the STANDING
// leg also lifts and bulges with the other thigh's flexion (its glute tightens): STAND_LIFT / STAND_BULGE at FLEX_DEG.
// That gives the reference's left / right alternation without walking (the author, 2026-11-05: strong ~3 cm).
const EXT_LIFT = 0.03, EXT_BULGE = 0.018, EXT_DEG = 20, FLEX_LIFT = -0.02, FLEX_BULGE = -0.012, FLEX_DEG = 25, SHAPE_CAP = 1.3;
const STAND_LIFT = 0.025, STAND_BULGE = 0.015;      // hips at rest height x: kneel 0.49, seiza 0.37, floor sits 0.14-0.22

export class Jiggle {
  constructor(vrm, level = 'weak') {
    this.vrm = vrm;
    this.bones = findChestBones(vrm).map((b) => {
      const child = b.children.find((c) => c.isBone);
      // tip in the bone's own space: its child, else 5 cm along the bone's forward
      const tip = child ? child.position.clone() : new THREE.Vector3(0, 0, 0.05);
      if (tip.length() < 0.02) tip.setLength(0.05);
      return { b, rest: b.quaternion.clone(), restP: b.position.clone(), tip, x: null, v: new THREE.Vector3(), prevA: null };
    });
    // their own spring joints (and those of the bones under them) go: this controller drives them
    const mine = new Set();
    for (const { b } of this.bones) b.traverse((o) => mine.add(o));
    const m = vrm.springBoneManager;
    if (m) for (const j of [...m.joints]) if (mine.has(j.bone)) m.deleteJoint(j);
    const B = vrm.scene.userData.hsinButt, U = vrm.scene.userData.hsinPush;
    this.butt = B && U?.hsinButt ? B.centres.map((c) => ({ c, x: null, v: new THREE.Vector3(), prevA: null })) : [];
    if (this.butt.length) {
      B.bone.updateWorldMatrix(true, false);
      this.buttRestY = B.bone.getWorldPosition(new THREE.Vector3()).y;
      const inv = B.bone.matrixWorld.clone().invert();
      // backward in the hips' frame, from the world at load (every model faces +z then; rotateVRM0 turns 0.x ones)
      this.buttBack = new THREE.Vector3(0, 0, -1).transformDirection(inv);
      this.buttDown = new THREE.Vector3(0, -1, 0).transformDirection(inv);
      const H = vrm.humanoid;
      this.thighs = [['leftUpperLeg', 'leftLowerLeg'], ['rightUpperLeg', 'rightLowerLeg']].map(([a, b]) => ({ a: H.getRawBoneNode(a), b: H.getRawBoneNode(b) }));
      this.thighRest = this.thighs.map((t) => this._thighAngle(t, B.bone));
      this.shape = [0, 1].map(() => ({ lift: [0, 0], bulge: [0, 0] }));   // spring state [value, velocity]
    }
    console.info(`jiggle: ${this.bones.length} chest bones, ${this.butt.length ? 'buttocks' : 'no buttocks'}`, this.bones.map((s) => s.b.name).join(' '));
    this.setLevel(level);
  }

  // the thigh's forward angle in the hips' sagittal plane (rad; + = forward / flexion)
  _thighAngle(t, hips) {
    const d = t.b.getWorldPosition(new THREE.Vector3()).sub(t.a.getWorldPosition(new THREE.Vector3()));
    const back = this.buttBack.clone().transformDirection(hips.matrixWorld), down = this.buttDown.clone().transformDirection(hips.matrixWorld);
    return Math.atan2(-d.dot(back), d.dot(down));
  }

  setLevel(level) {
    this.level = LEVELS[level] !== undefined ? level : 'weak';
    this.p = LEVELS[this.level];
    for (const s of this.bones) { s.x = null; s.b.quaternion.copy(s.rest); s.b.position.copy(s.restP); s.b.updateMatrix(); }
    for (const s of this.butt ?? []) s.x = null;
    for (const sh of this.shape ?? []) { sh.lift = [0, 0]; sh.bulge = [0, 0]; }
    for (const u of this.vrm.scene.userData.hsinPush?.hsinButtShape?.value ?? []) u.set(0, 0);
    for (const u of this.vrm.scene.userData.hsinPush?.hsinButt.value ?? []) u.set(0, 0, 0);
  }

  shift(d) { for (const s of [...this.bones, ...this.butt]) s.x?.sub(d); }   // the body moved by d: the masses stay behind

  // after vrm.update (the chest is posed), before rendering
  update(dt) {
    if (!this.p || dt <= 0) return;
    for (const s of this.bones) {
      s.b.quaternion.copy(s.rest); s.b.position.copy(s.restP); s.b.updateMatrix();
      s.b.parent.updateWorldMatrix(true, false);
      s.b.updateWorldMatrix(false, false);
      const A = _a.copy(s.tip).applyMatrix4(s.b.matrixWorld);          // where the tip is at rest now
      // move the bone by the lag: the breast mesh is weighted mostly to the bone's child, so tilting the bone moved it
      // little (15 deg: ~1 cm at the outline on test1); a shift moves every vertex weighted to the bone by its weight,
      // like the mass of the tissue lagging behind
      const lag = step(s, A, dt, this.p);
      const O = s.b.getWorldPosition(new THREE.Vector3());
      const local = s.b.parent.worldToLocal(O.clone().add(lag)).sub(s.b.parent.worldToLocal(O.clone()));
      s.b.position.copy(s.restP).add(local);
      s.b.updateMatrix();
      s.b.updateMatrixWorld(true);
    }
    if (this.butt.length) {
      const B = this.vrm.scene.userData.hsinButt, U = this.vrm.scene.userData.hsinPush.hsinButt.value;
      B.bone.updateWorldMatrix(true, false);
      const y = B.bone.getWorldPosition(_a).y / this.buttRestY;
      const t = Math.min(1, Math.max(0, (y - BUTT_FLOOR[0]) / (BUTT_FLOOR[1] - BUTT_FLOOR[0]))), fade = t * t * (3 - 2 * t);
      const p = { ...this.p, maxCm: this.p.maxCm * BUTT_SCALE };
      const back = this.vrm.scene.userData.hsinPush.hsinButtBack.value.copy(this.buttBack).transformDirection(B.bone.matrixWorld);
      this.vrm.scene.userData.hsinPush.hsinHipsRot.value.setFromMatrix4(B.bone.matrixWorld);   // limb_push.js hsinButtN
      this.butt.forEach((s, k) => {
        const A = _a.copy(s.c).applyMatrix4(B.bone.matrixWorld);
        U[k].copy(step(s, A, dt, p)).multiplyScalar(fade);
      });
      const P = this.vrm.scene.userData.hsinPush;
      P.hsinButtUp.value.copy(this.buttDown).transformDirection(B.bone.matrixWorld).negate();
      const w = 2 * Math.PI * HZ, z = this.p.zeta, n = Math.max(1, Math.ceil(dt / SUB)), h = dt / n;
      const spring = (st, target) => { for (let i = 0; i < n; i++) { st[1] += (w * w * (target - st[0]) - 2 * z * w * st[1]) * h; st[0] += st[1] * h; } return st[0]; };
      const deg = this.thighs.map((t, k) => (this._thighAngle(t, B.bone) - this.thighRest[k]) * 180 / Math.PI);
      const ext = deg.map((d) => Math.min(SHAPE_CAP, Math.max(0, -d) / EXT_DEG));
      const flex = deg.map((d) => Math.min(SHAPE_CAP, Math.max(0, d) / FLEX_DEG));
      this.thighs.forEach((t, k) => {
        const o = 1 - k, sc = this.p.shape ?? 0, sh = this.shape[k];
        const stand = Math.max(0, flex[o] - flex[k]);          // the other leg lifted, this one bears the weight
        const lift = spring(sh.lift, sc * (EXT_LIFT * ext[k] + FLEX_LIFT * flex[k] + STAND_LIFT * stand));
        const bulge = spring(sh.bulge, sc * (EXT_BULGE * ext[k] + FLEX_BULGE * flex[k] + STAND_BULGE * stand));
        P.hsinButtShape.value[k].set(lift * fade, bulge * fade);
      });
    }
  }
}

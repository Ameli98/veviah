// Applies a motion JSON to a VRM's normalized humanoid bones.
//
// Motion JSON: { fps, joints: [VRM humanoid bone names], quat_order: "xyzw", rotations: [T][J][4],
//               root: [T][3] (optional) }
// Rotations are local, in the VRM 1.0 normalized humanoid frame (zero = T-pose, faces +z).
// root = hips position minus its rest position, in units of the rest hips height, so one motion fits
// models of any size (this player multiplies by its own model's hips height). Without root the hips stay put.
// Only the bones listed in `joints` are driven, plus the finger bones (from the optional `hands`
// field, see HAND_SHAPES); expressions (mouth etc.) are never touched here.
import * as THREE from 'three';

// Optional VRM bones a model may lack, and the bone that takes over their rotation.
// 'post': parent' = parent * missing (torso chain). 'pre': child' = missing * child.
// Order matters: upperChest folds into chest before chest folds into spine.
const FOLD = [
  ['upperChest', 'chest', 'post'], ['chest', 'spine', 'post'], ['neck', 'head', 'pre'],
  ['leftShoulder', 'leftUpperArm', 'pre'], ['rightShoulder', 'rightUpperArm', 'pre'],
];

// Hand shapes. Fingers are not part of the 13 motion joints: a motion may carry, per frame and per
// hand, the name of one of these shapes, and the player poses the 15 finger bones from this table.
// Values: thumb fold (across the palm), curl of index, middle, ring, little (0 = straight, 1 = closed),
// finger spread, thumb drop (0 = level with the palm, 1 = swung down in front of the curled fingers).
export const HAND_SHAPES = {
  relaxed:   [0.35, 0.40, 0.50, 0.55, 0.60, 0.15, 0.35],   // (was 0.30 0.25 0.30 0.35 0.40 0.2 0.3: stiff, the author 2026-11-04)
  open:      [0.00, 0.03, 0.03, 0.03, 0.03, 1.0, 0.0],
  fist:      [0.60, 1.00, 1.00, 1.00, 1.00, 0.0, 1.0],
  point:     [0.60, 0.08, 1.00, 1.00, 1.00, 0.0, 1.0],
  peace:     [0.60, 0.08, 0.08, 1.00, 1.00, 1.0, 1.0],
  thumbs_up: [0.00, 1.00, 1.00, 1.00, 1.00, 0.0, 0.0],
  flat:      [0.10, 0.02, 0.02, 0.02, 0.02, -0.45, 0.0],   // straight, together (joined palms; 2026-11-06; -0.15 still left gaps)
  loose:     [0.15, 0.15, 0.20, 0.25, 0.30, 0.40, 0.15],   // lightly bent, a little spread (hands raised in surprise; Cecilia head_pat K3, 2026-11-07)
};
const DEFAULT_HAND = 'relaxed';
// Degrees at curl = 1. Tighter than this and the fingertips sink into the palm on the sample models.
const FINGER_SEGS = [['Proximal', 68], ['Intermediate', 88], ['Distal', 48]];
const THUMB_SEGS = [['Metacarpal', 25], ['Proximal', 30], ['Distal', 50]];
const FINGER_SPREAD = { Index: 1, Middle: 0, Ring: -1, Little: -2 };              // x 8 degrees at spread = 1
const RAD = Math.PI / 180;
const _qy = new THREE.Quaternion(), _qz = new THREE.Quaternion();
const Y = new THREE.Vector3(0, 1, 0);
const THUMB_DROP = 40;                                                              // degrees at drop = 1

// Pose one hand. In the T-pose the fingers point along the arm (left +x, right -x), palms down and
// thumbs forward (+z): fingers curl about z, the thumb folds across the palm about y and drops about x,
// fingers spread about y.
export function applyHandShape(vrm, side, p) {
  const s = vrm.meta?.metaVersion === '0' ? -1 : 1;          // same x/z flip as motions
  const d = side === 'left' ? -1 : 1;
  const z = new THREE.Vector3(0, 0, s);
  const node = (name) => vrm.humanoid.getNormalizedBoneNode(`${side}${name}`);
  for (const [seg, deg] of THUMB_SEGS) {
    const n = node(`Thumb${seg}`);
    if (!n) continue;
    n.quaternion.setFromAxisAngle(Y, -d * p[0] * deg * RAD);
    // the drop turns the thumb about the hand's long axis (x), at its base only
    if (seg === 'Metacarpal') n.quaternion.premultiply(_qz.setFromAxisAngle(new THREE.Vector3(s, 0, 0), p[6] * THUMB_DROP * RAD));
  }
  Object.entries(FINGER_SPREAD).forEach(([finger, spread], i) => {
    for (const [seg, deg] of FINGER_SEGS) {
      const n = node(`${finger}${seg}`);
      if (!n) continue;
      _qz.setFromAxisAngle(z, d * p[i + 1] * deg * RAD);
      if (seg === 'Proximal') n.quaternion.copy(_qy.setFromAxisAngle(Y, d * spread * 8 * p[5] * RAD)).multiply(_qz);
      else n.quaternion.copy(_qz);
    }
  });
}

export class MotionPlayer {
  constructor(vrm) {
    this.vrm = vrm;
    this.motion = null;
    // VRM 0.x models are turned 180 deg about y by VRMUtils.rotateVRM0, so a rotation written for
    // a +z-facing character must have its x and z components negated (checked on three-vrm-girl, a 0.x model).
    this.flipXZ = vrm.meta?.metaVersion === '0';
    this._a = new THREE.Quaternion();
    this._b = new THREE.Quaternion();
  }

  setMotion(motion) {
    const { fps, joints, rotations } = motion;
    if (!fps || !Array.isArray(joints) || !Array.isArray(rotations) || rotations.length === 0) {
      throw new Error('motion needs fps, joints and rotations');
    }
    if (motion.quat_order && motion.quat_order !== 'xyzw') throw new Error(`unsupported quat_order ${motion.quat_order}`);
    if (rotations[0].length !== joints.length) throw new Error('rotations[t] must have one quaternion per joint');
    this.bones = joints.map((name) => this.vrm.humanoid.getNormalizedBoneNode(name));
    this.quats = joints.map(() => new THREE.Quaternion());
    // [missing index, target index, mode] for bones this model lacks
    this.folds = [];
    for (const [missing, target, mode] of FOLD) {
      const m = joints.indexOf(missing), t = joints.indexOf(target);
      if (m >= 0 && !this.bones[m]) {
        if (t >= 0) this.folds.push([m, t, mode]);
        else console.warn(`model has no bone "${missing}" and the motion has no "${target}"; rotation dropped`);
      }
    }
    joints.forEach((name, j) => {
      if (!this.bones[j] && !FOLD.some(([m]) => m === name)) console.warn(`model has no bone "${name}", skipped`);
    });
    this.hands = this._handCurves(motion);
    this.hipsNode = this.vrm.humanoid.getNormalizedBoneNode('hips');
    this.hipsRest ??= this.hipsNode.position.clone();
    this.root = Array.isArray(motion.root) && motion.root.length === rotations.length ? motion.root : null;
    // motion.eyes = [T] closed weight 0..1 (optional; keyframes "eyes": closed): read back by eyesClosed after setTime
    this.eyes = Array.isArray(motion.eyes) && motion.eyes.length === rotations.length ? motion.eyes : null;
    this.eyesClosed = 0;
    this.motion = motion;
    this.setTime(0);
  }

  // motion.hands = { shapes: [names], left: [T indices], right: [T indices] } (optional).
  // Returns per side a [T][6] table of hand-shape values, averaged over +-3 frames so that a change
  // of shape takes about 0.2 s instead of snapping. Without motion.hands both hands stay relaxed.
  _handCurves(motion) {
    const T = motion.rotations.length, R = 3, out = {};
    for (const side of ['left', 'right']) {
      const ids = motion.hands?.[side];
      const raw = Array.from({ length: T }, (_, f) => {
        const name = ids ? motion.hands.shapes[ids[Math.min(f, ids.length - 1)]] : DEFAULT_HAND;
        if (!HAND_SHAPES[name]) console.warn(`unknown hand shape "${name}", using ${DEFAULT_HAND}`);
        return HAND_SHAPES[name] ?? HAND_SHAPES[DEFAULT_HAND];
      });
      out[side] = raw.map((_, f) => {
        const lo = Math.max(0, f - R), hi = Math.min(T - 1, f + R);
        return raw[0].map((_, i) => { let a = 0; for (let g = lo; g <= hi; g++) a += raw[g][i]; return a / (hi - lo + 1); });
      });
    }
    return out;
  }

  get duration() {
    return this.motion ? (this.motion.rotations.length - 1) / this.motion.fps : 0;
  }

  // Pose the model at time t (seconds), slerping between the two nearest frames.
  setTime(t) {
    if (!this.motion) return;
    const { fps, rotations } = this.motion;
    const f = Math.min(Math.max(t * fps, 0), rotations.length - 1);
    const i0 = Math.max(0, Math.min(Math.floor(f), rotations.length - 2));
    const i1 = Math.min(i0 + 1, rotations.length - 1);
    const u = f - i0;
    const s = this.flipXZ ? -1 : 1;
    this.quats.forEach((q, j) => {
      const a = rotations[i0][j];
      const b = rotations[i1][j];
      this._a.set(s * a[0], a[1], s * a[2], a[3]);
      this._b.set(s * b[0], b[1], s * b[2], b[3]);
      q.slerpQuaternions(this._a, this._b, u);
    });
    for (const [m, t, mode] of this.folds) {
      if (mode === 'post') this.quats[t].multiply(this.quats[m]);
      else this.quats[t].premultiply(this.quats[m]);
    }
    this.bones.forEach((node, j) => { if (node) node.quaternion.copy(this.quats[j]); });
    this.hipsNode.position.copy(this.hipsRest);
    if (this.root) {
      const a = this.root[i0], b = this.root[i1], h = this.hipsRest.y;
      this.hipsNode.position.x += s * h * (a[0] + (b[0] - a[0]) * u);
      this.hipsNode.position.y += h * (a[1] + (b[1] - a[1]) * u);
      this.hipsNode.position.z += s * h * (a[2] + (b[2] - a[2]) * u);
    }
    for (const side of ['left', 'right']) {
      const a = this.hands[side][i0], b = this.hands[side][i1];
      applyHandShape(this.vrm, side, a.map((v, i) => v + (b[i] - v) * u));
    }
    this.eyesClosed = this.eyes ? this.eyes[i0] + (this.eyes[i1] - this.eyes[i0]) * u : 0;
  }
}

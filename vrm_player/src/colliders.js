// Torso colliders that keep hair outside the clothes, for ANY model, by shape (2026-11-13, the author: no special cases by
// model name; replaces five capsules set by hand for the default model on 2026-10-23, whose own colliders were smaller
// than its loose shirt, so a bowed head let hair sink into the chest). Measured at load: every vertex of the body and
// its clothes whose strongest joint is the spine / chest / upperChest or a spring joint under them (a bust),
// in the body's frame (left, up, forward from the hips and the world up); per bone the 90th-percentile front, back and
// half width. Two capsules per bone, as the hand-set ones were: a flat front (small radius pushed forward, so bowed
// hair lies on the chest instead of sliding off to the sides) and a wide one for the sides. Added as one collider group
// to every spring joint hanging from the head / neck (hair). The arms' capsules are not here on purpose (2026-11-10:
// with the arms raised beside the head they pushed the side strands into an X over the chest).
import * as THREE from 'three';
import { VRMSpringBoneCollider, VRMSpringBoneColliderShapeCapsule } from '@pixiv/three-vrm';

const TORSO_BONES = ['upperChest', 'chest', 'spine'];
const pct = (a, q) => { if (!a.length) return 0; const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor(q * s.length))]; };

export function torsoShape(vrm) {
  const H = vrm.humanoid, V = THREE.Vector3;
  vrm.scene.updateMatrixWorld(true);
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const owner = new Map();
  // vertices on spring joints count too when those hang from the torso (a bust: hair lies on it), not from elsewhere
  const ownerOf = (node) => {
    if (owner.has(node)) return owner.get(node);
    let n = node;
    while (n && !rawToName.has(n)) n = n.parent;
    const r = n ? rawToName.get(n) : null; owner.set(node, r); return r;
  };
  const wp = (b) => H.getRawBoneNode(b)?.getWorldPosition(new V());
  const lL = wp('leftUpperLeg'), rL = wp('rightUpperLeg');
  if (!lL || !rL) return [];
  const up = new V(0, 1, 0), left = lL.clone().sub(rL).setY(0).normalize(), fwd = left.clone().cross(up).normalize();
  const pts = new Map(); const v = new V();
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const sk = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
    const used = o.geometry.index ? new Set(o.geometry.index.array) : Array.from({ length: o.geometry.attributes.position.count }, (_, i) => i);
    for (const i of used) {
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      const name = ownerOf(o.skeleton.bones[sk.getComponent(i, best)]);
      if (!TORSO_BONES.includes(name)) continue;
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      if (!pts.has(name)) pts.set(name, []);
      pts.get(name).push(v.clone());
    }
  });
  const out = [];
  for (const name of TORSO_BONES) {
    const c = wp(name), ps = pts.get(name) ?? [];
    if (!c || ps.length < 30) continue;
    const xs = [], ys = [], zs = [];
    for (const p of ps) { const d = p.clone().sub(c); xs.push(Math.abs(d.dot(left))); ys.push(d.y); zs.push(d.dot(fwd)); }
    out.push({ name, c, left, fwd, front: pct(zs, 0.9), back: -pct(zs, 0.1), half: pct(xs, 0.9), y: pct(ys, 0.5) });
  }
  return out;
}

export function addExtraColliders(vrm) {
  const m = vrm.springBoneManager;
  if (!m) return 0;
  const H = vrm.humanoid;
  const group = { colliders: [], name: 'hsin_torso' };
  const shapes = torsoShape(vrm);
  for (const t of shapes) {
    const bone = H.getRawBoneNode(t.name);
    const at = (x, z) => bone.worldToLocal(t.c.clone().addScaledVector(t.left, x).addScaledVector(t.fwd, z).add(new THREE.Vector3(0, t.y, 0)));
    const rf = 0.55 * t.front, xf = 0.45 * t.half;                  // flat front
    const rs = 0.53 * t.half, xs = Math.max(0, t.half - rs);         // sides
    const zs = (t.front - t.back) / 2;
    for (const [x, z, r] of [[xf, t.front - rf, rf], [xs, zs, rs]]) {
      const c = new VRMSpringBoneCollider(new VRMSpringBoneColliderShapeCapsule({ offset: at(x, z), tail: at(-x, z), radius: r }));
      bone.add(c);
      group.colliders.push(c);
    }
  }
  vrm.scene.userData.hsinTorso = shapes.map((t) => ({ name: t.name, front: +t.front.toFixed(3), back: +t.back.toFixed(3), half: +t.half.toFixed(3) }));
  // the hair: every spring joint hanging from the head / neck
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  for (const j of m.joints) {
    let n = j.bone; while (n && !rawToName.has(n)) n = n.parent;
    const h = n ? rawToName.get(n) : null;
    if (h === 'head' || h === 'neck') j.colliderGroups.push(group);
  }
  m._isSortedJointsDirty = true;          // re-collect the objects updated before the joints (incl. new colliders)
  return group.colliders.length;
}

// Body colliders for ANY model (2026-10-24, the author: skirts must not go through the legs or the floor, and
// hands should push hair / skirts aside instead of passing through them). Measured at load from the model's
// own skin: each vertex goes to the humanoid bone of its strongest joint (vertices hanging from spring joints
// are skipped), and per limb the 75th-percentile distance from the bone axis becomes a capsule radius
// (thighs, shins, forearms, hands). A plane at the floor (y = 0) keeps sitting skirts and long hair above it.
// They are added only to spring joints that hang from the hips / legs (skirts, tails) or from the head (hair):
// sleeves and busts hang from the arms / chest and would be pushed by their own arm.
import { VRMSpringBoneColliderShapePlane } from '@pixiv/three-vrm';
import { skinMesh } from './measure.js';

const LIMBS = [            // [from bone, to bone, which spring joints use it]
  ['leftUpperLeg', 'leftLowerLeg', 'low'], ['rightUpperLeg', 'rightLowerLeg', 'low'],
  ['leftLowerLeg', 'leftFoot', 'low'], ['rightLowerLeg', 'rightFoot', 'low'],
  ['leftLowerArm', 'leftHand', 'arm'], ['rightLowerArm', 'rightHand', 'arm'],   // forearms: skirts only (2026-11-10:
  // a forearm raised beside the head pushed the side strands inward and they crossed in an X over the chest)
  ['leftHand', 'leftMiddleDistal', 'all'], ['rightHand', 'rightMiddleDistal', 'all'],
];

function measureRadii(vrm) {
  const H = vrm.humanoid, V = THREE.Vector3;
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const spring = new Set(); for (const j of vrm.springBoneManager?.joints ?? []) spring.add(j.bone);
  const owner = new Map();
  const ownerOf = (node) => {
    if (owner.has(node)) return owner.get(node);
    let n = node, via = false;
    while (n && !rawToName.has(n)) { if (spring.has(n)) via = true; n = n.parent; }
    const r = via || !n ? null : rawToName.get(n); owner.set(node, r); return r;
  };
  const pos = (b) => H.getRawBoneNode(b)?.getWorldPosition(new V());
  vrm.scene.updateMatrixWorld(true);
  const verts = new Map();
  const v = new V();
  const skin = skinMesh(vrm);       // skin only (2026-10-29): a long skirt skinned to the legs made the leg capsules fat
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh || (skin && o !== skin)) return;
    const sk = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
    const used = o.geometry.index ? [...new Set(o.geometry.index.array)] : Array.from({ length: o.geometry.attributes.position.count }, (_, i) => i);
    for (const i of used) {
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      const name = ownerOf(o.skeleton.bones[sk.getComponent(i, best)]);
      if (!name) continue;
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      if (!verts.has(name)) verts.set(name, []);
      verts.get(name).push(v.clone());
    }
  });
  const out = [];
  for (const [from, to, use] of LIMBS) {
    const a = pos(from), b = pos(to) ?? (to.endsWith('Distal') ? pos(to.replace('Distal', 'Proximal')) : null);
    const vs = verts.get(from) ?? [];
    if (!a || !b || vs.length < 10) continue;
    const ab = b.clone().sub(a), L2 = ab.lengthSq(), d = [];
    for (const p of vs) {
      const t = p.clone().sub(a).dot(ab) / L2;
      if (t < 0.15 || t > 1) continue;                 // skip the joint ends (hip / pelvis mass)
      d.push(p.clone().sub(a).sub(ab.clone().multiplyScalar(t)).length());
    }
    if (d.length < 10) continue;
    d.sort((x, y) => x - y);
    out.push({ from, a, b, radius: d[Math.floor(0.75 * d.length)], use });
  }
  return out;
}

export function addBodyColliders(vrm) {
  const m = vrm.springBoneManager;
  if (!m) return null;
  const H = vrm.humanoid;
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const humanOf = (node) => { let n = node; while (n && !rawToName.has(n)) n = n.parent; return n ? rawToName.get(n) : null; };
  const limbs = measureRadii(vrm);
  const low = { colliders: [], name: 'hsin_body_low' }, all = { colliders: [], name: 'hsin_body_hands' },
    arm = { colliders: [], name: 'hsin_body_forearms' };
  for (const l of limbs) {
    const bone = H.getRawBoneNode(l.from);
    const c = new VRMSpringBoneCollider(new VRMSpringBoneColliderShapeCapsule({
      offset: bone.worldToLocal(l.a.clone()), tail: bone.worldToLocal(l.b.clone()), radius: l.radius }));
    bone.add(c);
    (l.use === 'low' ? low : l.use === 'arm' ? arm : all).colliders.push(c);
  }
  vrm.scene.userData.hsinLimbs = limbs.map((l) => {               // in the bone's frame, for skirt.js
    const bone = H.getRawBoneNode(l.from);
    // name = the humanoid bone (leftUpperLeg ...), never the model's own bone name (VRoid's J_Bip_L_UpperLeg)
    return { bone, name: l.from, a: bone.worldToLocal(l.a.clone()), b: bone.worldToLocal(l.b.clone()), radius: l.radius };
  });
  const floor = new VRMSpringBoneCollider(new VRMSpringBoneColliderShapePlane({
    offset: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(0, 1, 0) }));
  vrm.scene.parent ? vrm.scene.parent.add(floor) : vrm.scene.add(floor);   // world-fixed: the hips move, the floor does not
  low.colliders.push(floor);
  // A wall (2026-11-11, window snapping: leaning on a window's side): a vertical plane every spring joint (hair, skirt,
  // sleeves, tails) stays in front of; parked far away until setWall places it. The floor plane can be lowered (a ledge:
  // hair and hems hang below the edge she sits on).
  const wall = new VRMSpringBoneCollider(new VRMSpringBoneColliderShapePlane({
    offset: new THREE.Vector3(0, 0, 0), normal: new THREE.Vector3(1, 0, 0) }));
  wall.position.set(-100, 0, 0);
  vrm.scene.parent ? vrm.scene.parent.add(wall) : vrm.scene.add(wall);
  const wallGroup = { colliders: [wall], name: 'hsin_wall' };
  for (const j of m.joints) j.colliderGroups.push(wallGroup);
  vrm.scene.userData.hsinPlanes = { floor, wall };
  let nLow = 0, nHair = 0;
  for (const j of m.joints) {
    const h = humanOf(j.bone);
    if (h === 'hips' || (h && h.endsWith('UpperLeg')) || (h && h.endsWith('LowerLeg'))) { j.colliderGroups.push(low, arm, all); nLow++; }
    else if (h === 'head' || h === 'neck') { j.colliderGroups.push(all, { colliders: [floor], name: 'hsin_floor' }); nHair++; }
  }
  m._isSortedJointsDirty = true;
  return { limbs: limbs.map((l) => [l.from, Math.round(l.radius * 1000) / 1000]), skirtJoints: nLow, hairJoints: nHair };
}

// the wall plane at world x, the body on its +x side (side = 1) or -x side (side = -1); null = no wall
export function setWall(vrm, x, side = 1) {
  const w = vrm.scene.userData.hsinPlanes?.wall;
  if (!w) return;
  if (x === null) { w.position.set(-100, 0, 0); w.rotation.set(0, 0, 0); }
  else { w.position.set(x, 0, 0); w.rotation.set(0, side > 0 ? 0 : Math.PI, 0); }
  w.updateMatrixWorld(true);
}
// the floor plane's height (0 = the floor; far below while sitting on a ledge)
export function setFloorY(vrm, y) {
  const f = vrm.scene.userData.hsinPlanes?.floor;
  if (f) { f.position.y = y; f.updateMatrixWorld(true); }
}

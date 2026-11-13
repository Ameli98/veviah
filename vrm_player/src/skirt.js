// Skirts that sit with the body (2026-10-24, the author chose to place them by program; only clipping matters, a skirt
// may show what is under it). A spring skirt pulls back toward its standing cone, so sitting on the floor its cloth
// cuts through the folded legs. Once per (model, sitting posture) the skirt mesh is simulated as a cloth while the
// body sits down (simulateSit) and the settled shape is cached as a table; while sitting the skirt is drawn from it.
// Standing keeps the normal spring physics plus a per-vertex push out of the leg capsules (fixSkirtVertices).
// Everything is found from the skeleton, the spring joints and the skin weights, never from mesh / material names.
// What was tried and dropped (rope drape of the chains, pushing vertices out per posture, a mesh cloth relax from
// the draped chains, slow / two-phase sit-down, triangle-centre and self collisions, lifting hands onto the cloth,
// hiding outlines) is recorded in CLAUDE.md.
import * as THREE from 'three';
import { skinMesh } from './measure.js';

const V = THREE.Vector3;

const SKIRT_MIN_DROP = 0.5;       // thigh lengths below the hip joint a chain must reach to count as a skirt

// For skirtChains: chain -> true when the vertices it moves (strongest bone in the chain) all lie in mesh components
// that do not go round the hips (a tail, a ribbon behind). Components = glTF primitives of the skinned meshes.
function chainMeshesNotAround(vrm, hips, legs) {
  vrm.scene.updateMatrixWorld(true);
  const centre = hips.getWorldPosition(new V()), hipY = Math.min(...legs.map((l) => l.getWorldPosition(new V()).y));
  const round = new Map(), owners = new Map();           // mesh -> goes round; bone -> set of meshes it moves
  const v = new V();
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const used = g.index ? new Set(g.index.array) : null, dirs = new Set();
    for (let i = 0; i < si.count; i++) {
      if (used && !used.has(i)) continue;
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      const b = o.skeleton.bones[si.getComponent(i, best)];
      if (!owners.has(b)) owners.set(b, new Set());
      owners.get(b).add(o);
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      if (v.y < hipY) dirs.add(Math.floor(((Math.atan2(v.z - centre.z, v.x - centre.x) / (2 * Math.PI)) + 1) * 8) % 8);
    }
    round.set(o, dirs.size >= SKIRT_AROUND);
  });
  return (chain) => {
    const meshes = new Set();
    for (const k of chain) for (const o of owners.get(k.bone) ?? []) meshes.add(o);
    return meshes.size > 0 && [...meshes].every((o) => !round.get(o));
  };
}

function skirtChains(vrm) {
  const m = vrm.springBoneManager, H = vrm.humanoid;
  if (!m) return [];
  const raw = new Set(Object.keys(H.humanBones).map((n) => H.getRawBoneNode(n)).filter(Boolean));
  const humanOf = (node) => { let n = node; while (n && !raw.has(n)) n = n.parent; return n; };
  const hips = H.getRawBoneNode('hips');
  const legs = ['leftUpperLeg', 'rightUpperLeg'].map((n) => H.getRawBoneNode(n));
  const joints = [...m.joints].filter((j) => { const h = humanOf(j.bone); return h === hips || legs.includes(h); });
  const byBone = new Map(joints.map((j) => [j.bone, j]));
  vrm.scene.updateMatrixWorld(true);
  const legY = legs[0].getWorldPosition(new V()).y, kneeY = H.getRawBoneNode('leftLowerLeg').getWorldPosition(new V()).y;
  const notAround = chainMeshesNotAround(vrm, hips, legs);
  const chains = [];
  for (const j of joints) {
    if (byBone.has(j.bone.parent)) continue;          // not a chain root
    const chain = [];
    for (let k = j; k; k = k.child ? byBone.get(k.child) : null) chain.push(k);
    // tails also hang from the hips but are not cloth around the legs (AvatarSample_D's fox tail draped forward
    // over the thighs). By shape (2026-11-05, was by bone name tail / 尾 / しっぽ): a chain whose vertices all belong
    // to meshes that do not go round the body (fewer than SKIRT_AROUND of 8 directions round the hips) is not a
    // skirt; a chain that moves no vertex at all (test1's unused skirt chains) stays, as before
    if (notAround(chain)) continue;
    // a short chain is the hem of a top, not a skirt (2026-10-28: the default model's T-shirt has 6 chains reaching
    // 0.35 thigh lengths below the hip joint, skirts 0.8-1.0); simulated as cloth it spread flat on the floor
    // over the shorts in every sitting pose. Short hems keep plain springs + the leg / floor colliders.
    const ends = chain.map((k) => k.bone.getWorldPosition(new V()).y);
    if (chain.at(-1).child) ends.push(chain.at(-1).child.getWorldPosition(new V()).y);
    if (legY - Math.min(...ends) < SKIRT_MIN_DROP * (legY - kneeY)) continue;
    chains.push(chain);
  }
  return chains;
}

export function initSkirt(vrm) {
  const chains = skirtChains(vrm);
  for (const c of chains) for (const j of c) j.hsinRest = j.bone.quaternion.clone();
  vrm.scene.userData.hsinSkirt = { chains, table: new Map(), current: 'stand',
    hipsRest: vrm.humanoid.getNormalizedBoneNode('hips').position.clone() };
  return chains.length;
}

// Standing skirts swing at most maxDeg away from their rest (per chain joint, relative to its parent). A spring skirt
// is a few chains with long cloth edges between them: thrown far (a fast window drag) the cloth between two chains
// cut through a thigh although every chain joint stayed outside its collider (2026-11-05, the author; test2's dress, 6
// chains). Clamped joints also get their spring tail put where the clamped bone points, so the physics stops there
// instead of pulling past it. After vrm.update, before fixSkirtVertices. Returns how many joints were clamped.
export const SKIRT_SWING_MAX = 25;
const _qd = new THREE.Quaternion(), _inv4 = new THREE.Matrix4();
export function limitSkirtSwing(vrm, maxDeg = SKIRT_SWING_MAX) {
  const S = vrm.scene.userData.hsinSkirt;
  if (!S || S.current !== 'stand' || !S.chains.length) return 0;
  const max = maxDeg * Math.PI / 180;
  let n = 0;
  for (const c of S.chains) for (const j of c) {
    const rest = j.hsinRest;
    const ang = rest.angleTo(j.bone.quaternion);
    if (ang <= max) continue;
    _qd.copy(rest).slerp(j.bone.quaternion, max / ang);
    j.bone.quaternion.copy(_qd); j.bone.updateMatrix(); j.bone.updateMatrixWorld(true);
    if (j._currentTail && j._prevTail) {
      const t = j.bone.localToWorld(j.initialLocalChildPosition.clone());
      if (j.center) t.applyMatrix4(_inv4.copy(j.center.matrixWorld).invert());
      j._prevTail.lerp(t, 0.8); j._currentTail.copy(t);     // most of the swing's speed is lost at the limit
    }
    n++;
  }
  return n;
}

// three-vrm turns matrixAutoUpdate off on spring bones, so every quaternion written here needs updateMatrix()
// (without it the drape depended on the motion played before: stale matrices of the previous posture)
function restore(vrm, S) {
  for (const c of S.chains) for (const j of c) { j.bone.quaternion.copy(j.hsinRest); j.bone.updateMatrix(); }
  vrm.scene.updateMatrixWorld(true);
}


// The legs' real surface in the current (sitting) pose. The standing capsules are too thin for sitting: a thigh
// pressed on the floor flattens and spreads, so cloth lying on the capsule was still inside the visible thigh
// (the author: in seiza, side and cross-legged sits the legs went through the skirt). Per leg segment (thigh, shin)
// the leg vertices are skinned on the CPU and binned by position along the bone (10) and angle around it (16);
// each bin keeps the largest distance from the axis. Computed once per sitting posture.
const LEG_SEGS = [['leftUpperLeg', 'leftLowerLeg'], ['rightUpperLeg', 'rightLowerLeg'], ['leftLowerLeg', 'leftFoot'], ['rightLowerLeg', 'rightFoot']];
const NT = 10, NA = 16;

function legVertexLists(vrm) {
  const H = vrm.humanoid;
  const owners = new Map([[H.getRawBoneNode('hips'), 'hips']]);
  for (const [from] of LEG_SEGS) owners.set(H.getRawBoneNode(from), from);
  const skirtSet = new Map((vrm.scene.userData.hsinSkirtVerts ?? []).map((m) => [m.o.geometry.attributes.position, new Set(m.idx)]));
  const raw = new Set(Object.keys(H.humanBones).map((n) => H.getRawBoneNode(n)).filter(Boolean));
  const spring = new Set([...(vrm.springBoneManager?.joints ?? [])].map((j) => j.bone));
  const memo = new Map();
  const seg = (node) => {
    if (memo.has(node)) return memo.get(node);
    let n = node, via = false;
    while (n && !raw.has(n)) { if (spring.has(n)) via = true; n = n.parent; }
    const r = via ? null : owners.get(n) ?? null; memo.set(node, r); return r;
  };
  const seen = new Set(), lists = [], skin = SKIN_SURFACE ? skinMesh(vrm) : null;
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh || seen.has(o.geometry.attributes.position)) return;
    if (skin && o !== skin) return;      // the skin only (2026-10-29: test1's skirt top ring, weighted to the left thigh
    // more than the right, made the left leg's surface fatter near the hip)
    seen.add(o.geometry.attributes.position);
    const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight, by = {};
    const skip = skirtSet.get(o.geometry.attributes.position);
    const verts = skin ? new Set(o.geometry.index.array) : null;   // the skin's own vertices, not the shared buffer
    for (let i = 0; i < si.count; i++) {
      if (verts && !verts.has(i)) continue;
      if (skip?.has(i)) continue;                          // the skirt itself is not body
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      const name = seg(o.skeleton.bones[si.getComponent(i, best)]);
      if (name) (by[name] ??= []).push(i);
    }
    if (Object.keys(by).length) lists.push({ o, by });
  });
  return lists;
}

function legSurface(vrm) {
  const S = vrm.scene.userData.hsinSkirt, H = vrm.humanoid;
  S.legLists ??= legVertexLists(vrm);
  vrm.scene.updateMatrixWorld(true);
  const segs = [];
  // the pelvis / bottom: an axis from just above the hips joint down along the spine direction (the skirt
  // smoothing pulled cloth into the bottom, which then showed through the back of the dress)
  const hp = H.getRawBoneNode('hips').getWorldPosition(new V()), sp = H.getRawBoneNode('spine').getWorldPosition(new V());
  const up = sp.clone().sub(hp).normalize();
  const pelvis = ['hips', null, hp.clone().addScaledVector(up, 0.08), hp.clone().addScaledVector(up, -0.2)];
  for (const [from, to, a0, b0] of [pelvis, ...LEG_SEGS]) {
    const a = a0 ?? H.getRawBoneNode(from)?.getWorldPosition(new V()), b = b0 ?? H.getRawBoneNode(to)?.getWorldPosition(new V());
    if (!a || !b) continue;
    const u = b.clone().sub(a), L = u.length(); u.normalize();
    let e1 = new V(0, 1, 0).sub(u.clone().multiplyScalar(u.y));
    if (e1.lengthSq() < 1e-4) e1 = new V(0, 0, 1).sub(u.clone().multiplyScalar(u.z));
    e1.normalize();
    const e2 = u.clone().cross(e1);
    const R = new Float32Array(NT * NA);
    const v = new V(), w = new V();
    for (const { o, by } of S.legLists) {
      for (const i of by[from] ?? []) {
        o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
        w.subVectors(v, a);
        const t = w.dot(u) / L;
        if (t < -0.05 || t > 1.05) continue;
        w.addScaledVector(u, -w.dot(u));
        const ang = Math.atan2(w.dot(e2), w.dot(e1));
        const ti = Math.min(NT - 1, Math.max(0, Math.floor(t * NT)));
        const ai = ((Math.round((ang / (2 * Math.PI)) * NA) % NA) + NA) % NA;
        R[ti * NA + ai] = Math.max(R[ti * NA + ai], w.length());
      }
    }
    // empty bins: the neighbours around, then along, else the segment's median
    const filled = [...R].filter((x) => x > 0).sort((x, y) => x - y), med = filled[filled.length >> 1] ?? 0.05;
    for (let pass = 0; pass < 4; pass++) {
      for (let ti = 0; ti < NT; ti++) for (let ai = 0; ai < NA; ai++) {
        if (R[ti * NA + ai] > 0) continue;
        const nb = [R[ti * NA + (ai + 1) % NA], R[ti * NA + (ai + NA - 1) % NA], ti > 0 ? R[(ti - 1) * NA + ai] : 0, ti < NT - 1 ? R[(ti + 1) * NA + ai] : 0].filter((x) => x > 0);
        if (nb.length) R[ti * NA + ai] = Math.max(...nb);
      }
    }
    for (let k = 0; k < R.length; k++) if (!(R[k] > 0)) R[k] = med;
    segs.push({ a, u, L, e1, e2, R });
  }
  return segs;
}

// Moves p (world) out of the leg surface by r more than the surface; returns true if it was inside.
function pushOutLegs(segs, p, r) {
  let hit = false;
  const w = new V();
  for (const g of segs) {
    w.subVectors(p, g.a);
    const along = w.dot(g.u), t = along / g.L;
    if (t < -0.1 || t > 1.1) continue;
    const tc = Math.min(1, Math.max(0, t));
    w.addScaledVector(g.u, -along);                          // radial part
    const d = w.length();
    const ang = Math.atan2(w.dot(g.e2), w.dot(g.e1));
    const af = ((ang / (2 * Math.PI)) * NA + NA) % NA, a0 = Math.floor(af) % NA, a1 = (a0 + 1) % NA, f = af - Math.floor(af);
    const ti = Math.min(NT - 1, Math.floor(tc * NT));
    const R = (1 - f) * g.R[ti * NA + a0] + f * g.R[ti * NA + a1] + r;
    if (t !== tc) continue;                                  // beyond the ends: the neighbour segment / floor take it
                                                             // (round end caps tore side_right / knees_hugged cloth)
    if (d < R) {
      const n = d > 1e-6 ? w.divideScalar(d) : g.e1.clone();
      p.copy(g.a).addScaledVector(g.u, along).addScaledVector(n, R);
      hit = true;
    }
  }
  return hit;
}

// Where the legs are, on a 2 cm grid: the key of one draped skirt shape (the body posed as wanted).
const LEG_BONES = ['hips', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes'];
export function legKey(vrm) {
  const v = new V();
  return LEG_BONES.map((name) => {
    const n = vrm.humanoid.getNormalizedBoneNode(name);
    return n ? n.getWorldPosition(v).toArray().map((x) => Math.round(x / 0.02)).join(',') : '';
  }).join(';');
}

// Call with the body already posed at the motion's first frame. posture: 'stand' or a key of the sitting pose.
// motion (optional): { poseAt(f), frames, id }; when the legs change during a sitting motion the cloth is stepped
// through it (simulateMotion) and fixSkirtVertices(vrm, frame) draws the shape of that frame.
export function drapeSkirt(vrm, posture, motion = null) {
  const S = vrm.scene.userData.hsinSkirt;
  if (!S || !(S.chains.length || vrm.scene.userData.hsinSkirtVerts?.length)) return false;
  let moving = false;
  if (motion && posture !== 'stand') {
    const k0 = legKey(vrm);
    for (let f = 5; f < motion.frames && !moving; f += 5) { motion.poseAt(f); vrm.scene.updateMatrixWorld(true); moving = legKey(vrm) !== k0; }
    motion.poseAt(0); vrm.scene.updateMatrixWorld(true);
  }
  if (vrm.scene.userData.hsinSkirtVerts?.some((m) => m.weights))
    useSkirtWeights(vrm, posture !== 'stand' && wantsSmooth(vrm, motion) ? 'smooth' : 'orig');
  if (posture !== 'stand' && skinnedFor(vrm)) posture = 'stand';   // drawn as skinned (see skinnedFor)
  const hike = posture !== 'stand' ? hikeFor(vrm, motion) : 1;
  const key = (moving && posture !== 'stand' ? `${posture}|${motion.id}` : posture) + (hike < 1 ? `|hike${hike.toFixed(2)}` : '');
  if (S.current === key) return false;
  restore(vrm, S);
  for (const c of S.chains) for (const j of c) j.setInitState();
  vrm.springBoneManager.reset();
  S.current = key;
  S.frames = null;
  for (const k of [...S.table.keys()]) if (k.includes('|') && k !== key) S.table.delete(k);   // one motion's frames at a time
  if (posture !== 'stand' && vrm.scene.userData.hsinSkirtVerts?.length && !S.table.has(key))
    S.table.set(key, moving ? simulateMotion(vrm, legSurface(vrm), motion.poseAt, motion.frames, hike) : simulateSit(vrm, legSurface(vrm), hike));
  if (moving && posture !== 'stand') S.frames = S.table.get(key);
  return true;
}

// Skirts skinned to the legs (no spring joint moves them, test1's narrow long skirt) are never simulated (2026-11-03,
// the author, after looking at every kept sitting pose at 5 angles x 3 frames): simulated, the cloth cut through the
// bottom with raised knees (knees_hugged, T001, S019, T012, S065), showed it in legs_out and side_right, and tore in
// w_sit; a finer cloth mesh did not help. Skinned it follows the legs; limb_push.js keeps it out of the legs and the
// bottom, and smoothSkirtWeights keeps its two halves together in side sits. Skirts on spring joints (test2's dress,
// AvatarSample_D) are still simulated.
function skinnedFor(vrm) {
  const meshes = vrm.scene.userData.hsinSkirtVerts ?? [];
  return meshes.length > 0 && !meshes.some((m) => m.springs);
}

// A long skirt round legs spread wider than it can hold is hiked up onto the thighs first (2026-10-29, the author: in a
// narrow long skirt S029 cannot be done, a person pulls the skirt up). Two knees inside a loop of cloth can be at most
// half its perimeter apart; with HIKE_SLACK for the cloth's give and the legs' thickness. When the knees' outer edges
// (any frame of the motion) are further apart, the cloth is shortened toward its waistband so the hem rests at
// HIKE_HEM thigh lengths below the hip joints, then sits down as usual. The folds of hiked cloth are not modelled.
const HIKE_SLACK = 1.5, HIKE_HEM = 0.5, KNEE_R = 0.05;   // 1.25 hiked test1 for a side sit (knees 0.57 m apart), which a long skirt covers
function hikeFor(vrm, motion) {
  const R = vrm.scene.userData.hsinSkirtRing;
  if (!R) return 1;
  const H = vrm.humanoid, a = new V(), b = new V();
  const spread = () => { vrm.scene.updateMatrixWorld(true); H.getRawBoneNode('leftLowerLeg').getWorldPosition(a);
    H.getRawBoneNode('rightLowerLeg').getWorldPosition(b); return a.distanceTo(b) + 2 * KNEE_R * R.thigh / 0.4; };
  let most = spread();
  if (motion) { for (let f = 0; f < motion.frames; f += 3) { motion.poseAt(f); most = Math.max(most, spread()); } motion.poseAt(0); vrm.scene.updateMatrixWorld(true); }
  if (most <= HIKE_SLACK * R.perimeter / 2) return 1;
  return Math.min(1, (R.top - (R.hipY - HIKE_HEM * R.thigh)) / (R.top - R.hem));
}

// Skirt vertices. Standing: each frame, after the spring physics, every skirt vertex is skinned on the CPU; one a
// little inside a thigh / shin capsule (colliders.js radii + 3 mm) or under the floor is moved to the surface and
// mapped back through the inverse of its blended skin matrix (each chain moves its patch rigidly, and with few
// chains, test2: 6, the sides of a patch dip into a thigh). The original positions are restored before each frame.
// Sitting: the table from simulateSit is written instead.
const SLACK = 0.003;
// Sitting on a ledge (a window's top edge in the desktop app, 2026-11-11): the floor is the ledge only behind its
// front edge (world z < ledgeFront); in front of it the cloth hangs down over the legs. null = a floor everywhere.
let ledgeFront = null;
export function setLedge(front) { ledgeFront = front; SIM_GAP = front === null ? SIM_GAP_FLOOR : SIM_GAP_LEDGE; }
const onFloor = (p) => ledgeFront === null || p.z < ledgeFront;
const PREV_BELOW = 0.3;     // m: a hand this low decides over / under by the cloth's previous side (clothStep)

// Which cloth is a skirt is decided by its shape only (2026-10-29, the author: "不能單靠形狀嗎"; a skirt may hang from
// spring joints (test2, AvatarSample_D) or be skinned to the legs with its spring bones left unused (test1's long
// skirt: VRoid's J_Sec_*_Skirt* joints exist, no vertex is weighted to them)). A mesh component is a skirt piece when
//   - it is not the skin (measure.skinMesh);
//   - most of its vertices below the hip joints hang from the hips / legs (directly or through spring joints below
//     them; a dress's top hangs from the chest, test2);
//   - it reaches SKIRT_MIN_DROP thigh lengths below the hip joints (a top's hem does not);
//   - it hangs away from the thighs: between mid-thigh and knee the 95th-percentile distance from the nearest thigh
//     axis is above SKIRT_SPREAD x the skin's (skin 0.18-0.20 thigh lengths, close-fitting trousers 0.24, skirts
//     0.35 and more);
//   - it goes round the body: at least SKIRT_AROUND of 8 directions around the hips hold some of it (a tail hangs
//     only behind).
// Shorter rings around the body sharing a skirt's vertex buffer are its other layers (AvatarSample_D's tiers).
// Its vertices on the hips / legs take part up to the waist, except its top ring (SKIRT_TOP thigh lengths): that stays
// skinned, so the cloth is pinned to it (clothSetup pins vertices with an edge outside the set). Components that share one
// vertex buffer (VRoid's merged primitives) form one entry, so the layers of a tiered skirt are simulated together
// (layerFollowers). Replaces the spring-chain selection (vertices on the chains grown 8 rings into the hips / thighs).
const SKIRT_TOP = 0.08, SKIRT_SPREAD = 1.5, SKIRT_AROUND = 7;

// A skirt skinned to the legs splits down its middle when the legs go different ways (2026-11-03, test1 side sits: the
// front's left half on the left thigh, its right half on the right one, the triangles between turned over and the
// lining showed). Its leg weights are spread across the middle at load: for a vertex at lateral position u (-1 .. 1
// between the hip joints, 0 = the middle), the weight it has on the two thighs is shared left / right by
// smoothstep(u) and a share HIP_SHARE x (1 - |u|) goes to the hips; the same for the two shins. Outside the hip joints
// (|u| >= 1) nothing changes. At rest the skirt does not move (all bones at rest). Both weight sets are kept and
// drapeSkirt picks one per motion (wantsSmooth): the smoothed ones only when the thighs go apart with no knee raised.
const HIP_SHARE = 0.5, SPREAD_W = 1.6;   // SPREAD_W: the blend reaches this many half hip widths from the middle
function smoothSkirtWeights(vrm, m) {
  const H = vrm.humanoid, bones = m.o.skeleton.bones, at = (n) => bones.indexOf(H.getRawBoneNode(n));
  const hips = at('hips'), legs = [[at('leftUpperLeg'), at('rightUpperLeg')], [at('leftLowerLeg'), at('rightLowerLeg')]];
  if (hips < 0 || legs.flat().some((i) => i < 0)) return 0;
  vrm.scene.updateMatrixWorld(true);
  const L = H.getRawBoneNode('leftUpperLeg').getWorldPosition(new V()), R = H.getRawBoneNode('rightUpperLeg').getWorldPosition(new V());
  const c = L.clone().add(R).multiplyScalar(0.5), side = L.clone().sub(R), hx = side.length() / 2;
  side.normalize();
  const g = m.o.geometry, SI = g.attributes.skinIndex, SW = g.attributes.skinWeight, v = new V();
  const smooth = [], orig = [];
  let n = 0;
  for (const i of m.idx) {
    m.o.getVertexPosition(i, v); v.applyMatrix4(m.o.matrixWorld);
    const u = v.clone().sub(c).dot(side) / (hx * SPREAD_W);
    if (Math.abs(u) >= 1) continue;
    const w = new Map();
    for (let k = 0; k < 4; k++) { const b = SI.getComponent(i, k), x = SW.getComponent(i, k); if (x > 0) w.set(b, (w.get(b) ?? 0) + x); }
    const fl = THREE.MathUtils.smoothstep(u, -1, 1), share = HIP_SHARE * (1 - Math.abs(u));
    for (const [l, r] of legs) {
      const sum = (w.get(l) ?? 0) + (w.get(r) ?? 0);
      if (!sum) continue;
      w.set(l, sum * fl * (1 - share)); w.set(r, sum * (1 - fl) * (1 - share));
      w.set(hips, (w.get(hips) ?? 0) + sum * share);
    }
    const top = [...w.entries()].filter(([, x]) => x > 0).sort((a, b) => b[1] - a[1]).slice(0, 4), tot = top.reduce((t, [, x]) => t + x, 0);
    if (!tot) continue;
    smooth.push([i, [0, 1, 2, 3].map((k) => top[k]?.[0] ?? 0), [0, 1, 2, 3].map((k) => (top[k]?.[1] ?? 0) / tot)]);
    orig.push([i, [0, 1, 2, 3].map((k) => SI.getComponent(i, k)), [0, 1, 2, 3].map((k) => SW.getComponent(i, k))]);
    n++;
  }
  m.weights = { orig, smooth, now: 'orig' };
  return n;
}

// Which weights a skinned skirt draws with (see smoothSkirtWeights): 'smooth' or 'orig'.
function useSkirtWeights(vrm, which) {
  for (const m of vrm.scene.userData.hsinSkirtVerts ?? []) {
    if (!m.weights || m.weights.now === which) continue;
    const SI = m.o.geometry.attributes.skinIndex, SW = m.o.geometry.attributes.skinWeight;
    for (const [i, bi, bw] of m.weights[which]) for (let k = 0; k < 4; k++) { SI.setComponent(i, k, bi[k]); SW.setComponent(i, k, bw[k]); }
    SI.needsUpdate = SW.needsUpdate = true;
    m.weights.now = which;
  }
}

// When the smoothed weights are used: the thighs more than SPLIT_DEG apart in some frame (the halves would split:
// side sits 46 deg on test1) and no knee raised above its hip joint by KNEE_UP thigh lengths in any frame (raised
// knees came through the smoothed middle: knees_hugged, T001, S019, T012, S065 +0.75..+0.89; side sits -0.15). With
// the thighs together (seiza 7, legs_out 9, w_sit 3, kneel 6 deg) the smoothed middle let the knees through.
const KNEE_UP = 0.3, SPLIT_DEG = 25;
function wantsSmooth(vrm, motion) {
  const H = vrm.humanoid, P = (n) => H.getRawBoneNode(n).getWorldPosition(new V());
  let split = false, raised = false;
  const check = () => {
    vrm.scene.updateMatrixWorld(true);
    const th = ['left', 'right'].map((sd) => { const hip = P(sd + 'UpperLeg'), knee = P(sd + 'LowerLeg'); return [hip, knee, knee.clone().sub(hip)]; });
    if (th.some(([hip, knee, d]) => (knee.y - hip.y) / d.length() > KNEE_UP)) raised = true;
    if (th[0][2].angleTo(th[1][2]) > THREE.MathUtils.degToRad(SPLIT_DEG)) split = true;
  };
  check();
  if (motion) { for (let f = 0; f < motion.frames && !raised; f += 3) { motion.poseAt(f); check(); } motion.poseAt(0); vrm.scene.updateMatrixWorld(true); }
  return split && !raised;
}

export function initSkirtVertices(vrm) {
  const H = vrm.humanoid, skin = skinMesh(vrm);
  const raw = new Set(Object.keys(H.humanBones).map((n) => H.getRawBoneNode(n)).filter(Boolean));
  const low = new Set(['hips', 'leftUpperLeg', 'rightUpperLeg', 'leftLowerLeg', 'rightLowerLeg'].map((n) => H.getRawBoneNode(n)));
  vrm.scene.updateMatrixWorld(true);
  const P = (n) => H.getRawBoneNode(n).getWorldPosition(new V());
  const legs = ['left', 'right'].map((s) => [P(s + 'UpperLeg'), P(s + 'LowerLeg')]);
  const hip = legs[0][0], knee = legs[0][1], thigh = hip.y - knee.y, centre = P('hips');
  const used = (o) => [...new Set(o.geometry.index.array)];
  const world = (o, i) => { const v = new V(); o.getVertexPosition(i, v); return v.applyMatrix4(o.matrixWorld); };
  const spread = (o, us) => {
    const ab = new V(), d = [];
    for (const i of us) {
      const v = world(o, i);
      if (v.y > hip.y - 0.5 * thigh || v.y < knee.y) continue;
      let best = Infinity;
      for (const [a, b] of legs) {
        ab.subVectors(b, a);
        const t = THREE.MathUtils.clamp(v.clone().sub(a).dot(ab) / ab.lengthSq(), 0, 1);
        best = Math.min(best, v.distanceTo(a.clone().addScaledVector(ab, t)));
      }
      d.push(best);
    }
    d.sort((x, y) => x - y);
    return d.length > 10 ? d[Math.floor(0.95 * d.length)] : 0;
  };
  const skinSpread = skin?.geometry.index ? spread(skin, used(skin)) : 0.2 * thigh;
  const memo = new Map();
  const isLow = (node) => {
    if (memo.has(node)) return memo.get(node);
    let n = node; while (n && !raw.has(n)) n = n.parent;
    const r = low.has(n); memo.set(node, r); return r;
  };
  const spring = new Set([...(vrm.springBoneManager?.joints ?? [])].map((j) => j.bone)), smemo = new Map();
  const onSpring = (o, i) => {                              // the vertex's strongest bone is a spring joint or under one
    const g = o.geometry, si = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
    const node = o.skeleton.bones[si.getComponent(i, best)];
    if (smemo.has(node)) return smemo.get(node);
    let n = node, r = false; while (n && !raw.has(n)) { if (spring.has(n)) { r = true; break; } n = n.parent; }
    smemo.set(node, r); return r;
  };
  const cands = [];
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh || o === skin || !o.geometry.index) return;
    const g = o.geometry, si = g.attributes.skinIndex, sw = g.attributes.skinWeight, us = used(o);
    let nBelow = 0, nLowBelow = 0, minY = Infinity, topY = -Infinity;
    const low = [], dirs = new Set();
    for (const i of us) {
      const v = world(o, i);
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      const isL = isLow(o.skeleton.bones[si.getComponent(i, best)]);
      if (v.y < hip.y) { nBelow++; if (isL) nLowBelow++; }   // a dress: only its skirt part decides
      if (!isL) continue;
      low.push([i, v.y]);
      minY = Math.min(minY, v.y); topY = Math.max(topY, v.y);
      if (v.y < hip.y && v.y > knee.y) dirs.add(Math.floor(((Math.atan2(v.z - centre.z, v.x - centre.x) / (2 * Math.PI)) + 1) * 8) % 8);
    }
    // all of it on the hips / legs takes part, up to the waist (test2: cut at the thighs' top, the cloth between there
    // and the waist was pushed through by a raised thigh and turned inside out), but its top ring (SKIRT_TOP thigh
    // lengths) stays skinned and pins it
    const keep = low.filter(([, y]) => y < topY - SKIRT_TOP * thigh).map(([i]) => i);
    if (!nBelow || nLowBelow < 0.5 * nBelow || !keep.length || dirs.size < SKIRT_AROUND) return;
    const long = hip.y - minY >= SKIRT_MIN_DROP * thigh && spread(o, us) >= SKIRT_SPREAD * skinSpread;
    cands.push({ o, keep, long });
  });
  // a piece passing every rule makes its vertex buffer a skirt; the shorter rings around the body in that buffer are
  // its other layers (AvatarSample_D: three tiers and a frill, only the longest reaches half a thigh)
  const skirts = new Set(cands.filter((c) => c.long).map((c) => c.o.geometry.attributes.position));
  const byBuf = new Map();
  for (const { o, keep } of cands) {
    const key = o.geometry.attributes.position;
    if (!skirts.has(key)) continue;
    if (!byBuf.has(key)) byBuf.set(key, { o, others: [], idx: new Set() });
    const e = byBuf.get(key);
    if (e.o !== o) e.others.push(o);
    for (const i of keep) e.idx.add(i);
  }
  const meshes = [...byBuf.values()].map((e) => {
    const idx = [...e.idx], pos = e.o.geometry.attributes.position;
    return { o: e.o, others: e.others, idx, orig: Float32Array.from(idx.flatMap((i) => [pos.getX(i), pos.getY(i), pos.getZ(i)])),
      springs: idx.some((i) => onSpring(e.o, i)) };
  });
  vrm.scene.userData.hsinSkirtVerts = meshes;
  for (const m of meshes) if (!m.springs) smoothSkirtWeights(vrm, m);
  // the cloth round the knees (2026-10-29, for hiking a long skirt, see drapeSkirt): the median distance from the hips'
  // axis at knee height over 36 directions x 2 pi, and the hem; only for a skirt whose hem is below the knees
  const ring = new Array(36).fill(0);
  let hem = Infinity, top = -Infinity;
  for (const m of meshes) for (const i of m.idx) {
    const v = world(m.o, i);
    hem = Math.min(hem, v.y); top = Math.max(top, v.y);
    if (Math.abs(v.y - knee.y) < 0.125 * thigh) {
      const a = Math.floor(((Math.atan2(v.z - centre.z, v.x - centre.x) / (2 * Math.PI)) + 1) * 36) % 36;
      ring[a] = Math.max(ring[a], Math.hypot(v.x - centre.x, v.z - centre.z));
    }
  }
  const rs = ring.filter((r) => r > 0).sort((a, b) => a - b);
  vrm.scene.userData.hsinSkirtRing = meshes.length && hem < knee.y && rs.length > 8
    ? { perimeter: 2 * Math.PI * rs[Math.floor(rs.length / 2)], hem, top, hipY: hip.y, thigh } : null;
  return meshes.reduce((a, m) => a + m.idx.length, 0);
}

const _v = new V(), _cp = new V(), _ab = new V(), _M = new THREE.Matrix4(), _S = new THREE.Matrix4(), _B = new THREE.Matrix4();

export function fixSkirtVertices(vrm, frame = 0) {
  const meshes = vrm.scene.userData.hsinSkirtVerts;
  if (!meshes?.length) return 0;
  const S = vrm.scene.userData.hsinSkirt;
  const cloth = S && S.current !== 'stand'
    ? (S.frames ? S.frames[Math.max(0, Math.min(S.frames.length - 1, Math.round(frame)))] : S.table.get(S.current)) : null;
  if (cloth) {                               // sitting: the legs do not move, so the settled cloth is a table lookup;
    for (const c of S.chains) for (const j of c) j.reset();   // chains held at rest so the skin matrices are the
    vrm.scene.updateMatrixWorld(true);                         // ones the cloth was mapped with
    meshes.forEach((m, k) => {
      const pos = m.o.geometry.attributes.position, loc = cloth[k];
      m.idx.forEach((i, n) => pos.setXYZ(i, loc[3 * n], loc[3 * n + 1], loc[3 * n + 2]));
      pos.needsUpdate = true;
    });
    return -1;
  }
  const limbs = (vrm.scene.userData.hsinLimbs ?? []).filter((l) => /Leg$/.test(l.name))
    .map((l) => ({ a: l.bone.localToWorld(l.a.clone()), b: l.bone.localToWorld(l.b.clone()), r: l.radius + SLACK }));
  let moved = 0;
  for (const { o, idx, orig } of meshes) {
    const pos = o.geometry.attributes.position;
    idx.forEach((i, n) => pos.setXYZ(i, orig[3 * n], orig[3 * n + 1], orig[3 * n + 2]));
    o.updateMatrixWorld(true);
    const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight, bones = o.skeleton.bones, inv = o.skeleton.boneInverses;
    let changed = false;
    for (const i of idx) {
      o.getVertexPosition(i, _v); _v.applyMatrix4(o.matrixWorld);           // skinned, world
      const target = _v.clone();
      const depthIn = (p, l) => {            // how far p is inside capsule l, and its closest axis point in _cp
        _ab.subVectors(l.b, l.a);
        const t = THREE.MathUtils.clamp(p.clone().sub(l.a).dot(_ab) / _ab.lengthSq(), 0, 1);
        _cp.copy(l.a).addScaledVector(_ab, t);
        return l.r - p.distanceTo(_cp);
      };
      for (const l of limbs) {
        const dep = depthIn(target, l);
        if (dep <= 0) continue;
        if (dep < 0.5 * l.r) {               // shallow only: out the nearest way. Deeper points would be pushed
          const nrm = target.clone().sub(_cp).normalize(); target.copy(_cp).addScaledVector(nrm, l.r);   // to
        }                                    // whichever side is nearest and tear the cloth
      }
      if (target.y < SLACK && onFloor(target)) target.y = SLACK;
      if (target.distanceToSquared(_v) < 1e-10) continue;
      // world = matrixWorld * bindMatrixInverse * sum_k w_k (bone_k.matrixWorld * boneInverse_k) * bindMatrix * local
      _S.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
      for (let k = 0; k < 4; k++) {
        const w = sw.getComponent(i, k);
        if (!w) continue;
        const j = si.getComponent(i, k);
        _B.multiplyMatrices(bones[j].matrixWorld, inv[j]);
        for (let e = 0; e < 16; e++) _S.elements[e] += w * _B.elements[e];
      }
      _M.copy(o.matrixWorld).multiply(o.bindMatrixInverse).multiply(_S).multiply(o.bindMatrix).invert();
      target.applyMatrix4(_M);
      pos.setXYZ(i, target.x, target.y, target.z);
      changed = true; moved++;
    }
    pos.needsUpdate = true;                 // also when nothing moved: the originals were restored
  }
  return moved;
}

const _P = new THREE.Matrix4();
function skinMatrix(o, i, out) {
  const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight, bones = o.skeleton.bones, inv = o.skeleton.boneInverses;
  _S.set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  for (let k = 0; k < 4; k++) {
    const w = sw.getComponent(i, k);
    if (!w) continue;
    const j = si.getComponent(i, k);
    _P.multiplyMatrices(bones[j].matrixWorld, inv[j]);
    for (let e = 0; e < 16; e++) _S.elements[e] += w * _P.elements[e];
  }
  return out.copy(o.matrixWorld).multiply(o.bindMatrixInverse).multiply(_S).multiply(o.bindMatrix);
}

// Sit-down simulation (2026-10-24). Real cloth ends up outside the legs because the legs push it while sitting
// down, so once per sitting posture: start from the standing pose (the skirt hangs correctly around the legs),
// move the body to the sitting pose over SIM_MOVE steps, then let it settle for SIM_SETTLE steps. The skirt mesh is
// a cloth: Verlet particles = welded skirt vertices; edge length constraints; 2-ring neighbours kept >= SIM_BEND of
// rest; gravity; the posed leg / pelvis surface of legSurface carried by the bones (particles always, edge
// midpoints while settling: during the sit-down they snagged the cloth on rising knees); forearm / hand capsules
// press the cloth down while settling (people sit, then lay their hands on the skirt); the floor with friction;
// the seam vertices follow the body. Mapped to bind space with the final pose.
// 120 steps to sit down, 120 to settle, 4 constraint passes. SIM_GAP: the cloth stays this far off the leg surface
// (only particles and edge midpoints collide, so a knee between them poked through at 5 mm and w_sit's back hem
// flipped at 1.2 cm). SIM_BEND: least distance of 2-ring neighbours as a share of rest, against folding over (0.9
// folded w_sit's hem under the knees showing the inner side).
// MID_WHILE_SITTING (2026-10-29): edge midpoints also collide while sitting down; a raised knee tents the cloth (test2
// S029, knees_hugged: the knee passed between particles 5-8 cm apart and the dress tore); was settling only
let MOTION_PRESS = 'split', SPLIT_PREV = false, SIDE_SLIDE = true, SIDE_FRAC = 0.5;   // only for hands beside the body (h.beside; for hands on the thighs it flipped test2 T041's back hem)   // hands and the cloth through a motion (2026-10-29, see clothStep): 'split' | true | false
let SIM_MOVE = 120, MID_WHILE_SITTING = true, SKIN_SURFACE = true, PINS_ON_HIPS = false;   // SKIN_SURFACE: leg surface from the skin component only (2026-10-29, test1 side_left)
const SIM_SETTLE = 120, SIM_ITERS = 4, SIM_GAP_FLOOR = 0.016, SIM_BEND = 0.97;
// on a ledge the thighs are level and the cloth lies across them: a wider gap (the thighs showed through at their sides,
// 2026-11-11)
const SIM_GAP_LEDGE = 0.03;
let SIM_GAP = SIM_GAP_FLOOR;
export function setClothTrial(o) {
  if ('move' in o) SIM_MOVE = o.move; if ('mid' in o) MID_WHILE_SITTING = o.mid;
  if ('skin' in o) SKIN_SURFACE = o.skin; if ('pins' in o) PINS_ON_HIPS = o.pins;
  if ('press' in o) MOTION_PRESS = o.press; if ('prev' in o) SPLIT_PREV = o.prev; if ('side' in o) SIDE_SLIDE = o.side; if ('sidefrac' in o) SIDE_FRAC = o.sidefrac;
}   // experiments

function boneLocalSegs(vrm, segs) {     // leg surface axes in the frame of the bone that carries them
  const H = vrm.humanoid;
  const owners = ['hips', ...LEG_SEGS.map(([f]) => f)];
  return segs.map((g, k) => {
    const bone = H.getRawBoneNode(owners[k]);
    const inv = bone.matrixWorld.clone().invert();
    const dir = (v) => v.clone().transformDirection(inv);
    return { bone, a: g.a.clone().applyMatrix4(inv), u: dir(g.u), e1: dir(g.e1), e2: dir(g.e2), L: g.L, R: g.R };
  });
}
function worldSegs(local) {
  return local.map((g) => ({ a: g.a.clone().applyMatrix4(g.bone.matrixWorld), u: g.u.clone().transformDirection(g.bone.matrixWorld),
    e1: g.e1.clone().transformDirection(g.bone.matrixWorld), e2: g.e2.clone().transformDirection(g.bone.matrixWorld), L: g.L, R: g.R }));
}

function clothSetup(vrm, segs, hike = 1) {
  const S = vrm.scene.userData.hsinSkirt, H = vrm.humanoid;
  const names = Object.keys(H.humanBones).filter((n) => H.getNormalizedBoneNode(n));
  const sitQ = new Map(names.map((n) => [n, H.getNormalizedBoneNode(n).quaternion.clone()]));
  const hipsN = H.getNormalizedBoneNode('hips'), sitHips = hipsN.position.clone(), I = new THREE.Quaternion();
  const pose = (t) => {
    for (const n of names) H.getNormalizedBoneNode(n).quaternion.slerpQuaternions(I, sitQ.get(n), t);
    hipsN.position.lerpVectors(S.hipsRest, sitHips, t);
    H.update(); vrm.scene.updateMatrixWorld(true);
  };
  const local = boneLocalSegs(vrm, segs);
  const arms = (vrm.scene.userData.hsinLimbs ?? []).filter((l) => /(LowerArm|Hand)$/.test(l.name));   // forearms, hands
  pose(0);
  const meshes = vrm.scene.userData.hsinSkirtVerts, sims = [], R0 = vrm.scene.userData.hsinSkirtRing;
  for (const m of meshes) {                                 // particles, edges, pins at the standing pose
    const { o, idx, orig } = m, pos = o.geometry.attributes.position;
    idx.forEach((i, n) => pos.setXYZ(i, orig[3 * n], orig[3 * n + 1], orig[3 * n + 2]));
    const local = new Map(idx.map((i, n) => [i, n]));
    const pid = new Map(), part = [], vOf = new Int32Array(idx.length);
    idx.forEach((i, n) => {
      const key = [0, 1, 2].map((c) => Math.round(orig[3 * n + c] / 0.0005)).join(',');
      if (!pid.has(key)) {
        pid.set(key, part.length);
        const x = new V(); o.getVertexPosition(i, x); x.applyMatrix4(o.matrixWorld);
        if (hike < 1 && R0) x.y = R0.top - (R0.top - x.y) * hike;   // hiked: the cloth shortened toward its waistband
        part.push({ x, p: x.clone(), pin: false, nb: new Set(), rep: i });
      }
      vOf[n] = pid.get(key);
    });
    const edges = new Map(), tris = [];
    for (const mesh of [o, ...m.others]) {
      const ix = mesh.geometry.index;
      if (!ix) continue;
      for (let t = 0; t < ix.count; t += 3) {
        const l3 = [0, 1, 2].map((e) => local.get(ix.getX(t + e)));
        if (l3.every((x) => x !== undefined)) {
          const t3 = l3.map((x) => vOf[x]);
          if (t3[0] !== t3[1] && t3[1] !== t3[2] && t3[0] !== t3[2]) tris.push(t3);
        }
      }
      for (let t = 0; t < ix.count; t += 3) for (let e = 0; e < 3; e++) {
        const la = local.get(ix.getX(t + e)), lb = local.get(ix.getX(t + (e + 1) % 3));
        if (la === undefined && lb === undefined) continue;
        if (la === undefined || lb === undefined) { part[vOf[la ?? lb]].pin = true; continue; }
        const a = vOf[la], b = vOf[lb];
        if (a === b) continue;
        part[a].nb.add(b); part[b].nb.add(a);
        const key = a < b ? a * 1e6 + b : b * 1e6 + a;
        if (!edges.has(key)) edges.set(key, [a, b, part[a].x.distanceTo(part[b].x)]);
      }
    }
    const bends = new Map();                                // 2-ring pairs: keep at least 80 % apart
    part.forEach((P, a) => { for (const b of P.nb) for (const c of part[b].nb) {
      if (c === a || P.nb.has(c)) continue;
      const key = a < c ? a * 1e6 + c : c * 1e6 + a;
      if (!bends.has(key)) bends.set(key, [a, c, SIM_BEND * P.x.distanceTo(part[c].x)]);
    } });
    seamRings(part);
    const hipsBone = H.getRawBoneNode('hips'), inv = hipsBone.matrixWorld.clone().invert();
    for (const P of part) if (P.pin) P.hipRel = P.x.clone().applyMatrix4(inv);   // the seam rides on the hips alone
    sims.push({ m, part, vOf, E: [...edges.values()], B: [...bends.values()], follow: layerFollowers(part, tris) });
  }
  const finish = () => {                                    // back to the sitting pose the simulation started from
    for (const n of names) H.getNormalizedBoneNode(n).quaternion.copy(sitQ.get(n));
    hipsN.position.copy(sitHips); H.update(); vrm.scene.updateMatrixWorld(true);
  };
  // half the distance between the hip joints plus a thigh's skin radius: a hand further out than this (in the hips'
  // frame) is beside the body, not over the legs (SIDE_SLIDE)
  const hipsBone = H.getRawBoneNode('hips');
  const lj = H.getRawBoneNode('leftUpperLeg'), rj = H.getRawBoneNode('rightUpperLeg');
  const thighR = Math.max(0.03, ...(vrm.scene.userData.hsinLimbs ?? []).filter((l) => /UpperLeg$/.test(l.name)).map((l) => l.radius));
  const sideX = 0.5 * lj.getWorldPosition(new THREE.Vector3()).distanceTo(rj.getWorldPosition(new THREE.Vector3())) + thighR;
  return { sims, local, arms, pose, finish, hipsBone, sideX };
}

// Layered skirts (2026-10-28, AvatarSample_D: three tiers and a frill, each its own piece of cloth). Simulated side by
// side the layers passed through each other when sitting and the inner sides showed (dark red). So only the longest
// piece (lowest hem) is simulated; a piece lying on it (LAYER_SHARE of its particles within LAYER_GAP of its surface
// at rest) follows it: each particle is tied to its nearest triangle (barycentric point + offset along the triangle's
// normal) and placed from it after every step, so the layers keep their order. Pieces side by side (panels) are far
// from each other and are simulated as before.
const LAYER_GAP = 0.025, LAYER_SHARE = 0.8;

function closestOnTri(p, a, b, c, out) {            // closest point of triangle abc to p; returns barycentric [u, v, w]
  const ab = new V().subVectors(b, a), ac = new V().subVectors(c, a), ap = new V().subVectors(p, a);
  const d1 = ab.dot(ap), d2 = ac.dot(ap);
  if (d1 <= 0 && d2 <= 0) { out.copy(a); return [1, 0, 0]; }
  const bp = new V().subVectors(p, b), d3 = ab.dot(bp), d4 = ac.dot(bp);
  if (d3 >= 0 && d4 <= d3) { out.copy(b); return [0, 1, 0]; }
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) { const v = d1 / (d1 - d3); out.copy(a).addScaledVector(ab, v); return [1 - v, v, 0]; }
  const cp = new V().subVectors(p, c), d5 = ab.dot(cp), d6 = ac.dot(cp);
  if (d6 >= 0 && d5 <= d6) { out.copy(c); return [0, 0, 1]; }
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) { const w = d2 / (d2 - d6); out.copy(a).addScaledVector(ac, w); return [1 - w, 0, w]; }
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) {
    const w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); out.copy(b).addScaledVector(new V().subVectors(c, b), w); return [0, 1 - w, w];
  }
  const den = 1 / (va + vb + vc), v = vb * den, w = vc * den;
  out.copy(a).addScaledVector(ab, v).addScaledVector(ac, w); return [1 - v - w, v, w];
}

function triNormal(part, t, out) {
  const [a, b, c] = t.map((k) => part[k].x);
  return out.subVectors(b, a).cross(new V().subVectors(c, a)).normalize();
}

// Cloth next to the waist seam (pinned particles, skinned to the hips) rides with the body: pushed out of the legs it
// was torn from the seam (AvatarSample_D, whose thighs are thick at the hip: edges stretched 3-8x). Particles within
// SEAM_DIST of a pin, measured along the cloth at rest, skip the leg collision while that pin itself is inside the leg
// surface (D's seam is wrapped by the thick thighs; test2's seam is clear of them, and skipping there let its cloth
// sink into the legs and stretch).
const SEAM_DIST = 0.055;
const SEAM_PUSH = 0.3;
function seamRings(part) {
  part.forEach((P, i) => { P.seam = P.pin ? 0 : Infinity; P.seamPin = P.pin ? i : -1; });
  const todo = part.map((P, i) => i).filter((i) => part[i].pin);
  while (todo.length) {                       // small graphs: a plain relaxation queue is enough
    const i = todo.shift();
    for (const j of part[i].nb) {
      const d = part[i].seam + part[i].x.distanceTo(part[j].x);
      if (d < part[j].seam && d < SEAM_DIST) { part[j].seam = d; part[j].seamPin = part[i].seamPin; todo.push(j); }
    }
  }
}

function markSeamInside(part, W) {            // per step: P.ring = 0 = skip the leg collision
  const tmp = new V();
  const inside = new Map();
  for (const P of part) {
    if (P.seamPin < 0 || P.pin) { P.ring = Infinity; continue; }
    if (!inside.has(P.seamPin)) inside.set(P.seamPin, pushOutLegs(W, tmp.copy(part[P.seamPin].x), 0.05));
    P.ring = 0;
  }
}

function layerFollowers(part, tris) {
  const comp = new Int32Array(part.length).fill(-1); let nc = 0;     // pieces = connected particles
  for (let i = 0; i < part.length; i++) {
    if (comp[i] >= 0) continue;
    const st = [i]; comp[i] = nc;
    while (st.length) { const v = st.pop(); for (const w of part[v].nb) if (comp[w] < 0) { comp[w] = nc; st.push(w); } }
    nc++;
  }
  if (nc < 2) return [];
  const low = new Float64Array(nc).fill(9), size = new Int32Array(nc);
  part.forEach((P, i) => { size[comp[i]]++; low[comp[i]] = Math.min(low[comp[i]], P.x.y); });
  let master = -1;
  for (let c = 0; c < nc; c++) if (size[c] >= 20 && (master < 0 || low[c] < low[master])) master = c;
  if (master < 0) return [];
  const mt = tris.filter((t) => comp[t[0]] === master);
  if (!mt.length) return [];
  const follow = [], q = new V(), nrm = new V();
  for (let c = 0; c < nc; c++) {
    if (c === master) continue;
    const ties = [];
    let near = 0, count = 0;
    part.forEach((P, i) => {
      if (comp[i] !== c) return;
      count++;
      let best = null;
      for (const t of mt) {
        const bc = closestOnTri(P.x, part[t[0]].x, part[t[1]].x, part[t[2]].x, q);
        const d = q.distanceTo(P.x);
        if (!best || d < best.d) best = { d, t, bc, h: triNormal(part, t, nrm).dot(new V().subVectors(P.x, q)) };
      }
      if (best.d < LAYER_GAP) near++;
      ties.push({ i, t: best.t, bc: best.bc, h: best.h });
    });
    if (count && near / count >= LAYER_SHARE) follow.push(...ties.filter((x) => !part[x.i].pin));
  }
  return follow;
}

function placeFollowers(part, follow) {
  const n = new V();
  for (const f of follow) {
    const P = part[f.i];
    P.x.set(0, 0, 0);
    f.t.forEach((k, j) => P.x.addScaledVector(part[k].x, f.bc[j]));
    P.x.addScaledVector(triNormal(part, f.t, n), f.h);
    P.p.copy(P.x);
  }
}

const _g = new V(0, -9.8 / 3600, 0), _d = new V(), _q = new V();

// One cloth step with the body as it is posed now. settling: edge midpoints collide too; press: hands press the
// cloth down (the sit-down settle only).
function clothStep(C, settling, press = settling) {
  const { sims, local, arms } = C;
  const W = worldSegs(local);
  const hands = arms.map((l) => ({ a: l.bone.localToWorld(l.a.clone()), b: l.bone.localToWorld(l.b.clone()), r: l.radius + SLACK }));
  for (const h of hands) {           // beside the body (outside the thighs, in the hips' frame), see SIDE_SLIDE
    const m = C.hipsBone.worldToLocal(h.a.clone().add(h.b).multiplyScalar(0.5));
    h.beside = Math.abs(m.x) > C.sideX;
  }
  const g = _g, d = _d, q = _q;
  for (const { m, part } of sims) {
    for (const P of part) {
      if (!P.pin) continue;
      if (PINS_ON_HIPS && P.hipRel) { P.x.copy(P.hipRel).applyMatrix4(C.hipsBone.matrixWorld); P.p.copy(P.x); continue; }
      m.o.getVertexPosition(P.rep, P.x); P.x.applyMatrix4(m.o.matrixWorld); P.p.copy(P.x);
    }
    markSeamInside(part, W);
    for (const P of part) {
      if (P.pin) continue;
      q.copy(P.x).sub(P.p).multiplyScalar(0.96);           // Verlet with damping
      P.p.copy(P.x); P.x.add(q).add(g);
    }
  }
  for (let it = 0; it < SIM_ITERS; it++) {
    for (const { part, E, B } of sims) {
      for (const [a, b, L] of E) {
        const A = part[a], Bp = part[b];
        d.subVectors(Bp.x, A.x); const len = d.length();
        if (len < 1e-7) continue;
        const wa = A.pin ? 0 : 1, wb = Bp.pin ? 0 : 1;
        if (!(wa + wb)) continue;
        d.multiplyScalar((len - L) / (len * (wa + wb)));
        if (wa) A.x.add(d); if (wb) Bp.x.sub(d);
      }
      for (const [a, b, L] of B) {
        const A = part[a], Bp = part[b];
        d.subVectors(Bp.x, A.x); const len = d.length();
        if (len >= L || len < 1e-7) continue;
        const wa = A.pin ? 0 : 1, wb = Bp.pin ? 0 : 1;
        if (!(wa + wb)) continue;
        d.multiplyScalar(0.5 * (len - L) / (len * (wa + wb)));
        if (wa) A.x.add(d); if (wb) Bp.x.sub(d);
      }
      if (settling || MID_WHILE_SITTING) for (const [a, b] of E) {   // edge midpoints: a knee slipped between particles
        // (2026-10-24 they were settling only, a tent over rising knees looked wrong; for a raised knee it is right)
        const A = part[a], Bp = part[b];
        if (A.ring === 0 || Bp.ring === 0) continue;
        const mid = A.x.clone().add(Bp.x).multiplyScalar(0.5), m0 = mid.clone();
        if (!pushOutLegs(W, mid, SIM_GAP)) continue;
        mid.sub(m0);
        const wa = A.pin ? 0 : 1, wb = Bp.pin ? 0 : 1, k = 2 / (wa + wb);
        if (wa) A.x.addScaledVector(mid, 0.5 * k);
        if (wb) Bp.x.addScaledVector(mid, 0.5 * k);
      }
      for (const P of part) {
        if (P.pin) continue;
        if (press) for (const h of hands) {             // hands rest on top of the cloth (settling only:
          d.subVectors(h.b, h.a);                                    // while sitting down they folded cross-legged
          const tt = THREE.MathUtils.clamp(q.subVectors(P.x, h.a).dot(d) / d.lengthSq(), 0, 1);   // cloth under itself)
          q.copy(h.a).addScaledVector(d, tt);
          const dist = P.x.distanceTo(q);
          // pressed down under the hand (people sit, then lay their hands on the skirt), not pushed out sideways
          if (!(dist < h.r && dist > 1e-7)) continue;
          // cloth beside a hand that is beside the body (not above or below it) slides off to the side, out of the
          // capsule (2026-11-03, test2
          // T004: a hand going from the thigh down to the floor beside the hip pressed the dress side straight down
          // with it, and the dress tore open at the hip)
          if (SIDE_SLIDE && press !== true && h.beside && Math.abs(P.x.y - q.y) < SIDE_FRAC * dist) {
            P.x.sub(q).multiplyScalar(h.r / dist).add(q);
            continue;
          }
          const dy = Math.sqrt(Math.max(h.r * h.r - (P.x.x - q.x) ** 2 - (P.x.z - q.z) ** 2, 0));
          // through a motion (press 'split'): cloth above the hand's axis goes over it, below it under it. Pressing
          // all of it under dragged cloth lying on a hand through it step after step (test2 T001J: in the held pose
          // the dress turned inside out); without any press a hand on a raised knee let it slide off (S029).
          // trial 'prev' (off; 2026-11-03): for a low hand decide by the side the cloth came from (its previous
          // position). It fixed test2 T004 (hands going down beside the hips cut the dress side in two, torn open) but
          // tore T001J / S029 (a hand on the floor there too); for all capsules / forearms only / hands only it was
          // the same trade-off. T004's tear is left open.
          if (press === 'split' && (SPLIT_PREV && Math.min(h.a.y, h.b.y) < PREV_BELOW ? P.p.y : P.x.y) > q.y) P.x.y = Math.max(P.x.y, q.y + dy);
          else P.x.y = Math.min(P.x.y, q.y - dy);
        }
        if (P.ring !== 0) pushOutLegs(W, P.x, SIM_GAP);
        else {                               // near the seam: only part of the push, the edges to the seam win
          q.copy(P.x);
          if (pushOutLegs(W, q, SIM_GAP)) P.x.lerp(q, SEAM_PUSH);
        }
        if (P.x.y < SLACK && onFloor(P.x)) { P.x.y = SLACK; P.p.x = P.x.x * 0.8 + P.p.x * 0.2; P.p.z = P.x.z * 0.8 + P.p.z * 0.2; }   // floor friction
      }
    }
  }
  for (const { part, follow } of sims) placeFollowers(part, follow);
}

// The cloth now, mapped to bind space with the body's current pose (drawn with that pose it is exact).
function clothSnapshot(C) {
  const out = [];
  for (const { m, part, vOf } of C.sims) {
    const { o, idx, orig } = m, loc = new Float32Array(3 * idx.length), M = new THREE.Matrix4();
    idx.forEach((i, n) => {
      const P = part[vOf[n]];
      const v = P.pin && !PINS_ON_HIPS ? new V().fromArray(orig, 3 * n) : P.x.clone().applyMatrix4(skinMatrix(o, i, M).invert());
      loc.set([v.x, v.y, v.z], 3 * n);
    });
    out.push(loc);
  }
  return out;
}

// Sit down from standing into the current (sitting) pose and settle.
function sitDown(C) {
  for (let step = 0; step <= SIM_MOVE + SIM_SETTLE; step++) {
    const t = Math.min(1, step / SIM_MOVE);
    C.pose(t * t * (3 - 2 * t));
    clothStep(C, step > SIM_MOVE);
  }
  C.finish();
}

function simulateSit(vrm, segs, hike = 1) {
  const C = clothSetup(vrm, segs, hike);
  sitDown(C);
  return clothSnapshot(C);
}

// Legs that change while sitting (2026-10-28: sitting leg words, e.g. legs_out -> one knee up; the cloth draped once
// for the first frame's legs tore and turned inside out when a knee rose): sit down into frame 0 as above, then step
// the cloth through the motion, SIM_PER_FRAME steps per frame, and keep one shape per frame. poseAt(f) poses the
// body at frame f.
const SIM_PER_FRAME = 4;

function simulateMotion(vrm, segs, poseAt, nFrames, hike = 1) {
  poseAt(0); vrm.scene.updateMatrixWorld(true);
  const C = clothSetup(vrm, segs, hike);
  sitDown(C);
  const frames = [clothSnapshot(C)];
  for (let f = 1; f < nFrames; f++) {
    poseAt(f); vrm.scene.updateMatrixWorld(true);
    for (let k = 0; k < SIM_PER_FRAME; k++) clothStep(C, true, MOTION_PRESS);
    frames.push(clothSnapshot(C));
  }
  poseAt(0); vrm.scene.updateMatrixWorld(true);
  return frames;
}

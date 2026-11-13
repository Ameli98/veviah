// Measures a VRM in its rest pose for the motion solver (motion_loop/skeleton.py): bone positions and a fit of the
// body surface. Runs in the browser on a loaded model: by tools/skeleton.mjs and tools/fit_capsules.mjs, and by the
// MCP view (app/avatar.js) the first time a user's model is loaded, so any model works without a manual step.
// Both leave the model in its rest (T) pose.
import * as THREE from 'three';

export const SKELETON_BONES = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head',
  'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes', 'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes',
  'leftShoulder', 'leftUpperArm', 'leftLowerArm', 'leftHand', 'rightShoulder', 'rightUpperArm', 'rightLowerArm', 'rightHand'];

// Rest (T-pose, standing) world positions of the normalized humanoid bones -> motion_loop/skeletons/<model>.json
export function measureSkeleton(vrm) {
  vrm.humanoid.resetNormalizedPose(); vrm.update(0); vrm.scene.updateMatrixWorld(true);
  const r = {};
  for (const b of SKELETON_BONES) {
    const n = vrm.humanoid.getNormalizedBoneNode(b);
    if (!n) continue;
    const p = n.getWorldPosition(new THREE.Vector3());
    r[b] = [p.x, p.y, p.z].map((v) => Math.round(v * 10000) / 10000);   // rotateVRM0 already turned 0.x models to face +z
  }
  return r;
}

// Body surface fit -> motion_loop/body_fits/<model>.json (see tools/fit_capsules.mjs for what it holds)
const SEGMENTS = [          // [name, from bone, to bone (first one the model has)]
  ['hips', 'hips', ['spine']], ['spine', 'spine', ['chest', 'upperChest', 'neck']],
  ['chest', 'chest', ['upperChest', 'neck']], ['upperChest', 'upperChest', ['neck']], ['neck', 'neck', ['head']],
  ...['left', 'right'].flatMap((s) => [
    [`${s}UpperArm`, `${s}UpperArm`, [`${s}LowerArm`]], [`${s}LowerArm`, `${s}LowerArm`, [`${s}Hand`]],
    [`${s}Hand`, `${s}Hand`, [`${s}MiddleProximal`, `${s}IndexProximal`]],
    [`${s}UpperLeg`, `${s}UpperLeg`, [`${s}LowerLeg`]], [`${s}LowerLeg`, `${s}LowerLeg`, [`${s}Foot`]],
    [`${s}Foot`, `${s}Foot`, [`${s}Toes`]]]),
];

// The skin component (2026-10-29): the skinned mesh (glTF primitive) holding the most vertices that hang mostly from
// hand / finger bones, counting only the vertices its index uses. Gloves or clothes baked into it move with the bones
// too, so treating them as the body is safe. Returns the SkinnedMesh (or null).
export function skinMesh(vrm) {
  const H = vrm.humanoid, rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const humanOf = (node) => { let n = node; while (n && !rawToName.has(n)) n = n.parent; return n ? rawToName.get(n) : null; };
  let best = null, most = 0;
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const used = g.index ? new Set(g.index.array) : Array.from({ length: g.attributes.position.count }, (_, i) => i);
    let hands = 0;
    for (const i of used) {
      let b = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, b)) b = k;
      if (/Hand|Thumb|Index|Middle|Ring|Little/.test(humanOf(o.skeleton.bones[sk.getComponent(i, b)]) ?? '')) hands++;
    }
    if (hands > most) { most = hands; best = o; }
  });
  return best;
}

const TORSO_BONES = new Set(['hips', 'spine', 'chest', 'upperChest']);

export function fitBody(vrm) {
  const H = vrm.humanoid;
  H.resetNormalizedPose(); vrm.update(0); vrm.scene.updateMatrixWorld(true);
  const V = vrm.scene.position.constructor;
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const spring = new Set();
  for (const j of vrm.springBoneManager?.joints ?? []) spring.add(j.bone);
  const owner = new Map();            // joint node -> [humanoid name, passed a spring joint]
  const ownerOf = (node) => {
    if (owner.has(node)) return owner.get(node);
    let n = node, viaSpring = false;
    while (n && !rawToName.has(n)) { if (spring.has(n)) viaSpring = true; n = n.parent; }
    const r = [n ? rawToName.get(n) : null, viaSpring]; owner.set(node, r); return r;
  };
  const pos = (b) => { const n = H.getNormalizedBoneNode(b); return n ? n.getWorldPosition(new V()) : null; };
  const verts = new Map();            // humanoid name -> [V...]
  let floor = Infinity, hairN = 0, total = 0;
  const clothTorso = new Map();       // bone -> clothes vertices (see the fills below)
  const v = new V();
  // The body is measured from its skin only (2026-10-29, the author: the limbs are solved on the body, clothes are
  // pressed by the hands on the GPU, limb_push.js): the skin = the mesh component (glTF primitive) holding the most
  // vertices of the hands. Anything else in that component (gloves, clothes baked into it) moves with the bones too,
  // so avoiding it cannot clip. The head keeps every non-hair component (face, ears; hair: see HAIR_SPRING_MIN). Each component uses only the
  // vertices its index names (VRoid's merged primitives share one vertex buffer; counting the whole buffer for every
  // primitive made the per-mesh tallies wrong).
  // Hair = a component with at least HAIR_SPRING_MIN vertices hanging from spring joints under the head / neck; the
  // whole component goes, also its static part (bangs, the cap). By shape (2026-11-05, the author; was by mesh / material
  // name hair / bang / ahoge / 髪). On the six test models: every hair mesh has 45-8000 such vertices, faces / eyes /
  // the skin's head 0; VRoid's HairBack (the static layer on the scalp, no springs) now counts as head.
  const HAIR_SPRING_MIN = 20;
  const strongest = (sk, sw, i) => { let b = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, b)) b = k; return sk.getComponent(i, b); };
  const comps = [];
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const used = g.index ? [...new Set(g.index.array)] : Array.from({ length: g.attributes.position.count }, (_, i) => i);
    let hands = 0, headSpring = 0;
    for (const i of used) {
      const [nm, viaSpring] = ownerOf(o.skeleton.bones[strongest(sk, sw, i)]);
      if (nm && /Hand|Thumb|Index|Middle|Ring|Little/.test(nm)) hands++;
      if (viaSpring && (nm === 'head' || nm === 'neck')) headSpring++;
    }
    comps.push({ o, used, hands, hair: headSpring >= HAIR_SPRING_MIN });
  });
  const skin = comps.reduce((a, c) => (c.hands > (a?.hands ?? 0) ? c : a), null)?.o ?? null;
  for (const { o, used, hair: hairMesh } of comps) {
    const sk = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
    for (const i of used) {
      const [name, viaSpring] = ownerOf(o.skeleton.bones[strongest(sk, sw, i)]);
      total++;
      // spring-driven vertices are hair / skirts / ribbons, except the skin's own bust (VRoid's J_Sec_*_Bust springs
      // under the chest): dropping it left the chest without a front surface between the ribs and the collarbones, and
      // hands 'on the chest' lay on the ribs inside the bust (2026-11-03, test2)
      const bust = viaSpring && o === skin && (name === 'chest' || name === 'upperChest');
      if (!name || hairMesh || (viaSpring && !bust)) { hairN++; continue; }
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      floor = Math.min(floor, v.y);                 // shoes too
      if (skin && o !== skin && name !== 'head') {
        if (!clothTorso.has(name)) clothTorso.set(name, []);
        clothTorso.get(name).push(v.clone());
        continue;
      }
      if (!verts.has(name)) verts.set(name, []);
      const q = v.clone(); q.mesh = o.name + '/' + (o.material?.name ?? ''); verts.get(name).push(q);
    }
  }
  // a limb bone with no skin at all (AvatarSample_D's upper arms under the big sleeves): its clothes are its surface,
  // for the samples and the segment (without it the 'shoulder' target fell back to the chest, 2026-11-03). Arms and
  // legs only (hands / fingers keep their own samples: they set the hand lengths).
  for (const [b, qs] of clothTorso) if (/(Shoulder|UpperArm|LowerArm|UpperLeg|LowerLeg)$/.test(b) && !(verts.get(b)?.length)) verts.set(b, qs);
  const pct = (xs, p) => { if (!xs.length) return null; const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
  const r3 = (x) => (x == null ? null : Math.round(x * 1000) / 1000);
  const segs = {};
  for (const [seg, from, tos] of SEGMENTS) {
    const a = pos(from); const to = tos.find((b) => pos(b)); const vs = verts.get(from) ?? [];
    if (!a || !to || !vs.length) continue;
    const b = pos(to), ax = b.clone().sub(a), L = ax.length(); ax.normalize();
    const side = seg.startsWith('left') ? 1 : seg.startsWith('right') ? -1 : 0;
    const all = [], front = [], back = [], outw = [], inw = [], meshes = {};
    for (const p of vs) {
      const d = p.clone().sub(a), t = d.dot(ax) / L;
      if (t < -0.1 || t > 1.1) continue;
      const rad = d.sub(ax.clone().multiplyScalar(t * L)), r = rad.length();
      all.push(r); meshes[p.mesh] = (meshes[p.mesh] ?? 0) + 1;
      const dir = rad.clone().normalize();
      if (dir.z > 0.7) front.push(r); else if (dir.z < -0.7) back.push(r);
      const sx = side === 0 ? Math.sign(dir.x) || 1 : side;      // 'out' = away from the body's middle
      if (dir.x * sx > 0.7) outw.push(r); else if (dir.x * sx < -0.7) inw.push(r);
    }
    segs[seg] = { a: [a.x, a.y, a.z].map(r3), b: [b.x, b.y, b.z].map(r3), r: r3(pct(all, 0.9)), front: r3(pct(front, 0.9)),
      back: r3(pct(back, 0.9)), out: r3(pct(outw, 0.9)), in: r3(pct(inw, 0.9)), n: all.length, meshes };
  }
  const hv = verts.get('head') ?? [];
  const lo = ['x', 'y', 'z'].map((k) => pct(hv.map((p) => p[k]), 0.01)), hi = ['x', 'y', 'z'].map((k) => pct(hv.map((p) => p[k]), 0.99));
  const head = hv.length ? { c: lo.map((l, i) => r3((l + hi[i]) / 2)), r: lo.map((l, i) => r3((hi[i] - l) / 2)), n: hv.length } : null;
  // Surface samples per bone (at most 400, evenly strided), for contact targets: the target of a region is
  // the sample farthest out in the region's direction (a support point), which follows the real shape
  // (jaw, breasts, backpack) better than the capsules.
  const points = {};
  // Clothes fill the torso where the skin is missing (2026-11-03, test2: VRoid deletes the skin under clothes, so the
  // chest between the ribs and the collarbones had no front surface and hands 'on the chest' lay on the ribs). Only
  // the surface samples used by the contact targets get them, in 2 cm height bands with no skin vertex of the torso in
  // front of the spine; the segment radii (the filter's torso) stay skin-only.
  const front = (q) => q.z > 0.02;
  const skinBand = new Set();
  for (const b of TORSO_BONES) for (const q of verts.get(b) ?? []) if (front(q)) skinBand.add(Math.floor(q.y / 0.02));
  const filled = new Map();
  for (const [b, qs] of clothTorso) {
    if (!TORSO_BONES.has(b)) continue;
    const add = qs.filter((q) => front(q) && !skinBand.has(Math.floor(q.y / 0.02)));
    if (add.length) filled.set(b, add);
  }
  for (const [name, vs0] of verts) {
    const vs = TORSO_BONES.has(name) ? vs0.concat(filled.get(name) ?? []) : vs0;
    const step = Math.max(1, Math.floor(vs.length / 400));
    points[name] = vs.filter((_, i) => i % step === 0).map((p) => [r3(p.x), r3(p.y), r3(p.z)]);
  }
  // The head's outer surface in every direction (2026-11-05): the evenly strided samples above are mostly inside
  // (mouth, eyes) and sparse outside, so a head target (contact.py: the farthest sample in a 12 deg cone round a ray
  // from the head's centre) jumped by up to 4.5 cm when HairBack was added to the head. The outermost vertex per
  // direction bin (HEAD_BINS azimuth x elevation, seen from the ellipsoid centre) is added to the samples.
  if (head && hv.length) {
    const HEAD_BINS = [36, 18], c = new V(...head.c), best = new Map();
    for (const p of hv) {
      const d = p.clone().sub(c), r = d.length();
      if (r < 1e-6) continue;
      const az = Math.floor((Math.atan2(d.x, d.z) / Math.PI + 1) / 2 * HEAD_BINS[0]) % HEAD_BINS[0];
      const el = Math.min(HEAD_BINS[1] - 1, Math.floor((Math.asin(d.y / r) / Math.PI + 0.5) * HEAD_BINS[1]));
      const k = az * HEAD_BINS[1] + el;
      if (!best.has(k) || best.get(k).r < r) best.set(k, { p, r });
    }
    points.head = (points.head ?? []).concat([...best.values()].map(({ p }) => [r3(p.x), r3(p.y), r3(p.z)]));
  }
  // The torso's front INCLUDING clothes (2026-11-07, shy_cover: hands placed on the skin-only torso lay inside the
  // default model's loose T-shirt): per 2 cm x 2 cm cell (x across, y up) the frontmost z of every torso vertex, skin
  // and clothes (springs, hair excluded as above). Python: skeleton.clothed_front(x, y).
  const FG = 0.02, fg = new Map();
  const addF = (q) => { if (Math.abs(q.x) > 0.2) return; const k = Math.floor(q.x / FG) + ',' + Math.floor(q.y / FG); if (!fg.has(k) || fg.get(k) < q.z) fg.set(k, q.z); };
  for (const b of TORSO_BONES) { for (const q of verts.get(b) ?? []) addF(q); for (const q of clothTorso.get(b) ?? []) addF(q); }
  const front_grid = { cell: FG, z: Object.fromEntries([...fg].map(([k, z]) => [k, r3(z)])) };
  // a skirt (skirt.js, found by its shape at load): how low its hem hangs at rest, so a hand resting on a leg it covers
  // lands on the cloth, not on the skin under it (2026-10-29, test1's side sits: the hands sank into the skirt)
  let skirt = null;
  for (const e of vrm.scene.userData.hsinSkirtVerts ?? []) for (const i of e.idx) {
    e.o.getVertexPosition(i, v); v.applyMatrix4(e.o.matrixWorld);
    skirt = { hem: Math.min(skirt?.hem ?? Infinity, v.y) };
  }
  if (skirt) skirt.hem = r3(skirt.hem);
  // skinned = no skirt vertex on a spring joint, so skirt.js never simulates it (skinnedFor); the filter rejects raised
  // knees apart for such long skirts (filters.SKINNED_SKIRT, 2026-11-03)
  if (skirt) skirt.skinned = !(vrm.scene.userData.hsinSkirtVerts ?? []).some((m) => m.springs);
  return { floor: r3(floor), vertices: total, dropped_vertices: hairN, skin: skin ? skin.name + '/' + (skin.material?.name ?? '') : null,
    skirt, head, segments: segs, points, front_grid, mouth: mouthPoint(vrm), hair: hairChains(vrm), index: indexFinger(vrm) };
}

// The index finger relative to the hand bone at rest (2026-11-11, the author: a pointing index must sit ON the lips; the
// solver only knew the hand's own axis, and the index lies ~2 cm to the thumb side of it): base = the index's first
// joint, tip = its end (last joint + the last segment again), both minus the RIGHT hand bone, world axes of the rest
// pose (T-pose: the right hand points -x, the thumb side is +z). null without finger bones.
function indexFinger(vrm) {
  const H = vrm.humanoid, V = vrm.scene.position.constructor;
  const g = (n) => { const b = H.getRawBoneNode(n); return b ? b.getWorldPosition(new V()) : null; };
  const h = g('rightHand'), a = g('rightIndexProximal'), m = g('rightIndexIntermediate'), d = g('rightIndexDistal');
  if (!h || !a || !m || !d) return null;
  const tip = d.clone().add(d.clone().sub(m));
  const r = (v) => [r3m(v.x - h.x), r3m(v.y - h.y), r3m(v.z - h.z)];
  return { base: r(a), tip: r(tip) };
}

// The lips, by shape (2026-11-10, the author: an index upright on the lips; the 'mouth' target was a direction off the head
// centre that pointed chins pulled down): the vertices the 'aa' expression (open mouth) moves most. x, y = their mean
// weighted by the displacement, z = the 90th percentile of their z (the front of the lips). null without 'aa' morphs.
const r3m = (x) => Math.round(x * 1000) / 1000;
function mouthPoint(vrm) {
  const r3 = r3m;
  const em = vrm.expressionManager, aa = em?.getExpression?.('aa');
  const V = vrm.scene.position.constructor, v = new V(), pts = [];
  for (const b of aa?._binds ?? []) for (const m of b.primitives ?? []) {
    const d = m.geometry?.morphAttributes?.position?.[b.index];
    if (!d) continue;
    let mx = 0; for (let i = 0; i < d.count; i++) mx = Math.max(mx, Math.hypot(d.getX(i), d.getY(i), d.getZ(i)));
    for (let i = 0; i < d.count; i++) {
      const w = Math.hypot(d.getX(i), d.getY(i), d.getZ(i));
      if (w < 0.3 * mx) continue;
      m.getVertexPosition(i, v); v.applyMatrix4(m.matrixWorld); pts.push([v.x, v.y, v.z, w, d.getY(i)]);
    }
  }
  if (!pts.length) return null;
  const W = pts.reduce((a, p) => a + p[3], 0);
  const x = pts.reduce((a, p) => a + p[0] * p[3], 0) / W, y = pts.reduce((a, p) => a + p[1] * p[3], 0) / W;
  const zs = pts.map((p) => p[2]).sort((a, b) => a - b);
  const zf = zs[Math.floor(0.9 * (zs.length - 1))];
  // The upper lip (2026-11-11, the author: the shh fingertip goes to the upper lip): opening the mouth moves the lower lip
  // down and the upper lip up, so the upper lip = the moved vertices on the front of the face (within 1.2 cm of the
  // lips' front) that move UP; its top edge = the 90th percentile of their height. null when nothing moves up.
  const up = pts.filter((p) => p[4] > 0 && p[2] > zf - 0.012);
  let upper = null;
  if (up.length >= 4) {
    const ys = up.map((p) => p[1]).sort((a, b) => a - b), uz = up.map((p) => p[2]).sort((a, b) => a - b);
    upper = [r3(x), r3(ys[Math.floor(0.9 * (ys.length - 1))]), r3(uz[Math.floor(0.9 * (uz.length - 1))])];
  }
  return { p: [r3(x), r3(y), r3(zf)], upper, n: pts.length };
}

// Hair strands, by shape (2026-11-10): every spring chain that starts under the head bone, as the rest positions of its
// joints root -> tail (world, rest pose). Python (contact.hair_chains) sorts them into bangs / side hair / back hair /
// ponytail by where they grow and hang; chains whose tail is above their root (ears, ahoge) are dropped there.
function hairChains(vrm) {
  const r3 = r3m;
  const H = vrm.humanoid, head = H.getRawBoneNode('head');
  const joints = [...(vrm.springBoneManager?.joints ?? [])];
  if (!head || !joints.length) return [];
  const isJ = new Set(joints.map((j) => j.bone)), byBone = new Map(joints.map((j) => [j.bone, j]));
  const V = vrm.scene.position.constructor, out = [];
  for (const j of joints) {
    if (isJ.has(j.bone.parent)) continue;                       // chain roots only
    let under = false; for (let n = j.bone.parent; n; n = n.parent) if (n === head) { under = true; break; }
    if (!under) continue;
    const pts = [];
    for (let b = j.bone; b; ) {
      pts.push(b.getWorldPosition(new V()).toArray().map(r3));
      const nx = byBone.get(b)?.child ?? b.children.find((c) => isJ.has(c)) ?? null;
      if (!nx) break;
      if (!isJ.has(nx)) { pts.push(nx.getWorldPosition(new V()).toArray().map(r3)); break; }
      b = nx;
    }
    if (pts.length >= 2) out.push({ name: j.bone.name, pts });
  }
  return out;
}

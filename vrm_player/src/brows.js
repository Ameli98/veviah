// Eyebrows drawn over the bangs (2026-11-11, the author agreed): anime faces show the brows through the hair; with the bangs
// in front (test2, AvatarSample_D, the default model) angry / sad / shy hardly read. The brow part is found BY SHAPE,
// never by name: a small mesh part (< BROW_MAX_VERTS vertices) in front of the face, wider than BROW_MIN_W, its middle
// BROW_DY above the middle of what 'blink' moves (the eyes), and moved almost entirely by both 'angry' and 'sad'.
// While her face is turned toward the camera it is drawn last and without the depth test; turned away, normally.
import * as THREE from 'three';
const BROW_MAX_VERTS = 400, BROW_MIN_W = 0.06, BROW_DY = [0.012, 0.05], BROW_MOVED = 0.9, FACE_DOT = 0.25;

function movedSets(vrm, name) {
  const out = new Map();
  for (const b of vrm.expressionManager?.getExpression(name)?._binds ?? []) for (const m of b.primitives ?? []) {
    const d = m.geometry?.morphAttributes?.position?.[b.index];
    if (!d) continue;
    const s = out.get(m) ?? new Set();
    for (let i = 0; i < d.count; i++) if (Math.hypot(d.getX(i), d.getY(i), d.getZ(i)) > 1e-4) s.add(i);
    out.set(m, s);
  }
  return out;
}

export function findBrows(vrm) {
  vrm.scene.updateMatrixWorld(true);
  const v = new THREE.Vector3(), blink = movedSets(vrm, 'blink'), angry = movedSets(vrm, 'angry'), sad = movedSets(vrm, 'sad');
  let ey = 0, n = 0;
  for (const [m, s] of blink) for (const i of s) { m.getVertexPosition(i, v); v.applyMatrix4(m.matrixWorld); ey += v.y; n++; }
  if (!n) return [];
  ey /= n;
  const head = vrm.humanoid.getNormalizedBoneNode('head')?.getWorldPosition(new THREE.Vector3());
  const found = [];
  vrm.scene.traverse((o) => {
    if (!o.isMesh || !head) return;
    const g = o.geometry, ids = g.index ? [...new Set(g.index.array)] : [...Array(g.attributes.position.count).keys()];
    if (ids.length > BROW_MAX_VERTS || !angry.get(o) || !sad.get(o)) return;
    let cy = 0, cz = 0, lo = Infinity, hi = -Infinity, a = 0, s = 0;
    for (const i of ids) {
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      cy += v.y; cz += v.z; lo = Math.min(lo, v.x); hi = Math.max(hi, v.x);
      if (angry.get(o).has(i)) a++;
      if (sad.get(o).has(i)) s++;
    }
    cy /= ids.length; cz /= ids.length;
    const dy = cy - ey;
    if (hi - lo > BROW_MIN_W && dy > BROW_DY[0] && dy < BROW_DY[1] && cz > head.z && a / ids.length >= BROW_MOVED && s / ids.length >= BROW_MOVED) found.push(o);
  });
  return found;
}

const _f = new THREE.Vector3(), _c = new THREE.Vector3(), _h = new THREE.Vector3(), _q = new THREE.Quaternion();
// once per frame before rendering: over the hair while the face looks toward the camera
export function updateBrows(vrm, camera) {
  const brows = vrm.scene.userData.hsinBrows;
  if (!brows?.length) return;
  const head = vrm.humanoid.getNormalizedBoneNode('head');
  head.getWorldPosition(_h); head.getWorldQuaternion(_q);
  _f.set(0, 0, 1).applyQuaternion(_q);
  const facing = _f.dot(_c.copy(camera.position).sub(_h).normalize()) > FACE_DOT;
  for (const o of brows) {
    for (const mat of [].concat(o.material)) {
      if (mat.userData.hsinBrowOrig === undefined) mat.userData.hsinBrowOrig = { depthTest: mat.depthTest, transparent: mat.transparent };
      mat.depthTest = facing ? false : mat.userData.hsinBrowOrig.depthTest;
      mat.transparent = facing ? true : mat.userData.hsinBrowOrig.transparent;
    }
    o.renderOrder = facing ? 1000 : 0;
  }
}
export function initBrows(vrm) {
  vrm.scene.userData.hsinBrows = findBrows(vrm);
  return vrm.scene.userData.hsinBrows.map((o) => o.material?.name ?? o.name);
}

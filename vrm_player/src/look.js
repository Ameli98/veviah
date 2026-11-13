// Rendering look (2026-11-12, the author: game toon renders (ZZZ / WuWa / GFL2 / Snowbreak) look much better than plain
// MToon). Each part is a switch so they can be compared one by one; ?look=light,face,... or ?look=all.
// Works on the model's own MToon materials (the author's shade / outline colours stay); everything is found by shape.
//   light  = key light kept at a fixed place relative to the camera (upper front, viewer's left), exposure so a lit
//            surface shows its texture colour (key + ambient = 1, as VRoid models are authored in Unity) instead of
//            being clipped to white (the old sun pi + ambient 0.6 gave 1.19 x the texture)
//   face   = the face skin's normals point out of an ellipsoid round it (the trick behind games' SDF face shadows: the
//            light / shadow edge on the face becomes one smooth curve instead of patches round the nose / cheeks).
//            Face skin = the head's non-hair mesh component with the most morphed vertices (the eyelids, mouth).
//   outline = the author's outlines kept where they are (materials without one stay without), but as a fixed width on
//            the screen (OUTLINE_PX, so zooming does not thicken them) and coloured by the surface under them, darkened
//            (OUTLINE_DARK x the lit texture colour) instead of the usual black
//   bangs  = hair casts a shadow on the face / eyes: a hair-only depth pass; a face pixel whose point a little toward
//            the light (on the screen) is covered by hair just in front of it is drawn in its shade colour
import * as THREE from 'three';

// the author's pick (2026-11-12, one-at-a-time comparisons): on by default; a depth rim and warmer skin shades were tried
// and dropped (barely visible), as was 'edge' (the transparent window has no fringes)
export const LOOK_PARTS = ['light', 'face', 'bangs', 'outline'];
export function lookFromQuery(q) {
  const s = q.get('look');
  if (s === 'none' || s === '') return new Set();
  return new Set(s === null || s === 'all' ? LOOK_PARTS : s.split(','));
}

// camera frame: viewer's left, up, toward the viewer; ?lookkey=x,y,z for tests (e.g. a light from the side)
const KEY_Q = new URLSearchParams(globalThis.location?.search ?? '').get('lookkey');
const KEY_DIR = new THREE.Vector3(...(KEY_Q ? KEY_Q.split(',').map(Number) : [-0.45, 0.75, 1.0])).normalize();
const KEY_I = 0.85 * Math.PI, AMBIENT_I = 0.45;

// mesh components (one per skinned mesh = glTF primitive) with their humanoid owner per vertex and the hair flag
// (measure.js's rule: >= 20 vertices hanging from spring joints under the head / neck)
export function components(vrm) {
  const H = vrm.humanoid, rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const spring = new Set([...(vrm.springBoneManager?.joints ?? [])].map((j) => j.bone));
  const cache = new Map();
  const ownerOf = (node) => {
    if (cache.has(node)) return cache.get(node);
    let n = node, via = false;
    while (n && !rawToName.has(n)) { if (spring.has(n)) via = true; n = n.parent; }
    const r = [n ? rawToName.get(n) : null, via]; cache.set(node, r); return r;
  };
  const out = [];
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const used = g.index ? [...new Set(g.index.array)] : Array.from({ length: g.attributes.position.count }, (_, i) => i);
    const owner = new Map(); let headSpring = 0, hands = 0;
    for (const i of used) {
      let b = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, b)) b = k;
      const [nm, via] = ownerOf(o.skeleton.bones[sk.getComponent(i, b)]);
      owner.set(i, [nm, via]);
      if (via && (nm === 'head' || nm === 'neck')) headSpring++;
      if (nm && /Hand|Thumb|Index|Middle|Ring|Little/.test(nm)) hands++;
    }
    out.push({ o, used, owner, hair: headSpring >= 20, hands });
  });
  return out;
}

const FACE_W = 0.9;      // how far the face normals go toward the ellipsoid's (the rest keeps the nose / lips' shape)
function findFace(comps) {
  let face = null, most = 0;
  for (const c of comps) {
    if (c.hair) continue;
    const m = c.o.geometry.morphAttributes.position;
    if (!m?.length) continue;
    let head = 0, moved = 0;
    for (const i of c.used) {
      if (c.owner.get(i)[0] !== 'head') continue;
      head++;
      if (m.some((a) => Math.abs(a.getX(i)) + Math.abs(a.getY(i)) + Math.abs(a.getZ(i)) > 1e-5)) moved++;
    }
    if (head > c.used.length / 2 && moved > most) { most = moved; face = c; }
  }
  return face;
}
function radialFace(comps) {
  let face = null, most = 0;
  for (const c of comps) {
    if (c.hair) continue;
    const m = c.o.geometry.morphAttributes.position;
    if (!m?.length) continue;
    let head = 0, moved = 0;
    for (const i of c.used) {
      if (c.owner.get(i)[0] !== 'head') continue;
      head++;
      if (m.some((a) => Math.abs(a.getX(i)) + Math.abs(a.getY(i)) + Math.abs(a.getZ(i)) > 1e-5)) moved++;
    }
    if (head > c.used.length / 2 && moved > most) { most = moved; face = c; }
  }
  radialFace.last = face;
  if (!face) return 'no face';
  const g = face.o.geometry, P = g.attributes.position, N = g.attributes.normal;
  const ids = face.used.filter((i) => face.owner.get(i)[0] === 'head');
  const box = new THREE.Box3(), v = new THREE.Vector3();
  for (const i of ids) box.expandByPoint(v.fromBufferAttribute(P, i));
  const c = box.getCenter(new THREE.Vector3()), r = box.getSize(new THREE.Vector3()).multiplyScalar(0.5);
  const n = new THREE.Vector3(), e = new THREE.Vector3();
  for (const i of ids) {
    v.fromBufferAttribute(P, i).sub(c);
    e.set(v.x / (r.x * r.x), v.y / (r.y * r.y), v.z / (r.z * r.z)).normalize();
    n.fromBufferAttribute(N, i).lerp(e, FACE_W).normalize();
    N.setXYZ(i, n.x, n.y, n.z);
    // the morphs' normal changes no longer apply (but a morph without its own normals IS the base attribute in
    // GLTFLoader: zeroing that wiped the face's normals, 2026-11-12)
    for (const a of g.morphAttributes.normal ?? []) if (a !== N && a.array !== N.array) a.setXYZ(i, 0, 0, 0);
  }
  N.needsUpdate = true;
  for (const a of g.morphAttributes.normal ?? []) a.needsUpdate = true;     // (the morph texture is built at the first draw, after this)
  return `face ${face.o.name} ${ids.length} vertices`;
}

const OUTLINE_PX = 1.3, OUTLINE_DARK = 0.42;
function mtoonMaterials(vrm) {
  const out = new Set();
  vrm.scene.traverse((o) => { if (!o.isMesh) return; for (const m of [o.material].flat()) if (m?.isMToonMaterial) out.add(m); });
  return [...out];
}
function setupOutlines(vrm) {
  const mats = mtoonMaterials(vrm).filter((m) => m.isOutline && m.outlineWidthMode !== 'none' && m.outlineWidthFactor > 0);
  for (const m of mats) {
    m.outlineWidthMode = 'screenCoordinates';
    m.outlineColorFactor.setScalar(OUTLINE_DARK);
    m.outlineLightingMixFactor = 1;
    m.needsUpdate = true;
  }
  return mats;
}

const BANG_K = 0.7, BANG_TAP_M = 0.004;   // shadow strength, tap spacing (m at the head)
const BANG_M = 0.018, BANG_RANGE = [0.035, 0.09];          // shadow offset (m at the head), hair within this in front
const HAIR_LAYER = 3;

// shared uniforms of the patched materials
function lookUniforms() {
  return {
    hsinHairDepth: { value: null }, hsinRes: { value: new THREE.Vector2(1, 1) },
    hsinNF: { value: new THREE.Vector2(0.1, 20) },
    hsinKeyScr: { value: new THREE.Vector2(-0.5, 0.8) }, hsinBangOff: { value: new THREE.Vector2(0, 0) },
    hsinShadeK: { value: 1 }, hsinBangTap: { value: 2 },
  };
}
const FRAG_PARS = `
uniform sampler2D hsinHairDepth; uniform vec2 hsinRes; uniform vec2 hsinNF;
uniform vec2 hsinKeyScr; uniform vec2 hsinBangOff; uniform float hsinShadeK;
uniform float hsinBangOn; uniform float hsinBangTap;
float hsinLin(float d) { return hsinNF.x * hsinNF.y / (hsinNF.y - d * (hsinNF.y - hsinNF.x)); }
`;
const FRAG_MAIN = `
  #ifndef OUTLINE
  {
    vec2 hsinUv = gl_FragCoord.xy / hsinRes;
    float hsinMe = vViewPosition.z;
    if (hsinBangOn > 0.5) {
      // 3 x 3 taps round the offset point, averaged: thin strands gave hatched stripes (AvatarSample_D)
      float s = 0.0;
      for (int i = -1; i <= 1; i++) for (int j = -1; j <= 1; j++) {
        float h = texture2D(hsinHairDepth, hsinUv + (hsinBangOff + vec2(float(i), float(j)) * hsinBangTap) / hsinRes).x;
        if (h < 1.0) {
          float gap = hsinMe - hsinLin(h);
          s += step(0.002, gap) * (1.0 - smoothstep(${BANG_RANGE[0]}, ${BANG_RANGE[1]}, gap));
        }
      }
      s = smoothstep(0.15, 0.7, s / 9.0);
      col = mix(col, material.shadeColor * hsinShadeK + reflectedLight.indirectDiffuse, s * ${BANG_K});
    }
  }
  #endif
`;
function patchMaterial(m, U, bang) {
  const prev = m.onBeforeCompile;
  m.onBeforeCompile = (shader, r) => {
    prev?.call(m, shader, r);
    Object.assign(shader.uniforms, U, { hsinBangOn: { value: bang ? 1 : 0 } });
    shader.fragmentShader = shader.fragmentShader
      .replace('void main() {', FRAG_PARS + '\nvoid main() {')
      .replace('  col += totalEmissiveRadiance;', FRAG_MAIN + '\n  col += totalEmissiveRadiance;');
  };
  const key = m.customProgramCacheKey?.bind(m);
  m.customProgramCacheKey = () => (key ? key() : '') + '|hsinLook' + (bang ? 'B' : '');
  m.needsUpdate = true;
}

export function createLook(parts, { scene, camera, sun, ambient, renderer }) {
  const on = (p) => parts.has(p);
  const tmp = new THREE.Vector3();
  let outlines = [];
  const U = lookUniforms();
  const depthMat = new THREE.MeshDepthMaterial();
  let rtHair = null, head = null;
  const size = new THREE.Vector2();
  const target = (rt, w, h) => {
    if (rt && rt.width === w && rt.height === h) return rt;
    rt?.dispose();
    const t = new THREE.WebGLRenderTarget(w, h, { depthBuffer: true });
    t.depthTexture = new THREE.DepthTexture(w, h);
    return t;
  };
  const depthPass = (rt, layer) => {
    const prevT = renderer.getRenderTarget(), prevO = scene.overrideMaterial, prevL = camera.layers.mask;
    const cc = renderer.getClearColor(new THREE.Color()), ca = renderer.getClearAlpha();
    scene.overrideMaterial = depthMat;
    if (layer !== null) camera.layers.set(layer);
    renderer.setRenderTarget(rt); renderer.setClearColor(0, 0); renderer.clear(); renderer.render(scene, camera);
    renderer.setRenderTarget(prevT); renderer.setClearColor(cc, ca);
    scene.overrideMaterial = prevO; camera.layers.mask = prevL;
  };
  if (on('light')) { sun.intensity = KEY_I; ambient.intensity = AMBIENT_I; }
  return {
    parts,
    onModel(vrm) {
      const comps = components(vrm), log = [];
      if (on('face')) log.push(radialFace(comps));
      if (on('outline')) { outlines = setupOutlines(vrm); log.push(`outline ${outlines.length} materials`); }
      if (on('bangs')) {
        // faces / eyes = the head's non-hair components in front of the head (they get the hair shadow)
        const faceMats = new Set();
        let hairN = 0;
        for (const c of comps) {
          if (c.hair) { c.o.layers.enable(HAIR_LAYER); hairN++; continue; }
          let headN = 0; for (const i of c.used) if (c.owner.get(i)[0] === 'head') headN++;
          if (headN > c.used.length / 2) for (const m of [c.o.material].flat()) faceMats.add(m);
        }
        for (const m of faceMats) if (m?.isMToonMaterial && !m.isOutline) patchMaterial(m, U, true);
        head = vrm.humanoid.getRawBoneNode('head');
        log.push(`bangs on ${hairN} hair meshes, ${faceMats.size} face materials`);
      }
      return log.join('; ');
    },
    update(vrm) {
      if (outlines.length) {
        // screenCoordinates: the offset is outlineWidthFactor x half the view height at that depth = that fraction of
        // half the canvas height
        const f = 2 * OUTLINE_PX / Math.max(1, renderer.domElement.clientHeight);
        for (const m of outlines) m.outlineWidthFactor = f;
      }
      if (on('bangs')) {
        renderer.getDrawingBufferSize(size);
        const w = Math.max(1, Math.round(size.x)), h = Math.max(1, Math.round(size.y));
        U.hsinRes.value.set(w, h);
        U.hsinNF.value.set(camera.near, camera.far);
        tmp.copy(KEY_DIR); U.hsinKeyScr.value.set(tmp.x, tmp.y).normalize();
        U.hsinShadeK.value = sun.intensity / Math.PI;
        if (on('bangs') && head) {
          rtHair = target(rtHair, w, h); depthPass(rtHair, HAIR_LAYER); U.hsinHairDepth.value = rtHair.depthTexture;
          // BANG_M at the head's distance, in drawing-buffer px, toward the light on the screen
          const dist = head.getWorldPosition(tmp).distanceTo(camera.position);
          const px = BANG_M / (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2))) * h;
          U.hsinBangOff.value.copy(U.hsinKeyScr.value).multiplyScalar(px);
          U.hsinBangTap.value = px * BANG_TAP_M / BANG_M;
        }
      }
      if (on('light')) {
        // a directional light shines from position toward target (origin)
        tmp.copy(KEY_DIR).applyQuaternion(camera.quaternion);
        sun.position.copy(tmp);
        sun.target.position.set(0, 0, 0);
        sun.target.updateMatrixWorld();
      }
    },
  };
}

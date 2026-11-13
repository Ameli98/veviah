// Phase 1 VRM player. Loads a VRM, plays a motion JSON, and exposes window.hsin so a headless
// browser (the batch renderer) can drive the same page.
//
// URL parameters: model, motion (paths under public/), t (seconds; pauses there),
//                 view=full|upper|front|side|close, ui=0 (hide controls and labels),
//                 skirt=physics (no draping of skirts when sitting, see skirt.js),
//                 face=rig (build every expression from morph targets, ignoring the presets: checks face_rig.js),
//                 colliders=model (only the model's own colliders; default adds the bigger hair ones and the body /
//                 floor ones of colliders.js),
//                 (the legs and the hips position come from the motion: standing or sitting on the floor)
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { MotionPlayer } from './player.js';
import { addBodyColliders, addExtraColliders } from './colliders.js';
import { completeExpressions } from './face_rig.js';
import { FaceController } from './face.js';
import { initBrows, updateBrows } from './brows.js';
import { fitBody, measureSkeleton, skinMesh } from './measure.js';
import { drapeSkirt, fixSkirtVertices, initSkirt, initSkirtVertices, legKey, setClothTrial } from './skirt.js';
import { addLimbPush, updateLimbPush } from './limb_push.js';
import { Jiggle, dampSprings, followVertical } from './jiggle.js';

const params = new URLSearchParams(location.search);
const $ = (id) => document.getElementById(id);

// ---------- scene ----------
const canvas = $('view');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
renderer.setPixelRatio(window.devicePixelRatio);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe9edf2);
const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
const controls = new OrbitControls(camera, canvas);
const sun = new THREE.DirectionalLight(0xffffff, Math.PI);
sun.position.set(1, 1.5, 2);
scene.add(sun, new THREE.AmbientLight(0xffffff, 0.6));
scene.add(new THREE.GridHelper(4, 8, 0xb0b8c4, 0xd0d6de));

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  renderer.setSize(w, h, false);
  if (crop) {                    // the canvas shows only this part of a square view (see setCrop)
    camera.aspect = 1;
    camera.setViewOffset(1000, 1000, crop[0] * 1000, crop[1] * 1000, crop[2] * 1000, crop[3] * 1000);
  } else {
    camera.aspect = w / h;
    camera.clearViewOffset();
  }
  camera.updateProjectionMatrix();
}
// Crop for the batch renderer: [x, y, w, h] as fractions of a square view, e.g. [0.125, 0, 0.75, 0.9]
// keeps the middle 3/4 in width and the top 9/10 in height. The camera framing stays that of the square
// view, so a crop is pixel for pixel the same as rendering the square and cutting it. null = no crop.
let crop = null;
function setCrop(c) { crop = c; resize(); }

// Spring bones (hair, sleeves) in renderFrames. Off = they stay in their rest shape relative to their
// parent bones, so a judge does not mark a motion down for hair passing through the shirt (2026-10-22).
let springs = true;
function setSprings(on) { springs = on; }
window.addEventListener('resize', resize);

// The camera sits on +z looking back at the character, who faces +z. So the character's right
// side (-x) is on the LEFT of the screen, as when you face a real person.
function frameCamera(view) {
  const head = { y: state.headY };          // head height at the start of the motion, so the camera does not follow it
  if (view === 'upper') {                    // close-up for watching
    controls.target.set(0, head.y - 0.22, 0);
    camera.position.set(0, head.y - 0.12, 2.3);
  } else if (view === 'front' || view === 'side') {
    // Judge views: the whole body from the floor to a raised hand (fingertips of a hand stretched
    // straight up may leave the frame; the author wanted the body larger). The frame is set per posture, from
    // where the legs are at frame 0 (postureFromMotion), never from the arms, so all candidates of one
    // posture are shown at the same scale. Only the top 9/10 of the square view are kept by the tall crop
    // (setCrop, TALL_CROP in vrm_render.py), and its middle 3/4 in width.
    // 'side' is from the character's left, so it faces screen-left.
    const box = state.legBox ?? { x0: -0.2, x1: 0.2, z0: -0.15, z1: 0.2 };
    const bottom = state.sitting ? -0.12 : -0.05, top = head.y + 0.4;
    let width, cx = 0, cz = 0;
    if (view === 'front') width = 2 * Math.max(Math.abs(box.x0) + 0.12, Math.abs(box.x1) + 0.12, 0.55);
    else {
      const z0 = Math.min(box.z0 - 0.12, -0.3), z1 = Math.max(box.z1 + 0.12, 0.4);
      width = z1 - z0; cz = (z0 + z1) / 2;
    }
    const h = Math.max((top - bottom) / 0.9, width / 0.75);
    // visible rows: [cy + h/2 - 0.9h, cy + h/2]; centre the body in them
    const cy = ((top - h / 2) + (bottom + 0.4 * h)) / 2;
    const dist = h / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    controls.target.set(cx, cy, cz);
    if (view === 'front') camera.position.set(cx, cy, dist);
    else camera.position.set(dist, cy, cz);
  } else if (view.startsWith('orbit:')) {
    // Inspection view of the lower body (2026-10-24, skirt / leg clipping): 'orbit:az:el:dist' = azimuth from
    // the front toward the character's left (deg), elevation (deg), distance (m), looking at the hips / lap.
    // An optional 4th number is the height looked at (m), e.g. hands over the head.
    const [az, el, d, ty0] = view.split(':').slice(1).map(Number);
    const ty = Number.isFinite(ty0) ? ty0 : state.sitting ? 0.22 : 0.75;
    const a = THREE.MathUtils.degToRad(az), e = THREE.MathUtils.degToRad(el);
    controls.target.set(0, ty, 0.05);
    camera.position.set(d * Math.cos(e) * Math.sin(a), ty + d * Math.sin(e), 0.05 + d * Math.cos(e) * Math.cos(a));
  } else if (view === 'close') {
    // Judge close-up: head and shoulders from the front, about 0.6 m tall, so a shrug or a small
    // nod is visible. Raised hands leave this frame; the front and side views cover them.
    controls.target.set(0, head.y - 0.08, 0);
    camera.position.set(0, head.y - 0.08, 1.16);
  } else {
    controls.target.set(0, head.y * 0.55, 0);
    camera.position.set(0, head.y * 0.6, 4.6);
  }
  controls.update();
  document.body.classList.toggle('sideview', view === 'side');   // left/right labels only fit front views
  $('viewsel').value = view;
}

// ---------- model + motion ----------
const state = { vrm: null, player: null, motion: null, time: 0, playing: false, loop: true, headY: 1.4, manual: false, motionSerial: 0 };
const loader = new GLTFLoader();
loader.register((parser) => new VRMLoaderPlugin(parser));

async function loadModel(url) {
  const gltf = await loader.loadAsync(url);
  const vrm = gltf.userData.vrm;
  if (!vrm) throw new Error(`${url} is not a VRM`);
  VRMUtils.rotateVRM0(vrm);                 // 0.x models face -z; turn them to +z. No-op for 1.0
  if (state.vrm) {
    scene.remove(state.vrm.scene);
    VRMUtils.deepDispose(state.vrm.scene);
  }
  if (params.get('colliders') !== 'model') {
    addExtraColliders(vrm, url);   // default since 2026-10-23 (the author)
    state.bodyColliders = addBodyColliders(vrm);   // legs / hands / floor for skirts and hair, any model (2026-10-24)
    if (params.get('limbpush') !== 'off') addLimbPush(vrm);   // clothes / flesh pressed by the hands, on the GPU (limb_push.js)
    if (params.get('clothtrial')) setClothTrial(JSON.parse(params.get('clothtrial')));   // experiments (skirt.js)
    if (params.get('skirt') !== 'physics') state.skirtChains = initSkirt(vrm);   // skirts sit with the body (skirt.js)
    if (params.get('skirt') !== 'physics') state.skirtVerts = initSkirtVertices(vrm);   // their vertices (skirt.js)
  }
  // chest sway (jiggle.js): off unless ?jiggle=weak|medium|strong (judge renders and data stay as they were)
  state.jiggle = params.get('jiggle') && params.get('jiggle') !== 'off' ? new Jiggle(vrm, params.get('jiggle')) : null;
  if (params.get('jiggle')) dampSprings(vrm);   // as the app does (stage.js), so ?jiggle renders preview what the app shows
  state.vfollow = params.get('jiggle') ? {} : null;   // the same: hair follows the body up / down
  state.faceRig = completeExpressions(vrm, { force: params.get('face') === 'rig' });
  if (params.get('brows') === 'over') state.brows = initBrows(vrm);     // only on request: judge renders unchanged   // missing faces (face_rig.js)
  // the face as the app shows it (2026-11-10, review renders): motion.emotion = the face word, motion.eyes = the eyes
  // track; no random blinks, so a render stays deterministic (judge renders carry neither: unchanged)
  state.face = new FaceController(vrm);
  state.face.blinks = false;
  vrm.scene.traverse((o) => { o.frustumCulled = false; });
  scene.add(vrm.scene);
  state.vrm = vrm;
  state.restHeadY = vrm.humanoid.getNormalizedBoneNode('head').getWorldPosition(new THREE.Vector3()).y;
  state.hipsY = vrm.humanoid.getNormalizedBoneNode('hips').position.y;
  postureFromMotion();
  state.player = new MotionPlayer(vrm);
  if (state.motion) state.player.setMotion(state.motion);
  frameCamera(params.get('view') || 'full');
  setTime(state.time);
  $('info').textContent = `VRM ${vrm.meta?.metaVersion ?? '?'} · ${vrm.meta?.name ?? vrm.meta?.title ?? url}`;
  return vrm.meta;
}

// motion: a URL or an already-parsed motion object
// The camera frames the posture at the start of the motion (standing, or sitting on the floor: the
// hips are far down) and then stays put, so the judge sees the motion against a fixed frame.
const LEG_BONES = ['hips', 'leftUpperLeg', 'leftLowerLeg', 'leftFoot', 'leftToes',
  'rightUpperLeg', 'rightLowerLeg', 'rightFoot', 'rightToes'];
function postureFromMotion() {
  const r0 = state.motion?.root?.[0];
  state.sitting = !!r0 && r0[1] < -0.3;
  state.headY = (state.restHeadY ?? 1.4) + (r0 ? r0[1] * (state.hipsY ?? 0.9) : 0);
  state.legBox = null;
  const vrm = state.vrm;
  if (!vrm || !state.player?.motion) return;
  state.player.setTime(0);                 // measure the legs and head in the first frame
  vrm.update(0);
  vrm.scene.updateMatrixWorld(true);
  const v = new THREE.Vector3(), box = { x0: 0, x1: 0, z0: 0, z1: 0 };
  for (const name of LEG_BONES) {
    const n = vrm.humanoid.getNormalizedBoneNode(name);
    if (!n) continue;
    n.getWorldPosition(v);
    box.x0 = Math.min(box.x0, v.x); box.x1 = Math.max(box.x1, v.x);
    box.z0 = Math.min(box.z0, v.z); box.z1 = Math.max(box.z1, v.z);
  }
  state.legBox = box;
  state.headY = vrm.humanoid.getNormalizedBoneNode('head').getWorldPosition(v).y;
  // skirts: one draped shape per sitting posture, keyed by where the legs are (2 cm grid); per frame when the legs
  // move while sitting (skirt.js)
  const fps = state.player.motion.fps ?? 30, frames = state.player.motion.rotations.length;
  const poseAt = (f) => { state.player.setTime(f / fps); vrm.humanoid.update(); };
  drapeSkirt(vrm, state.sitting ? legKey(vrm) : 'stand', { poseAt, frames, id: ++state.motionSerial });
  state.player.setTime(0); vrm.update(0); vrm.scene.updateMatrixWorld(true);
}

async function loadMotion(motion) {
  if (typeof motion === 'string') motion = await (await fetch(motion)).json();
  state.motion = motion;
  state.player?.setMotion(motion);
  state.face?.setEmotion(motion.emotion ?? 'neutral');
  postureFromMotion();
  $('seek').max = state.player?.duration ?? 0;
  setTime(0);
  return { frames: motion.rotations.length, fps: motion.fps };
}

// Pose at time t and draw one frame. dt = 0 keeps spring bones (hair) from advancing, so the same
// t gives the same body pose every time.
function setTime(t, dt = 0) {
  state.time = t;
  if (!state.vrm) return;
  state.player.setTime(t);
  if (state.vfollow && dt > 0) followVertical(state.vrm, state.vfollow);
  if (state.face) { state.face.setEyes(state.player.eyesClosed); state.face.update(dt || 1 / 30); }
  state.vrm.update(dt);
  state.jiggle?.update(dt);
  fixSkirtVertices(state.vrm, t * (state.motion?.fps ?? 30));
  updateLimbPush(state.vrm);
  if (state.brows) updateBrows(state.vrm, camera);
  renderer.render(scene, camera);
  $('seek').value = t;
  $('time').textContent = `${t.toFixed(2)} / ${state.player.duration.toFixed(2)} s`;
}

const timer = new THREE.Timer();
function tick() {
  requestAnimationFrame(tick);
  timer.update();
  const dt = timer.getDelta();
  if (!state.vrm || state.manual) return;
  if (state.playing && state.motion) {
    let t = state.time + dt;
    const d = state.player.duration;
    if (t > d) {
      if (state.loop) t = d > 0 ? t % d : 0;
      else { t = d; setPlaying(false); }
    }
    setTime(t, dt);
  } else {
    state.vrm.update(dt);                   // hair keeps settling while paused
    state.jiggle?.update(dt);
    fixSkirtVertices(state.vrm, state.time * (state.motion?.fps ?? 30));
    updateLimbPush(state.vrm);
    if (state.brows) updateBrows(state.vrm, camera);
  renderer.render(scene, camera);
  }
}

// Batch rendering for the judge. Plays the motion from frame 0 at a fixed time step, so hair and
// sleeves move as they do in playback, and captures the requested frame indices from each view.
// Returns [index][view] PNG data URLs. Same input -> same poses (no wall-clock time involved).
function renderFrames(indices, views = ['front', 'side'], settleSteps = 30) {
  state.manual = true;                      // stop the real-time loop from advancing physics
  setPlaying(false);
  const dt = 1 / state.motion.fps;
  const want = new Map();
  indices.forEach((f, k) => { if (!want.has(f)) want.set(f, []); want.get(f).push(k); });
  const out = new Array(indices.length);
  state.player.setTime(0);
  state.vrm.update(0);
  state.vrm.springBoneManager?.reset();
  if (state.vfollow) state.vfollow = {};
  state.jiggle?.setLevel(state.jiggle.level);   // masses at rest
  for (let i = 0; i < settleSteps; i++) {                      // let hair and sleeves hang first; the face eases in
    if (state.face) { state.face.setEyes(state.player.eyesClosed); state.face.update(dt); }
    state.vrm.update(dt); state.jiggle?.update(dt);
  }
  const last = Math.max(...indices);
  for (let f = 0; f <= last; f++) {
    state.player.setTime(f * dt);
    if (state.vfollow && springs) followVertical(state.vrm, state.vfollow);
    if (state.face) { state.face.setEyes(state.player.eyesClosed); state.face.update(dt); }
    state.vrm.update(springs ? dt : 0);
    if (!springs) state.vrm.springBoneManager?.reset();   // after update: dt = 0 still carries inertia
    state.jiggle?.update(dt);
    if (!want.has(f)) continue;
    fixSkirtVertices(state.vrm, f);         // skirt vertices out of the legs (skirt.js), only for frames drawn
    updateLimbPush(state.vrm);
    const shots = views.map((v) => {
      frameCamera(v);
      if (state.brows) updateBrows(state.vrm, camera);
  renderer.render(scene, camera);
      return canvas.toDataURL('image/png');
    });
    want.get(f).forEach((k) => { out[k] = shots; });
  }
  return out;
}

function setPlaying(on) {
  state.playing = on;
  $('play').textContent = on ? 'Pause' : 'Play';
}

// ---------- controls ----------
$('play').onclick = () => setPlaying(!state.playing);
$('seek').oninput = (e) => { setPlaying(false); setTime(parseFloat(e.target.value)); };
$('loop').onchange = (e) => { state.loop = e.target.checked; };
$('motion').onchange = (e) => loadMotion(e.target.value).then(() => setPlaying(true));
$('viewsel').onchange = (e) => frameCamera(e.target.value);
$('modelfile').onchange = (e) => {
  const f = e.target.files[0];
  if (f) loadModel(URL.createObjectURL(f));
};
$('motionfile').onchange = async (e) => {
  const f = e.target.files[0];
  if (f) { await loadMotion(JSON.parse(await f.text())); setPlaying(true); }
};
if (params.get('ui') === '0') document.body.classList.add('bare');

// ---------- start ----------
resize();
const ready = (async () => {
  await loadModel(params.get('model') || 'models/VRM1_Constraint_Twist_Sample.vrm');
  const m = params.get('motion') || $('motion').value;
  // motion=none: no sample motion (the Windows app's cloth window: the samples in public/ are not bundled)
  if (m !== 'none') { $('motion').value = m; await loadMotion(m); }
  if (params.has('t')) setTime(parseFloat(params.get('t')));
  else setPlaying(true);
  tick();
})();
ready.catch((err) => { $('info').textContent = `Error: ${err.message}`; console.error(err); });

// For the headless batch renderer: await hsin.ready, then loadModel / loadMotion / setTime.
// The skirt's cloth at some frames of the loaded motion (2026-10-29, the author: the legs first, then the cloth, then the
// hands): world positions of the cloth vertices after the drape (skirt.js), at most `most` per frame. The solver
// puts hands resting on a covered leg on this surface (motion_loop/refine.py).
function clothSurface(indices, most = 1500) {
  const vrm = state.vrm, fps = state.motion?.fps ?? 30, meshes = vrm.scene.userData.hsinSkirtVerts ?? [], v = new THREE.Vector3();
  return indices.map((f) => {
    state.player.setTime(f / fps); vrm.update(0); vrm.scene.updateMatrixWorld(true);
    fixSkirtVertices(vrm, f);
    const all = [];
    for (const m of meshes) for (const i of m.idx) { m.o.getVertexPosition(i, v); v.applyMatrix4(m.o.matrixWorld); all.push([v.x, v.y, v.z]); }
    const step = Math.max(1, Math.ceil(all.length / most));
    return all.filter((_, k) => k % step === 0).map((p) => p.map((x) => Math.round(x * 10000) / 10000));
  });
}

// debugging the cloth: the skin component's vertices (posed) at these frames, every k-th
function bodySurface(indices, most = 6000) {
  const vrm = state.vrm, fps = state.motion?.fps ?? 30, skin = skinMesh(vrm), v = new THREE.Vector3();
  const ids = [...new Set(skin.geometry.index.array)], step = Math.max(1, Math.ceil(ids.length / most));
  return indices.map((f) => {
    state.player.setTime(f / fps); vrm.update(0); vrm.scene.updateMatrixWorld(true);
    return ids.filter((_, k) => k % step === 0).map((i) => { skin.getVertexPosition(i, v); v.applyMatrix4(skin.matrixWorld); return [v.x, v.y, v.z].map((x) => Math.round(x * 10000) / 10000); });
  });
}

// world positions of humanoid bones at these frames (2026-11-03: measuring where the fingers really are, the Python side
// has no finger joints); missing bones -> null
function bonePoints(indices, names) {
  const vrm = state.vrm, fps = state.motion?.fps ?? 30, v = new THREE.Vector3();
  return indices.map((f) => {
    state.player.setTime(f / fps); vrm.update(0); vrm.scene.updateMatrixWorld(true);
    return Object.fromEntries(names.map((n) => {
      const b = vrm.humanoid.getNormalizedBoneNode(n);
      return [n, b ? b.getWorldPosition(v).toArray().map((x) => Math.round(x * 10000) / 10000) : null];
    }));
  });
}

window.hsin = { ready, clothSurface, bodySurface, bonePoints, measureSkeleton: () => measureSkeleton(state.vrm), fitBody: () => fitBody(state.vrm), loadModel, loadMotion, setTime, frameCamera, renderFrames, setCrop, setSprings, state, camera, controls };

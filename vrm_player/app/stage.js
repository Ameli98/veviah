// The avatar stage shared by the MCP Apps view (avatar.js) and the desktop companion window (companion.js):
// scene, camera framing, loading a VRM from bytes (colliders, limb push, skirt, face rig; measuring a new model),
// playing a solved motion once with its emotion, then holding the last pose with hair still settling.
// Where the model, the motions and the measuring go is up to the caller (MCP resources / the companion's HTTP).
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';
import { MotionPlayer } from '../src/player.js';
import { addBodyColliders, addExtraColliders, setFloorY, setWall as setWallPlane } from '../src/colliders.js';
import { FaceController } from '../src/face.js';
import { completeExpressions } from '../src/face_rig.js';
import { fitBody, measureSkeleton } from '../src/measure.js';
import { drapeSkirt, fixSkirtVertices, initSkirt, initSkirtVertices, legKey, limitSkirtSwing, setLedge } from '../src/skirt.js';
import { addLimbPush, updateLimbPush } from '../src/limb_push.js';
import { initBrows, updateBrows } from '../src/brows.js';
import { createLook, lookFromQuery } from '../src/look.js';
const BROWS = new URLSearchParams(location.search).get('brows') !== 'hair';      // ?brows=hair = under the bangs as modelled
import { Jiggle, dampSprings, springTailsShift, followVertical } from '../src/jiggle.js';

// opts: background (hex, or null = transparent), headroom (m above the head in view), onEnd(wasIdle) when a motion
// has played to its end
// pickable: keep the drawn frame so alphaAt() can read it (the desktop window's click-through)
// footroom: m of floor shown under the feet (the MCP view keeps room for its subtitle there)
// jiggle: chest sway level (src/jiggle.js) 'off' | 'weak' | 'medium' | 'strong'
// windowInertia: the window moving on the screen (dragged) moves the body for the physics (hair, skirt, chest lag
// behind), while the picture stays where it is in the window (the desktop app)
// tests: ?timescale=0.1 slows everything down so screenshots catch transitions frame by frame
const TIME_SCALE = Number(new URLSearchParams(location.search).get('timescale')) || 1;
export function createStage(canvas, { background = 0xe9edf2, headroom = 0.3, footroom = null, onEnd = () => {}, pickable = false,
                                      jiggle = 'weak', windowInertia = false } = {}) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: background === null,
                                             preserveDrawingBuffer: pickable });
  renderer.setPixelRatio(window.devicePixelRatio);
  if (background === null) renderer.setClearColor(0x000000, 0);
  const scene = new THREE.Scene();
  if (background !== null) scene.background = new THREE.Color(background);
  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  const sun = new THREE.DirectionalLight(0xffffff, Math.PI);
  sun.position.set(1, 1.5, 2);
  const ambient = new THREE.AmbientLight(0xffffff, 0.6);
  scene.add(sun, ambient);
  // rendering look switches (src/look.js, 2026-11-12): ?look=light,face,... / all
  const look = createLook(lookFromQuery(new URLSearchParams(location.search)), { scene, camera, sun, ambient, renderer });

  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    // draw at once: resizing clears the drawing buffer, and waiting for the next frame showed an empty window while
    // the desktop app's window was being scaled (it flickered, 2026-11-04)
    if (state?.vrm) renderer.render(scene, camera);
  }
  new ResizeObserver(resize).observe(canvas);

  // Whole body, floor to above the head, at the posture of the motion's first frame. scale < 1 = the body smaller in
  // the view with its feet staying at the bottom (the desktop app's wheel zoom: the window keeps its size).
  // Above 1 the headroom shrinks, up to the top of the head (+ HEAD_TOP) at the top of the view (2026-11-04, the author:
  // bigger than "whole body + room for raised hands").
  // headTop = how far the top of the model (hair incl.) is above the head bone, measured at load; the head bone sits
  // at the base of the skull (12 cm cut the top of the head off, 2026-11-04)
  let legDrop = 0;    // m the legs hang below the floor line (a ledge), see setMotion
  let headTop = 0.3, footPts = [], scale = 1, lastFrame = null, viewH = 2, standHead = null, camShift = 0;
  const maxScale = (headY, sitting) => { const b = -(footroom ?? (sitting ? 0.35 : 0.3)) - footDrop() - legDrop; return (headY + headroom - b) / (headY + headTop + 0.03 - b); };
  // The view for a frame posture: { cy, h } = camera height and view height (m). rise = how far the head goes above
  // its rest height in a jump (2026-11-07, the author Q9 (a)): up to maxScale the view zooms out just enough for it, about
  // the floor, so the floor stays on the same pixel row (she lands on the taskbar again); past maxScale (legs already
  // out of the bottom) the camera rises instead.
  // the floor under the shoe tips (nearer the camera) shows lower than the floor under the hips: about tan(fov / 2)
  // x how much nearer (the camera looks level from about the middle of the view); so the view keeps that much more
  // room under the floor, for the farthest shoe point in any facing (2026-11-07: the shoe tips were cut by the
  // window's bottom and went behind the taskbar)
  const footDrop = () => Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) * footPts.reduce((a, f) => Math.max(a, f.length()), 0);
  function viewFor(headY, sitting, rise = 0, sc = scale) {
    const bottom = -(footroom ?? (sitting ? 0.35 : 0.3)) - footDrop() - legDrop, top = headY + headroom;   // room for the subtitle under the feet
    // up to maxScale the feet stay at the bottom; past it the top of the head stays at the top and the legs go out of
    // the bottom of the view (2026-11-04, the author: no limit on zooming in)
    const m = maxScale(headY, sitting), h0 = (top - bottom) * 1.05 / sc;
    if (sc > m) return { cy: bottom + (top - bottom) * 1.05 / m - h0 / 2 + rise, h: h0 };
    const fr = -bottom, need = headY + headTop + 0.03 + rise;
    const h = Math.max(h0, need / (1 - fr / h0)), b = -fr * h / h0;       // floor keeps its pixel row: fr / h0 of the view
    return { cy: b + h / 2, h };
  }
  function placeCamera(cy, h) {
    viewH = h;
    const dist = h / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    camera.position.set(0, cy, dist);
    camera.lookAt(0, cy, 0);
  }
  function frame(headY, sitting) {
    lastFrame = [headY, sitting];
    const v = viewFor(headY, sitting);
    placeCamera(v.cy, v.h);
  }
  // every frame: the jump zoom (riseNow) and, while held, the standing whole-body view (she is lifted out of any
  // posture; zoomed in past the whole body, the swinging legs left the window at the sides) plus a camera that rises
  // with her as far as her head needs (2026-11-07, the author Q8: off the floor the taskbar does not matter)
  // WHOLE: the whole-body view for a held body / a jump when zoomed in, a little smaller than the full-body zoom (at
  // the full-body zoom the arms, out to the sides, touched the window's sides in a 10 deg swing; view.mjs)
  const WHOLE = 0.9;
  let viewW = 0, jumpW = 0, jumpLift = 0;
  const JUMP_VIEW_AFTER = 1.0;
  const HELD_VIEW_AFTER = 1.0;
  const HELD_PTS = [['head', 0], ['leftHand', 0.12], ['rightHand', 0.12], ['leftFoot', 0.06], ['rightFoot', 0.06],
                    ['leftToes', 0.05], ['rightToes', 0.05], ['leftLowerLeg', 0.08], ['rightLowerLeg', 0.08]];
  function heldFit(v) {
    let top = -Infinity, bot = Infinity, side = 0;
    const p = new THREE.Vector3();
    for (const [n, r] of HELD_PTS) {
      const b = state.vrm.humanoid.getNormalizedBoneNode(n);
      if (!b) continue;
      b.getWorldPosition(p);
      const up = n === 'head' ? headTop + 0.03 : r;
      top = Math.max(top, p.y + up); bot = Math.min(bot, p.y - r); side = Math.max(side, Math.abs(p.x) + r);
    }
    const aspect = camera.aspect || 0.62;
    const h = Math.max(v.h, (top - bot) * 1.04, 2 * side * 1.04 / aspect);
    let cy = v.cy;
    if (cy + h / 2 < top) cy = top - h / 2;
    if (cy - h / 2 > bot) cy = bot + h / 2;
    return { cy, h };
  }
  // One size (2026-11-11, the author: she grew and shrank while dragged, sitting on a ledge, jumping): the view height is
  // always the STANDING whole-body view at the wheel's zoom (baseView); every other case only moves the camera:
  // a ledge lowers it by the legs' drop (the floor line, = the window's edge, rises in the window), a jump raises it
  // with the head, a lifted body is kept in the window by moving, never by zooming out; what still does not fit (zoomed
  // in a lot) is cut by the window's edge.
  function baseView() {
    const saved = legDrop; legDrop = 0;
    const v = viewFor(standHead ?? lastFrame[0], false, 0);
    legDrop = saved;
    return v;
  }
  function heldRange() {               // top / bottom of the lifted body (bones of the last frame, with margins)
    let top = -Infinity, bot = Infinity;
    const p = new THREE.Vector3();
    for (const [n, r] of HELD_PTS) {
      const b = state.vrm.humanoid.getNormalizedBoneNode(n);
      if (!b) continue;
      b.getWorldPosition(p);
      top = Math.max(top, p.y + (n === 'head' ? headTop + 0.03 : r)); bot = Math.min(bot, p.y - r);
    }
    return { top, bot };
  }
  function updateView(riseNow, dt) {
    if (!lastFrame) return;
    const b = baseView();
    let cy = b.cy - legDrop;
    // a jump: the camera rises only as far as the head would leave the window (with the spare room above her it does
    // not move at all, so she visibly leaves the floor); eased
    if (riseNow > 0) {
      const top = headY() + headTop + 0.03, room = cy + b.h / 2;
      jumpLift += (Math.max(0, top - room) - jumpLift) * Math.min(1, dt / 0.08);
    } else jumpLift += (0 - jumpLift) * Math.min(1, dt / 0.25);
    cy += jumpLift;
    if (held.on) held.offAt = clock;
    const want = held.on || Math.abs(held.ang) > 0.03 || clock - (held.offAt ?? -1e9) < HELD_VIEW_AFTER ? 1 : 0;
    viewW += Math.sign(want - viewW) * Math.min(Math.abs(want - viewW), dt / (want ? 0.12 : 0.3));
    const e = smooth(viewW); held.viewW = viewW;
    camShift = 0;
    if (e > 0) {
      // lifted: the camera moves (eased) so the body stays inside: bottom first, the head wins if both cannot
      const { top, bot } = heldRange();
      let c = cy;
      if (c - b.h / 2 > bot) c = bot + b.h / 2;
      if (c + b.h / 2 < top) c = top - b.h / 2;
      held.fitCy = (held.fitCy ?? c) + (c - (held.fitCy ?? c)) * 0.5;
      cy = cy + (held.fitCy - cy) * e;
    } else { held.fitCy = undefined; }
    placeCamera(cy, b.h);
  }
  // per motion frame: how far the head rises past the view's spare room (a jump), held a little before and after the
  // rise and eased in / out (RISE_PRE / RISE_POST / RISE_RAMP s), so the zoom is calm
  const RISE_PRE = 0.35, RISE_POST = 0.2, RISE_RAMP = 0.3, RISE_FREE = 0.03, RISE_HAIR = 0.06;
  function riseEnvelope(motion, headAt, head0) {
    const fps = motion.fps ?? 30, T = motion.rotations.length, out = new Float32Array(T);
    if (!motion.root || !motion.root.some((r) => r[1] > RISE_FREE / 2)) return out;     // nothing rises: skip the sampling
    const rise = [];
    // + RISE_HAIR once it rises: long hair flies up above the head on landing (test1's tuck jump, view.mjs)
    for (let f = 0; f < T; f++) { const r = Math.max(0, headAt(f) - head0 - RISE_FREE); rise.push(r + RISE_HAIR * Math.min(1, r / 0.02)); }
    for (let k = 0; k < T; k++) {
      if (rise[k] <= 0) continue;
      const lo = Math.max(0, Math.floor(k - (RISE_PRE + RISE_RAMP) * fps)), hi = Math.min(T - 1, Math.ceil(k + (RISE_POST + RISE_RAMP) * fps));
      for (let f = lo; f <= hi; f++) {
        const d = f < k ? (k - f) / fps - RISE_PRE : (f - k) / fps - RISE_POST;
        const w = d <= 0 ? 1 : 1 - smooth(Math.min(1, d / RISE_RAMP));
        out[f] = Math.max(out[f], rise[k] * w);
      }
    }
    return out;
  }

  const state = { vrm: null, player: null, face: null, emotion: 'neutral', t: 0, playing: false, motionSerial: 0,
                  modelName: null, measured: false };
  const headY = () => state.vrm.humanoid.getNormalizedBoneNode('head').getWorldPosition(new THREE.Vector3()).y;

  // bytes: ArrayBuffer of the .vrm; measured: the solver already has this model's skeleton / body fit;
  // saveFit({skeleton, fit}) -> Promise<bool> stores a new model's measurements (called only when !measured)
  async function loadModel(bytes, name, { measured, saveFit, status = () => {} }) {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.parseAsync(bytes, '');
    const vrm = gltf.userData.vrm;
    VRMUtils.rotateVRM0(vrm);
    addExtraColliders(vrm, `${name}.vrm`);
    addBodyColliders(vrm);
    addLimbPush(vrm);
    initSkirt(vrm);
    initSkirtVertices(vrm);
    state.jiggle = new Jiggle(vrm, jiggleLevel);            // chest sway (jiggle.js), replaces the model's own chest springs
    dampSprings(vrm);                                        // hair / cloth settle without bouncing (jiggle.js)
    vrm.scene.traverse((o) => { o.frustumCulled = false; });
    baseYaw = vrm.scene.rotation.y;
    vrm.scene.rotation.y = baseYaw + yaw * Math.PI / 180;
    scene.add(vrm.scene);
    state.vrm = vrm;
    state.modelName = name;
    state.measured = measured;
    if (!measured) {         // a new model: measure it once for the solver (src/measure.js)
      status('measuring the model…');
      state.measured = await saveFit({ skeleton: measureSkeleton(vrm), fit: fitBody(vrm) });
    }
    state.player = new MotionPlayer(vrm);
    held.restHips = vrm.humanoid.getNormalizedBoneNode('hips').position.clone();   // T-pose hips: standing, legs straight
    heldQ = null;
    console.info('face rig', completeExpressions(vrm));
    if (BROWS) console.info('brows over hair', initBrows(vrm));   // brows.js (2026-11-11)
    console.info('look', [...look.parts].join(',') || 'off', look.onModel(vrm) ?? '');      // hand-made expressions the model lacks (face_rig.js)
    state.face = new FaceController(vrm);
    vrm.scene.updateMatrixWorld(true);
    const top = new THREE.Box3().setFromObject(vrm.scene, true).max.y;     // rest pose, hair incl.
    if (Number.isFinite(top)) headTop = Math.min(Math.max(top - headY(), 0.1), 0.6);
    standHead = headY();                                    // T-pose = standing: the view while held
    // the soles on the floor (rest mesh vertices near the floor; model space, relative to the hips): the snap row is the
    // floor point nearest the camera, the lowest row of her on the screen (the hips' floor row was ~10 px higher and
    // the shoe tips went behind the taskbar, the author 2026-11-07)
    footPts = [];                // every vertex within FOOT_Y of the floor at rest (the soles), subsampled
    { const v = new THREE.Vector3(), all = [];
      vrm.scene.updateMatrixWorld(true);
      vrm.scene.traverse((o) => {
        if (!o.isSkinnedMesh) return;
        const n = o.geometry.attributes.position.count;
        for (let i = 0; i < n; i++) {
          o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
          if (v.y < 0.04) all.push(vrm.scene.worldToLocal(v.clone()));
        }
      });
      const step = Math.max(1, Math.floor(all.length / 400));
      for (let k = 0; k < all.length; k += step) footPts.push(new THREE.Vector2(all[k].x - held.restHips.x, all[k].z - held.restHips.z));
    }
    frame(headY(), false);
  }

  // ---- blending and the idle layer (2026-11-04, spec in CLAUDE.md) ----
  // A new motion starts from wherever the body is (BLEND_S). While no main motion plays (idle actions count as
  // idle), a continuous layer breathes and lets the head drift; it fades out while a main motion plays.
  const BLEND_S = 0.3, IDLE_FADE_S = 0.4;
  const BREATH_DEG = 2.5, BREATH_HZ = 0.25, DRIFT_DEG = 3;
  let blendFrom = null, blendT = 0, blendS = 0.3, idleW = 0, clock = 0;
  const node = (n) => state.vrm.humanoid.getNormalizedBoneNode(n);
  function snapshot() {
    const out = {};
    for (const n of Object.keys(state.vrm.humanoid.humanBones)) {
      const b = node(n);
      if (b) out[n] = { q: b.quaternion.clone(), p: n === 'hips' ? b.position.clone() : null };
    }
    return out;
  }
  const smooth = (x) => x * x * (3 - 2 * x);
  const qa = new THREE.Quaternion(), ea = new THREE.Euler();

  // ---- picked up by a window drag (2026-11-06, grilling with the author: Q1 (b), Q2 (a), Q3) ----
  // While the user drags the window the body takes one "lifted" pose whatever it was doing (absolute rotations from
  // the T-pose, so standing and sitting alike): feet off the floor (hips HELD_LIFT up from the standing rest), legs
  // hanging with the knees a little bent and the toes down, arms a little out, a surprised face. It hangs from a point
  // just above the head and swings there like a pendulum (the author's reference: Yixuan carried by a bird swings most
  // when the direction changes): angle'' = -(g/L) sin a - (window acceleration / L) cos a - damping, the window's
  // speed changes applied as impulses (HANG_L, HANG_ZETA, HANG_GAIN, at most HANG_MAX_DEG); the legs trail a
  // little more (HELD_LEG_LAG). Amplitude halved 2026-11-07 (the author: swings too much; was gain 0.5, max 25). HANG_ZETA 0.12 swung on for seconds; the author (2026-11-06, the Yixuan clip): damped
  // much more, about one swing back and it has settled. The first version tilted about the hips toward the window's speed: the feet flew out
  // and it pivoted at the waist. On release the pose blends back over HELD_OUT_S with a landing dip (LAND_DIP m,
  // LAND_S): she lands where she was dropped (falling to something to stand on comes with window snapping).
  const HELD_IN_S = 0.18, HELD_OUT_S = 0.22, HELD_LIFT = 0.10;
  const HANG_L = 0.9, HANG_ZETA = 0.4, HANG_MAX_DEG = 12, HANG_GAIN = 0.25, HELD_LEG_LAG = 0.25, G = 9.81;
  const LAND_DIP = 0.06, LAND_S = 0.4;
  const held = { on: false, w: 0, ang: 0, vel: 0, winV: 0, smoothV: 0, lastV: 0, land: -1, restHips: null };
  const d2r = Math.PI / 180;
  // [bone, euler x, y, z in degrees] from the T-pose, for a +z-facing VRM 1.0 rig (0.x: x / z negated below)
  const HELD_POSE = [
    ['leftUpperLeg', -14, 0, 3], ['rightUpperLeg', -26, 0, -3], ['leftLowerLeg', 40, 0, 0], ['rightLowerLeg', 52, 0, 0],
    ['leftFoot', 35, 0, 0], ['rightFoot', 35, 0, 0],
    ['leftUpperArm', 0, -12, -48], ['rightUpperArm', 0, 12, 48], ['leftLowerArm', 0, -28, 0], ['rightLowerArm', 0, 28, 0],
    ['spine', -3, 0, 0], ['chest', -3, 0, 0], ['head', 6, 0, 0],
  ];
  let heldQ = null;
  function heldTargets() {
    const f = state.vrm.meta?.metaVersion === '0' ? -1 : 1;
    heldQ = HELD_POSE.filter(([n]) => node(n)).map(([n, x, y, z]) => {
      const q = new THREE.Quaternion().setFromEuler(new THREE.Euler(x * d2r, y * d2r, z * d2r));
      q.x *= f; q.z *= f;
      return [n, q];
    });
  }
  function setHeld(on) {
    if (!state.vrm || held.on === on) return;
    held.on = on;
    if (on) { state.face?.setEmotion('surprised', 0.8); held.land = -1; }
    else { held.land = 0; state.face?.setEmotion(state.playing && !state.idleMotion ? state.emotion : 'neutral'); }
  }
  function applyHeld(dt) {
    const target = held.on ? 1 : 0;
    held.w += Math.sign(target - held.w) * Math.min(Math.abs(target - held.w), dt / (held.on ? HELD_IN_S : HELD_OUT_S));
    // the window's sideways speed, smoothed (pixel steps are noisy) and fading when no movement comes; a change of it
    // is an impulse on the pendulum (exact for a pivot that changes speed: angle' -= dv cos(a) / L), times HANG_GAIN
    held.smoothV += (held.winV - held.smoothV) * Math.min(1, dt / 0.05);
    held.winV *= Math.exp(-dt / 0.08);
    const dv = held.on ? held.smoothV - held.lastV : 0;
    held.lastV = held.smoothV;
    const w0 = Math.sqrt(G / HANG_L);
    held.vel -= HANG_GAIN * dv * Math.cos(held.ang) / HANG_L;
    held.vel += (-w0 * w0 * Math.sin(held.ang) - 2 * HANG_ZETA * w0 * held.vel) * dt;
    held.ang = THREE.MathUtils.clamp(held.ang + held.vel * dt, -HANG_MAX_DEG * d2r, HANG_MAX_DEG * d2r);
    if (!held.on && held.w <= 0) { held.ang *= Math.exp(-dt / 0.15); held.vel = 0; }
    let dip = 0;
    if (held.land >= 0) { held.land += dt; const u = held.land / LAND_S; dip = u < 1 ? Math.sin(Math.PI * u) : 0; if (u >= 1) held.land = -1; }
    if (held.w <= 0 && !dip) return;
    if (!heldQ) heldTargets();
    const e = smooth(held.w), f = state.vrm.meta?.metaVersion === '0' ? -1 : 1;
    for (const [n, q] of heldQ) node(n).quaternion.slerp(q, e);
    const hips = node('hips');
    if (held.restHips) hips.position.lerp(new THREE.Vector3(held.restHips.x, held.restHips.y + HELD_LIFT, held.restHips.z), e);
    if (Math.abs(held.ang) > 1e-4 && e > 0) {
      // swing about a point just above the head, in the hips' parent space, about the axis toward the camera
      const parent = hips.parent;
      parent.updateMatrixWorld(true);
      const inv = new THREE.Matrix4().copy(parent.matrixWorld).invert();
      const top = node('head').getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, headTop + 0.05, 0));
      // the head's world position is from the last frame; good enough for a pivot
      const pivot = top.applyMatrix4(inv);
      const axis = new THREE.Vector3(0, 0, 1).transformDirection(inv);
      const R = qa.setFromAxisAngle(axis, held.ang * e);
      hips.position.sub(pivot).applyQuaternion(R).add(pivot);
      hips.quaternion.premultiply(R);
      for (const n of ['leftUpperLeg', 'rightUpperLeg'])            // the legs trail the swing a little
        node(n)?.quaternion.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, f), -HELD_LEG_LAG * held.vel * 0.1 * e));
    }
    if (dip) {                                          // landing: the knees give and the hips drop, then straighten
      hips.position.y -= LAND_DIP * dip;
      for (const [n, a2] of [['leftUpperLeg', -22], ['rightUpperLeg', -22], ['leftLowerLeg', 42], ['rightLowerLeg', 42], ['leftFoot', -18], ['rightFoot', -18]])
        node(n)?.quaternion.multiply(qa.setFromEuler(ea.set(a2 * d2r * dip * f, 0, 0)));
    }
  }

  function applyBlendAndIdle(dt) {
    if (blendFrom) {
      const w = smooth(Math.min(blendT / blendS, 1));
      for (const [n, f] of Object.entries(blendFrom)) {
        const b = node(n);
        b.quaternion.slerpQuaternions(f.q, b.quaternion.clone(), w);
        if (f.p) b.position.lerpVectors(f.p, b.position.clone(), w);
      }
      blendT += dt;
      if (blendT >= blendS) blendFrom = null;
    }
    const target = state.playing && !state.idleMotion ? 0 : 1;
    idleW += Math.sign(target - idleW) * Math.min(Math.abs(target - idleW), dt / IDLE_FADE_S);
    clock += dt;
    if (idleW <= 0) return;
    const d2r = Math.PI / 180, t = clock;
    const breath = Math.sin(2 * Math.PI * BREATH_HZ * t) * BREATH_DEG * d2r * idleW;
    const chest = node('upperChest') ?? node('chest');
    if (chest) chest.quaternion.multiply(qa.setFromEuler(ea.set(breath, 0, 0)));
    // two slow sines per axis, never in step
    const yaw = (Math.sin(2 * Math.PI * 0.07 * t) + 0.5 * Math.sin(2 * Math.PI * 0.17 * t + 1.3)) / 1.5;
    const pitch = (Math.sin(2 * Math.PI * 0.05 * t + 2.1) + 0.5 * Math.sin(2 * Math.PI * 0.13 * t + 0.4)) / 1.5;
    const roll = Math.sin(2 * Math.PI * 0.04 * t + 4.0);
    const k = DRIFT_DEG * d2r * idleW;
    for (const [n, share] of [['neck', 0.35], ['head', 0.65]]) {
      const b = node(n);
      if (b) b.quaternion.multiply(qa.setFromEuler(ea.set(pitch * k * share, yaw * k * share, roll * k * 0.5 * share)));
    }
  }

  // opts.idle: an idle action (the idle layer stays on); opts.blend false = jump (e.g. behind a fade)
  function setMotion(motion, play = true, emotion = state.emotion, opts = {}) {
    const { vrm, player } = state;
    state.emotion = emotion;
    blendFrom = player.motion && opts.blend !== false ? snapshot() : null;
    blendS = opts.blendS ?? BLEND_S;          // a longer blend = a posture transition step (companion.js, 2026-11-11)
    if (opts.blend === false) state.jiggle?.setLevel(jiggleLevel);     // a jump (behind a fade): the chest starts at rest
    blendT = 0;
    state.idleMotion = !!opts.idle;
    player.setMotion(motion);
    player.setTime(0);
    // the new motion's pose is only MEASURED here (legs, head height, skirt key); the hair / cloth springs must not see
    // it: vrm.update(0) ran the springs with the bones jumped to the new pose, and the blend then pulled them back, so
    // every posture change flung the hair out (2026-11-11). humanoid.update() moves the bones without the springs.
    const settle = () => { vrm.humanoid.update(); vrm.scene.updateMatrixWorld(true); };
    settle();
    const sitting = !!motion.root && motion.root[0][1] < -0.3;
    // on a ledge the legs hang below the floor line (the window's top edge): the view keeps room for them and the
    // skirt's floor ends at the knees (2026-11-11)
    legDrop = 0;
    let kneeZ = null;
    for (const n of ['leftFoot', 'rightFoot', 'leftToes', 'rightToes']) {
      const b = vrm.humanoid.getNormalizedBoneNode(n);
      if (b) legDrop = Math.max(legDrop, -b.getWorldPosition(new THREE.Vector3()).y + 0.06);
    }
    if (legDrop < 0.12) legDrop = 0;
    else for (const n of ['leftLowerLeg', 'rightLowerLeg']) {
      const b = vrm.humanoid.getNormalizedBoneNode(n);
      if (b) kneeZ = Math.max(kneeZ ?? -Infinity, b.getWorldPosition(new THREE.Vector3()).z);
    }
    // a window's top edge is a line, not a floor: no cloth floor at all (the skirt lies on the thighs and hangs), and
    // hair / spring hems may hang below it (the first version kept a floor behind the knees: the dress stood out in a ring)
    setLedge(legDrop ? -Infinity : null);
    setFloorY(vrm, legDrop ? -5 : 0);
    // one draped skirt shape per sitting posture, keyed by the leg positions; per frame when the legs move (skirt.js)
    const fps = motion.fps ?? 30;
    const poseAt = (f) => { player.setTime(f / fps); vrm.humanoid.update(); };
    drapeSkirt(vrm, sitting ? legKey(vrm) : 'stand', { poseAt, frames: motion.rotations.length, id: ++state.motionSerial });
    player.setTime(0); settle();
    frame(headY(), sitting);
    state.rise = riseEnvelope(motion, (f) => { poseAt(f); vrm.scene.updateMatrixWorld(true); return headY(); }, headY());
    player.setTime(0); settle();
    state.box0 = bodyBox();                 // the body's sides in the NEW pose (window snapping lines up at once)
    // back to the pose the body is in (the blend's start) before the next frame, so nothing sees the jump
    if (blendFrom) { for (const [n, f] of Object.entries(blendFrom)) { const b = node(n); if (b) { b.quaternion.copy(f.q); if (f.p) b.position.copy(f.p); } } settle(); }
    state.t = 0;
    state.playing = play;
    if (play) state.face?.setEmotion(state.emotion);
  }

  function replay() {
    if (!state.player?.motion) return;
    state.t = 0; state.playing = true;
    state.face?.setEmotion(state.emotion);
  }

  const timer = new THREE.Timer();
  let slowAcc = 0;
  function tick() {
    requestAnimationFrame(tick);
    timer.update();
    let dt = Math.min(timer.getDelta(), 0.1);
    if (TIME_SCALE !== 1) {               // tests: her time runs slower, in fixed 1/60 s steps (a scaled dt left the
      slowAcc += dt * TIME_SCALE;         // springs with tiny steps and the hair floated about)
      if (slowAcc < 1 / 60) return;
      slowAcc -= 1 / 60; dt = 1 / 60;
    }
    const { vrm, player } = state;
    if (!vrm || !state.face) return;             // still loading / measuring
    if (state.playing) {
      state.t += dt;
      if (state.t >= player.duration) {
        state.t = player.duration; state.playing = false;
        state.face.setEmotion(state.emotion, 0.3);            // the feeling lingers a little after the line
        onEnd(state.idleMotion);
      }
    }
    if (player.motion) player.setTime(state.t);              // every frame: the layers below are added on top
    else vrm.humanoid.resetNormalizedPose();                 // no motion yet (2026-11-13): start each frame from the rest of the
                                                             // model, else the breathing / head drift piled up frame after frame
                                                             // and folded the head and chest forward until the first motion came
    applyBlendAndIdle(dt);
    state.face.setEyes(player.motion ? player.eyesClosed : 0);  // "eyes": closed in the keyframes (2026-11-07)
    state.face.update(dt);                                    // before vrm.update, which applies expressions
    if (windowInertia) dragShift(dt);
    applyHeld(dt);
    const rf = state.rise?.length ? Math.min(state.rise.length - 1, Math.round(state.t * (player.motion?.fps ?? 30))) : -1;
    updateView(rf >= 0 ? state.rise[rf] : 0, dt);
    // hair follows the body up / down (jiggle.js); fully during a posture transition step (sitting down at 0.7 the
    // hair flew up over the head, 2026-11-11)
    const FK = new URLSearchParams(location.search).get('followk');      // tests: try other follow amounts
    followVertical(vrm, state.vfollow ??= {}, FK !== null ? Number(FK) : blendFrom && blendS > 0.4 ? 1 : undefined);
    vrm.update(dt);
    // a thrown skirt cuts through the thighs between its few chains: while / just after a window drag its swing is
    // limited (skirt.js limitSkirtSwing; not during motions, where raised thighs push the skirt further on purpose)
    if (windowInertia && clock - lastDragAt < DRAG_SKIRT_S) limitSkirtSwing(vrm, DRAG_SKIRT_DEG);
    state.jiggle?.update(dt);
    fixSkirtVertices(vrm, state.t * (player?.motion?.fps ?? 30));
    updateLimbPush(vrm);
    if (BROWS) updateBrows(state.vrm, camera);
    look.update(vrm);
    renderer.render(scene, camera);
  }
  // The window moved by (dx, dy) screen px since the last frame = the body moved by that much in metres (view height /
  // canvas height per px) as far as the physics is concerned: spring tails and chest masses go back by it. Capped at
  // MAX_DRAG_V (m/s after the gain); a jump of over half the screen (the window put back home, another screen) is ignored.
  // DRAG_GAIN: how much of the window's movement the physics feels (0.5 was tried 2026-11-05 for 'swings too much';
  // what the author meant was the bouncing back and forth, fixed by dampSprings instead)
  const MAX_DRAG_V = 3, DRAG_GAIN = 1;
  // skirt swing limit while dragging (tools/skirt_drag_check.mjs, test2: none 5.3 cm into a thigh, 25 deg 3.2, 18 deg 1.2)
  const DRAG_SKIRT_DEG = 18, DRAG_SKIRT_S = 1.5;
  let lastDragAt = -1e9;
  let lastX = null, lastY = null;
  function dragShift(dt) {
    const x = window.screenX, y = window.screenY;
    if (lastX === null) { lastX = x; lastY = y; return; }
    const dx = x - lastX, dy = y - lastY;
    lastX = x; lastY = y;
    if (!dx && !dy) return;
    if (state.following) return;           // the window follows a window she sits / leans on: she rides with it, no sway
    if (Math.hypot(dx, dy) > 0.5 * (window.screen?.height || 1000)) return;
    const k = DRAG_GAIN * viewH / Math.max(canvas.clientHeight, 1);
    const d = new THREE.Vector3(dx * k, -dy * k, 0);
    const max = MAX_DRAG_V * Math.max(dt, 1 / 120);
    if (d.length() > max) d.setLength(max);
    lastDragAt = clock;
    held.winV = dx * k / Math.max(dt, 1 / 120);          // the window's sideways speed, m/s (the held sway)
    springTailsShift(state.vrm, d);
    state.jiggle?.shift(d);
  }
  let jiggleLevel = jiggle;
  function setJiggle(level) { jiggleLevel = level; state.jiggle?.setLevel(level); return jiggleLevel; }
  // opacity (0..1) of the drawn frame at CSS pixel (x, y) of the canvas
  const px = new Uint8Array(4);
  function alphaAt(x, y) {
    const gl = renderer.getContext(), r = canvas.getBoundingClientRect(), k = gl.drawingBufferWidth / r.width;
    const X = Math.floor((x - r.left) * k), Y = Math.floor(gl.drawingBufferHeight - (y - r.top) * k);
    if (X < 0 || Y < 0 || X >= gl.drawingBufferWidth || Y >= gl.drawingBufferHeight) return 0;
    gl.readPixels(X, Y, 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, px);
    return px[3] / 255;
  }
  // Which body part is under canvas CSS pixel (x, y) (touch feedback, 2026-11-06, Q5): every part is a bone segment
  // with a radius; the nearest one on screen within its radius wins (distance / radius). The head is a ball round
  // the middle of the skull (head bone + half the measured head top), its upper half 'head', its lower half 'cheek'.
  // Sides are the character's own; front / back from where the chest faces. Returns {part, side} or null.
  // part: head, cheek, chest, belly, hips (front) / bottom (back), back, arm, hand, skirt (upper thigh), legs.
  const PART_SEGS = [   // [part, from, to (null = a ball), radius m, side]
    ['chest', 'upperChest', 'neck', 0.13], ['chest', 'chest', 'upperChest', 0.13], ['belly', 'spine', 'chest', 0.12],
    ['hips', 'hips', 'spine', 0.14],
    ['arm', 'leftUpperArm', 'leftLowerArm', 0.05, 'left'], ['arm', 'rightUpperArm', 'rightLowerArm', 0.05, 'right'],
    ['arm', 'leftLowerArm', 'leftHand', 0.045, 'left'], ['arm', 'rightLowerArm', 'rightHand', 0.045, 'right'],
    ['hand', 'leftHand', 'leftHand+', 0.05, 'left'], ['hand', 'rightHand', 'rightHand+', 0.05, 'right'],
    ['skirt', 'leftUpperLeg', 'leftUpperLeg~', 0.09, 'left'], ['skirt', 'rightUpperLeg', 'rightUpperLeg~', 0.09, 'right'],
    ['legs', 'leftUpperLeg~', 'leftLowerLeg', 0.07, 'left'], ['legs', 'rightUpperLeg~', 'rightLowerLeg', 0.07, 'right'],
    ['legs', 'leftLowerLeg', 'leftFoot', 0.06, 'left'], ['legs', 'rightLowerLeg', 'rightFoot', 0.06, 'right'],
    ['legs', 'leftFoot', 'leftToes', 0.06, 'left'], ['legs', 'rightFoot', 'rightToes', 0.06, 'right'],
  ];
  const wp = (n) => { const b = node(n); return b ? b.getWorldPosition(new THREE.Vector3()) : null; };
  function boneAt(n) {
    if (n.endsWith('+')) {                     // a hand: from the wrist 9 cm on along the forearm
      const side = n.startsWith('left') ? 'left' : 'right', h = wp(`${side}Hand`), l = wp(`${side}LowerArm`);
      return h && l ? h.clone().add(h.clone().sub(l).setLength(0.09)) : null;
    }
    if (n.endsWith('~')) {                     // half way down the thigh
      const side = n.startsWith('left') ? 'left' : 'right', a = wp(`${side}UpperLeg`), b = wp(`${side}LowerLeg`);
      return a && b ? a.lerp(b, 0.5) : null;
    }
    return wp(n);
  }
  function toScreen(p) {
    const q = p.clone().project(camera);
    return new THREE.Vector2((q.x + 1) / 2 * canvas.clientWidth, (1 - q.y) / 2 * canvas.clientHeight);
  }
  function pxRadius(p, r) {                    // a world radius at p in CSS pixels
    const right = new THREE.Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    return toScreen(p).distanceTo(toScreen(p.clone().addScaledVector(right, r)));
  }
  function bodyPartAt(x, y) {
    if (!state.vrm) return null;
    state.vrm.scene.updateMatrixWorld(true);
    const P = new THREE.Vector2(x, y);
    // facing: the character's left (left - right shoulder) x up = its forward; front if that points at the camera
    const L = wp('leftUpperArm'), R = wp('rightUpperArm'), hips = wp('hips');
    const fwd = L && R ? new THREE.Vector3().crossVectors(L.clone().sub(R), new THREE.Vector3(0, 1, 0)) : new THREE.Vector3(0, 0, 1);
    const front = fwd.dot(camera.position.clone().sub(hips)) >= 0;
    let best = null, under = null;            // under: the best torso / leg part, also when a hand lies over it
    const consider = (part, a, b, r, side) => {
      const A = toScreen(a), B = b ? toScreen(b) : A, rp = pxRadius(a, r);
      const AB = B.clone().sub(A), t = AB.lengthSq() ? THREE.MathUtils.clamp(P.clone().sub(A).dot(AB) / AB.lengthSq(), 0, 1) : 0;
      const d = P.distanceTo(A.clone().addScaledVector(AB, t)) / Math.max(rp, 1);
      if (d <= 1 && (!best || d < best.d)) best = { part, side, d, A };
      if (d <= 1 && part !== 'hand' && part !== 'arm' && (!under || d < under.d)) under = { part, d };
    };
    const head = wp('head');
    if (head) {
      const c = head.clone().add(new THREE.Vector3(0, headTop * 0.45, 0));
      consider('head', c, null, Math.max(headTop * 0.62, 0.08));
    }
    for (const [part, a, b, r, side] of PART_SEGS) {
      const pa = boneAt(a), pb = b ? boneAt(b) : null;
      if (pa && (pb || !b)) consider(part, pa, pb, r, side);
    }
    if (!best) return null;
    let { part, side } = best;
    if (part === 'head') {                     // lower half of the head ball = the face's cheeks, by screen side
      const c = toScreen(head.clone().add(new THREE.Vector3(0, headTop * 0.45, 0)));
      if (P.y > c.y) { part = 'cheek'; side = (P.x < c.x) === front ? 'right' : 'left'; }
    }
    let below = under?.part ?? null;
    if (below === 'hips') below = front ? 'hips' : 'bottom';
    if (part === 'hips') part = front ? 'hips' : 'bottom';
    if ((part === 'chest' || part === 'belly') && !front) {
      part = 'back';
      const c = toScreen(hips);
      side = (P.x < c.x) === front ? 'right' : 'left';
    }
    return { part, side: side ?? null, under: below };
  }
  // where the top of the head is on the canvas (CSS pixels), for a speech bubble above it
  function headScreen() {
    if (!state.vrm) return null;
    const p = state.vrm.humanoid.getNormalizedBoneNode('head').getWorldPosition(new THREE.Vector3());
    p.y += 0.2;
    p.project(camera);
    return { x: (p.x + 1) / 2 * canvas.clientWidth, y: (1 - p.y) / 2 * canvas.clientHeight };
  }
  // the hands on the canvas (CSS px): wrist and fingertip of each, and a hand's size in px, for keeping the bubble
  // from hiding them
  const HAND_PTS = [['leftHand', 'leftMiddleDistal'], ['rightHand', 'rightMiddleDistal']];
  function handsScreen() {
    if (!state.vrm) return null;
    const pts = [], H = state.vrm.humanoid, v = new THREE.Vector3();
    for (const names of HAND_PTS) for (const n of names) {
      const b = H.getNormalizedBoneNode(n);
      if (!b) continue;
      b.getWorldPosition(v).project(camera);
      pts.push({ x: (v.x + 1) / 2 * canvas.clientWidth, y: (1 - v.y) / 2 * canvas.clientHeight });
    }
    const r = 0.05 / (viewH / Math.max(canvas.clientHeight, 1));         // ~5 cm in px
    return { pts, r };
  }
  resize();
  tick();
  function setScale(s) {
    scale = Math.min(Math.max(s, 0.25), 6);
    if (lastFrame) frame(...lastFrame);
    return scale;
  }
  // the avatar's facing (deg, + = turned toward its left), on top of its own (a VRM 0.x model is turned at load)
  let yaw = 0, baseYaw = 0;
  function setYaw(deg) {
    yaw = ((deg % 360) + 540) % 360 - 180;
    if (state.vrm) state.vrm.scene.rotation.y = baseYaw + yaw * Math.PI / 180;
    return yaw;
  }
  // where the floor under her is on the canvas (CSS px from the top), for snapping her feet onto the taskbar
  // where she will stand (canvas CSS px): the floor under her in the view she will stand in (no jump zoom, no held
  // lift; held = she lands standing, under where she was lifted from) and px per m there
  function landing() {
    if (!state.vrm) return null;
    const lifted = held.on || held.w > 0;
    const h = lifted && held.restHips ? held.restHips.clone()
      : state.vrm.humanoid.getNormalizedBoneNode('hips').getWorldPosition(new THREE.Vector3());
    const v0 = baseView(), v = held.on ? v0 : { cy: v0.cy - legDrop, h: v0.h };   // held.on only: right after a drop onto a ledge the floor row is already the ledge's      // one size (see updateView)
    const cam = camera.clone();
    cam.position.set(0, v.cy, v.h / 2 / Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
    cam.lookAt(0, v.cy, 0); cam.updateMatrixWorld(true);
    const ry = state.vrm.scene.rotation.y, c = Math.cos(ry), sn = Math.sin(ry);
    let fz = 0;          // the shoe point nearest the camera (+z), turned with her facing
    for (const f of footPts) fz = Math.max(fz, -sn * f.x + c * f.y);
    const p = new THREE.Vector3(h.x, 0, h.z + fz).project(cam);
    return { x: (p.x + 1) / 2 * canvas.clientWidth, y: (1 - p.y) / 2 * canvas.clientHeight, pxPerM: canvas.clientHeight / v.h };
  }
  function groundScreen() { return landing()?.y ?? null; }
  // her shoulders on the canvas (CSS px): the outer edges of the upper arms' roots and their height, and the body's
  // middle; the desktop app leans her shoulder against a window's side with it (window snapping, 2026-11-11)
  // the body's outermost points on the canvas (CSS px) and in the world (m): shoulders, arms, hands, hips with their
  // thickness; sy = the shoulders' height; the desktop app lines her up with a window's side by it (2026-11-11)
  // the shoulders and hips (the same points as the solver's wall plane, skeleton.WALL_PTS / WALL_HIP_R): the arms may
  // move in front of the wall, these lie on it
  const BOX_PTS = [['leftUpperArm', 0.05], ['rightUpperArm', 0.05], ['leftUpperLeg', 0.1], ['rightUpperLeg', 0.1]];
  function bodyBox() {
    if (!state.vrm) return null;
    const H = state.vrm.humanoid, v = new THREE.Vector3();
    let wl = Infinity, wr = -Infinity, sy = 0, n = 0;
    for (const [name, r] of BOX_PTS) {
      const b = H.getNormalizedBoneNode(name);
      if (!b) continue;
      b.getWorldPosition(v);
      wl = Math.min(wl, v.x - r); wr = Math.max(wr, v.x + r);
      if (name.endsWith('UpperArm')) { sy += v.y; n++; }
    }
    if (!n) return null;
    const px = (x, y) => { v.set(x, y, 0).project(camera); return [(v.x + 1) / 2 * canvas.clientWidth, (1 - v.y) / 2 * canvas.clientHeight]; };
    const [l, y] = px(wl, sy / n), [r] = px(wr, sy / n), [cx] = px(0, 0);
    // the bottom: the lowest point of the pelvis (the hip joints minus the thigh's thickness), where she sits on a ledge
    const hp = H.getNormalizedBoneNode('hips').getWorldPosition(new THREE.Vector3());
    const [, seat] = px(hp.x, hp.y - 0.1);
    return { l, r, sy: y, cx, wl, wr, seat };
  }
  // a wall at the body's outermost point on that side of the SCREEN ('left' | 'right'), or none: hair and clothes on
  // springs stay on her side of it
  const WALL_IN = 0.04;
  function setWall(screenSide) {
    if (!state.vrm) return;
    const b = screenSide ? bodyBox() : null;
    if (!b) setWallPlane(state.vrm, null);
    // WALL_IN inside the body's edge: the plane stops spring JOINTS, the cloth around them is a few cm wider (skirt hems
    // crossed the wall by that much)
    else if (screenSide === 'right') setWallPlane(state.vrm, b.wr - WALL_IN, -1);
    else setWallPlane(state.vrm, b.wl + WALL_IN, 1);
  }

  // the floor's row on the canvas now (the real camera; tests)
  const floorRow = () => (1 - new THREE.Vector3(0, 0, 0).project(camera).y) / 2 * canvas.clientHeight;
  const sittingNow = () => !!state.player?.motion?.root && state.player.motion.root[0][1] < -0.3;
  // render cost (2026-11-12, the look switches' budget): n full draws incl. the look passes, waiting for the GPU;
  // ms per frame + the WebGL renderer in use
  function bench(n = 60) {
    const gl = renderer.getContext();
    if (!state.vrm) return null;
    renderer.render(scene, camera); gl.finish();
    const t0 = performance.now();
    for (let i = 0; i < n; i++) { look.update(state.vrm); renderer.render(scene, camera); }
    gl.finish();
    const ext = gl.getExtension('WEBGL_debug_renderer_info');
    return { ms: (performance.now() - t0) / n, gpu: ext ? gl.getParameter(ext.UNMASKED_RENDERER_WEBGL) : null,
             px: [gl.drawingBufferWidth, gl.drawingBufferHeight], look: [...look.parts] };
  }
  return { bench, state, bodyPartAt, setHeld, isHeld: () => held.on, _held: held, setJiggle, handsScreen, loadModel, setMotion, replay, alphaAt, headScreen, sittingNow, setScale, getScale: () => scale, groundScreen, bodyBox, poseBox: () => state.box0, setWall, landing, floorRow, setYaw, getYaw: () => yaw, hasMotion: () => !!state.player?.motion };
}

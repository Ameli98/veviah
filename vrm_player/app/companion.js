// Desktop companion page (Decision 11 (B), 2026-11-03): the avatar fed by the companion server (companion/server.mjs).
// The MCP server tells the companion about every solved motion; the companion pushes it here over server-sent events
// (/events); this page fetches the motion (/motion/<id>) and plays it (stage.js), the line in a speech bubble.
// ?bg=none = transparent background; ?app=1 = inside the desktop app (desktop/main.js), whose preload gives
// window.hsinApp: click-through on transparent pixels, dragging the window, wheel = window size, right-click menu.
import { createStage } from './stage.js';
import { LipSync } from '../src/lipsync.js';

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const status = (s) => { $('status').textContent = s; };
const clear = q.get('bg') === 'none';
const appMode = q.get('app') === '1' && !!window.hsinApp;
if (clear) document.documentElement.classList.add('clear');
if (appMode) document.body.classList.add('app');
const BUBBLE_HOLD = 3.0;          // s the bubble stays after the motion has ended
const IDLE_GAP = (q.get('idlegap') ?? '20,60').split(',').map(Number);   // s between idle actions (random); ?idlegap= for tests
const FADE_MS = 250;              // a change of posture (no transition motion yet) fades out and in, each half

let bubbleTimer = null, motionDone = true;
const stage = createStage($('view'), {
  background: clear ? null : 0xe9edf2, headroom: 0.55, pickable: appMode,
  footroom: appMode ? 0.03 : null,           // in the app the feet stand at the window's bottom (the line is above the head)
  jiggle: q.get('jiggle') ?? 'weak', windowInertia: appMode,   // chest sway; dragging the window sways hair / skirt / chest
  onEnd: (wasIdle) => {
    realign();                         // snapped to a window: back onto its edge after every motion (2026-11-11)
    if (!wasIdle) {
      $('replay').style.display = 'block';
      motionDone = true;
      holdBubble();
    }
    scheduleIdle();
  },
});
window.hsinStage = stage;          // for tests (mcp_server/test/companion.mjs, speech.mjs)
const getJSON = async (url) => { const r = await fetch(url); if (!r.ok) throw new Error(`${url}: ${r.status}`); return r.json(); };

function say(text) {
  clearTimeout(bubbleTimer);
  $('say').textContent = text ?? '';
  $('say').classList.toggle('on', !!text);
}
// a hint from the app (2026-11-13, Q14: until the user has opened the right-click menu once, each start says where the
// settings are): shown for HINT_S unless a line is playing
const HINT_S = 6;
window.hsinApp?.onHint?.((text) => {
  if (!motionDone || speaker.playing || lineQueue.length) return;
  say(text);
  bubbleTimer = setTimeout(() => $('say').classList.remove('on'), HINT_S * 1000);
});
// the bubble goes BUBBLE_HOLD s after both the motion and the voice have ended
function holdBubble() {
  if (!motionDone || speaker.playing) return;
  clearTimeout(bubbleTimer);
  bubbleTimer = setTimeout(() => $('say').classList.remove('on'), BUBBLE_HOLD * 1000);
}

// ---------- voice (companion /speech: one wav per sentence, speech.mjs) ----------
// The motion and the bubble start when the first sentence's audio is there (FIRST_WAIT at most, then without it);
// the next sentences are fetched while one plays. Lines queue (pumpLines). The mouth follows the audio (lipsync.js).
const FIRST_WAIT = 4000;
const speaker = {
  ctx: null, ls: null, src: null, serial: 0, playing: false,
  init() {
    if (this.ctx) return;
    this.ctx = new AudioContext();
    this.ls = new LipSync(this.ctx);
    this.ls.input.connect(this.ctx.destination);
  },
  stop() {
    this.serial++;
    try { this.src?.stop(); } catch { /* not started */ }
    this.src = null; this.playing = false;
    if (this.ls) this.ls.active = false;
  },
  async fetchWav(sp, k) {
    const r = await fetch(`/speech/${sp.serial}/${k}`);
    if (r.status !== 200) return null;
    return this.ctx.decodeAudioData(await r.arrayBuffer());
  },
  // resolves when the first sentence can start (or FIRST_WAIT passed); then plays every sentence in turn
  async start(sp, onSentence) {
    this.stop(); this.init();
    const my = this.serial;
    if (this.ctx.state === 'suspended') this.ctx.resume().catch(() => {});
    const bufs = sp.sentences.map((_, k) => (k === 0 ? this.fetchWav(sp, 0) : null));
    const first = await Promise.race([bufs[0].catch(() => null), new Promise((r) => setTimeout(() => r('late'), FIRST_WAIT))]);
    if (my !== this.serial) return;
    this.playing = true;
    (async () => {
      for (let k = 0; k < sp.sentences.length && my === this.serial; k++) {
        if (k + 1 < sp.sentences.length) bufs[k + 1] = this.fetchWav(sp, k + 1);     // next one while this plays
        const buf = k === 0 && first !== 'late' ? first : await bufs[k].catch(() => null);
        if (my !== this.serial) return;
        onSentence(k);
        if (!buf) continue;
        await new Promise((resolve) => {
          const src = this.ctx.createBufferSource();
          src.buffer = buf; src.connect(this.ls.input); src.onended = resolve;
          this.src = src; this.ls.active = true; src.start();
        });
        this.ls.active = false;
      }
      if (my === this.serial) { this.playing = false; holdBubble(); }
    })();
  },
};
window.hsinSpeaker = speaker;      // for tests (mcp_server/test/speech.mjs)
let mouthT = performance.now();
(function mouth() {           // every frame: the lip sync's mouth into the face (before stage's own frame update)
  requestAnimationFrame(mouth);
  const now = performance.now(), dt = Math.min(0.1, (now - mouthT) / 1000);   // real time: frames may be slow
  mouthT = now;
  const f = stage.state.face;
  if (!f) return;
  f.setMouth(speaker.ls ? speaker.ls.update(dt) : null, speaker.playing);
})();
// keep the bubble above the head, inside the window
(function follow() {
  requestAnimationFrame(follow);
  const h = stage.headScreen(), el = $('say');
  if (!h || !el.classList.contains('on')) return;
  const w = el.offsetWidth, hh = el.offsetHeight, W = innerWidth;
  el.style.left = `${Math.min(Math.max(h.x, w / 2 + 6), W - w / 2 - 6)}px`;
  el.style.top = `${Math.max(h.y, hh + 6)}px`;
  // a hand behind the bubble: the bubble fades (stays near the head, where the user looks for it; the author 2026-11-05)
  const hs = stage.handsScreen(), b = el.getBoundingClientRect();
  const hidden = !!hs && hs.pts.some((p) => p.x > b.left - hs.r && p.x < b.right + hs.r && p.y > b.top - hs.r && p.y < b.bottom + hs.r);
  el.classList.toggle('behind', hidden);
})();

async function loadModel() {
  status('loading model…');
  const info = await getJSON('/model/info');
  const bytes = await (await fetch('/model.vrm')).arrayBuffer();
  await stage.loadModel(bytes, info.name, {
    measured: info.measured, status,
    saveFit: async (args) => {
      const r = await fetch('/fit', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(args) });
      if (!r.ok) console.warn('model not measured:', await r.text());
      return r.ok;
    },
  });
  try { if (!stage.hasMotion()) stage.setMotion(await getJSON('/motion/rest'), false); }
  catch (e) { console.warn('no rest pose', e); }
  status('');
  idleList(posture);                  // solve / fetch the idle actions ahead
  touchList(posture);
  scheduleIdle();
}

// ---------- idle actions (companion /idle: solved for this model and posture) ----------
// Personality sets (2026-11-08, the author): ?set=name (the app's setting, changed at runtime through hsinApp.onSet) picks
// sets/<name>.json for idle actions and touch reactions; what the set lacks for a posture (or the filter dropped on this
// model) comes from the default set.
let posture = 'stand', turn = 0, idleTimer = null, lastIdle = null;   // turn: the facing the last line left her in
let setName = q.get('set') || 'default';
const libCache = new Map();
// Window snapping (2026-11-11): sitting on a window's top edge (posture 'ledge') or leaning on a window's side
// (standing, place.wall = the side of HER body the window is on). The companion keeps it (/state) for the MCP server.
let place = { ledge: false, wall: null };
const postPlace = () => fetch('/state', { method: 'POST', body: JSON.stringify({ posture, wall: place.wall, ledge: place.ledge }) }).catch(() => {});
function libList(lib, p, set = setName) {
  const wall = p === 'stand' && place.wall ? place.wall : '';
  const key = `${lib}:${set}:${p}:${p === 'stand' ? turn : 0}:${wall}`;
  if (!libCache.has(key)) libCache.set(key, getJSON(`/${lib}?posture=${p}&turn=${p === 'stand' ? turn : 0}&set=${set}&wall=${wall}`).catch((e) => { libCache.delete(key); console.warn(lib, e); return []; }));
  return libCache.get(key);
}
async function idleList(p) {
  const own = await libList('idle', p);
  return own.length || setName === 'default' ? own : libList('idle', p, 'default');
}
function touchList(p) { return libList('touch', p); }
function setSet(name) {
  if (!name || name === setName) return;
  setName = name; libCache.clear(); lastIdle = null;
  idleList(posture); touchList(posture);
}
window.hsinApp?.onSet?.(setSet);
window.hsinSetSet = setSet;         // for tests (mcp_server/test/sets.mjs)
function scheduleIdle() {
  clearTimeout(idleTimer);
  const s = IDLE_GAP[0] + Math.random() * (IDLE_GAP[1] - IDLE_GAP[0]);
  idleTimer = setTimeout(playIdle, s * 1000);
}
async function playIdle() {
  if (stage.state.playing || speaker.playing) { scheduleIdle(); return; }
  const all = await idleList(posture);
  const list = all.length > 1 ? all.filter((a) => a.name !== lastIdle) : all;     // a posture's only idle action may repeat (2026-11-11)
  if (!list.length) { scheduleIdle(); return; }
  const a = list[Math.floor(Math.random() * list.length)];
  try {
    const m = await getJSON(`/motion/${a.id}`);
    if (stage.state.playing) { scheduleIdle(); return; }     // a line arrived meanwhile: it wins
    lastIdle = a.name;
    stage.setMotion(m, true, 'neutral', { idle: true });
  } catch (e) { console.warn('idle', e); libCache.clear(); scheduleIdle(); }
}
const canvasEl = $('view');
canvasEl.style.transition = `opacity ${FADE_MS}ms`;
const TIME_SCALE = Number(q.get('timescale')) || 1;     // tests: slowed down (stage.js reads the same)
const wait = (ms) => new Promise((r) => setTimeout(r, ms / TIME_SCALE));

const ready = loadModel();
ready.catch((e) => { status(`model error: ${e.message ?? e}`); console.error(e); });

const POSTURES = new Set(['stand', 'cross_legged', 'w_sit', 'seiza', 'side_right', 'side_left', 'knees_hugged', 'legs_out', 'kneel', 'ledge']);
let last = 0;
function listen() {
  const es = new EventSource('/events');
  es.addEventListener('play', (ev) => {
    const d = JSON.parse(ev.data);
    if (d.serial <= last) return;                 // a replayed event after reconnecting
    last = d.serial;
    // lines queue (2026-11-13, the author: Gemini sends one reply as two perform calls ~4 s apart and the second cut the
    // first off): a line starts when the previous one's voice and motion have ended; with more than MAX_QUEUED
    // waiting, the oldest waiting one is dropped
    lineQueue.push(d);
    while (lineQueue.length > MAX_QUEUED) lineQueue.shift();
    pumpLines();
  });
  es.onerror = () => status('waiting for the companion server…');
  es.onopen = () => { if ($('status').textContent.startsWith('waiting')) status(''); };
}
const MAX_QUEUED = 2, LINE_MAX_MS = 60000;
const lineQueue = [];
let lineBusy = false;
async function pumpLines() {
  if (lineBusy) return;
  lineBusy = true;
  try {
    while (lineQueue.length) {
      await playLine(lineQueue.shift());
      const t0 = performance.now();                // until this line's voice and motion are over (at most LINE_MAX_MS)
      while ((!motionDone || speaker.playing) && performance.now() - t0 < LINE_MAX_MS) await wait(100);
    }
  } finally { lineBusy = false; }
}
async function playLine(d) {
    try {
      await ready;
      clearTimeout(idleTimer);
      $('replay').style.display = 'none';
      const m = await getJSON(`/motion/${d.id}`);
      speaker.stop();
      if (d.speech) {             // the voice first: motion and bubble start with its first sentence
        const sep = /[\u3000-\u9fff\uff00-\uffef]/.test(d.say ?? '') ? '' : ' ';      // CJK sentences join without a space
        const text = (k) => (d.subtitle ? d.subtitle : d.speech.sentences.slice(0, k + 1).join(sep));
        await speaker.start(d.speech, (k) => say(text(k)));
      }
      motionDone = false;
      const p = d.posture ?? (m.root && m.root[0][1] < -0.3 ? 'sit?' : 'stand');
      if (place.ledge && p === 'stand') await unsnap(true);            // asked to stand on a ledge: she hops down
      else if (place.wall && p !== 'stand') { place.wall = null; stage.setWall(null); postPlace(); libCache.clear(); }   // sat down: off the wall
      else if (place.ledge && p !== 'ledge') { /* another way of sitting, still on the edge */ }
      if (p !== posture) {                         // a change of posture: the transition steps, then the line's motion
        if (!d.speech) say(d.subtitle || d.say);    // the line shows from the start of the transition (Q3)
        if (await transitionTo(p)) stage.setMotion(m, true, d.emotion ?? 'neutral');
        else {                                     // no path (e.g. onto a ledge): fade over it as before
          canvasEl.style.opacity = '0';
          await wait(FADE_MS);
          stage.setMotion(m, true, d.emotion ?? 'neutral', { blend: false });
          canvasEl.style.opacity = '1';
        }
        posture = p;
        if (POSTURES.has(p)) { idleList(p); touchList(p); }
      } else {
        stage.setMotion(m, true, d.emotion ?? 'neutral');
      }
      if (!d.speech) say(d.subtitle || d.say);
      else if (!$('say').classList.contains('on')) say(d.subtitle || d.speech.sentences[0]);
      const t = Math.round(Number(d.endTurn ?? 0) / 15) * 15;
      if (posture === 'stand' && t !== turn) { turn = t; idleList(posture); touchList(posture); }
      status(d.problems?.length ? `filter: ${d.problems.join(', ')}` : '');
      if (d.solvedFor && d.solvedFor !== stage.state.modelName) status(`solved for ${d.solvedFor} (model was not measured yet)`);
    } catch (e) { status(`error: ${e.message ?? e}`); console.error(e); }
}
listen();
$('replay').onclick = () => { stage.replay(); $('replay').style.display = 'none'; };

// ---------- touch feedback (2026-11-06, grilling with the author Q4-Q9; escalation removed 2026-11-08) ----------
// A click on the body (stage.bodyPartAt) plays a reaction from the companion's /touch library, solved for this model
// and posture like the idle actions. A reaction may have versions (<name>, <name>~2, <name>~3) that play in turn on
// repeated clicks; there is no escalation and no angry face from repeated clicks (the author, 2026-11-08). While Claude's
// line plays only the face reacts (Q9); while she is held nothing does. A reaction the filter dropped for this model /
// posture falls back along TOUCH_FALLBACK, and with none left the face alone reacts.
const TOUCH_WINDOW = 5;             // s: a hand over a shy part just touched still counts as that part
const SHY = new Set(['chest', 'hips', 'bottom', 'skirt']);
const TOUCH_FALLBACK = { shy_bottom: 'shy_skirt', shy_skirt: 'shy_cover' };
const touchTurn = new Map();
let lastShy = -1e9;
let faceTimer = null;
function reactionFor(part, side) {
  // chest -> shy_cover (hands on the chest), skirt / front of the hips -> shy_skirt (pressing the skirt down), bottom ->
  // shy_bottom (hands behind, looking back): covered where touched (the author, 2026-11-06: 碰哪護哪)
  if (SHY.has(part)) return { name: part === 'chest' ? 'shy_cover' : part === 'bottom' ? 'shy_bottom' : 'shy_skirt', emotion: 'shy' };
  if (part === 'head') return { name: 'head_pat', emotion: 'happy' };
  if (part === 'cheek') return { name: `cheek_poke_${side === 'left' ? 'l' : 'r'}`, emotion: 'surprised' };
  if (part === 'hand') return { name: `hand_${side === 'left' ? 'l' : 'r'}`, emotion: 'happy' };
  if (part === 'arm' || part === 'back') return { name: `shoulder_${side === 'left' ? 'l' : 'r'}`, emotion: 'neutral' };
  if (part === 'belly') return { name: 'belly', emotion: 'happy' };
  return { name: 'legs', emotion: 'neutral' };
}
const baseName = (n) => n.split('~')[0];
function pickVersion(list, name) {  // the versions of one reaction, in turn
  const vs = list.filter((x) => baseName(x.name) === name);
  if (!vs.length) return null;
  const kk = `${vs[0].set}:${name}`, k = touchTurn.get(kk) ?? 0;
  touchTurn.set(kk, k + 1);
  return vs[k % vs.length];
}
function faceOnly(emotion) {        // a short look on the face, then back to the line's / the current feeling
  const f = stage.state.face;
  if (!f) return;
  f.setEmotion(emotion, 0.8);
  clearTimeout(faceTimer);
  faceTimer = setTimeout(() => f.setEmotion(stage.state.emotion, stage.state.playing ? 1 : 0.3), 1200);
}
async function touch(x, y) {
  let hit = stage.bodyPartAt(x, y);
  if (!hit || stage.isHeld()) return null;
  // a hand lying over a shy part she was just touched on (shy_cover puts the hands on the chest) counts as that part
  const now = performance.now() / 1000;
  if ((hit.part === 'hand' || hit.part === 'arm') && SHY.has(hit.under) && now - lastShy < TOUCH_WINDOW) hit = { part: hit.under, side: null };
  if (SHY.has(hit.part)) lastShy = now;
  const r = reactionFor(hit.part, hit.side);
  const lineOn = stage.state.playing && !stage.state.idleMotion;
  if (lineOn || speaker.playing) { faceOnly(r.emotion); return { ...hit, reaction: 'face' }; }
  // the chosen set first, then the default set, then the fallback items (in both)
  const lists = [await touchList(posture)];
  if (setName !== 'default') lists.push(await libList('touch', posture, 'default'));
  const pick = (name) => { for (const l of lists) { const a = pickVersion(l, name); if (a) return a; } return null; };
  let a = pick(r.name);
  for (let alt = TOUCH_FALLBACK[r.name]; !a && alt; alt = TOUCH_FALLBACK[alt]) a = pick(alt);
  if (!a) { faceOnly(r.emotion); return { ...hit, reaction: 'face' }; }
  try {
    const m = await getJSON(`/motion/${a.id}`);
    if (stage.state.playing && !stage.state.idleMotion) { faceOnly(r.emotion); return { ...hit, reaction: 'face' }; }
    clearTimeout(idleTimer);
    stage.setMotion(m, true, a.emotion ?? r.emotion, { idle: true });
    lastIdle = null;
    return { ...hit, reaction: a.name, set: a.set };
  } catch (e) { console.warn('touch', e); return null; }
}
window.hsinTouch = touch;           // also for tests (mcp_server/test/touch.mjs)

function onTouch(x, y) { touch(x, y); }

// ---------- window snapping (2026-11-11) ----------
// The desktop app finds the window under her when she is dropped (main.js): its top edge -> she sits on it (posture
// 'ledge', the legs hanging); its side next to her shoulder -> she leans on it (standing, place.wall). It moves the
// window with the target and tells the page when the target goes away (minimized, closed, maximized).
async function restMotion(p, wall = null, pose = '') {
  const list = await getJSON(`/rest?posture=${p}&wall=${wall ?? ''}&pose=${pose}`);
  return list?.[0] ? getJSON(`/motion/${list[0].id}`) : getJSON('/motion/rest');
}
// ---------- posture transitions (2026-11-11, the author: a change of posture faded out and in) ----------
// A transition walks a small graph of rest poses, blending from one to the next (stage.setMotion blendS): standing
// goes down through a squat, then either back onto the bottom (knees_hugged, from there legs_out / cross_legged) or
// down onto the knees (kneel, seiza, from there w_sit / side_*). Each step is a solved rest pose of this model.
// sitting on the bottom and on the knees meet at the side sits (the knees fall to one side), so changing between
// two sitting ways never stands up half way (cross_legged -> side_left went up through the squat at first)
const T_EDGES = [['stand', 'kneel', 0.9], ['kneel', 'seiza', 0.6], ['seiza', 'w_sit', 0.6], ['seiza', 'side_right', 0.7],
                 ['seiza', 'side_left', 0.7], ['stand', 'squat', 0.8], ['squat', 'cross_legged', 1.0], ['squat', 'legs_out', 1.0],
                 ['squat', 'knees_hugged', 0.9], ['knees_hugged', 'legs_out', 0.7], ['knees_hugged', 'cross_legged', 0.8],
                 ['knees_hugged', 'side_right', 0.8], ['knees_hugged', 'side_left', 0.8],
                 ['cross_legged', 'side_right', 0.8], ['cross_legged', 'side_left', 0.8], ['legs_out', 'cross_legged', 0.8]];
// (2026-11-11, the author: sitting down sank and came back up: stand -> squat -> kneel rose again from the squat to the
// upright kneel, and stand -> squat -> knees_hugged -> cross_legged bowed then straightened; every step now only goes
// down when sitting and only up when standing)
function transitionPath(a, b) {
  const adj = new Map();
  for (const [x, y, d] of T_EDGES) { (adj.get(x) ?? adj.set(x, []).get(x)).push([y, d]); (adj.get(y) ?? adj.set(y, []).get(y)).push([x, d]); }
  const prev = new Map([[a, null]]), q = [a];
  while (q.length) { const x = q.shift(); if (x === b) break; for (const [y, d] of adj.get(x) ?? []) if (!prev.has(y)) { prev.set(y, [x, d]); q.push(y); } }
  if (!prev.has(b)) return null;
  const path = [];
  for (let x = b; prev.get(x); x = prev.get(x)[0]) path.unshift([x, prev.get(x)[1]]);
  return path;                                     // [[pose, seconds], ...] after a
}
// null when this model has no such rest (e.g. cross_legged / side sits on a long skirt skinned to the legs: rejected)
const poseMotion = async (p) => {
  const list = await getJSON(p === 'squat' ? '/rest?posture=stand&pose=squat' : `/rest?posture=${p}`);
  return list?.[0] ? getJSON(`/motion/${list[0].id}`) : null;
};
// The steps become ONE motion (each rest pose's first frame, joints slerped with a smoothstep per step, the hips
// moved linearly), so the player treats it like any motion: a skirt / dress is simulated frame by frame through it
// (step-by-step blends showed each step's finished drape on legs still on the way: test2's dress tore, 2026-11-11).
function slerpQ(a, b, t) {
  let d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
  const s = d < 0 ? -1 : 1; d = Math.abs(d);
  if (d > 0.9995) { const q = a.map((x, i) => x + (s * b[i] - x) * t); const n = Math.hypot(...q); return q.map((x) => x / n); }
  const th = Math.acos(d), k0 = Math.sin((1 - t) * th) / Math.sin(th), k1 = s * Math.sin(t * th) / Math.sin(th);
  return a.map((x, i) => k0 * x + k1 * b[i]);
}
const smoothT = (x) => x * x * (3 - 2 * x);
function transitionMotion(motions, durs) {
  const fps = 30, rotations = [], root = [], left = [], right = [];
  const first = (m) => ({ q: m.rotations[0], r: m.root?.[0] ?? [0, 0, 0], l: m.hands?.left?.[0] ?? 0, rt: m.hands?.right?.[0] ?? 0 });
  for (let i = 1; i < motions.length; i++) {
    const a = first(motions[i - 1]), b = first(motions[i]), n = Math.max(2, Math.round(durs[i - 1] * fps));
    for (let f = 0; f < n; f++) {
      const t = smoothT(f / n);
      rotations.push(a.q.map((q, j) => slerpQ(q, b.q[j], t)));
      root.push(a.r.map((x, k) => x + (b.r[k] - x) * t));
      left.push(t < 0.5 ? a.l : b.l); right.push(t < 0.5 ? a.rt : b.rt);
    }
  }
  const last = first(motions[motions.length - 1]);
  rotations.push(last.q); root.push(last.r); left.push(last.l); right.push(last.rt);
  const m0 = motions[motions.length - 1];
  return { fps, joints: m0.joints, quat_order: m0.quat_order ?? 'xyzw', rotations, root,
           hands: { shapes: m0.hands?.shapes, left, right } };
}
// play the transition from the current posture to p; resolves when it has finished (false: no path)
async function transitionTo(p) {
  const path = transitionPath(posture, p);
  if (!path) return false;
  const motions = await Promise.all([poseMotion(posture), ...path.map(([x]) => poseMotion(x))]);
  if (motions.some((m) => !m)) return false;        // a step this model cannot take: fade over it instead (test1)
  const m = transitionMotion(motions, path.map(([, d]) => d));
  stage.setMotion(m, true, stage.state.emotion, { idle: true });
  await wait((m.rotations.length - 1) / 30 * 1000);
  return true;
}
async function swapTo(p, m) {         // a posture change without a line: fade out, the new rest, fade in
  canvasEl.style.opacity = '0';
  await wait(FADE_MS);
  stage.setMotion(m, false, stage.state.emotion, { blend: false });
  canvasEl.style.opacity = '1';
  posture = p; libCache.clear(); lastIdle = null;
  if (POSTURES.has(p)) { idleList(p); touchList(p); }
}
// main.js names the wall by the side of her body it is on IF she faces the viewer (a window at the screen's right =
// her left); turned round (her facing + the last line's turn) it is her other side
const facingBack = () => Math.cos((stage.getYaw() + turn) * Math.PI / 180) < 0;
// from the lifted pose straight into sitting on the edge / leaning (no fade; 2026-11-11)
async function blendTo(p, m, sec) {
  stage.setMotion(m, false, stage.state.emotion, { blendS: sec });
  // the window lines up with the edge / wall NOW, from the new pose's own measures, and the body blends into it
  // (lining up after the blend, she first sank in place and the window then jumped 165 px; the author 2026-11-11)
  if (place.ledge || place.wall) window.hsinApp?.attached?.(stage.groundScreen(), stage.poseBox() ?? stage.bodyBox());
  posture = p; libCache.clear(); lastIdle = null;
  if (POSTURES.has(p)) { idleList(p); touchList(p); }
  await wait(sec * 1000);
}
async function snapTo(t) {            // t = { kind: 'ledge' | 'wall', side }
  stage.setWall(null);
  if (t.kind === 'ledge') { place = { ledge: true, wall: null }; await blendTo('ledge', await restMotion('ledge'), 0.6); }
  else {
    const side = facingBack() ? (t.side === 'left' ? 'right' : 'left') : t.side;
    place = { ledge: false, wall: side, screen: t.side === 'left' ? 'right' : 'left' };
    await blendTo('stand', await restMotion('stand', side), 0.8);
  }
  postPlace();
  // (no plane pushing hair / clothes: it folded test2's dress; the author 2026-11-11 Q3)
  window.hsinApp?.attached?.(stage.groundScreen(), stage.bodyBox());
}
// fall: the window drops onto the taskbar (main.js) and she lands there standing
async function unsnap(fall) {
  const was = place;
  stage.setWall(null);
  place = { ledge: false, wall: null };
  if (was.ledge || posture === 'ledge' || was.wall) await blendTo('stand', await restMotion('stand'), 0.5);
  postPlace();
  if (fall) {
    await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
    window.hsinApp?.fall?.(stage.groundScreen());
    stage.setHeld(true); setTimeout(() => stage.setHeld(false), 120);     // the landing dip
  }
}
window.hsinApp?.onAttach?.((t) => { if (t) snapTo(t); else unsnap(false); });
// while snapped, every drawn frame asks the app to follow the target window (synced to the display: a separate 16 ms
// timer in main.js stepped out of time with the frames and the movement stuttered, the author 2026-11-11)
(function followLoop() {
  requestAnimationFrame(followLoop);
  const on = (place.ledge || place.wall) && !stage.isHeld();
  stage.state.following = on;
  if (on) window.hsinApp?.follow?.();
})();
window.hsinApp?.onDetach?.(() => unsnap(place.ledge));     // off a ledge she falls; off a wall she just stands up straight
// line her up with the window again (after a zoom or a motion: her size / pose in the window changed)
let realignTimer = null;
function realign() {
  if (!place.ledge && !place.wall) return;
  clearTimeout(realignTimer);
  realignTimer = setTimeout(() => window.hsinApp?.attached?.(stage.groundScreen(), stage.bodyBox()), 120);
}
window.hsinSnapTo = snapTo; window.hsinUnsnap = unsnap;     // tests
window.hsinTransition = async (p) => { const ok = await transitionTo(p); posture = p; return ok; };   // tests

// ---------- inside the desktop app ----------
if (appMode) {
  const app = window.hsinApp;
  let solid = true, drag = null;
  const ALPHA_MIN = 0.1;                         // an opaque-enough pixel catches the mouse
  // A press on the avatar is a click (touch, below) unless the mouse moves CLICK_PX or it is held CLICK_MS: then it
  // is a drag that moves the window and lifts her (stage.setHeld; 2026-11-06, Q4: a drag always moves the window)
  const CLICK_MS = 400, CLICK_PX = 5;     // 250 -> 400 ms (2026-11-11, the author: slow clicks did nothing)
  let press = null;
  window.addEventListener('mousemove', (e) => {
    if (press && !drag && Math.hypot(e.screenX - press.x, e.screenY - press.y) >= CLICK_PX) {
      drag = true; app.dragStart(press.x, press.y); stage.setHeld(true);
      // (place stays until the drop: main.js answers with attach(null) and unsnap() puts her back to standing; clearing
      // it here made unsnap think she had not been leaning, and she kept the lean after being dragged away, 2026-11-11)
    }
    if (drag) { app.dragMove(e.screenX, e.screenY, stage.groundScreen(), stage.bodyBox()); return; }
    const on = stage.alphaAt(e.clientX, e.clientY) > ALPHA_MIN;
    if (on !== solid) { solid = on; app.setSolid(on); }
  });
  // what the page sees (button / wheel), so problems can be read in the app's log
  window.addEventListener('mousedown', (e) => app.log?.('mousedown', e.button, 'solid', solid, 'alpha', stage.alphaAt(e.clientX, e.clientY).toFixed(2)), true);
  window.addEventListener('wheel', (e) => app.log?.('wheel', e.deltaY, 'solid', solid), { passive: true, capture: true });
  window.addEventListener('mousedown', (e) => {
    if (e.button === 0 && solid) press = { x: e.screenX, y: e.screenY, cx: e.clientX, cy: e.clientY, t: performance.now() };
  });
  window.addEventListener('mouseup', () => {
    const p = press; press = null;
    if (drag) {
      drag = null; app.dragEnd(stage.groundScreen(), stage.bodyBox()); stage.setHeld(false); landSnap = 0;
      // main.js answers with attach (a target, or null = none): off a ledge without a new one she stands
      // she lands in the posture she was in (2026-11-11, the author: sitting must last until she is told to stand; before,
      // a drag always ended standing: landStanding)
    } else if (p && performance.now() - p.t < CLICK_MS) onTouch(p.cx, p.cy);
  });
  // where she will land: a soft shadow on the floor row she will stand on (stage.landing), moved onto the taskbar
  // and coloured when the app says the drop will snap there (main.js sends the offset in px, or null)
  const land = document.getElementById('land');
  let landSnap = 0, snapping = false;
  let target = null;
  app.onSnap?.((dy) => { snapping = dy !== null; landSnap = dy ?? 0; });
  app.onTarget?.((t) => { target = t; });
  const glow = document.getElementById('glow');
  (function landLoop() {
    requestAnimationFrame(landLoop);
    const on = stage.isHeld();
    land.classList.toggle('on', on && !target);
    if (!on) { glow.classList.remove('on'); return; }
    const l = stage.landing();
    if (!l) return;
    const w = 0.42 * l.pxPerM, h = w * 0.24;
    land.style.width = `${w}px`; land.style.height = `${h}px`;
    land.style.left = `${l.x}px`;
    land.style.top = `${Math.min(l.y + landSnap, innerHeight - h / 2)}px`;
    land.classList.toggle('snap', snapping && !target);
    const bx = target ? stage.bodyBox() : null;
    glow.classList.toggle('on', !!bx);
    if (bx) {
      if (target.kind === 'ledge') {             // a line under her bottom, where it will sit on the edge
        Object.assign(glow.style, { left: `${bx.l - 25}px`, width: `${bx.r - bx.l + 50}px`, top: `${bx.seat - 2}px`, height: '4px' });
      } else {                                   // a line down the side she will lean on (her left = the screen's right)
        const x = target.side === 'left' ? bx.r : bx.l;
        Object.assign(glow.style, { left: `${x - 2}px`, width: '4px', top: `${bx.sy - 60}px`, height: `${bx.seat - bx.sy + 90}px` });
      }
    }
  })();
  // the wheel scales the avatar inside the window (the window keeps its size: resizing a transparent window jumped
  // and flickered on Windows, 2026-11-04); the scale is kept by the app
  if (q.get('scale')) stage.setScale(Number(q.get('scale')));
  if (q.get('yaw')) stage.setYaw(Number(q.get('yaw')));
  // Shift + wheel turns her (15 deg a notch, kept by the app); Chromium turns Shift + wheel into a sideways scroll,
  // so deltaX counts too
  window.addEventListener('wheel', (e) => {
    if (!solid) return;
    const d = e.deltaY || e.deltaX;
    if (!d) return;
    if (e.shiftKey) app.turn(stage.setYaw(stage.getYaw() + (d < 0 ? 15 : -15)));
    else { app.zoom(stage.setScale(stage.getScale() * (d < 0 ? 1.06 : 1 / 1.06))); realign(); }
  }, { passive: true });
  window.addEventListener('contextmenu', (e) => { e.preventDefault(); if (solid) app.menu(); });
  app.onJiggle?.((level) => stage.setJiggle(level));
  app.setSolid(true);
}

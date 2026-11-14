// Desktop companion server (Decision 11 (B), 2026-11-03): the avatar in its own window instead of inside the chat
// (Claude Desktop stopped showing MCP Apps views of local servers). No dependencies beyond Node.
//
//   node companion.mjs [model.vrm]     (or companion.sh; port: env HSIN_COMPANION_PORT, default 8790)
//   then open http://localhost:8790/ (the Windows browser reaches WSL's localhost)
//
//   GET  /               the page (vrm_player/dist-companion/companion.html, npm run build:companion)
//   GET  /model/info     {name, measured}       GET /model.vrm   the VRM file
//   POST /fit            {skeleton, fit} from the page: a new model's measurements (as the MCP view's save_model_fit)
//   GET  /motion/<id>    a solved motion (written by the MCP server; 'rest' = the standing rest pose)
//   POST /play           from the MCP server after each perform: {id, say, emotion, problems, solvedFor}
//   GET  /events         server-sent events to the page: 'play' with the same data (+ serial)
//   POST /quit           stop (the desktop app does this when it quits or switches the model)
//   GET|POST /settings   speech settings (speech.mjs; the app posts them at start and on every change) and state
//   GET  /speech/<serial>/<k>   the k-th sentence of line <serial> as wav (waits until it is synthesized; 204 = none)
//   GET  /touch?posture=p[&turn=deg][&set=name] touch reactions, the same way, each with its emotion; GET /sets lists the sets
//   GET  /idle?posture=p[&turn=deg][&set=name] idle actions (standing ones at that facing, rounded to 15) for that posture (sets/<name>.json, 'sit' ones for any floor posture), solved for this
//                        model once and cached (~/.cache/hsin/motions/idle/); only those that pass the rule filter:
//                        [{id, name, duration}], each playable through /motion/<id>
import http from 'node:http';
import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Speech, VOICES } from './speech.mjs';
import { Chat } from './chat.mjs';
import { VoiceInstaller, NEED_GB } from './voice_install.mjs';
import { COMPANION_PORT, MODEL_HASH, MODEL_NAME, MODEL_PATH, ROOT, STORE, measured, motionFile, saveFit, PYTHON, PY_ENV, CACHE_DIR, CONFIG_DIR, PORT_FILE } from './model.mjs';

const PAGE = path.join(ROOT, 'vrm_player/dist-companion/companion.html');
const PLAYER = path.join(ROOT, 'vrm_player/dist-player/index.html');   // the hidden cloth window's page (Windows app)

// Cloth first (2026-11-13, Windows app): the solver asks POST /cloth; the app (Electron main, our parent, through the
// IPC channel it opened) drapes the skirt in a hidden window running the player page and answers with the points.
// Without the channel (WSL / tests) or for another model: 503, and the solver falls back to the SKIRT_GAP rule.
const clothWaits = new Map();
let clothId = 0;
process.on?.('message', (m) => {
  if (m?.type !== 'cloth-result') return;
  const w = clothWaits.get(m.id); if (!w) return;
  clothWaits.delete(m.id); clearTimeout(w.timer);
  m.error ? w.reject(new Error(m.error)) : w.resolve(m.points);
});
function askCloth(motion, frames) {
  if (!process.send) return Promise.reject(new Error('no app'));
  const id = ++clothId;
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => { clothWaits.delete(id); reject(new Error('cloth timeout')); }, 90000);
    clothWaits.set(id, { resolve, reject, timer });
    process.send({ type: 'cloth', id, motion, frames });
  });
}
const HERE = path.dirname(fileURLToPath(import.meta.url));   // (URL.pathname gave /C:/... on Windows)
// libraries of short motions solved per model: idle actions and touch reactions, in personality sets (sets/<name>.json,
// 2026-11-08: idle.json / touch.json became sets/default.json; a set entry {name, use: 'other' | 'set:other'} plays
// another entry's motion under its own name; the page falls back to the default set for items a set lacks)
const SETS_DIR = path.join(HERE, 'sets');
async function setNames() {
  return (await fs.readdir(SETS_DIR)).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)).sort((a, b) => (a === 'default' ? -1 : b === 'default' ? 1 : a.localeCompare(b)));
}
const readSet = async (name) => JSON.parse(await fs.readFile(path.join(SETS_DIR, `${name.replace(/[^\w-]/g, '')}.json`), 'utf-8'));
// the entries of one set's library ('idle' | 'touch') for 'stand' | 'sit', `use` references resolved (one level)
async function setActs(setName, libName, kind) {
  const set = await readSet(setName);
  const acts = set[libName]?.[kind] ?? [];
  const out = [];
  for (const a of acts) {
    if (!a.use) { out.push(a); continue; }
    const [otherSet, otherName] = a.use.includes(':') ? a.use.split(':') : [setName, a.use];
    const src = (otherSet === setName ? acts : (await readSet(otherSet))[libName]?.[kind] ?? []).find((b) => b.name === otherName && !b.use);
    if (!src) { log(`${setName}/${libName}/${kind}/${a.name}: use ${a.use} not found`); continue; }
    out.push({ ...src, name: a.name, emotion: a.emotion ?? src.emotion });
  }
  return out;
}
const IDLE_DIR = path.join(STORE, 'idle');
const POSTURES = ['stand', 'cross_legged', 'w_sit', 'seiza', 'side_right', 'side_left', 'knees_hugged', 'legs_out', 'kneel', 'ledge'];
const idleJobs = new Map();
let place = { posture: 'stand', wall: null, ledge: false };   // see /state       // lib:posture:turn -> Promise of its list (one solve at a time each)

function batchSolve(reqs) {
  return new Promise((resolve, reject) => {
    const p = spawn(PYTHON, ['-m', 'motion_loop.perform', 'batch'], { cwd: path.join(ROOT, 'motion_loop'), env: PY_ENV });
    let out = '', err = '';
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    p.on('error', reject);
    p.on('close', (code) => (code === 0 ? resolve(JSON.parse(out)) : reject(new Error(`solver exited ${code}: ${err.slice(-300)}`))));
    p.stdin.end(JSON.stringify({ reqs }));
  });
}
// The idle actions of one posture for this model: from the cache (same model file, same idle.json) or solved now.
// Standing ones keep her facing: every keyframe gets legs.turn = turn + its own (an own turn that would pass 180 goes
// the other way), so an idle action does not swing her back to the front (2026-11-04, the author: after Claude turned her
// round, the first idle action turned her straight back).
function turnedActs(acts, turn) {
  if (!turn) return acts;
  return acts.map((a) => ({ ...a, keyframes: a.keyframes.map((k, i) => {
    const own = Number(k.legs?.turn ?? 0);
    if (i > 0 && k.legs?.turn === undefined) return k;
    let t = turn + own;
    if (Math.abs(t) > 180) t = turn - own;
    return { ...k, legs: { ...(k.legs ?? {}), turn: t } };
  }) }));
}
// the solver's own code is part of the cache key: an idle action solved by an older solver is solved again (2026-11-05:
// the turning steps changed; the old cache files stay, unused)
let solverHashP = null;
function solverHash() {
  solverHashP ??= (async () => {
    const dir = path.join(ROOT, 'motion_loop/motion_loop'), h = createHash('sha256');
    for (const f of (await fs.readdir(dir)).filter((f) => f.endsWith('.py')).sort()) h.update(await fs.readFile(path.join(dir, f)));
    return h.digest('hex');
  })();
  return solverHashP;
}
// wall: 'left' | 'right' = leaning on a window's side (window snapping, 2026-11-11): standing motions are solved with the
// wall's body (keyframes.WALL_BODY fills what they leave out). libName 'rest' = the posture's plain rest pose.
const REST_ACTS = [{ name: 'rest', emotion: 'neutral', keyframes: [{ t: 0 }, { t: 1 }] }];
// in-between poses of posture transitions (companion.js transitionTo): /rest?posture=stand&pose=squat
const SQUAT = { legs: { height: 'squat' }, right_arm: { touch: 'knee', with: 'palm' }, left_arm: { touch: 'knee', with: 'palm' }, torso: { lean: 15 } };
const POSE_ACTS = { squat: [{ name: 'squat', emotion: 'neutral', keyframes: [{ t: 0, ...SQUAT }, { t: 1, ...SQUAT }] }] };
async function idleFor(posture, turn = 0, libName = 'idle', setName = 'default', wall = null) {
  if (posture !== 'stand') wall = null;
  // leaning on a wall: the set's touch.wall reactions (arms stay folded; 2026-11-11), else the standing ones
  const kind = posture === 'stand' ? (wall && libName === 'touch' && (await readSet(setName)).touch?.wall?.length ? 'wall' : 'stand') : 'sit';
  const base = POSE_ACTS[libName] ?? (libName === 'rest' ? REST_ACTS : await setActs(setName, libName, kind));
  const acts = turnedActs(base, posture === 'stand' ? turn : 0);
  const key = createHash('sha256').update(JSON.stringify(acts)).update(MODEL_HASH).update(await solverHash()).update(wall ?? '').digest('hex').slice(0, 12);
  const tag = `${libName === 'idle' ? '' : libName + '_'}${setName === 'default' ? '' : setName + '_'}${wall ? 'wall' + wall + '_' : ''}`;
  const list = path.join(IDLE_DIR, `${tag}${MODEL_NAME}_${posture}_${key}.json`);
  try { return JSON.parse(await fs.readFile(list, 'utf-8')); } catch { /* not solved yet */ }
  await fs.mkdir(IDLE_DIR, { recursive: true });
  const t0 = Date.now();
  const res = await batchSolve(acts.map((a) => ({ posture, keyframes: a.keyframes, interp: a.interp, wall, model: MODEL_NAME, model_path: MODEL_PATH, cloth: false })));
  const out = [];
  for (let i = 0; i < acts.length; i++) {
    const r = res[i];
    if (r.error || r.problems?.length) { log(`${setName}/${libName} ${posture}/${acts[i].name} skipped: ${r.error ?? r.problems.join(',')}`); continue; }
    const id = `idle_${tag}${MODEL_NAME}_${posture}_${acts[i].name}_${key}`.replace(/[^\w-]/g, '');
    await fs.writeFile(motionFile(id), JSON.stringify(r.motion));
    out.push({ id, name: acts[i].name, duration: r.duration, emotion: acts[i].emotion, set: setName });
  }
  await fs.writeFile(list, JSON.stringify(out));
  log(`${setName}/${libName} ${posture}: ${out.length}/${acts.length} solved for ${MODEL_NAME} in ${Date.now() - t0} ms`);
  return out;
}
// Stale cache (2026-11-08, the author): a list file of THIS model whose key no longer matches (older solver, older set file,
// re-exported model) and the idle_* motion files no list refers to any more are deleted at startup; other models' files
// stay until that model is loaded again.
async function pruneIdle() {
  let lists;
  try { lists = (await fs.readdir(IDLE_DIR)).filter((f) => f.endsWith('.json')); } catch { return; }
  const valid = new Set(), hash = await solverHash();
  for (const setName of await setNames()) {
    for (const libName of ['idle', 'touch', 'rest', ...Object.keys(POSE_ACTS)]) {
      for (const kind of ['stand', 'sit', 'wall']) {
        if (kind === 'wall' && libName !== 'touch') continue;
        const acts = POSE_ACTS[libName] ?? (libName === 'rest' ? REST_ACTS : await setActs(setName, libName, kind));
        const turns = kind !== 'sit' ? Array.from({ length: 25 }, (_, i) => -180 + 15 * i) : [0];
        for (const turn of turns) for (const wall of kind === 'stand' ? ['', 'left', 'right'] : kind === 'wall' ? ['left', 'right'] : [''])
          valid.add(createHash('sha256').update(JSON.stringify(turnedActs(acts, turn))).update(MODEL_HASH).update(hash).update(wall).digest('hex').slice(0, 12));
      }
    }
  }
  const mine = new RegExp(`^(?:touch_|rest_|squat_)?(?:[\\w-]+_)?${MODEL_NAME}_[a-z_]+_([0-9a-f]{12})\\.json$`);
  let gone = 0;
  const keep = new Set();
  for (const f of lists) {
    const m = mine.exec(f);
    if (m && !valid.has(m[1])) { await fs.unlink(path.join(IDLE_DIR, f)); gone++; continue; }
    try { for (const a of JSON.parse(await fs.readFile(path.join(IDLE_DIR, f), 'utf-8'))) keep.add(a.id); } catch { /* unreadable list */ }
  }
  let motions = 0;
  for (const f of await fs.readdir(STORE)) {
    if (!f.startsWith('idle_') || !f.endsWith('.json') || !f.includes(`_${MODEL_NAME}_`)) continue;
    if (!keep.has(f.slice(0, -5))) { await fs.unlink(path.join(STORE, f)); motions++; }
  }
  if (gone || motions) log(`pruned ${gone} stale list(s) and ${motions} unused idle motion(s) of ${MODEL_NAME}`);
}
const log = (...a) => console.error('[companion]', ...a);
const speech = new Speech({ log, root: ROOT });
const chat = new Chat({ port: COMPANION_PORT, configDir: CONFIG_DIR, log,
  langs: () => ({ voice: speech.settings.voice ? (speech.settings.voiceLang ?? speech.voice?.lang ?? null) : null, ui: speech.settings.uiLang }) });
// first-voice-on download of the TTS environment (Windows only; voice_install.mjs); default folder next to the cache
const TTS_DIR = path.join(path.dirname(CACHE_DIR), 'tts');
const installer = process.platform === 'win32' ? new VoiceInstaller({ dir: TTS_DIR, root: ROOT, voices: VOICES, log }) : null;
installer?.load();
const gpuCheck = installer ? installer.gpu() : Promise.resolve({ ok: true });   // once per start (Q15)
if (installer) speech.prepare = (v) => installer.ensureVersion(v.version, v.root);
// a voice is usable when it is a URL or its python exists; with none the settings page offers the download
async function usableVoice() {
  for (const v of await speech.voices()) {
    if (v.url) return true;
    try { await fs.access(v.python); return true; } catch {}
  }
  return false;
}
async function installInfo() {
  if (!installer) return { supported: false };
  let freeGB = null;
  try { const st = await fs.statfs(path.parse(installer.dir).root); freeGB = Math.round(st.bavail * st.bsize / 1e9); } catch {}
  return { supported: true, ...installer.info(), defaultDir: TTS_DIR, freeGB, needGB: NEED_GB, needed: !(await usableVoice()),
           gpu: await gpuCheck };
}
const clients = new Set();
let serial = 0;

const body = (req) => new Promise((resolve, reject) => {
  let b = ''; req.on('data', (d) => { b += d; if (b.length > 50e6) reject(new Error('too large')); });
  req.on('end', () => resolve(b)); req.on('error', reject);
});
const send = (res, code, type, data) => { res.writeHead(code, { 'content-type': type, 'cache-control': 'no-store' }); res.end(data); };
const json = (res, obj, code = 200) => send(res, code, 'application/json', JSON.stringify(obj));

function broadcast(event, data) {
  const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
  for (const c of clients) c.write(msg);
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://x');
  try {
    if (req.method === 'GET' && url.pathname === '/') return send(res, 200, 'text/html; charset=utf-8', await fs.readFile(PAGE));
    if (req.method === 'GET' && url.pathname === '/player/') return send(res, 200, 'text/html; charset=utf-8', await fs.readFile(PLAYER));
    if (req.method === 'POST' && url.pathname === '/cloth') {
      const b = JSON.parse(await body(req) || '{}');
      if (b.model && b.model !== MODEL_NAME) return json(res, { error: `cloth only for the app's model ${MODEL_NAME}` }, 503);
      try { return json(res, { points: await askCloth(b.motion ?? null, b.frames ?? []) }); }
      catch (e) { return json(res, { error: String(e.message ?? e) }, 503); }
    }
    if (req.method === 'GET' && url.pathname === '/model/info') return json(res, { name: MODEL_NAME, path: MODEL_PATH, measured: await measured() });
    if (req.method === 'GET' && url.pathname === '/model.vrm') return send(res, 200, 'model/gltf-binary', await fs.readFile(MODEL_PATH));
    // the standing rest pose of THIS model (shown before any motion; 2026-11-13: a fresh install that no AI app had
    // started yet had no rest.json, the MCP server wrote it, and the character stood in a T-pose; it was also the
    // last model's rest after a model change). Solved and cached like the rest lists.
    if (req.method === 'GET' && url.pathname === '/motion/rest') {
      const job = 'default:rest:stand:0:null';
      if (!idleJobs.has(job)) idleJobs.set(job, idleFor('stand', 0, 'rest', 'default', null).catch((e) => { idleJobs.delete(job); throw e; }));
      try {
        const id = (await idleJobs.get(job))?.[0]?.id;
        if (id) return send(res, 200, 'application/json', await fs.readFile(motionFile(id)));
      } catch (e) { log('rest pose failed', e.message ?? e); }
    }
    if (req.method === 'GET' && url.pathname.startsWith('/motion/')) {
      try { return send(res, 200, 'application/json', await fs.readFile(motionFile(url.pathname.slice(8)))); }
      catch { return json(res, { error: 'no such motion' }, 404); }
    }
    if (req.method === 'POST' && url.pathname === '/fit') {
      const { skeleton, fit } = JSON.parse(await body(req));
      const err = await saveFit(skeleton, fit);
      if (err) return json(res, { error: err }, 400);
      log(`measured ${MODEL_NAME}: hips ${skeleton.hips[1]} m`);
      return json(res, { ok: true });
    }
    if (req.method === 'POST' && url.pathname === '/play') {
      const msg = { ...JSON.parse(await body(req)), serial: ++serial };
      const sp = speech.say(msg.serial, msg.say, msg.emotion);
      if (sp) msg.speech = sp;
      broadcast('play', msg);
      log(`play ${msg.id} to ${clients.size} window(s): ${String(msg.say ?? '').slice(0, 40)}`);
      return json(res, { ok: true, windows: clients.size });
    }
    // built-in chat (the API version, chat.mjs): the app posts {text, cfg, system}; the key comes with each request
    // from the app (kept in Electron safeStorage), never stored here
    if (url.pathname.startsWith('/chat')) {
      const b = req.method === 'POST' ? JSON.parse(await body(req) || '{}') : {};
      if (url.pathname === '/chat' && req.method === 'POST') return json(res, await chat.send(b.text, b.cfg ?? {}, String(b.system ?? '')));
      if (url.pathname === '/chat/history') return json(res, await chat.history());
      if (url.pathname === '/chat/clear' && req.method === 'POST') { await chat.clear(); return json(res, await chat.history()); }
      if (url.pathname === '/chat/test' && req.method === 'POST') return json(res, await chat.test(b.cfg ?? {}));
      if (url.pathname === '/chat/models' && req.method === 'POST') return json(res, await chat.models(b.cfg ?? {}));
    }
    // AI programs that started the MCP server (server.mjs posts its client's name; Q13b), kept in the config folder
    if (url.pathname === '/client' || url.pathname === '/clients') {
      const f = path.join(CONFIG_DIR, 'clients.json');
      let all = {};
      try { all = JSON.parse(await fs.readFile(f, 'utf-8')); } catch { /* none yet */ }
      if (req.method === 'POST') {
        const b = JSON.parse(await body(req) || '{}');
        if (b.name) { all[String(b.name).slice(0, 80)] = { version: String(b.version ?? '').slice(0, 40), at: Date.now() }; }
        await fs.mkdir(CONFIG_DIR, { recursive: true }); await fs.writeFile(f, JSON.stringify(all, null, 1));
        log(`client ${b.name}`);
      }
      return json(res, all);
    }
    if (url.pathname === '/state') {           // where she is: posture + wall (the page posts it, the MCP server reads it)
      if (req.method === 'POST') { const b = JSON.parse(await body(req) || '{}'); place = { posture: POSTURES.includes(b.posture) ? b.posture : 'stand', wall: ['left', 'right'].includes(b.wall) ? b.wall : null, ledge: !!b.ledge }; log('state', JSON.stringify(place)); }
      return json(res, place);
    }
    if (req.method === 'GET' && (url.pathname === '/idle' || url.pathname === '/touch' || url.pathname === '/rest')) {
      const pose = url.searchParams.get('pose');
      const libName = pose && POSE_ACTS[pose] ? pose : url.pathname.slice(1);
      const wall = ['left', 'right'].includes(url.searchParams.get('wall')) ? url.searchParams.get('wall') : null;
      const posture = url.searchParams.get('posture') ?? 'stand';
      if (!POSTURES.includes(posture)) return json(res, { error: 'unknown posture' }, 400);
      const turn = posture === 'stand' ? Math.round(Number(url.searchParams.get('turn') ?? 0) / 15) * 15 : 0;
      const setName = url.searchParams.get('set') || 'default';
      if (!(await setNames()).includes(setName)) return json(res, { error: 'unknown set' }, 400);
      const job = `${setName}:${libName}:${posture}:${turn}:${wall}`;
      if (!idleJobs.has(job)) idleJobs.set(job, idleFor(posture, turn, libName, setName, wall).catch((e) => { idleJobs.delete(job); throw e; }));
      return json(res, await idleJobs.get(job));
    }
    if (req.method === 'GET' && url.pathname === '/sets') {          // the personality sets (sets/<name>.json): name + title per language
      const out = [];
      for (const n of await setNames()) { const s = await readSet(n); out.push({ name: n, title: s.title ?? { zh: n, en: n, ja: n } }); }
      return json(res, out);
    }
    if (url.pathname === '/settings') {
      if (req.method === 'POST') return json(res, await speech.set(JSON.parse(await body(req))));
      return json(res, await speech.info());
    }
    // reference clips per emotion (2026-11-12): the settings page reads / changes them, plays clips and samples
    if (url.pathname.startsWith('/voice/')) {
      const q = Object.fromEntries(url.searchParams);
      const b = req.method === 'POST' ? JSON.parse(await body(req) || '{}') : {};
      try {
        if (url.pathname === '/voice/install' && req.method === 'GET') return json(res, await installInfo());
        if (url.pathname === '/voice/install' && installer) {
          if (installer.info().status !== 'running') {
            installer.dir = b.dir || TTS_DIR;
            installer.run().then(async (r) => { if (r.status === 'installed') await speech.set({}); });
          }
          return json(res, await installInfo());
        }
        if (url.pathname === '/voice/install/cancel' && installer) { installer.cancel(); return json(res, await installInfo()); }
        if (url.pathname === '/voice/install/remove' && installer) {
          // voice off, the environment gone, and the default voice unregistered (it pointed into the removed folder)
          await speech.set({ voice: false });
          installer.dir = b.dir || installer.dir;
          await installer.remove();
          try { const all = JSON.parse(await fs.readFile(VOICES, 'utf-8')); all.voices = all.voices.filter((v) => v.name !== 'default'); await fs.writeFile(VOICES, JSON.stringify(all, null, 2)); } catch {}
          return json(res, await installInfo());
        }
        if (url.pathname === '/voice/refs') return json(res, await speech.refTable(q.voice));
        if (url.pathname === '/voice/ref') return json(res, await speech.setRef(b.voice, b.emotion, b.wav ?? null));
        if (url.pathname === '/voice/clip') return json(res, await speech.addClip(b.voice, b.file, b.group));
        if (url.pathname === '/voice/text') { await speech.editClipText(b.voice, b.wav, b.text); return json(res, await speech.refTable(b.voice)); }
        if (url.pathname === '/voice/add') return json(res, await speech.addVoice(b));
        if (url.pathname === '/voice/audio') {          // only a clip of that voice's pool
          const t = await speech.refTable(q.voice);
          if (!t?.pool.some((c) => c.wav === q.wav)) return json(res, { error: 'not a clip of this voice' }, 404);
          return send(res, 200, /\.flac$/i.test(q.wav) ? 'audio/flac' : 'audio/wav', await fs.readFile(q.wav));
        }
        if (url.pathname === '/voice/preview') return send(res, 200, 'audio/wav', await speech.preview(q.voice, q.wav));
      } catch (e) { return json(res, { error: String(e.message ?? e) }, 400); }
    }
    if (req.method === 'GET' && url.pathname.startsWith('/speech/')) {
      const [sr, k] = url.pathname.slice(8).split('/').map(Number);
      const job = speech.audio(sr, k);
      const wav = job ? await job : null;
      if (!wav) { res.writeHead(204); res.end(); return; }
      return send(res, 200, 'audio/wav', wav);
    }
    if (req.method === 'POST' && url.pathname === '/quit') {         // the desktop app stops its server
      json(res, { ok: true });
      log('quit');
      await speech.stop(); await chat.close();
      setTimeout(() => process.exit(0), 100);
      return;
    }
    if (req.method === 'GET' && url.pathname === '/events') {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-store', connection: 'keep-alive' });
      res.write(': hello\n\n');
      clients.add(res);
      req.on('close', () => clients.delete(res));
      return;
    }
    send(res, 404, 'text/plain', 'not found');
  } catch (e) {
    log('error', e.message);
    if (!res.headersSent) json(res, { error: String(e.message ?? e) }, 500);
  }
});
setInterval(() => { for (const c of clients) c.write(': ping\n\n'); }, 20000);     // keep idle connections open

for (const sig of ['SIGTERM', 'SIGINT']) process.on(sig, async () => { await speech.stop(); process.exit(0); });
server.on('error', (e) => { log(e.code === 'EADDRINUSE' ? `port ${COMPANION_PORT} is in use (already running?)` : e.message); process.exit(1); });
pruneIdle().catch((e) => log('prune', e.message));
server.listen(COMPANION_PORT, '127.0.0.1', () => {
  log(`model ${MODEL_PATH}; open http://localhost:${COMPANION_PORT}/`);
  // the AI apps' MCP servers find this port here (model.mjs companionPort); only the app's own companion writes it
  if (process.env.HSIN_PORT_FILE === '1') fs.mkdir(CONFIG_DIR, { recursive: true }).then(() => fs.writeFile(PORT_FILE, JSON.stringify({ port: COMPANION_PORT, pid: process.pid }))).catch((e) => log('port file', e.message));
});

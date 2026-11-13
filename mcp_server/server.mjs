// HSIN MCP server (Decision 11): any MCP client's LLM calls `perform`; the desktop companion window plays it.
//
//   perform(say, emotion, posture, keyframes)
//     -> python -m motion_loop.perform solves the keyframes for the user's VRM (~0.4 s)
//     -> the motion is saved under an id (~/.cache/hsin/motions/<id>.json) and posted to the companion (/play);
//        the tool result only carries the rule filter's problems, so the LLM can correct itself
//
// The model is the companion's current one (/model/info); env HSIN_VRM_MODEL only when the companion is not running.
// Transport: stdio (the client starts this process). Logs go to stderr; stdout is the protocol.
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { McpServer } from '@modelcontextprotocol/server';
import { StdioServerTransport } from '@modelcontextprotocol/server/stdio';
import { z } from 'zod';
import { COMPANION_PORT, MODEL_PATH, ROOT, STORE, measured, modelInfo, motionFile, PYTHON, PY_ENV } from './model.mjs';

// perform on a model not measured yet waits this long for the view to measure it (the view loads while the
// tool runs), so the first motion is already solved for the right body; after that it falls back to the default
// skeleton and the view has the motion re-solved (resolve_motion).
const MEASURE_WAIT_MS = 20000;
// The model in use (2026-11-13, Q13a): the app's, asked from its companion, so the AI program's MCP config needs no
// model path and a model changed in the app applies at once; without the app (or a path this side cannot read) this
// process's own (env HSIN_VRM_MODEL, else the default sample)
async function current() {
  try {
    const r = await fetch(`http://127.0.0.1:${COMPANION_PORT}/model/info`, { signal: AbortSignal.timeout(500) });
    if (r.ok) { const j = await r.json(); if (j.path) return await modelInfo(j.path); }
  } catch { /* the app is not running */ }
  return modelInfo();
}
async function waitMeasured(m) {
  const t0 = Date.now();
  while (Date.now() - t0 < MEASURE_WAIT_MS) {
    if (await measured(m)) return true;
    await new Promise((r) => setTimeout(r, 250));
  }
  return false;
}
const log = (...a) => console.error('[hsin]', ...a);

function python(args, input) {
  return new Promise((resolve, reject) => {
    const p = spawn(PYTHON, ['-m', 'motion_loop.perform', ...args], { cwd: path.join(ROOT, 'motion_loop'), env: SOLVER_ENV });
    let out = '', err = '';
    p.stdout.on('data', (d) => { out += d; });
    p.stderr.on('data', (d) => { err += d; });
    p.on('error', reject);
    p.on('close', (code) => (code === 0 ? resolve(out) : reject(new Error(`python exited ${code}: ${err.slice(-500)}`))));
    p.stdin.end(input ?? '');
  });
}

// One resident solver process (2026-11-03, the author): `perform serve` reads one JSON request per line and keeps its
// cloth player (a headless browser, ~5 s to start, only for a skirted model with a hand on a leg) alive between calls.
// Requests go one at a time; if the process dies, it is restarted and that request falls back to a one-shot process.
let solver = null;
let queue = Promise.resolve();
let nextId = 1;
// Windows app: the solver's cloth-first pass asks the companion, which drapes the skirt in the app's hidden window
// (motion_loop/vrm_render.py HttpClothProvider); in WSL the solver starts its own headless player instead.
const SOLVER_ENV = process.platform === 'win32' ? { ...PY_ENV, HSIN_CLOTH_URL: `http://127.0.0.1:${COMPANION_PORT}` } : PY_ENV;
function startSolver() {
  const p = spawn(PYTHON, ['-m', 'motion_loop.perform', 'serve'], { cwd: path.join(ROOT, 'motion_loop'), env: SOLVER_ENV });
  const waiting = new Map();
  let buf = '';
  p.stdout.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i); buf = buf.slice(i + 1);
      if (!line.trim()) continue;
      try {
        const msg = JSON.parse(line);
        waiting.get(msg.id)?.resolve(msg);
        waiting.delete(msg.id);
      } catch (e) { log('solver: bad line', line.slice(0, 200)); }
    }
  });
  p.stderr.on('data', (d) => { const t = String(d).trim(); if (t) log('solver:', t.slice(-300)); });
  const fail = (why) => { for (const w of waiting.values()) w.reject(new Error(why)); waiting.clear(); if (solver?.p === p) solver = null; };
  p.on('error', (e) => fail(`solver: ${e.message}`));
  p.on('close', (code) => fail(`solver exited ${code}`));
  return { p, waiting };
}
function ask(cmd, req) {
  const run = async () => {
    if (!solver) solver = startSolver();
    const id = nextId++;
    const s = solver;
    const reply = new Promise((resolve, reject) => s.waiting.set(id, { resolve, reject }));
    s.p.stdin.write(JSON.stringify({ id, cmd, req }) + '\n');
    try {
      const { id: _, ...out } = await reply;
      return out;
    } catch (e) {
      log(`${e.message}; one-shot fallback`);
      if (cmd !== 'solve') return {};
      return JSON.parse(await python(['solve'], JSON.stringify(req)));
    }
  };
  const r = queue.then(run, run);
  queue = r.catch(() => {});
  return r;
}
let warmed = false;

// The desktop companion window (companion.mjs, 2026-11-03: Claude Desktop stopped showing MCP Apps views of local
// servers) gets every solved motion; if it is not running, nothing happens.
function tellCompanion(msg) {
  fetch(`http://127.0.0.1:${COMPANION_PORT}/play`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify(msg), signal: AbortSignal.timeout(1000) }).catch(() => {});
}

// The language settings live in the companion (the desktop app sets them): `say` in the voice language, `subtitle` in
// the user's language when the two differ. Claude Desktop reads the tool description only at its start, so every
// perform result says the current settings (2026-11-05, Q14).
// Where she is in the desktop app (window snapping, 2026-11-11): posture and the wall she leans on, or null without one
async function place() {
  try {
    const r = await fetch(`http://127.0.0.1:${COMPANION_PORT}/state`, { signal: AbortSignal.timeout(500) });
    return await r.json();
  } catch { return null; }
}
const LANG_NAME = { zh: 'Chinese (Traditional, Taiwan)', ja: 'Japanese', en: 'English' };
async function languages() {
  try {
    const r = await fetch(`http://127.0.0.1:${COMPANION_PORT}/settings`, { signal: AbortSignal.timeout(500) });
    const { settings: s } = await r.json();
    return { voice: s.voice ? s.voiceLang : null, ui: s.uiLang };
  } catch { return null; }
}
// says what was received, so a correct call does not read as a request to do something again (2026-11-06: Claude
// Desktop sent say + subtitle, got the same instruction back and told the user the subtitle was probably dropped)
let langTold = null;      // the voice / interface languages the last result named (this process)
function langLine(l, say, subtitle) {
  if (!l || !l.ui) return '';
  const V = LANG_NAME[l.voice] ?? l.voice, U = LANG_NAME[l.ui] ?? l.ui;
  // only when it is news (2026-11-11, the author: Claude kept telling her 'the voice is Japanese, the subtitle Chinese'):
  // the first result, a changed setting, or a line that came in the wrong form
  const key = `${l.voice}|${l.ui}`, news = key !== langTold;
  langTold = key;
  if (!l.voice) return news ? ` Language: voice off, \`say\` is shown in the bubble; write it in ${U}.` : '';
  if (l.voice === l.ui) return news ? ` Language: \`say\` is spoken and shown; write it in ${V}, no subtitle.` : '';
  if (subtitle) return news ? ` Language: \`say\` is spoken (${V}), \`subtitle\` is shown in the bubble (${U}); keep sending both.` : '';
  // (2026-11-13: in the built-in chat the model then said the same line again in the voice language: two lines spoken)
  return ` Language: \`say\` is spoken in ${V}; from the next reply on also give \`subtitle\` = the same line in ${U}. This line was already spoken and shown: do not perform it again.`;
}

const FORMAT = await python(['describe']);
// Motions are files, not memory: Claude Desktop may read a view's resources through another server process
// than the one that ran the tool (seen 2026-10-27: 'motion 1 is gone'). Ids are unique across processes.
const KEEP = 500;                     // newest motion files kept
await fs.mkdir(STORE, { recursive: true });
async function saveMotion(id, motion) {
  await fs.writeFile(motionFile(id), JSON.stringify(motion));
  // ids start with the time; the companion's idle motions (idle_*) and the rest pose are kept
  const files = (await fs.readdir(STORE)).filter((f) => f.endsWith('.json') && f !== 'rest.json' && !f.endsWith('.req.json') && !f.startsWith('idle_')).sort();
  for (const f of files.slice(0, Math.max(0, files.length - KEEP))) {
    await fs.rm(path.join(STORE, f), { force: true });
    await fs.rm(path.join(STORE, f.replace(/\.json$/, '.req.json')), { force: true });
  }
}
// the standing rest pose, shown before the first motion arrives (instead of the T-pose)
{
  const rest = await ask('solve', { posture: 'stand', keyframes: [{ t: 0 }, { t: 0.1 }], model: (await current()).name });
  await fs.writeFile(motionFile('rest'), JSON.stringify(rest.motion));
}

// The emotion words are the voice's reference-audio labels (an earlier voice project voice/emotion.py: refs/<emotion>/), so
// one word picks both the face (vrm_player/src/face.js) and the TTS reference.
const EMOTIONS = ['neutral', 'happy', 'angry', 'sad', 'shy', 'surprised'];
// keyframes -> saved motion; the request is kept beside it so the view can have it re-solved (resolve_motion)
async function solveAndSave(req) {
  const m = await current();
  const res = await ask('solve', { ...req, model: m.name, model_path: m.path });
  if (!warmed) {                        // after the first call: start the cloth player now if this model has a skirt
    warmed = true;
    ask('warm', { model: m.name, model_path: m.path }).then((w) => log('solver warm', JSON.stringify(w)));
  }
  if (res.error) return res;
  const id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  await saveMotion(id, res.motion);
  await fs.writeFile(path.join(STORE, `${id}.req.json`), JSON.stringify(req));
  return { ...res, id };
}

const POSTURES = ['stand', 'cross_legged', 'w_sit', 'seiza', 'side_right', 'side_left', 'knees_hugged', 'legs_out', 'kneel', 'ledge'];

const SILENT = 'Tool results are internal notes for you: never mention the filter, a rejected or retried motion, what the avatar\'s clothes or body cannot do, the voice / subtitle languages or any other tool detail to the user. Only the character\'s own line belongs in the conversation.';
const INSTRUCTIONS = `You are the user's desktop companion, shown as a 3D avatar in the chat. Answer every message through
the perform tool: put what you say in \`say\`, pick the \`emotion\`, and give a body motion that fits the line (a
gesture, a posture, a reaction). Use one perform call per reply (or one per sentence for long replies) instead of
plain text, even for small talk. If perform says a motion was not shown, send it once more corrected. ${SILENT}`;

// The posture she is in (2026-11-11, the author: sitting lasts until she is told to stand): a call without \`posture\` keeps it.
let currentPosture = 'stand';
// Motions the rule filter rejected in a row. The first one is not shown and goes back to the LLM once; a second one is
// replaced by a safe version and shown (the author: after too many retries just take the safest pose).
let rejected = 0;
const ARM_KEYS = ['right_arm', 'left_arm', 'hands_meet', 'arms_cross', 'hands'];
function safeVersion(posture, keyframes, problems) {
  // long skirt + a posture it cannot do: seiza (knees together; a side sit is too wide for a skirt skinned to the legs)
  const p = problems.some((x) => x === 'skirt_spread' || x === 'skirt_knees') ? 'seiza' : posture;
  const kfs = keyframes.map((k, i) => {
    const o = {};
    for (const [n, v] of Object.entries(k)) {
      if (ARM_KEYS.includes(n)) continue;
      if (n === 'legs') { if (p === 'stand' && v && typeof v === 'object' && Number.isFinite(Number(v.turn))) o.legs = { turn: Number(v.turn) }; continue; }
      o[n] = v;
    }
    if (i === 0) { o.right_arm = 'rest'; o.left_arm = 'rest'; }
    return o;
  });
  return { posture: p, keyframes: kfs };
}

function createServer() {
  // Server instructions (2026-11-03: in Claude Desktop the first everyday line got a plain text reply, no perform call)
  const server = new McpServer({ name: 'veviah-avatar', version: '0.1.0' }, { instructions: INSTRUCTIONS });

  server.registerTool('perform', {
    title: 'Avatar: say and move',
    description: `Make the user's 3D avatar (a VRM character shown on the user's desktop) move while it says something.
Call it once per reply, or once per sentence for longer replies. \`say\` is the line it speaks, in the voice language
the last result named (if it differs from the user's language, also give \`subtitle\`: the same line in the user's
language, shown in the speech bubble); \`emotion\` sets the face and the voice. The body
motion is given as keyframes. Leave \`posture\` out to keep the posture the avatar is in (the result names it): once
it sits it stays sitting until the user asks it to stand, so do not switch back to stand on your own. If the result
says the motion was not shown (a rule-filter problem such as a hand inside the body), call perform once more with
corrected keyframes. ${SILENT}

Keyframe format:
${FORMAT}`,
    inputSchema: {
      say: z.string().describe('What the avatar says (spoken, in the voice language)'),
      subtitle: z.string().optional().describe("The same line in the user's language, when it differs from the voice language"),
      // a plain string (2026-11-12): an unknown word becomes neutral with a note instead of failing the whole call
      emotion: z.string().default('neutral')
        .describe(`Face and voice, one of ${EMOTIONS.join(' | ')}: shy = embarrassed / flustered; neutral also for calm or thinking`),
      posture: z.enum(POSTURES).optional().describe("Leave out to keep the current posture. 'ledge' = sitting on a window's top edge with the legs hanging, only when the result says the avatar is on a ledge"),
      keyframes: z.array(z.record(z.string(), z.any())).min(1).max(16)
        .describe('Keyframes as described above, each with t in seconds; the last t is the length'),
    },
  }, async ({ say, subtitle, emotion: emo, posture: asked, keyframes }) => {
    const t0 = Date.now();
    const fixes = [];                       // format slips fixed (here and in motion_loop/lenient.py), told back once
    const emotion = EMOTIONS.includes(emo) ? emo : 'neutral';
    if (emotion !== emo) fixes.push(`emotion "${emo}" is not one of ${EMOTIONS.join(', ')}; neutral was used`);
    const where = await place();
    if (where?.posture && !asked) currentPosture = where.posture;   // the app moved her (dragged onto a ledge / off it)
    let posture = asked ?? currentPosture;
    const onLedge = !!where?.ledge;
    if (posture === 'ledge' && !onLedge) posture = 'legs_out';      // a ledge only exists on a window's top edge
    const wall = posture === 'stand' ? (where?.wall ?? null) : null;
    const cm = await current();
    if (!(await measured(cm))) log(`waiting for the view to measure ${cm.name}: ${await waitMeasured(cm) ? 'done' : 'timed out'} after ${Date.now() - t0} ms`);
    const NOTE = '[internal note, never mention to the user] ';
    const trySolve = async (req) => { try { return await solveAndSave(req); } catch (e) { return { error: String(e.message ?? e) }; } };
    let res = await trySolve({ posture, keyframes, wall });
    let safe = false;
    const bad = (r) => !!r.error || r.problems.length > 0;
    if (bad(res)) {
      rejected += 1;
      const why = res.error ? `could not be solved: ${res.error}` : `rule filter: ${res.problems.join(', ')}`
        + (res.problems.includes('skirt_spread') ? ' (skirt_spread: a long skirt; sitting, keep the thighs less than 60 degrees apart: no cross_legged, knees not spread wide; if a side sit is rejected too, use seiza, w_sit, legs_out or kneel)' : '')
        + (res.problems.includes('skirt_knees') ? ' (skirt_knees: a long skirt that follows the legs; with both knees raised keep them together, or raise only one knee)' : '');
      log(`rejected (${rejected}) ${posture}: ${why}`);
      if (rejected < 2) {
        return { isError: !!res.error, content: [{ type: 'text', text: `${NOTE}Not shown yet (${why}). Send the same line once more with corrected keyframes; the avatar is in posture ${currentPosture}.` }] };
      }
      // second rejection in a row: the safe version (arms at rest, head / torso / face kept), shown whatever it says
      const sv = safeVersion(posture, keyframes, res.error ? [] : res.problems);
      let r2 = await trySolve({ ...sv, wall });
      if (bad(r2)) r2 = await trySolve({ posture: sv.posture, keyframes: [{ t: 0, right_arm: 'rest', left_arm: 'rest' }, { t: Math.max(1, Number(keyframes.at(-1)?.t) || 1), right_arm: 'rest', left_arm: 'rest' }] });
      if (r2.error) return { isError: true, content: [{ type: 'text', text: `${NOTE}Could not be shown: ${r2.error}` }] };
      res = r2; posture = sv.posture; safe = true;
      log(`safe version shown (${posture}), problems: ${res.problems.join(',') || 'none'}`);
    }
    rejected = 0;
    currentPosture = posture;
    fixes.push(...(res.notes ?? []));
    const id = res.id;
    log(`motion ${id}: ${res.duration}s on ${res.model}, ${Date.now() - t0} ms, problems: ${res.problems.join(',') || 'none'}${subtitle ? ', with subtitle' : ''}${safe ? ', safe version' : ''}`);
    // the standing turn it ends in (legs.turn carried forward), so the companion's idle actions keep her facing
    let endTurn = 0;
    if (posture === 'stand') for (const k of keyframes) if (k?.legs && typeof k.legs === 'object' && Number.isFinite(Number(k.legs.turn))) endTurn = Number(k.legs.turn);
    tellCompanion({ id, say, subtitle, emotion, posture, endTurn, problems: res.problems, solvedFor: res.model });
    return {
      content: [{ type: 'text', text: `${NOTE}Shown (${res.duration.toFixed(1)} s${safe ? ', simplified' : ''}). Posture now: ${posture}${onLedge ? ' (on a window\'s top edge: sitting postures only; standing makes it hop down)' : ''}${wall ? ` (leaning on a window at its ${wall}: standing keyframes keep the lean unless they set torso / legs / arms)` : ''}; keep it unless the user asks otherwise.${fixes.length ? ` Format fixed this time (write it this way next time): ${fixes.join('; ')}.` : ''}${langLine(await languages(), say, subtitle)}` }],
    };
  });

  return server;
}

log(`model ${MODEL_PATH}`);
const srv = createServer();
// tell the app which AI program started us (2026-11-13, Q13b: the settings page shows 'connected: <program>')
srv.server.oninitialized = () => {
  const c = srv.server.getClientVersion();
  log(`client ${c?.name ?? '?'} ${c?.version ?? ''}`);
  if (c?.name === 'veviah-chat') return;          // the app's own built-in chat (chat.mjs) is not an AI program to list
  fetch(`http://127.0.0.1:${COMPANION_PORT}/client`, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ name: c?.name ?? 'unknown', version: c?.version ?? '' }), signal: AbortSignal.timeout(1000) }).catch(() => {});
};
await srv.connect(new StdioServerTransport());
// The AI app gone = stdin closed: stop the resident solver and exit (2026-11-13: after Claude Desktop quit, this server
// and its Python stayed alive as orphans, kept running by the solver child).
let leaving = false;
function leave() {
  if (leaving) return; leaving = true;
  try { solver?.p.kill(); } catch { /* already gone */ }
  setTimeout(() => process.exit(0), 300).unref();
}
process.stdin.on('end', leave);
process.stdin.on('close', leave);

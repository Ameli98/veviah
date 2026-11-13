// The one set of idle actions and touch reactions (sets/default.json; personality sets removed 2026-11-11): /sets lists
// it, /touch and /idle serve its entries, and the page plays a slot's versions (name, name~2, ...) in turn.
// Run: node test/sets.mjs [model]
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = '8796', base = `http://127.0.0.1:${PORT}`;
const model = process.argv[2];
let bad = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) bad++; };
const env = { ...process.env, HSIN_COMPANION_PORT: PORT };
if (model) env.HSIN_VRM_MODEL = path.join(HERE, `../../vrm_player/public/models/${model}.vrm`);
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'inherit', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const get = async (u) => { const r = await fetch(base + u); return { status: r.status, body: await r.json() }; };

const sets = (await get('/sets')).body;
check(sets.length === 1 && sets[0].name === 'default' && sets[0].title?.zh, `sets: ${sets.map((s) => s.name).join(' ')}`);
check((await get('/touch?posture=stand&set=nope')).status === 400, 'unknown set -> 400');
const st = (await get('/touch?posture=stand')).body, names = st.map((a) => a.name);
for (const n of ['head_pat', 'head_pat~2', 'cheek_poke_l', 'cheek_poke_r', 'hand_r', 'hand_r~2', 'hand_l~2', 'shoulder_l', 'legs', 'shy_cover', 'shy_cover~2', 'shy_cover~3', 'shy_skirt', 'shy_bottom', 'shy_bottom~2'])
  check(names.includes(n), `stand touch has ${n}`);
const sit = (await get('/touch?posture=w_sit')).body;
check(sit.some((a) => a.name === 'head_pat') && sit.some((a) => a.name === 'hand_r~2'), `sit touch: ${sit.map((a) => a.name).join(' ')}`);
const idle = (await get('/idle?posture=stand')).body, idleSit = (await get('/idle?posture=w_sit')).body;
check(idle.length >= 5, `stand idle: ${idle.map((a) => a.name).join(' ')}`);
check(idleSit.length === 1 && idleSit[0].name === 'finger_to_lips', `sit idle: ${idleSit.map((a) => a.name).join(' ')}`);

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 360, height: 600 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/?idlegap=600,601`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 60000 });
await page.waitForTimeout(800);
const map = await page.evaluate(() => {
  const out = {};
  for (let y = 4; y < innerHeight; y += 6) for (let x = 4; x < innerWidth; x += 6) {
    const h = window.hsinStage.bodyPartAt(x, y);
    if (!h) continue;
    const k = `${h.part}${h.side ? ':' + h.side : ''}`;
    (out[k] ??= []).push([x, y]);
  }
  return Object.fromEntries(Object.entries(out).map(([k, v]) => [k, v[Math.floor(v.length / 2)]]));
});
const click = (k) => page.evaluate(([x, y]) => window.hsinTouch(x, y), map[k]);
const settle = () => page.waitForFunction(() => !window.hsinStage.state.playing, null, { timeout: 20000 });
// a slot's versions play in turn (2026-11-11: one set; no personality sets)
let r = await click('head'); check(r?.reaction === 'head_pat', `head -> ${r?.reaction}`); await settle();
r = await click('head'); check(r?.reaction === 'head_pat~2', `head again -> ${r?.reaction}`); await settle();
r = await click('head'); check(r?.reaction === 'head_pat', `head a third time -> ${r?.reaction}`); await settle();
r = await click('hand:right'); check(r?.reaction === 'hand_r', `hand -> ${r?.reaction}`); await settle();
r = await click('hand:right'); check(r?.reaction === 'hand_r~2', `hand again -> ${r?.reaction}`); await settle();
r = await click('arm:left'); check(r?.reaction === 'shoulder_l', `arm -> ${r?.reaction}`); await settle();
await browser.close();
comp.kill();
console.log(bad ? `${bad} FAILED` : 'all ok');
process.exit(bad ? 1 : 0);

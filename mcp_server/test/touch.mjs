// Touch feedback in the companion page (2026-11-06): a map of which part is under each pixel (stage.bodyPartAt),
// a click on each part plays its reaction, repeated clicks play the versions in turn, a line playing gets a face-only reaction, a held
// (dragged) body none, and the lifted pose goes on / off. Run: node test/touch.mjs [model]
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = '8795', base = `http://127.0.0.1:${PORT}`;
const model = process.argv[2];
let bad = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) bad++; };
const env = { ...process.env, HSIN_COMPANION_PORT: PORT };
if (model) env.HSIN_VRM_MODEL = path.join(HERE, `../../vrm_player/public/models/${model}.vrm`);
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'inherit', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 360, height: 600 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/?idlegap=600,601`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 60000 });
await page.waitForTimeout(800);
// where each part is: scan the canvas
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
console.log('parts found:', Object.keys(map).sort().join(' '));
for (const want of ['head', 'cheek:left', 'cheek:right', 'chest', 'belly', 'hips', 'arm:left', 'arm:right', 'hand:left', 'hand:right', 'skirt:left', 'legs:right'])
  check(!!map[want], `part ${want} on screen`);
const click = (k) => page.evaluate(([x, y]) => window.hsinTouch(x, y), map[k]);
const settle = () => page.waitForFunction(() => !window.hsinStage.state.playing, null, { timeout: 15000 });
const EXPECT = { head: 'head_pat', 'cheek:left': 'cheek_poke_l', 'cheek:right': 'cheek_poke_r', 'hand:right': 'hand_r', 'arm:left': 'shoulder_l', 'legs:right': 'legs', chest: 'shy_cover', 'skirt:left': 'shy_skirt', hips: 'shy_skirt' };
// (2026-11-11: no belly slot: a belly click is the face alone; skirt / hips = the crotch cover)
for (const [k, name] of Object.entries(EXPECT)) {
  if (!map[k]) continue;
  const r = await click(k);
  const playing = await page.evaluate(() => window.hsinStage.state.playing);
  check(String(r?.reaction).split('~')[0] === name && playing, `click ${k} -> ${r?.reaction} (want ${name})`);
  await settle();
}
if (map.belly) { const rb = await click('belly'); check(rb?.reaction === 'face', `click belly -> ${rb?.reaction} (face only)`); }
// versions in turn (no escalation, 2026-11-08): the head was clicked once above, so the next is head_pat~2, then
// head_pat again; chest clicked many times stays shy_cover
let r;
r = await click('head'); check(r?.reaction === 'head_pat~2', `head 2nd -> ${r?.reaction}`); await settle();
r = await click('head'); check(r?.reaction === 'head_pat', `head 3rd -> ${r?.reaction}`); await settle();
for (let i = 0; i < 5; i++) { r = await click('chest'); await settle(); }
check(String(r?.reaction).startsWith('shy_cover'), `chest x5 -> ${r?.reaction} (a shy_cover version, no turn away)`);
// a line playing: face only
await fetch(`${base}/play`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: 'rest', say: 'test', emotion: 'neutral' }) });
await page.waitForTimeout(400);
await page.evaluate(() => { window.hsinStage.state.playing = true; window.hsinStage.state.idleMotion = false; });
r = await click('belly');
check(r?.reaction === 'face', `click during a line -> ${r?.reaction} (face only)`);
await page.evaluate(() => { window.hsinStage.state.playing = false; });
// held: no reaction; lifted pose on, then off with a landing
await page.evaluate(() => window.hsinStage.setHeld(true));
r = await click('head');
check(r === null, 'no reaction while held');
const hipsUp = await page.evaluate(async () => { await new Promise((r) => setTimeout(r, 600)); return window.hsinStage.state.vrm.humanoid.getNormalizedBoneNode('hips').position.y; });
await page.evaluate(() => window.hsinStage.setHeld(false));
const hipsDown = await page.evaluate(async () => { await new Promise((r) => setTimeout(r, 1000)); return window.hsinStage.state.vrm.humanoid.getNormalizedBoneNode('hips').position.y; });
check(hipsUp - hipsDown > 0.05, `held lifts the hips ${(100 * (hipsUp - hipsDown)).toFixed(1)} cm, landing brings them back`);
await browser.close(); comp.kill();
console.log(bad ? `${bad} FAILED` : 'all ok');
process.exit(bad ? 1 : 0);

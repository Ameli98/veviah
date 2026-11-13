// One size, the camera only moves (2026-11-11, replaces the zooms of 2026-11-07): up to the full-body zoom the body
// stays inside the window when held or jumping; past it (scale > 1.1) parts may be cut, not checked.
// The body stays inside the companion window (2026-11-07, the author Q8-Q10): held (dragged, swinging back and forth)
// and big / tucked jumps, at several zooms incl. past the full-body one. Checks per frame that no opaque pixel
// touches the top / left / right edge (and the bottom while the feet stand in the view), and that the floor is on
// the same pixel row before a jump, during it and after a drag (she lands on the taskbar again).
// Jump motions: ~/hsin_runs/view_check/jump_{big,tuck}.json (python -m motion_loop.perform solve).
// Run: node test/view.mjs [model]
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = '8794', base = `http://127.0.0.1:${PORT}`;
const model = process.argv[2];
let bad = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) bad++; };
const env = { ...process.env, HSIN_COMPANION_PORT: PORT };
if (model) env.HSIN_VRM_MODEL = path.join(HERE, `../../vrm_player/public/models/${model}.vrm`);
const jumps = {};
for (const j of ['big', 'tuck']) jumps[j] = JSON.parse(await fs.readFile(path.join(os.homedir(), `hsin_runs/view_check/jump_${j}.json`))).motion;

const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'inherit', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 372, height: 600 } });   // the app window's aspect (0.62)
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.addInitScript(() => {          // a stand-in for the desktop app's preload, so the page runs in app mode
  const nop = () => {};
  window.hsinApp = { dragStart: nop, dragMove: nop, dragEnd: nop, setSolid: nop, menu: nop, zoom: nop, turn: nop,
                     onJiggle: nop, onSpeech: nop, log: nop };
});
await page.goto(`${base}/?app=1&bg=none&idlegap=600,601`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 60000 });
await page.waitForTimeout(800);
await page.evaluate(() => {
  const c = document.querySelector('canvas'), tmp = document.createElement('canvas'), g = tmp.getContext('2d', { willReadFrequently: true });
  // opaque pixels on each edge of the drawn frame (1 px band)
  window.edges = () => {
    tmp.width = c.width; tmp.height = c.height; g.clearRect(0, 0, c.width, c.height); g.drawImage(c, 0, 0);
    const d = g.getImageData(0, 0, c.width, c.height).data, W = c.width, H = c.height, a = (x, y) => d[4 * (y * W + x) + 3] > 40;
    const out = { top: 0, bottom: 0, left: 0, right: 0 };
    for (let x = 0; x < W; x++) { out.top += a(x, 0); out.bottom += a(x, H - 1); }
    for (let y = 0; y < H; y++) { out.left += a(0, y); out.right += a(W - 1, y); }
    return out;
  };
  window.rec = null;
  (function f() { requestAnimationFrame(f); if (window.rec) { const e = window.edges(); e.ground = window.hsinStage.floorRow(); const H = window.hsinStage._held; e.dbg = [performance.now() | 0, H.w.toFixed(2), (H.viewW ?? 0).toFixed(2), (H.ang * 57.3).toFixed(1), window.hsinStage.state.t.toFixed(2)]; window.rec.push(e); } })();
});
const sum = (r, k) => r.reduce((s, x) => s + x[k], 0);

for (const scale of [0.6, 1.0, 1.1, 1.3, 2.0]) {
  await page.evaluate((s) => window.hsinStage.setScale(s), scale);
  // stand at rest: is the full body in view (feet above the bottom)?
  await page.waitForTimeout(300);
  const rest = await page.evaluate(() => ({ ...window.edges(), ground: window.hsinStage.groundScreen() }));
  const feetIn = rest.ground < 599;   // the floor in view (the soles touch the bottom row: not checked)
  console.log(`scale ${scale} rest: edges ${JSON.stringify(rest)}`);
  // jumps
  for (const [name, m] of Object.entries(jumps)) {
    await page.evaluate((mo) => { window.hsinStage.setMotion(mo, true, 'happy'); window.rec = []; }, m);
    await page.waitForFunction(() => !window.hsinStage.state.playing, null, { timeout: 10000 });
    await page.waitForTimeout(200);
    const r = await page.evaluate(() => { const r = window.rec; window.rec = null; return r; });
    const g = r.map((x) => x.ground);
    if (scale <= 1.1) check(sum(r, 'top') + sum(r, 'left') + sum(r, 'right') === 0,
      `scale ${scale} ${name} jump: inside the window (edge px top ${sum(r, 'top')} left ${sum(r, 'left')} right ${sum(r, 'right')})`);
    if (feetIn) check(Math.abs(g[g.length - 1] - g[0]) <= 1, `scale ${scale} ${name} jump: floor row back after landing (${g[0].toFixed(1)} -> ${g[g.length - 1].toFixed(1)}; during ${Math.min(...g).toFixed(1)}..${Math.max(...g).toFixed(1)} px)`);
  }
  // held: lifted, then the window swings left / right / reverses (window speed in m/s set directly), then released
  await page.evaluate(() => {
    const s = window.hsinStage; s.setMotion(s.state.player.motion, false); s.setHeld(true); window.rec = [];
    const t0 = performance.now();
    window.drive = setInterval(() => { const t = (performance.now() - t0) / 1000;
      s._held.winV = t < 2.4 ? (Math.floor(t / 0.6) % 2 ? -4 : 4) : 0; }, 16);
  });
  await page.waitForTimeout(3200);
  const g0 = rest.ground;
  await page.evaluate(() => { clearInterval(window.drive); window.groundAtRelease = window.hsinStage.groundScreen(); window.hsinStage.setHeld(false); });
  await page.waitForTimeout(1800);
  const r = await page.evaluate(() => { const r = window.rec; window.rec = null; return r; });
  const rel = await page.evaluate(() => window.groundAtRelease);
  const maxAng = await page.evaluate(() => 0);
  const lifted = r.filter((x) => x.dbg[1] === '1.00');      // fully lifted: the feet must be in view too
  if (scale <= 1.1) check(sum(r, 'top') + sum(r, 'left') + sum(r, 'right') + sum(lifted, 'bottom') === 0,
    `scale ${scale} held + swing: inside the window (edge px top ${sum(r, 'top')} left ${sum(r, 'left')} right ${sum(r, 'right')} bottom while lifted ${sum(lifted, 'bottom')}); max swing ${Math.max(...r.map((x) => Math.abs(+x.dbg[3]))).toFixed(1)} deg`);
  for (const x of r) if (x.left + x.right + x.top) console.log('   edge', x.left, x.right, x.dbg.join(' '));
  const after = await page.evaluate(() => window.hsinStage.groundScreen());   // the same measure as before / at release
  if (feetIn) check(Math.abs(rel - g0) <= 1 && Math.abs(after - g0) <= 1,
    `scale ${scale} held: floor reported at release ${rel.toFixed(1)}, after landing ${after.toFixed(1)}, before ${g0.toFixed(1)} px`);
}
await browser.close(); comp.kill();
console.log(bad ? `${bad} FAILED` : 'all ok');
process.exit(bad ? 1 : 0);

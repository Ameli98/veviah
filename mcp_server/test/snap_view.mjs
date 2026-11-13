// Window snapping, the page (2026-11-11): the ledge rest (legs below the floor line, in view) and the wall rest, as
// screenshots for checking by eye. Run: node test/snap_view.mjs MODEL OUTDIR
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const [model = 'test2', out = '.'] = process.argv.slice(2);
const PORT = '8798';
const env = { ...process.env, HSIN_COMPANION_PORT: PORT, HSIN_VRM_MODEL: path.join(HERE, `../../vrm_player/public/models/${model}.vrm`) };
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'ignore', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 420, height: 680 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${PORT}/?idlegap=600,601`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 90000 });
await page.waitForTimeout(1500);
// a translucent box where the window would be (its top edge under her / its side beside her)
const box = (css) => page.evaluate((css) => {
  let d = document.getElementById('fakewin');
  if (!d) { d = document.createElement('div'); d.id = 'fakewin'; document.body.appendChild(d); }
  d.style.cssText = `position:fixed;background:rgba(40,40,60,.45);border:1px solid #223;pointer-events:none;${css}`;
}, css);
for (const [name, t] of [['ledge', { kind: 'ledge' }], ['wall_left', { kind: 'wall', side: 'left' }], ['wall_right', { kind: 'wall', side: 'right' }]]) {
  await box('display:none');
  await page.evaluate((t) => window.hsinSnapTo(t), t);
  await page.waitForTimeout(3500);
  const [g, b] = await page.evaluate(() => [window.hsinStage.groundScreen(), window.hsinStage.bodyBox()]);
  console.log(name, 'floor row', g?.toFixed(0), 'body', b ? `${b.l.toFixed(0)}..${b.r.toFixed(0)} px, shoulders y ${b.sy.toFixed(0)}` : null);
  if (name === 'ledge') await box(`left:40px;right:40px;top:${g}px;bottom:0`);
  else if (t.side === 'left') await box(`left:${b.r}px;right:0;top:120px;bottom:0`);      // her left = the screen's right
  else await box(`left:0;width:${b.l}px;top:120px;bottom:0`);
  await page.waitForTimeout(300);
  await page.screenshot({ path: path.join(out, `snap_${model}_${name}.png`) });
}
await browser.close(); comp.kill();

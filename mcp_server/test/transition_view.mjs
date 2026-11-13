// Posture transitions (2026-11-11): frames every STEP s of each transition, for checking by eye.
// Run: node test/transition_view.mjs MODEL OUTDIR [from:to,...]
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const [model = 'VRM1_Constraint_Twist_Sample', out = '.', list = 'stand:cross_legged,cross_legged:side_left,side_left:stand,stand:seiza,seiza:w_sit,w_sit:legs_out,legs_out:stand,stand:knees_hugged,knees_hugged:kneel,kneel:stand'] = process.argv.slice(2);
const PORT = '8796', STEP = 0.2;
const env = { ...process.env, HSIN_COMPANION_PORT: PORT, HSIN_VRM_MODEL: path.join(HERE, `../../vrm_player/public/models/${model}.vrm`) };
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'ignore', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 300, height: 480 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
const TS = 0.1;                  // the page runs at a tenth of the speed: a screenshot every STEP s of her time
await page.goto(`http://localhost:${PORT}/?idlegap=600,601&timescale=${TS}${process.env.EXTRA ?? ''}`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 90000 });
await page.waitForTimeout(1000);
for (const pair of list.split(',')) {
  const [a, b] = pair.split(':');
  // start from a (warm the cache first: solving a pose the first time takes seconds)
  await page.evaluate(([a, b]) => Promise.all([fetch(`/rest?posture=${a === 'squat' ? 'stand' : a}`), fetch(`/rest?posture=${b}`), fetch('/rest?posture=stand&pose=squat'),
    ...['knees_hugged', 'kneel', 'seiza', 'stand'].map((p) => fetch(`/rest?posture=${p}`))]), [a, b]);
  await page.evaluate((a) => window.hsinTransition(a), a);
  await page.waitForTimeout(800 / TS);
  const done = page.evaluate((b) => window.hsinTransition(b), b);
  const shots = [];
  const t0 = Date.now();
  for (let k = 0; k < 16; k++) {
    await page.waitForTimeout(Math.max(0, k * STEP * 1000 / TS - (Date.now() - t0)));
    shots.push(await page.screenshot());
  }
  const ok = await done;
  const fs = await import('node:fs');
  shots.forEach((png, k) => fs.writeFileSync(path.join(out, `tr_${model}_${a}_${b}_${String(k).padStart(2, '0')}.png`), png));
  console.log(`${a} -> ${b}: ${ok ? 'path' : 'NO PATH (fade)'}`);
}
await browser.close(); comp.kill();

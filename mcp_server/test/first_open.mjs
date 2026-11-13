// Frames of the first seconds after the companion page opens on a fresh install (empty caches). Run: node test/first_open.mjs OUTDIR
// The first seconds after the page opens on a fresh install (empty caches): frames every 150 ms.
import { fileURLToPath } from 'node:url';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises'; import os from 'node:os';
import path from 'node:path';
import { chromium } from 'playwright';
const R = fileURLToPath(new URL('..', import.meta.url)).replace(/\/$/, ''), out = process.argv[2];
const data = await fs.mkdtemp(path.join(os.tmpdir(), 'vfirst-'));
const env = { ...process.env, HSIN_COMPANION_PORT: '8798', HSIN_DATA: data, HSIN_VRM_MODEL: R + '/../vrm_player/public/models/VRM1_Constraint_Twist_Sample.vrm' };
const comp = spawn(process.execPath, [R + '/companion.mjs'], { env, stdio: ['ignore', 'ignore', 'pipe'] });
await new Promise((r) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) r(); }));
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 300, height: 480 } });
const log = [];
page.on('console', (m) => log.push(m.text()));
const t0 = Date.now();
await page.goto('http://localhost:8798/');
// per animation frame: the bend of spine / chest / upperChest / neck / head away from their rest (deg), and which motion
await page.evaluate(() => {
  window.__trace = [];
  const t0 = performance.now();
  const names = ['spine', 'chest', 'upperChest', 'neck', 'head'];
  const step = () => {
    const vrm = window.hsinStage?.state?.vrm;
    if (vrm) {
      const row = [((performance.now() - t0) / 1000).toFixed(2)];
      for (const n of names) {
        const b = vrm.humanoid.getNormalizedBoneNode(n);
        row.push(b ? (2 * Math.acos(Math.min(1, Math.abs(b.quaternion.w))) * 180 / Math.PI).toFixed(1) : '-');
      }
      row.push(window.hsinStage.hasMotion() ? 'M' : '-');
      window.__trace.push(row.join(' '));
    }
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
});
await page.waitForTimeout(25000);
const tr = await page.evaluate(() => window.__trace);
console.log('t spine chest upperChest neck head motion');
for (let i = 0; i < tr.length; i += Math.max(1, Math.floor(tr.length / 120))) console.log(tr[i]);
console.log(log.slice(0, 20).join('\n'));
await browser.close(); comp.kill(); await fs.rm(data, { recursive: true, force: true });

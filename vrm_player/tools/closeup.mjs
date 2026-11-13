// Close-up screenshots of one bone at given times of a motion, for checking hands and wrists by eye.
// Usage: node tools/closeup.mjs <motion.json> <out dir> <bone> <t1,t2,...> [dx,dy,dz camera direction] [distance m]
import { mkdirSync, readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const [motionPath, outDir, bone, times, dir = '0,0,1', dist = '0.6'] = process.argv.slice(2);
const out = resolve(outDir);
mkdirSync(out, { recursive: true });
const motion = JSON.parse(readFileSync(motionPath, 'utf8'));
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5449, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 420, height: 420 }, deviceScaleFactor: 1 });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0`);
await page.evaluate(() => window.hsin.ready);
await page.evaluate((m) => window.hsin.loadMotion(m), motion);
for (const t of times.split(',').map(Number)) {
  await page.evaluate(({ t, bone, d, dist }) => {
    const { state, camera, controls } = window.hsin;
    state.manual = true;
    window.hsin.setTime(t);
    const c = state.vrm.humanoid.getNormalizedBoneNode(bone).getWorldPosition(controls.target.clone());
    const n = Math.hypot(...d);
    controls.target.copy(c);
    camera.position.set(c.x + dist * d[0] / n, c.y + dist * d[1] / n, c.z + dist * d[2] / n);
    controls.update();
    window.hsin.setTime(t);
  }, { t, bone, d: dir.split(',').map(Number), dist: Number(dist) });
  await page.screenshot({ path: `${out}/${bone}_${t.toFixed(2)}.png` });
}
await browser.close();
await server.close();

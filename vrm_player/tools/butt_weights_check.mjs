// Buttock sway weights (limb_push.js buttWeights): the sway set by hand to DY m down (and DZ back) on both sides,
// (loose clothes pushed where it reaches them, limb_push.js), back / side / back-oblique views with and without it,
// per model -> one PNG per model. Also prints how many vertices
// follow it. Usage: node tools/butt_weights_check.mjs OUT_DIR [dy] [dz] [models...]
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdirSync } from 'node:fs';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const OUT = resolve(process.argv[2] ?? 'butt_check'), DY = Number(process.argv[3] ?? -0.04), DZ = Number(process.argv[4] ?? 0);
const MODELS = process.argv.slice(5).length ? process.argv.slice(5) : ['VRM1_Constraint_Twist_Sample', 'test1', 'test2', 'AvatarSample_D'];
mkdirSync(OUT, { recursive: true });
const server = await createServer({ root, server: { port: 5388, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
for (const m of MODELS) {
  const page = await browser.newPage({ viewport: { width: 360, height: 360 } });
  page.on('pageerror', (e) => console.log(m, 'ERR', e.message));
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&jiggle=weak&model=models/${m}.vrm`);
  await page.evaluate(() => window.hsin.ready);
  const info = await page.evaluate(() => {
    const { state } = window.hsin; state.manual = true; state.jiggle.p = null;
    let n = 0, mx = 0;
    state.vrm.scene.traverse((o) => { const a = o.geometry?.attributes?.hsinButtW; if (!a) return; for (let i = 0; i < a.count; i++) { const w = Math.max(a.getX(i), a.getY(i)); if (w > 0.05) n++; mx = Math.max(mx, w); } });
    return { n, mx, hips: state.vrm.humanoid.getRawBoneNode('hips').getWorldPosition(state.vrm.scene.position.clone()).y };
  });
  console.log(m, `${info.n} vertices follow (weight > 0.05), max ${info.mx.toFixed(2)}`);
  const shots = [];
  for (const view of [`orbit:180:5:1.0:${(info.hips - 0.05).toFixed(2)}`, `orbit:90:5:1.0:${(info.hips - 0.05).toFixed(2)}`, `orbit:145:20:1.0:${(info.hips - 0.05).toFixed(2)}`])
    for (const on of [0, 1]) {
      await page.evaluate(([view, on, dy, dz]) => {
        const { state } = window.hsin; window.hsin.frameCamera(view); const u0 = () => state.vrm.scene.position.clone().set(0, 0, 0);
        for (const u of state.vrm.scene.userData.hsinPush.hsinButt.value) u.set(0, on * dy, on * dz);
        window.hsin.setTime(0, 0);
      }, [view, on, DY, DZ]);
      shots.push(await page.screenshot());
    }
  const fs = await import('node:fs');
  shots.forEach((b, k) => fs.writeFileSync(`${OUT}/${m}_${k}.png`, b));
  await page.close();
}
await browser.close(); await server.close();

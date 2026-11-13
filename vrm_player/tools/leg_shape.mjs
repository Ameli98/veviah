// Measures the default model's leg thickness in the T-pose, for the hand-leg clipping check in
// motion_loop/skeleton.py: per 4 cm height slice of the left leg, the centre (x, z) and the 50th / 90th
// percentile distance of the skin vertices from it, per mesh (a skirt shows up as a separate mesh).
// Usage: node tools/leg_shape.mjs [models/X.vrm]
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5296, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
const model = process.argv[2] ? `&model=${process.argv[2]}` : '';
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0${model}`);
const out = await page.evaluate(async () => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm;
  vrm.humanoid.resetNormalizedPose();
  vrm.update(0); vrm.scene.updateMatrixWorld(true);
  const V = vrm.scene.position.constructor;
  const v = new V(), per = {};
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh || /hair|bang|ahoge/i.test(o.name)) return;
    const pts = [];
    for (let i = 0; i < o.geometry.attributes.position.count; i++) {
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      if (v.x > 0.0 && v.x < 0.25 && v.y > 0.08 && v.y < 0.9) pts.push([v.x, v.y, v.z]);
    }
    if (pts.length) per[o.name] = pts;
  });
  const res = {};
  for (const [name, pts] of Object.entries(per)) {
    const bins = new Map();
    for (const p of pts) { const k = Math.round(p[1] / 0.04); (bins.get(k) ?? bins.set(k, []).get(k)).push(p); }
    res[name] = [...bins.entries()].sort((a, b) => b[0] - a[0]).map(([k, ps]) => {
      const cx = ps.reduce((s, p) => s + p[0], 0) / ps.length, cz = ps.reduce((s, p) => s + p[2], 0) / ps.length;
      const d = ps.map((p) => Math.hypot(p[0] - cx, p[2] - cz)).sort((a, b) => a - b);
      const q = (f) => d[Math.min(d.length - 1, Math.floor(f * d.length))];
      return [+(k * 0.04).toFixed(2), +cx.toFixed(3), +cz.toFixed(3), +q(0.5).toFixed(3), +q(0.9).toFixed(3), ps.length];
    });
  }
  return res;
});
console.log(JSON.stringify(out));
await browser.close(); await server.close();

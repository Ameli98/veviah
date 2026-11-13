// Central vertices the 'aa' morph moves, with their vertical move: to tell the upper lip from the lower. node tools/lip_probe.mjs models/A.vrm ...
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5303, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
for (const model of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${model}`);
  const out = await page.evaluate(async () => {
    await window.hsin.ready; const vrm = window.hsin.state.vrm; vrm.scene.updateMatrixWorld(true);
    const aa = vrm.expressionManager?.getExpression?.('aa'); const V = vrm.scene.position.constructor, v = new V(); const rows = [];
    for (const b of aa?._binds ?? []) for (const m of b.primitives ?? []) {
      const d = m.geometry?.morphAttributes?.position?.[b.index]; if (!d) continue;
      let mx = 0; for (let i = 0; i < d.count; i++) mx = Math.max(mx, Math.hypot(d.getX(i), d.getY(i), d.getZ(i)));
      for (let i = 0; i < d.count; i++) { const w = Math.hypot(d.getX(i), d.getY(i), d.getZ(i)); if (w < 0.02 * mx) continue;
        m.getVertexPosition(i, v); v.applyMatrix4(m.matrixWorld); if (Math.abs(v.x) > 0.006) continue;
        rows.push([+v.y.toFixed(4), +v.z.toFixed(4), +(d.getY(i) / mx).toFixed(2), +(w / mx).toFixed(2)]); }
    }
    rows.sort((a, b) => b[0] - a[0]); return { fit: window.hsin.fitBody().mouth, rows };
  });
  console.log(model, JSON.stringify(out.fit)); const seen = new Set();
  for (const r of out.rows) { const k = r.join(','); if (seen.has(k)) continue; seen.add(k); console.log('  y', r[0], 'z', r[1], 'dy', r[2], 'w', r[3]); }
}
await browser.close(); await server.close();

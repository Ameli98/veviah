// Edge lengths of the simulated skirt cloth (skirt.js hsinSkirtVerts) at rest, to choose the subdivision size
// (2026-11-03, test1's long skirt: long edges cut through the bottom when sitting). Usage: node tools/skirt_edges.mjs models/X.vrm
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5307, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${process.argv[2]}`);
const out = await page.evaluate(async () => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm, res = [];
  vrm.scene.updateMatrixWorld(true);
  for (const m of vrm.scene.userData.hsinSkirtVerts ?? []) {
    const keep = new Set(m.idx), L = [], seen = new Set();
    let tris = 0;
    for (const o of [m.o, ...m.others]) {
      const ix = o.geometry.index, pos = o.geometry.attributes.position;
      for (let t = 0; t < ix.count; t += 3) {
        const v = [0, 1, 2].map((k) => ix.getX(t + k));
        if (!v.every((i) => keep.has(i))) continue;
        tris++;
        for (let k = 0; k < 3; k++) {
          const a = v[k], b = v[(k + 1) % 3], key = a < b ? `${a},${b}` : `${b},${a}`;
          if (seen.has(key)) continue; seen.add(key);
          const p = [a, b].map((i) => [pos.getX(i), pos.getY(i), pos.getZ(i)]);
          L.push(Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1], p[0][2] - p[1][2]));
        }
      }
    }
    L.sort((x, y) => x - y);
    const q = (f) => +(L[Math.floor(f * (L.length - 1))] * 100).toFixed(1);
    res.push({ mesh: m.o.name, verts: m.idx.length, tris, edges: L.length, cm: { p10: q(0.1), p50: q(0.5), p90: q(0.9), max: q(1) } });
  }
  return res;
});
console.log(JSON.stringify(out, null, 1));
await browser.close(); await server.close();

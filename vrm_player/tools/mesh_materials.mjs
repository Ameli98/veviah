// Lists a model's meshes with material name, side (0 front, 1 back, 2 double) and triangle count. Usage: node tools/mesh_materials.mjs models/X.vrm

import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const root = fileURLToPath(new URL('..', import.meta.url));
const server = await createServer({ root, server: { port: 5308, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 64, height: 64 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${process.argv[2]}`);
const r = await page.evaluate(async () => {
  await window.hsin.ready;
  const out = [];
  window.hsin.state.vrm.scene.traverse((o) => { if (o.isMesh) { const m = Array.isArray(o.material) ? o.material[0] : o.material;
    out.push([o.name, m.name, m.side, m.color ? m.color.getHexString() : null, m.uniforms?.litFactor?.value?.getHexString?.(), o.geometry.index.count / 3]); } });
  return out;
});
console.log(r.map((x) => x.join(' | ')).join('\n'));
await browser.close(); await server.close();

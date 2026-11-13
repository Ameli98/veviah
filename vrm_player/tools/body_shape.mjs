// Measures a model's torso and head surface in the T-pose, for the hand-body collision check in
// motion_loop/filters.py: per 2 cm height slice, the torso's extent (x half-width, front z, back z) from the
// vertices of every skinned mesh except hair and hands/arms (|x| beyond the shoulder joint is dropped).
// Usage: node tools/body_shape.mjs models/X.vrm   -> JSON {meshes, slices: [[y, xhalf, zfront, zback], ...]}
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5295, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${process.argv[2]}`);
const out = await page.evaluate(async () => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm;
  vrm.humanoid.resetNormalizedPose();
  vrm.update(0); vrm.scene.updateMatrixWorld(true);
  const V = vrm.scene.position.constructor;
  const shoulderX = Math.abs(vrm.humanoid.getNormalizedBoneNode('leftUpperArm').getWorldPosition(new V()).x);
  const bins = new Map(), meshes = [];
  const v = new V();
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const name = (o.name + ' ' + (o.material?.name ?? '')).toLowerCase();
    const skip = /hair|bang|ahoge/.test(name);
    meshes.push([o.name, o.material?.name ?? '', skip]);
    if (skip) return;
    const n = o.geometry.attributes.position.count;
    for (let i = 0; i < n; i++) {
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      if (Math.abs(v.x) > shoulderX * 0.9 || v.y < 0.8 || v.y > 1.6) continue;
      const k = Math.round(v.y / 0.02);
      const b = bins.get(k) ?? [0, Infinity, -Infinity]; b[0] = Math.max(b[0], Math.abs(v.x)); b[1] = Math.min(b[1], v.z); b[2] = Math.max(b[2], v.z);
      bins.set(k, b);
    }
  });
  const slices = [...bins.entries()].sort((a, b) => a[0] - b[0]).map(([k, b]) => [+(k * 0.02).toFixed(2), +b[0].toFixed(3), +b[2].toFixed(3), +b[1].toFixed(3)]);
  return { meshes, shoulderX, slices };
});
console.log(JSON.stringify(out));
await browser.close(); await server.close();

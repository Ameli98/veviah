// Lists a model's skinned meshes: vertex count, which humanoid bones own them, and how many hang from spring
// joints (hair, skirts, ribbons) -- to see where a skirt is and how it moves.
// Usage: node tools/mesh_probe.mjs models/X.vrm
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5303, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${process.argv[2]}`);
const out = await page.evaluate(async () => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm, H = vrm.humanoid;
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const spring = new Set(); for (const j of vrm.springBoneManager?.joints ?? []) spring.add(j.bone);
  const rows = [];
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const sk = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight;
    const owners = {}, springOwners = {}, springJoints = {};
    for (let i = 0; i < o.geometry.attributes.position.count; i++) {
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      let n = o.skeleton.bones[sk.getComponent(i, best)], via = null;
      while (n && !rawToName.has(n)) { if (spring.has(n) && !via) via = n.name; n = n.parent; }
      const name = n ? rawToName.get(n) : '?';
      if (via) { springOwners[name] = (springOwners[name] ?? 0) + 1; const key = via.replace(/[0-9_.]+$/, ''); springJoints[key] = (springJoints[key] ?? 0) + 1; }
      else owners[name] = (owners[name] ?? 0) + 1;
    }
    rows.push({ mesh: o.name, material: o.material?.name ?? (o.material?.[0]?.name), n: o.geometry.attributes.position.count, owners, springOwners,
      springJoints: Object.fromEntries(Object.entries(springJoints).sort((a, b) => b[1] - a[1]).slice(0, 6)) });
  });
  return rows;
});
console.log(JSON.stringify(out, null, 1));
await browser.close(); await server.close();

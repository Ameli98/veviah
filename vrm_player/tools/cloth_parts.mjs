// Lists a model's mesh pieces (triangle-connected components of each skinned mesh) around the hips at rest: vertex
// count, height range, how many are in skirt.js's simulated skirt set, which humanoid bones own them, and whether a
// spring joint moves them. To see which layers of a skirt the cloth simulation leaves out (2026-10-28).
// Usage: node tools/cloth_parts.mjs models/X.vrm
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5306, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${process.argv[2]}`);
const out = await page.evaluate(async () => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm, H = vrm.humanoid;
  vrm.scene.updateMatrixWorld(true);
  const raw = new Map(Object.keys(H.humanBones).map((n) => [H.getRawBoneNode(n), n]).filter(([b]) => b));
  const spring = new Set([...(vrm.springBoneManager?.joints ?? [])].map((j) => j.bone));
  const own = (b) => { let n = b, via = false; while (n && !raw.has(n)) { if (spring.has(n)) via = true; n = n.parent; } return [raw.get(n) ?? '?', via]; };
  const hipY = H.getRawBoneNode('leftUpperLeg').getWorldPosition(new vrm.scene.position.constructor()).y;
  const skirt = new Map((vrm.scene.userData.hsinSkirtVerts ?? []).map((e) => [e.o.geometry.attributes.position, new Set(e.idx)]));
  const res = [], seen = new Set();
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const pos = o.geometry.attributes.position;
    const ix = o.geometry.index; if (!ix) return;
    const key = `${pos.uuid ?? ''}:${ix.uuid ?? ''}:${o.uuid}`;
    if (seen.has(o)) return; seen.add(o);
    const nb = new Map(), used = new Set();
    for (let t = 0; t < ix.count; t += 3) for (let k = 0; k < 3; k++) {
      const a = ix.getX(t + k), b = ix.getX(t + (k + 1) % 3); used.add(a);
      if (!nb.has(a)) nb.set(a, []); nb.get(a).push(b);
    }
    const comp = new Map(); const comps = [];
    for (const i of used) {
      if (comp.has(i)) continue;
      const st = [i], list = [i]; comp.set(i, comps.length);
      while (st.length) { const v = st.pop(); for (const w of nb.get(v) ?? []) if (!comp.has(w)) { comp.set(w, comps.length); st.push(w); list.push(w); } }
      comps.push(list);
    }
    const si = o.geometry.attributes.skinIndex, sw = o.geometry.attributes.skinWeight, v = new vrm.scene.position.constructor();
    const set = skirt.get(pos) ?? new Set();
    for (const list of comps) {
      let y0 = 9, y1 = -9, inSet = 0, viaN = 0; const owners = {};
      for (const i of list) {
        o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
        y0 = Math.min(y0, v.y); y1 = Math.max(y1, v.y);
        if (set.has(i)) inSet++;
        let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
        const [n, via] = own(o.skeleton.bones[si.getComponent(i, best)]);
        owners[n] = (owners[n] ?? 0) + 1; if (via) viaN++;
      }
      if (y0 > hipY - 0.03 || y1 < hipY - 0.1 || list.length < 8 || owners.head || /Tail/.test(o.material?.name ?? "")) continue;   // pieces reaching below the hips
      res.push({ mesh: o.name, mat: o.material?.name, n: list.length, y: [+y0.toFixed(2), +y1.toFixed(2)], inSet, spring: viaN, owners });
    }
  });
  return { hipY: +hipY.toFixed(3), parts: res };
});
console.log(JSON.stringify(out.hipY)); for (const p of out.parts) console.log(JSON.stringify(p));
await browser.close(); await server.close();

// Skirt cloth diagnostics (2026-10-28, AvatarSample_D's layered skirt tore when sitting): loads a model and a motion,
// drapes the skirt as the player does, and reports per connected piece of the skirt cloth: vertices, and how much its
// edges are stretched at a frame (skinned positions vs the rest mesh).
// Usage: node tools/skirt_probe.mjs models/X.vrm motion.json [frame]
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5305, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${process.argv[2]}`);
const motion = JSON.parse(readFileSync(process.argv[3], 'utf8'));
const frame = Number(process.argv[4] ?? motion.rotations.length - 1);
const out = await page.evaluate(async ({ motion, frame }) => {
  await window.hsin.ready;
  await window.hsin.loadMotion(motion);
  window.hsin.setTime(frame / motion.fps);
  const vrm = window.hsin.state.vrm;
  vrm.scene.updateMatrixWorld(true);
  const res = [];
  for (const e of vrm.scene.userData.hsinSkirtVerts ?? []) {
    const o = e.o, pos = o.geometry.attributes.position;
    const set = new Set(e.idx), nb = new Map();
    for (const mesh of [o, ...e.others]) {
      const ix = mesh.geometry.index;
      for (let t = 0; t < ix.count; t += 3) for (let k = 0; k < 3; k++) {
        const a = ix.getX(t + k), b = ix.getX(t + (k + 1) % 3);
        if (!set.has(a) || !set.has(b)) continue;
        if (!nb.has(a)) nb.set(a, new Set()); if (!nb.has(b)) nb.set(b, new Set());
        nb.get(a).add(b); nb.get(b).add(a);
      }
    }
    const comp = new Map(); let nc = 0;
    for (const i of e.idx) {
      if (comp.has(i)) continue;
      const st = [i]; comp.set(i, nc);
      while (st.length) { const v = st.pop(); for (const w of nb.get(v) ?? []) if (!comp.has(w)) { comp.set(w, nc); st.push(w); } }
      nc++;
    }
    const orig = new Map(e.idx.map((i, n) => [i, [e.orig[3 * n], e.orig[3 * n + 1], e.orig[3 * n + 2]]]));
    const P = new Map(), V = vrm.scene.position.constructor;
    for (const i of e.idx) { const v = new V(); o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld); P.set(i, v); }
    const stats = Array.from({ length: nc }, () => ({ n: 0, edges: 0, over2: 0, max: 0, ymin: 9, mats: new Set() }));
    for (const i of e.idx) stats[comp.get(i)].n++;
    for (const [a, ns] of nb) for (const b of ns) {
      if (b < a) continue;
      const r = Math.hypot(...orig.get(a).map((x, k) => x - orig.get(b)[k]));
      if (r < 1e-5) continue;
      const s = P.get(a).distanceTo(P.get(b)) / r, st = stats[comp.get(a)];
      st.edges++; if (s > 2) st.over2++; st.max = Math.max(st.max, s);
    }
    for (const i of e.idx) stats[comp.get(i)].ymin = Math.min(stats[comp.get(i)].ymin, P.get(i).y);
    const worst = [];
    for (const [a, ns] of nb) for (const b of ns) {
      if (b < a) continue;
      const r = Math.hypot(...orig.get(a).map((x, k) => x - orig.get(b)[k]));
      if (r < 1e-5) continue;
      worst.push({ s: +(P.get(a).distanceTo(P.get(b)) / r).toFixed(2), rest: +r.toFixed(3), a: P.get(a).toArray().map((x) => +x.toFixed(3)),
        b: P.get(b).toArray().map((x) => +x.toFixed(3)), restYa: +orig.get(a)[1].toFixed(3), comp: comp.get(a) });
    }
    worst.sort((x, y) => y.s - x.s);
    res.push({ worst: worst.slice(0, 12) });
    for (const mesh of [o, ...e.others]) {
      const ix = mesh.geometry.index;
      for (let t = 0; t < ix.count; t++) { const v = ix.getX(t); if (comp.has(v)) stats[comp.get(v)].mats.add(mesh.material?.name ?? '?'); }
    }
    res.push(stats.filter((s) => s.n > 3).map((s) => ({ n: s.n, edges: s.edges, over2: s.over2, max: +s.max.toFixed(2), ymin: +s.ymin.toFixed(3), mats: [...s.mats].join(' ') })));
  }
  return res;
}, { motion, frame });
console.log(JSON.stringify(out, null, 0));
await browser.close(); await server.close();

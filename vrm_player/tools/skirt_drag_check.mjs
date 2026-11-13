// Skirt vs legs while the body is dragged (jiggle.js springTailsShift + dampSprings, standing): per frame after
// skirt.js fixSkirtVertices, how many skirt vertices are still inside a thigh / shin capsule and how deep.
// Usage: node tools/skirt_drag_check.mjs [model] [dx_m] [dir x|z] [shot.png]
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const M = process.argv[2] ?? 'test2', DX = Number(process.argv[3] ?? 0.3), DIR = process.argv[4] ?? 'x', SWING = Number(process.argv[5] ?? 0), SHOT = process.argv[6];
const server = await createServer({ root, server: { port: 5390, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 500, height: 500 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&model=models/${M}.vrm`);
await page.evaluate(() => window.hsin.ready);
const r = await page.evaluate(async ([dx, dir, swing]) => {
  const { springTailsShift, dampSprings } = await import('/src/jiggle.js');
  const { fixSkirtVertices, limitSkirtSwing } = await import('/src/skirt.js');
  const { state } = window.hsin; state.manual = true;
  const v = state.vrm; const T = { Vector3: v.scene.position.constructor }; dampSprings(v);
  state.player.setTime(0);
  for (let i = 0; i < 120; i++) v.update(1 / 60);
  const meshes = v.scene.userData.hsinSkirtVerts ?? [];
  const limbs = () => (v.scene.userData.hsinLimbs ?? []).filter((l) => /Leg/.test(l.bone.name))
    .map((l) => ({ a: l.bone.localToWorld(l.a.clone()), b: l.bone.localToWorld(l.b.clone()), r: l.radius }));
  const p = new T.Vector3(), cp = new T.Vector3(), ab = new T.Vector3();
  const out = []; let worst = { d: 0, f: -1 };
  for (let f = 0; f < 180; f++) {
    if (dir === 'shake') { if (f < 120) { const w = 2 * Math.PI * 2; springTailsShift(v, new T.Vector3(dx * w * Math.cos(w * f / 60) / 60, 0, 0)); } }
    else if (f < 12) springTailsShift(v, new T.Vector3(dir === 'x' ? dx / 12 : 0, 0, dir === 'z' ? dx / 12 : 0));
    v.update(1 / 60); if (swing) limitSkirtSwing(v, swing); v.scene.updateMatrixWorld(true);
    fixSkirtVertices(v, 0);
    const L = limbs(); let n = 0, dmax = 0;
    const pts = [];
    for (const { o, idx } of meshes) {
      const W = new Map(); for (const i of idx) { const q = new T.Vector3(); o.getVertexPosition(i, q); q.applyMatrix4(o.matrixWorld); W.set(i, q); pts.push(q); }
      const ix = o.geometry.index.array;
      for (let t = 0; t < ix.length; t += 3) for (let e = 0; e < 3; e++) { const a = W.get(ix[t + e]), b = W.get(ix[t + (e + 1) % 3]); if (a && b) pts.push(a.clone().add(b).multiplyScalar(0.5)); }
    }
    for (const q of pts) { p.copy(q);
      for (const l of L) {
        ab.subVectors(l.b, l.a); const t = Math.min(1, Math.max(0, p.clone().sub(l.a).dot(ab) / ab.lengthSq()));
        cp.copy(l.a).addScaledVector(ab, t); const d = l.r - p.distanceTo(cp);
        if (d > 0.005) { n++; dmax = Math.max(dmax, d); break; }
      }
    }
    out.push([n, +(dmax * 100).toFixed(1)]);
    if (dmax > worst.d) worst = { d: dmax, f };
  }
  return { nVerts: meshes.reduce((a, m) => a + m.idx.length, 0), out, worst };
}, [DX, DIR, SWING]);
const peak = r.out.reduce((a, x) => (x[0] > a[0] ? x : a), [0, 0]);
console.log(`${M} drag ${DX} m along ${DIR}, swing limit ${SWING || 'none'}: ${r.nVerts} skirt vertices; worst frame ${r.worst.f}: ${(r.worst.d * 100).toFixed(1)} cm deep; most inside at once ${peak[0]}`);
console.log('(vertices + edge midpoints) per 6 frames [inside, cm]:', r.out.filter((_, i) => i % 6 === 0).map((x) => x.join('/')).join(' '));
if (SHOT) {
  await page.evaluate(async ([dx, dir, f, swing]) => { /* re-run to the worst frame and draw */
    const { springTailsShift } = await import('/src/jiggle.js'); const { fixSkirtVertices, limitSkirtSwing } = await import('/src/skirt.js');
    const { state } = window.hsin, v = state.vrm; const T = { Vector3: v.scene.position.constructor }; v.springBoneManager.reset(); state.player.setTime(0);
    for (let i = 0; i < 120; i++) v.update(1 / 60);
    for (let k = 0; k <= f; k++) { if (dir === 'shake') { if (k < 120) { const w = 4 * Math.PI; springTailsShift(v, new T.Vector3(dx * w * Math.cos(w * k / 60) / 60, 0, 0)); } } else if (k < 12) springTailsShift(v, new T.Vector3(dir === 'x' ? dx / 12 : 0, 0, dir === 'z' ? dx / 12 : 0)); v.update(1 / 60); if (swing) limitSkirtSwing(v, swing); }
    window.hsin.frameCamera('orbit:20:5:1.4:0.6'); window.hsin.setTime(0, 0);
  }, [DX, DIR, r.worst.f, SWING]);
  await page.screenshot({ path: SHOT });
}
await browser.close(); await server.close();

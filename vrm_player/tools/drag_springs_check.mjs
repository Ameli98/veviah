// Window drag -> spring physics (jiggle.js springTailsShift), deterministic: from the same settled rest pose, step
// the springs at 60 Hz with and without a drag (the body "moved" right by DX m over 0.2 s, then still) and report how
// far the spring bones (hair, sleeves, skirts) turned away from the no-drag run.
// Usage: node tools/drag_springs_check.mjs [model] [dx_m]
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const M = process.argv[2] ?? 'test1', DX = Number(process.argv[3] ?? 0.3), MIN_DRAG = Number(process.argv[4] ?? 0);
const server = await createServer({ root, server: { port: 5391, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const run = async (dx) => {
  const page = await browser.newPage();
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&model=models/${M}.vrm`);
  await page.evaluate(() => window.hsin.ready);
  const r = await page.evaluate(async ([dx, minDrag]) => {
    const { springTailsShift } = await import('/src/jiggle.js');
    const { state } = window.hsin; state.manual = true;
    const v = state.vrm, V = [...v.springBoneManager.joints];
    if (minDrag) for (const j of V) j.settings.dragForce = Math.max(j.settings.dragForce, minDrag);
    state.player.setTime(0);
    for (let i = 0; i < 120; i++) v.update(1 / 60);                       // settle
    const out = [];
    for (let f = 0; f < 90; f++) {
      if (f < 12) springTailsShift(v, { x: dx / 12, y: 0, z: 0, clone() { return { ...this }; } , isVector3: true });
      v.update(1 / 60);
      out.push(V.map((j) => [j.bone.name, j.bone.getWorldQuaternion(j.bone.quaternion.clone()).toArray()]));
    }
    return out;
  }, [dx, MIN_DRAG]);
  await page.close();
  return r;
};
const a = await run(0), b = await run(DX);
const ang = (p, q) => { const d = Math.abs(p[0] * q[0] + p[1] * q[1] + p[2] * q[2] + p[3] * q[3]); return 2 * Math.acos(Math.min(1, d)) * 180 / Math.PI; };
const groups = {};
// overshoot: the bone's rotation relative to the no-drag run, as an axis-angle; projected on the axis at its first
// peak. A swing that comes back past zero (negative) and the size of that back-swing = oscillation.
const rel = (p, q) => { // q * p^-1 as [x, y, z, w]
  const [px, py, pz, pw] = [-p[0], -p[1], -p[2], p[3]], [qx, qy, qz, qw] = q;
  return [qw * px + qx * pw + qy * pz - qz * py, qw * py - qx * pz + qy * pw + qz * px, qw * pz + qx * py - qy * px + qz * pw, qw * pw - qx * px - qy * py - qz * pz];
};
const osc = {};
a[0].forEach(([n], k) => {
  const g = n.replace(/_?\d+$/, '').replace(/_(L|R)_/, '_').replace(/J_Sec_|J_/, '').replace(/(Hair\d*|Skirt|Sleeve|Bust).*/, '$1');
  const tr = a.map((fr, f) => { let r = rel(fr[k][1], b[f][k][1]); if (r[3] < 0) r = r.map((x) => -x); return r.slice(0, 3).map((x) => 2 * x * 180 / Math.PI); });
  let pk = 0, ax = null;
  tr.forEach((v) => { const m = Math.hypot(...v); if (m > pk) { pk = m; ax = v.map((x) => x / m); } });
  if (!ax || pk < 1) return;
  const proj = tr.map((v) => v[0] * ax[0] + v[1] * ax[1] + v[2] * ax[2]);
  const back = -Math.min(0, ...proj.slice(proj.indexOf(Math.max(...proj))));
  const o = osc[g] ?? { peak: 0, back: 0 }; osc[g] = { peak: Math.max(o.peak, pk), back: Math.max(o.back, back) };
});
for (let f = 0; f < a.length; f++) a[f].forEach(([n, q], k) => {
  const g = n.replace(/_?\d+$/, '').replace(/_(L|R)_/, '_').replace(/J_Sec_|J_/, '').replace(/(Hair\d*|Skirt|Sleeve|Bust).*/, '$1');
  const d = ang(q, b[f][k][1]); groups[g] = groups[g] ?? { max: 0, at: 0 }; if (d > groups[g].max) groups[g] = { max: d, at: f };
});
console.log(`${M}, body moved ${DX} m right over 0.2 s, min dragForce ${MIN_DRAG}:`);
for (const [g, { max, at }] of Object.entries(groups)) console.log(`  ${g}: up to ${max.toFixed(1)} deg (frame ${at}), swings back past rest by ${(osc[g]?.back ?? 0).toFixed(1)} deg`);
await browser.close(); await server.close();

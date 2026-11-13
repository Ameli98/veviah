// head-area mesh primitives: material name, vertex count, centroid relative to the blink region, share moved by angry
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { port: 5314, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 100, height: 100 } });
for (const m of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${m}`);
  const r = await page.evaluate(async () => {
    await window.hsin.ready; const vrm = window.hsin.state.vrm; vrm.scene.updateMatrixWorld(true);
    const V = vrm.scene.position.constructor, v = new V();
    const em = vrm.expressionManager, moved = (name) => {        // mesh -> Set of vertex indices the expression moves
      const out = new Map();
      for (const b of em.getExpression(name)?._binds ?? []) for (const mm of b.primitives ?? []) {
        const d = mm.geometry.morphAttributes.position?.[b.index]; if (!d) continue;
        const s = out.get(mm) ?? new Set(); for (let i = 0; i < d.count; i++) if (Math.hypot(d.getX(i), d.getY(i), d.getZ(i)) > 1e-4) s.add(i); out.set(mm, s);
      }
      return out;
    };
    const blink = moved('blink'), angry = moved('angry'), sad = moved('sad');
    let ey = 0, n = 0;
    for (const [mm, s] of blink) for (const i of s) { mm.getVertexPosition(i, v); v.applyMatrix4(mm.matrixWorld); ey += v.y; n++; }
    ey /= n || 1;
    const head = vrm.humanoid.getNormalizedBoneNode('head').getWorldPosition(new V());
    const rows = [];
    vrm.scene.traverse((o) => {
      if (!o.isMesh) return;
      const g = o.geometry, idx = g.index, cnt = idx ? new Set(idx.array).size : g.attributes.position.count;
      const ids = idx ? [...new Set(idx.array)] : [...Array(cnt).keys()];
      let cy = 0, cz = 0, cx = 0, minx = 1e9, maxx = -1e9;
      for (const i of ids) { o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld); cy += v.y; cz += v.z; minx = Math.min(minx, v.x); maxx = Math.max(maxx, v.x); }
      cy /= ids.length; cz /= ids.length;
      if (Math.abs(cy - head.y) > 0.25) return;
      const share = (M) => { const s = M.get(o); if (!s) return 0; let k = 0; for (const i of ids) if (s.has(i)) k++; return +(k / ids.length).toFixed(2); };
      rows.push({ mat: o.material?.name?.slice(0, 40), n: ids.length, dy: +(cy - ey).toFixed(3), z: +(cz - head.z).toFixed(3), w: +(maxx - minx).toFixed(3), blink: share(blink), angry: share(angry), sad: share(sad) });
    });
    return rows;
  });
  console.log(m); for (const x of r) console.log('  ', JSON.stringify(x));
}
await browser.close(); await server.close();

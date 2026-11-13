// Which vertices lie in a height band in front of the torso, per mesh component and owner bone (2026-11-03: the body
// fit had no chest front samples between the ribs and the collarbones).
// Usage: node tools/band_probe.mjs models/X.vrm y0 y1
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const root = fileURLToPath(new URL('..', import.meta.url));
const server = await createServer({ root, server: { port: 5311, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 64, height: 64 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${process.argv[2]}`);
const [y0, y1] = [Number(process.argv[3]), Number(process.argv[4])];
const r = await page.evaluate(async ([y0, y1]) => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm, H = vrm.humanoid;
  H.resetNormalizedPose(); vrm.update(0); vrm.scene.updateMatrixWorld(true);
  const spring = new Set(); for (const j of vrm.springBoneManager?.joints ?? []) spring.add(j.bone);
  const out = {}; const v = vrm.scene.position.clone();
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const used = new Set(g.index ? g.index.array : []);
    for (const i of used) {
      o.getVertexPosition(i, v); v.applyMatrix4(o.matrixWorld);
      if (v.y < y0 || v.y > y1 || v.z < 0.04) continue;
      let b = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, b)) b = k;
      const bone = o.skeleton.bones[sk.getComponent(i, b)];
      let n = bone, via = false; while (n && !Object.values(H.humanBones).some((h) => h.node === n)) { if (spring.has(n)) via = true; n = n.parent; }
      const key = `${o.name}/${o.material?.name} <- ${bone.name}${via ? ' (spring)' : ''}`;
      out[key] = out[key] ?? { n: 0, zmax: -1 }; out[key].n++; out[key].zmax = Math.max(out[key].zmax, +v.z.toFixed(3));
    }
  });
  return out;
}, [y0, y1]);
console.log(JSON.stringify(r, null, 1));
await browser.close(); await server.close();

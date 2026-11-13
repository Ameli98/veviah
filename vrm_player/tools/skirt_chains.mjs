import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const root = fileURLToPath(new URL('..', import.meta.url));
const server = await createServer({ root, server: { port: 5304, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
for (const model of process.argv.slice(2)) {
  const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${model}`);
  const out = await page.evaluate(async () => {
    await window.hsin.ready;
    const vrm = window.hsin.state.vrm, H = vrm.humanoid, m = vrm.springBoneManager;
    vrm.scene.updateMatrixWorld(true);
    const raw = new Set(Object.keys(H.humanBones).map((n) => H.getRawBoneNode(n)).filter(Boolean));
    const humanOf = (node) => { let n = node; while (n && !raw.has(n)) n = n.parent; return n; };
    const hips = H.getRawBoneNode('hips'), legs = ['leftUpperLeg', 'rightUpperLeg'].map((n) => H.getRawBoneNode(n));
    const y = (n) => n.getWorldPosition(hips.position.clone()).y;
    const legY = y(legs[0]), kneeY = y(H.getRawBoneNode('leftLowerLeg'));
    const joints = [...m.joints].filter((j) => { const h = humanOf(j.bone); return h === hips || legs.includes(h); });
    const byBone = new Map(joints.map((j) => [j.bone, j]));
    const chains = [];
    for (const j of joints) {
      if (byBone.has(j.bone.parent)) continue;
      const c = []; for (let k = j; k; k = k.child ? byBone.get(k.child) : null) c.push(k);
      const ys = c.map((k) => y(k.bone)); if (c.at(-1).child) ys.push(y(c.at(-1).child));
      chains.push({ name: j.bone.name, n: c.length, top: +((Math.max(...ys) - legY) / (legY - kneeY)).toFixed(2), bottom: +((Math.min(...ys) - legY) / (legY - kneeY)).toFixed(2) });
    }
    return { legY: +legY.toFixed(3), kneeY: +kneeY.toFixed(3), chains };
  }).catch((e) => String(e));
  console.log(model, JSON.stringify(out));
  await page.close();
}
await browser.close(); await server.close();

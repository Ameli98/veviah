// Prints the torso hair colliders addExtraColliders (src/colliders.js) measured for each model (2026-11-13).
// Usage: node tools/torso_colliders.mjs models/X.vrm [...]
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5305, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
for (const model of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${model}`);
  const r = await page.evaluate(async () => {
    await window.hsin.ready;
    const vrm = window.hsin.state.vrm, m = vrm.springBoneManager;
    const g = [...m.joints].filter((j) => j.colliderGroups.some((x) => x.name === 'hsin_torso'));
    const caps = [...m.joints].flatMap((j) => j.colliderGroups.filter((x) => x.name === 'hsin_torso')).slice(0, 1)
      .flatMap((x) => x.colliders.map((c) => ({ bone: c.parent.name, r: +c.shape.radius.toFixed(3),
        off: c.shape.offset.toArray().map((v) => +v.toFixed(3)), tail: c.shape.tail.toArray().map((v) => +v.toFixed(3)) })));
    return { torso: vrm.scene.userData.hsinTorso, hairJoints: g.length, capsules: caps };
  });
  console.log(model, JSON.stringify(r, null, 0));
}
await browser.close(); await server.close();

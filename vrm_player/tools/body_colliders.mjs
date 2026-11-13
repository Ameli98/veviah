// Prints what addBodyColliders (src/colliders.js) measured and attached for a model.
// Usage: node tools/body_colliders.mjs models/X.vrm [...]
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5304, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
for (const model of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${model}`);
  const r = await page.evaluate(async () => {
    await window.hsin.ready;
    const vrm = window.hsin.state.vrm, m = vrm.springBoneManager;
    const skirt = [...m.joints].filter((j) => j.colliderGroups.some((g) => g.name === 'hsin_body_low'));
    return { measured: window.hsin.state.bodyColliders, skirtJointNames: [...new Set(skirt.map((j) => j.bone.name.replace(/[0-9_]+$/, '')))].slice(0, 12),
      hitRadius: [...new Set(skirt.map((j) => Math.round(j.settings.hitRadius * 1000) / 1000))] };
  });
  console.log(model, JSON.stringify(r));
}
await browser.close(); await server.close();

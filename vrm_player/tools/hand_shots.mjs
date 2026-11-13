// Close-up pictures of every hand shape, for checking HAND_SHAPES by eye.
// Usage: node tools/hand_shots.mjs <out dir> [model path under public/]
import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const out = resolve(process.argv[2] ?? 'hand_shots');
mkdirSync(out, { recursive: true });
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5399, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 300, height: 300 }, deviceScaleFactor: 1 });
page.on('console', (m) => { if (m.type() !== 'log') console.error('[browser]', m.text()); });
const q = new URLSearchParams({ ui: '0', t: '0' });
if (process.argv[3]) q.set('model', process.argv[3]);
await page.goto(`${server.resolvedUrls.local[0]}?${q}`);
await page.evaluate(() => window.hsin.ready);
const shapes = await page.evaluate(async () => Object.keys((await import('/src/player.js')).HAND_SHAPES));
// Both forearms raised beside the head (identity elsewhere), so both hands are near the face.
const I = [0, 0, 0, 1], h = Math.SQRT1_2;
const joints = ['leftUpperArm', 'leftLowerArm', 'rightUpperArm', 'rightLowerArm'];
const pose = [[0, 0, -0.5, 0.866], [0, -h, 0, h], [0, 0, 0.5, 0.866], [0, h, 0, h]];
for (const [k, name] of shapes.entries()) {
  for (const [view, dir] of [['palm', [0.15, -0.25, 1]], ['back', [-0.6, 0.8, 0.5]], ['thumb', [0.2, 0.3, 1]], ['under', [0.3, -1, 0.2]]]) {
    await page.evaluate(async ({ joints, pose, shapes, k, dir }) => {
      const T = 30, ids = Array(T).fill(k);
      await window.hsin.loadMotion({ fps: 30, joints, quat_order: 'xyzw', rotations: Array(T).fill(pose), hands: { shapes, left: ids, right: ids } });
      const { state, camera, controls } = window.hsin;
      state.manual = true;
      window.hsin.setTime(0.5);
      for (let i = 0; i < 60; i++) state.vrm.update(1 / 30);            // let hair settle
      // aim at the right hand's middle knuckle, from 0.4 m away along `dir`
      const c = state.vrm.humanoid.getNormalizedBoneNode('rightMiddleProximal').getWorldPosition(controls.target.clone());
      const n = Math.hypot(...dir);
      controls.target.copy(c);
      camera.position.set(c.x + 0.4 * dir[0] / n, c.y + 0.4 * dir[1] / n, c.z + 0.4 * dir[2] / n);
      controls.update();
      window.hsin.setTime(0.5);
    }, { joints, pose, shapes, k, dir });
    await page.screenshot({ path: `${out}/${name}_${view}.png` });
  }
  console.log(name);
}
await browser.close();
await server.close();

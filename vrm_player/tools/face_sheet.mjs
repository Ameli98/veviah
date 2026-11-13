// Close-ups of every face expression of a model, with its own presets and with face_rig.js's hand-made ones
// (?face=rig). Usage: node tools/face_sheet.mjs models/X.vrm OUT_DIR  -> OUT_DIR/<preset|rig>_<name>.png + rig.json
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const [modelArg, out] = process.argv.slice(2);
fs.mkdirSync(out, { recursive: true });
const root = fileURLToPath(new URL('..', import.meta.url));
const server = await createServer({ root, server: { port: 5310, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const NAMES = ['neutral', 'happy', 'angry', 'sad', 'relaxed', 'surprised', 'shy', 'blink', 'aa', 'ih', 'ou', 'ee', 'oh'];
for (const mode of ['preset', 'rig']) {
  const page = await browser.newPage({ viewport: { width: 300, height: 300 } });
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${modelArg}${mode === 'rig' ? '&face=rig' : ''}`);
  const shots = await page.evaluate(async (names) => {
    const h = window.hsin; await h.ready;
    const em = h.state.vrm.expressionManager;
    h.frameCamera('close');
    const y = h.state.headY + 0.06;               // tighter than the judge close-up: the face fills the tile
    h.controls.target.set(0, y, 0); h.camera.position.set(0, y, 0.55); h.controls.update();
    const res = {};
    for (const n of names) {
      for (const e of em.expressions) em.setValue(e.expressionName, 0);
      if (n !== 'neutral') { if (!em.getExpression(n)) { res[n] = null; continue; } em.setValue(n, 1); }
      h.setTime(0);
      res[n] = document.getElementById('view').toDataURL('image/png');
    }
    return { res, rig: h.state.faceRig };
  }, NAMES);
  for (const [n, url] of Object.entries(shots.res)) {
    if (url) fs.writeFileSync(`${out}/${mode}_${n}.png`, Buffer.from(url.split(',')[1], 'base64'));
  }
  fs.writeFileSync(`${out}/${mode}.json`, JSON.stringify(shots.rig, null, 1));
  await page.close();
}
await browser.close(); await server.close();

// upperStrength of each preset / blink per model. node tools/face_strength.mjs models/A.vrm ...
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { port: 5313, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 100, height: 100 } });
for (const m of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${m}`);
  const r = await page.evaluate(async () => {
    await window.hsin.ready; const { upperStrength } = await import('/src/face.js');
    const v = window.hsin.state.vrm, out = {};
    for (const n of ['blink', 'happy', 'angry', 'sad', 'shy', 'surprised', 'relaxed']) out[n] = upperStrength(v, n);
    const b = out.blink || 1; for (const k in out) out[k] = +(out[k] / b).toFixed(3);
    return out;
  });
  console.log(m.split('/').pop(), JSON.stringify(r));
}
await browser.close(); await server.close();

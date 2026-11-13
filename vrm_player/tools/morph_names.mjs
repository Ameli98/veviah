// morph target names of a model's face meshes that look like a blush / cheek / shy. node tools/morph_names.mjs models/X.vrm ...
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { port: 5312, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 100, height: 100 } });
for (const m of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${m}`);
  const r = await page.evaluate(async () => {
    await window.hsin.ready; const names = new Set();
    window.hsin.state.vrm.scene.traverse((o) => { if (o.morphTargetDictionary) for (const n of Object.keys(o.morphTargetDictionary)) names.add(n); });
    const all = [...names];
    return { n: all.length, fcl: all.filter((n) => /Fcl_(BRW|EYE|HA|ALL)|cheek|blush/i.test(n)), hit: all.filter((n) => /cheek|blush|shy|照|頬|redden|hoho|teare/i.test(n)), eye: all.filter((n) => /EYE_/i.test(n)).slice(0, 30), brw: all.filter((n) => /BRW/i.test(n)) };
  });
  console.log(m, JSON.stringify(r));
}
await browser.close(); await server.close();

// Only the mouth part of the body fit (measure.js mouthPoint). node tools/mouth_fit.mjs models/A.vrm ... -> JSON {model: mouth}
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5304, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
const out = {};
for (const model of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${model}`);
  out[model] = await page.evaluate(async () => { await window.hsin.ready; const f = window.hsin.fitBody(); return { mouth: f.mouth, index: f.index }; });
}
console.log(JSON.stringify(out));
await browser.close(); await server.close();

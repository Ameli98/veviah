// Prints the rest (T-pose, standing) world positions of the normalized humanoid bones of each model,
// for motion_loop/skeleton.py (leg IK and floor sitting need bone lengths).
// Usage: node tools/skeleton.mjs models/A.vrm [models/B.vrm ...]   -> JSON on stdout
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5298, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
const out = {};
for (const model of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${model}`);
  out[model] = await page.evaluate(async () => { await window.hsin.ready; return window.hsin.measureSkeleton(); });   // src/measure.js
}
console.log(JSON.stringify(out, null, 1));
await browser.close();
await server.close();

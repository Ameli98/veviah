// Fits the body surface of any VRM model in its rest pose, for contact targets (touch words) and the
// hand-body collision check, without hand-tuned numbers (the author, 2026-10-23: must work on any model).
// Each skinned vertex goes to the humanoid bone that owns its strongest skin joint (climbing up from
// non-humanoid joints). Vertices moved by spring bones (hair, ribbons, skirts, loose sleeves) and hair meshes (a
// mesh with >= 20 vertices on springs under the head, measure.js HAIR_SPRING_MIN) are left out: they swing, and the
// spring-bone colliders push them aside.
// (Seed-san's long hair hangs from the chest, so 'spring joints below the head' was not enough.)
// Per segment (bone -> child bone): the axis a-b and, around it, the 90th percentile of the vertex
// distance in four directions (front +z, back -z, out = away from the middle, in) and overall.
// Head: an axis-aligned ellipsoid (centre, radii) from its 1st / 99th percentiles. Floor: lowest vertex.
// The fit itself is fitBody() in src/measure.js (shared with the MCP view).
// Usage: node tools/fit_capsules.mjs models/A.vrm [models/B.vrm ...]   -> JSON on stdout, one entry per model
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5302, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
const out = {};
for (const model of process.argv.slice(2)) {
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${model}`);
  out[model] = await page.evaluate(async () => { await window.hsin.ready; return window.hsin.fitBody(); });   // src/measure.js
}
console.log(JSON.stringify(out, null, 1));
await browser.close(); await server.close();

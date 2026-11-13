// Lists the spring-bone colliders of a model (bone, shape, offset, radius) and which joints use them.
// Usage: node tools/colliders.mjs [models/X.vrm]
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const server = await createServer({ root, server: { port: 5299, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 200, height: 200 } });
const model = process.argv[2];
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0${model ? `&model=${model}` : ''}`);
const out = await page.evaluate(async () => {
  await window.hsin.ready;
  const m = window.hsin.state.vrm.springBoneManager;
  const r2 = (v) => Math.round(v * 1000) / 1000;
  const colliders = [...m.colliders].map((c) => ({
    bone: c.parent?.name, type: c.shape.type, radius: r2(c.shape.radius),
    offset: c.shape.offset?.toArray().map(r2), tail: c.shape.tail?.toArray().map(r2), inside: c.shape.inside,
  }));
  const groups = [...m.colliderGroups].map((g) => g.colliders.map((c) => [...m.colliders].indexOf(c)));
  const joints = {};
  for (const j of m.joints) {
    const key = `${j.bone.name} r=${r2(j.settings.hitRadius)} groups=${j.colliderGroups.map((g) => [...m.colliderGroups].indexOf(g)).join(',')}`;
    const top = key.replace(/\d+(?=[_\s.])/g, '#');
    joints[top] = (joints[top] ?? 0) + 1;
  }
  return { colliders, groups, joints };
});
console.log(JSON.stringify(out, null, 1));
await browser.close();
await server.close();

// Lists a model's expressions (preset / custom, with what each binds) and every morph target name per mesh.
// Usage: node tools/face_probe.mjs models/X.vrm [--brief]
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const root = fileURLToPath(new URL('..', import.meta.url));
const server = await createServer({ root, server: { port: 5309, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 64, height: 64 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&model=${process.argv[2]}`);
const r = await page.evaluate(async () => {
  await window.hsin.ready;
  const vrm = window.hsin.state.vrm, em = vrm.expressionManager;
  const exprs = (em?.expressions ?? []).map((e) => ({
    name: e.expressionName, binds: e._binds?.length ?? 0,
    morphs: (e._binds ?? []).map((b) => b.primitives ? `${b.primitives[0]?.name}[${b.index}]*${b.weight}` : b.constructor.name).slice(0, 6),
    overrideMouth: e.overrideMouth, overrideBlink: e.overrideBlink }));
  const meshes = [];
  vrm.scene.traverse((o) => { if (o.isMesh && o.morphTargetDictionary) meshes.push({ mesh: o.name, morphs: Object.keys(o.morphTargetDictionary) }); });
  return { meta: vrm.meta?.metaVersion, exprs, meshes };
});
const brief = process.argv.includes('--brief');
console.log(`VRM ${r.meta}; expressions (${r.exprs.length}): ${r.exprs.map((e) => `${e.name}(${e.binds})`).join(' ')}`);
if (!brief) for (const e of r.exprs) console.log(`  ${e.name}: ${e.morphs.join(', ')}  mouth=${e.overrideMouth} blink=${e.overrideBlink}`);
const seen = new Set();
for (const m of r.meshes) {
  const key = m.morphs.join(',');
  if (seen.has(key)) continue; seen.add(key);
  console.log(`mesh ${m.mesh}: ${m.morphs.length} morphs` + (brief ? `: ${m.morphs.slice(0, 12).join(' ')}…` : `\n  ${m.morphs.join(' ')}`));
}
await browser.close(); await server.close();

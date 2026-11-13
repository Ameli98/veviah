// Per mesh component (glTF primitive = one SkinnedMesh in three.js): the vertices it really uses (through its index;
// VRoid's merged primitives share one vertex buffer), and how many of them hang mostly from hand / finger bones,
// feet / toes, head. 2026-10-29: telling skin from clothes by the component that holds the hands, not by names.
// Usage: node tools/components.mjs models/X.vrm
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
  const vrm = window.hsin.state.vrm, H = vrm.humanoid;
  const rawToName = new Map();
  for (const name of Object.keys(H.humanBones)) { const n = H.getRawBoneNode(name); if (n) rawToName.set(n, name); }
  const humanOf = (node) => { let n = node; while (n && !rawToName.has(n)) n = n.parent; return n ? rawToName.get(n) : null; };
  const out = [];
  vrm.scene.traverse((o) => {
    if (!o.isSkinnedMesh) return;
    const g = o.geometry, sk = g.attributes.skinIndex, sw = g.attributes.skinWeight;
    const used = new Set(g.index ? g.index.array : Array.from({ length: g.attributes.position.count }, (_, i) => i));
    const c = { hand: 0, foot: 0, head: 0, leg: 0, arm: 0, torso: 0, other: 0 };
    for (const i of used) {
      let best = 0; for (let k = 1; k < 4; k++) if (sw.getComponent(i, k) > sw.getComponent(i, best)) best = k;
      const n = humanOf(o.skeleton.bones[sk.getComponent(i, best)]) ?? '';
      const kind = /Hand|Thumb|Index|Middle|Ring|Little/.test(n) ? 'hand' : /Foot|Toes/.test(n) ? 'foot' : /head|neck|Eye|Jaw/.test(n) ? 'head'
        : /Leg/.test(n) ? 'leg' : /Arm|Shoulder/.test(n) ? 'arm' : /hips|spine|chest|Chest/.test(n) ? 'torso' : 'other';
      c[kind]++;
    }
    const m = Array.isArray(o.material) ? o.material[0] : o.material;
    out.push([o.name, m?.name, g.attributes.position.count, used.size, JSON.stringify(c)]);
  });
  return out;
});
console.log('mesh | material | buffer vertices | used | by bone');
console.log(r.map((x) => x.join(' | ')).join('\n'));
await browser.close(); await server.close();

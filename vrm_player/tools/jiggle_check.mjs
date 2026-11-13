// Checks for src/jiggle.js (2026-11-04): chest bones found by structure on every test model (2 on the VRoid ones,
// 0 on Seed-san, whose backpack ropes sit in front of its chest), and a sudden 9 cm drop of the body makes the chest
// lag (> 0.3 cm) and settle (< 0.05 cm within 2 s); both buttocks sway (> 0.2 cm), settle, and are still when the
// hips are down at the floor (2026-11-05). Usage: node tools/jiggle_check.mjs   (exit 1 on a failure)
import { chromium } from 'playwright';
import { createServer } from 'vite';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const WANT = { VRM1_Constraint_Twist_Sample: 2, test1: 2, test2: 2, AvatarSample_D: 2, 'three-vrm-girl_vrm0': 2, 'Seed-san': 0 };
const server = await createServer({ root, server: { port: 5392, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
let bad = 0;
for (const [m, want] of Object.entries(WANT)) {
  const page = await browser.newPage();
  await page.goto(`${server.resolvedUrls.local[0]}?ui=0&jiggle=weak&model=models/${m}.vrm`);
  await page.evaluate(() => window.hsin.ready);
  const r = await page.evaluate(() => {
    const { state } = window.hsin; state.manual = true;
    const j = state.jiggle, v = state.vrm, cm = [], bt = [];
    const U = v.scene.userData.hsinPush?.hsinButt.value ?? [];
    for (let f = 0; f < 80; f++) {
      v.scene.position.y = f < 10 ? 0 : -0.03 * Math.min(f - 9, 3);           // 9 cm down over 0.1 s, then still
      v.update(1 / 30); j.update(1 / 30);
      cm.push(Math.max(0, ...j.bones.map((b) => b.cm ?? 0)));
      bt.push(Math.max(0, ...U.map((u) => u.length() * 100)));
    }
    // sitting on the floor (the hips lowered to 0.2 of their height): the buttock sway must be off
    v.scene.position.y = -0.8 * j.buttRestY;
    for (let f = 0; f < 20; f++) { v.update(1 / 30); j.update(1 / 30); }
    v.scene.position.y = -0.8 * j.buttRestY - 0.09; v.update(1 / 30); j.update(1 / 30);
    const sit = Math.max(0, ...U.map((u) => u.length() * 100));
    v.scene.position.y = 0;
    return { n: j.bones.length, peak: Math.max(...cm), end: Math.max(...cm.slice(-10)), butt: j.butt.length, bpeak: Math.max(...bt), bend: Math.max(...bt.slice(-10)), sit };
  });
  const ok = r.n === want && (want === 0 || (r.peak > 0.3 && r.end < 0.05)) && r.butt === 2 && r.bpeak > 0.2 && r.bend < 0.05 && r.sit < 0.01;
  if (!ok) bad++;
  console.log(`${ok ? 'ok  ' : 'FAIL'} ${m}: ${r.n} chest bones (want ${want}), lag peak ${r.peak.toFixed(2)} cm, last 0.3 s ${r.end.toFixed(3)} cm; buttocks ${r.butt}, sway peak ${r.bpeak.toFixed(2)} cm, last 0.3 s ${r.bend.toFixed(3)}, sitting ${r.sit.toFixed(3)}`);
  await page.close();
}
await browser.close(); await server.close();
process.exit(bad ? 1 : 0);

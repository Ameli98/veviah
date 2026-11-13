// Rendering look (2026-11-12): screenshots of the companion page for comparing render switches by eye.
// Run: node test/look_shots.mjs MODEL OUTDIR TAG [extra query, e.g. "look=all"]
// Writes OUTDIR/<model>_<TAG>_<view>.png with a transparent background (bg=none), views:
//   body = whole body front, face = head close-up front, side = turned 60 deg, face_side = close-up turned 45.
import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const [model = 'VRM1_Constraint_Twist_Sample', out = '.', tag = 'base', extra = ''] = process.argv.slice(2);
const PORT = '8797';
const env = { ...process.env, HSIN_COMPANION_PORT: PORT, HSIN_VRM_MODEL: path.join(HERE, `../../vrm_player/public/models/${model}.vrm`) };
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'ignore', 'pipe'] });
process.on('exit', () => comp.kill());
for (const sig of ['SIGINT', 'SIGTERM']) process.on(sig, () => process.exit(1));
await new Promise((resolve) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) resolve(); }));
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
try {
const page = await browser.newPage({ viewport: { width: 480, height: 760 }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
page.on('console', (m) => { if (/look|error/i.test(m.text())) console.log('[page]', m.text()); });
await page.goto(`http://localhost:${PORT}/?bg=none&idlegap=600,601&jiggle=off&${extra}`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 120000 });
await page.waitForTimeout(2500);
await page.addStyleTag({ content: '#say,#replay,#land,#glow{display:none!important}' });
for (const [view, scale, yaw] of [['body', 1, 0], ['face', 5, 0], ['side', 1, 60], ['face_side', 5, 45]]) {
  await page.evaluate(([s, y]) => { window.hsinStage.setScale(s); window.hsinStage.setYaw(y); }, [scale, yaw]);
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(out, `${model}_${tag}_${view}.png`), omitBackground: true });
}
} finally { await browser.close(); comp.kill(); }

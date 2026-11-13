// Long-lived renderer for motion_loop (see motion_loop/vrm_render.py). Starts the player once in a
// headless browser, then answers one JSON request per stdin line with one JSON line on stdout:
//   request  {"motion": <motion JSON object>, "frames": [frame indices], "views": ["front", "side"]}
//   response {"ok": true, "images": [frame][view] base64 PNG}  or  {"ok": false, "error": "..."}
// Usage: node tools/render_server.mjs [--model models/X.vrm] [--size 320] [--crop x,y,w,h] [--gpu] [--no-springs]
// --no-springs: hair and sleeves keep their rest shape (see setSprings in src/main.js).
// --model-colliders: only the model's own hair colliders (default adds the bigger ones of src/colliders.js).
// --crop: fractions of the square size x size view to keep (see setCrop in src/main.js); images are then w*size x h*size.
// --gpu: render on the NVIDIA GPU (~12x faster than the default SwiftShader CPU rasteriser; images differ only
//   in antialiased edge pixels). In WSL2 only the full Chromium build in a real window reaches the GPU: Mesa's
//   d3d12 GL driver (WSLg) under ANGLE's GL backend; the headless shell always falls back to SwiftShader.
//   The window is placed off screen. Falls back to SwiftShader if the GPU is not reached; the ready line says which.
import { createInterface } from 'node:readline';
import { existsSync } from 'node:fs';
import { dirname, isAbsolute, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
import { createServer } from 'vite';

const args = process.argv.slice(2);
const opt = (name, def) => { const i = args.indexOf(`--${name}`); return i >= 0 ? args[i + 1] : def; };
const size = parseInt(opt('size', '320'), 10);
const crop = opt('crop') ? opt('crop').split(',').map(Number) : null;
const say = (obj) => process.stdout.write(`${JSON.stringify(obj)}\n`);

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
// --model is a path under public/ (models/X.vrm) or an absolute file anywhere (the MCP user's own VRM: served as /@fs)
const modelFile = opt('model') && isAbsolute(opt('model')) && existsSync(opt('model')) ? opt('model') : null;
const fsAllow = modelFile ? { fs: { allow: [root, dirname(modelFile)] } } : {};
const server = await createServer({ root, server: { port: 5299, strictPort: false, ...fsAllow }, logLevel: 'silent' });
await server.listen();
const CPU = { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] };
const GPU = {
  channel: 'chromium', headless: false,
  args: ['--ignore-gpu-blocklist', '--enable-gpu', '--window-position=-4000,-4000', '--window-size=200,200'],
  env: { ...process.env, GALLIUM_DRIVER: 'd3d12', MESA_D3D12_DEFAULT_ADAPTER_NAME: 'NVIDIA',
         LD_LIBRARY_PATH: ['/usr/lib/wsl/lib', process.env.LD_LIBRARY_PATH].filter(Boolean).join(':') },
};
let browser = null;
if (args.includes('--gpu')) browser = await chromium.launch(GPU).catch((e) => { console.error('[gpu]', String(e).split('\n')[0]); return null; });
browser ??= await chromium.launch(CPU);
const vw = crop ? Math.round(size * crop[2]) : size, vh = crop ? Math.round(size * crop[3]) : size;
const page = await browser.newPage({ viewport: { width: vw, height: vh }, deviceScaleFactor: 1 });
page.on('pageerror', (e) => console.error('[page error]', e.message));
page.on('console', (m) => { if (m.type() === 'error') console.error('[browser]', m.text()); });

const q = new URLSearchParams({ ui: '0', t: '0' });
if (opt('model')) q.set('model', modelFile ? `/@fs${modelFile}` : opt('model'));
if (args.includes('--model-colliders')) q.set('colliders', 'model');
if (args.includes('--skirt-physics')) q.set('skirt', 'physics');   // plain spring physics for skirts (no skirt.js)
if (args.includes('--no-limb-push')) q.set('limbpush', 'off');   // hands do not press clothes / flesh (limb_push.js)
if (opt('jiggle')) q.set('jiggle', opt('jiggle'));   // chest sway level (jiggle.js), off by default
if (args.includes('--cloth-trial')) q.set('clothtrial', args[args.indexOf('--cloth-trial') + 1]);   // skirt.js experiments
await page.goto(`${server.resolvedUrls.local[0]}?${q}`);
await page.evaluate(() => window.hsin.ready);
// The page's own 60 fps loop (hair settling, redraw) is useless here: every command poses and draws explicitly.
// Left running it kept a GPU Chromium at 45-65 % CPU while idle (seen at ~1300 % after a day in the MCP server's
// cloth provider, 2026-11-08), so stop it as renderFrames does.
await page.evaluate(() => { window.hsin.state.manual = true; });
if (crop) await page.evaluate((c) => window.hsin.setCrop(c), crop);
if (args.includes('--no-springs')) await page.evaluate(() => window.hsin.setSprings(false));
const gl = await page.evaluate(() => {
  const g = document.createElement('canvas').getContext('webgl2'), e = g?.getExtension('WEBGL_debug_renderer_info');
  return e ? g.getParameter(e.UNMASKED_RENDERER_WEBGL) : 'unknown';
});
say({ ready: true, model: await page.evaluate(() => document.getElementById('info').textContent), gl });

for await (const line of createInterface({ input: process.stdin })) {
  if (!line.trim()) continue;
  try {
    const { motion, frames, views = ['front', 'side'], cmd } = JSON.parse(line);
    if (cmd === 'cloth' || cmd === 'body' || cmd === 'bones') {   // the skirt's cloth / the skin / bone positions (main.js)
      const bones = JSON.parse(line).bones;
      const points = await page.evaluate(async ({ motion, frames, cmd, bones }) => {
        await window.hsin.loadMotion(motion);
        return cmd === 'cloth' ? window.hsin.clothSurface(frames) : cmd === 'body' ? window.hsin.bodySurface(frames)
          : window.hsin.bonePoints(frames, bones);
      }, { motion, frames, cmd, bones });
      say({ ok: true, points });
      continue;
    }
    const urls = await page.evaluate(async ({ motion, frames, views }) => {
      await window.hsin.loadMotion(motion);
      return window.hsin.renderFrames(frames, views);
    }, { motion, frames, views });
    say({ ok: true, images: urls.map((row) => row.map((u) => u.slice(u.indexOf(',') + 1))) });
  } catch (err) {
    say({ ok: false, error: String(err.message ?? err) });
  }
}
await browser.close();
await server.close();

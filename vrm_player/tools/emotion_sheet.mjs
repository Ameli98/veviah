// The six emotion words as the app shows them (FaceController: presets, talking twins, blinks off), quiet and talking
// (mouth 'aa' 0.3, as lipsync's loud syllable) for one model, plus eyes closed. Usage:
// node tools/emotion_sheet.mjs models/X.vrm OUT_DIR -> OUT_DIR/<emotion>_<quiet|talk>.png
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import { chromium } from 'playwright';
import { createServer } from 'vite';
const [modelArg, out, gainArg = '1', extra = ''] = process.argv.slice(2);
const GAIN = Number(gainArg);
fs.mkdirSync(out, { recursive: true });
const server = await createServer({ root: fileURLToPath(new URL('..', import.meta.url)), server: { port: 5311, strictPort: false }, logLevel: 'silent' });
await server.listen();
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] });
const page = await browser.newPage({ viewport: { width: 260, height: 300 } });
await page.goto(`${server.resolvedUrls.local[0]}?ui=0&t=0&pose=stand&model=${modelArg}${extra}`);
const shots = await page.evaluate(async (GAIN) => {
  const h = window.hsin; await h.ready;
  h.frameCamera('close');
  const y = h.state.headY + 0.05;
  h.controls.target.set(0, y, 0); h.camera.position.set(0, y, 0.6); h.controls.update();
  const f = h.state.face, res = {};
  const settle = () => { for (let i = 0; i < 40; i++) f.update(1 / 30); h.setTime(0); };
  for (const em of ['neutral', 'happy', 'angry', 'sad', 'shy', 'surprised']) {
    for (const talk of [false, true]) {
      f.setEyes(0); f.setEmotion(em, GAIN); f.setMouth(talk ? { aa: 0.3 } : {}, talk); settle();
      res[`${em}_${talk ? 'talk' : 'quiet'}`] = document.getElementById('view').toDataURL('image/png');
    }
  }
  f.setEmotion('neutral'); f.setMouth({}, false); f.setEyes(1); settle();
  res.eyes_closed = document.getElementById('view').toDataURL('image/png');
  f.setEyes(0); settle();
  return res;
}, GAIN);
for (const [k, v] of Object.entries(shots)) fs.writeFileSync(`${out}/${k}.png`, Buffer.from(v.split(',')[1], 'base64'));
await browser.close(); await server.close();
console.log('ok', Object.keys(shots).length);

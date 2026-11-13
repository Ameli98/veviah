// Screenshot the player at given times, to check a motion by eye without opening a browser.
// Usage: node tools/shots.mjs <motion name> <t1,t2,...> [view] [model path under public/]
import { chromium } from 'playwright';
import { createServer } from 'vite';

const [name = 'raise_right_hand', times = '0,1.5', view = 'full', model] = process.argv.slice(2);
const server = await createServer({ server: { port: 5199, strictPort: false }, logLevel: 'error' });
await server.listen();
const base = server.resolvedUrls.local[0];
const browser = await chromium.launch({ args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'] });
const page = await browser.newPage({ viewport: { width: 640, height: 640 } });
page.on('console', (m) => { if (m.type() === 'error' || m.type() === 'warning') console.log(`[browser ${m.type()}]`, m.text()); });
page.on('pageerror', (e) => console.log('[page error]', e.message));
const q = new URLSearchParams({ motion: `motions/${name}.json`, t: '0', view });
if (model) q.set('model', model);
await page.goto(`${base}?${q}`);
await page.evaluate(() => window.hsin.ready);
console.log(await page.evaluate(() => document.getElementById('info').textContent));
for (const t of times.split(',')) {
  await page.evaluate((t) => window.hsin.setTime(t), parseFloat(t));
  const out = `shots/${name}_${view}_${t}.png`;
  await page.screenshot({ path: out });
  console.log(out);
}
await browser.close();
await server.close();

// Drives the built labelling page in a headless browser: players, ranking by drag and by tap, the
// broken zone, resume after reload, and the answer code. Usage: node check.mjs <site url> <screenshot dir>
import { createRequire } from 'node:module';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url));
const { chromium } = createRequire(resolve(here, '../../vrm_player/package.json'))('playwright');
const [url, shots] = process.argv.slice(2);
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1536, height: 760 } });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
const vs = (f) => page.evaluate((src) => [...document.querySelectorAll('video')].map(new Function('v', 'return ' + src)), f);
const ans = () => page.evaluate(() => JSON.parse(JSON.stringify(state.answers[DATA.groups[state.i].id])));
async function dragTo(letter, zone, beforeLetter) {          // real mouse drag from a letter to a zone
  const a = await page.locator(`#chip-${letter}`).boundingBox();
  let b;
  if (beforeLetter) { const c = await page.locator(`#chip-${beforeLetter}`).boundingBox(); b = { x: c.x + 4, y: c.y + c.height / 2 }; }
  else { const z = await page.locator(`.zone.${zone}`).boundingBox(); b = { x: z.x + z.width - 20, y: z.y + z.height / 2 }; }
  await page.mouse.move(a.x + a.width / 2, a.y + a.height / 2); await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 8 }); await page.mouse.up(); await page.waitForTimeout(100);
}

await page.goto(url);
await page.click('#start');
const refused = await page.textContent('#intro-hint');
await page.fill('#name', '測試員 A'); await page.click('#start');
await page.waitForSelector('#cards .card video');
// the drag test needs four clips: go to the first group that has four (groups lose clips to the filter)
await page.evaluate(() => show(DATA.groups.findIndex((g) => g.vids.length >= 4)));
await page.waitForTimeout(1500);
const startPaused = await vs('v.paused');
await page.keyboard.press('Space'); await page.waitForTimeout(1500);
const spacePlays = await vs('!v.paused');
await page.keyboard.press('Space'); await page.waitForTimeout(200);
const spacePauses = await vs('!v.paused');
const nextDisabledAtStart = await page.isDisabled('#next');
// ranking: tap C (goes to rank 1), drag A to the end, drag D in front of C, drag B to broken
await page.click('#chip-C');
await dragTo('A', 'rank');
await dragTo('D', 'rank', 'C');
const order1 = (await ans()).order;
await dragTo('B', 'broken');
const afterBroken = await ans();
const nextEnabled = !(await page.isDisabled('#next'));
await page.screenshot({ path: `${shots}/9_dock.png` });
await page.screenshot({ path: `${shots}/9_dock_full.png`, fullPage: true });
// move A from rank back to the tray, then tap it back in
await dragTo('A', 'tray');
const trayBack = await ans();
await page.click('#chip-A');
await page.click('#next');
await page.reload();                                          // resume
const resumeLabel = await page.textContent('#start');
await page.click('#start'); await page.waitForSelector('#cards .card video');
const resumedAt = await page.evaluate(() => state.i);
const total = await page.evaluate(() => DATA.groups.length);
for (let i = resumedAt; i < total; i++) {                     // the rest: tap letters in a rotating order
  const n = await page.evaluate(() => DATA.groups[state.i].vids.length);
  for (let j = 0; j < n; j++) await page.click(`#chip-${'ABCD'[(i + j) % n]}`);
  await page.click('#next');
}
await page.waitForSelector('#done:not([hidden])');
const code = await page.inputValue('#code');
const phone = await browser.newContext({ viewport: { width: 400, height: 800 }, storageState: await ctx.storageState() });
const p2 = await phone.newPage();
await p2.goto(url); await p2.click('#start'); await p2.waitForSelector('#cards .card video'); await p2.waitForTimeout(800);
const overflow = await p2.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
await p2.screenshot({ path: `${shots}/10_phone_dock.png` });
console.log(JSON.stringify({ refused, startPaused, spacePlays, spacePauses, nextDisabledAtStart, order1, afterBroken, nextEnabled, trayBack,
  resumeLabel, resumedAt, phoneOverflowPx: overflow, errors, codeLength: code.length }));
console.log(code);
await browser.close();

// End-to-end check of the desktop companion (2026-11-03): start companion.mjs, open its page in a headless
// browser, call `perform` through server.mjs over stdio (as Claude Desktop does), and check that the page plays
// the motion with its line. Screenshots in test/out/companion_*.png. Run: node test/companion.mjs
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const OUT = path.join(HERE, 'out');
await fs.mkdir(OUT, { recursive: true });
const PORT = '8799';
const env = { ...process.env, HSIN_COMPANION_PORT: PORT };

const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'inherit', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { process.stderr.write(d); if (String(d).includes('open http')) resolve(); }));

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 480, height: 640 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`http://localhost:${PORT}/?idlegap=1,2`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 60000 });
await page.screenshot({ path: path.join(OUT, 'companion_rest.png') });

const client = new Client({ name: 'hsin-test', version: '0' });
await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, '../server.mjs')], env, stderr: 'inherit' }));
const t0 = Date.now();
const r = await client.callTool({ name: 'perform', arguments: {
  say: '早安！一起伸個懶腰吧', emotion: 'happy', posture: 'stand',
  // palms together straight above the head: the hands pass where the bubble is
  keyframes: [{ t: 0 }, { t: 0.8, hands_meet: { at: 'above_head far', how: 'palms' } },
              { t: 1.8, hands_meet: { at: 'above_head far', how: 'palms' } },
              { t: 2.6, left_arm: 'rest', right_arm: 'rest' }] } });
console.log('perform ->', r.content[0].text);
await page.waitForFunction(() => document.getElementById('say').textContent.includes('伸個懶腰'), null, { timeout: 10000 });
console.log('page got the line after', Date.now() - t0, 'ms');
// hands raised behind the bubble: it fades (class 'behind') and comes back when they are down (2026-11-05)
await page.waitForFunction(() => document.getElementById('say').classList.contains('behind'), null, { timeout: 4000 })
  .then(() => console.log('bubble faded while the hands were behind it'), () => { console.log('FAIL: bubble never faded'); process.exitCode = 1; });
await page.screenshot({ path: path.join(OUT, 'companion_play.png') });
await page.waitForFunction(() => !document.getElementById('say').classList.contains('behind'), null, { timeout: 4000 })
  .then(() => console.log('bubble back once the hands came down'), () => { console.log('FAIL: bubble stayed faded'); process.exitCode = 1; });
await page.screenshot({ path: path.join(OUT, 'companion_play_after.png') });
await page.waitForFunction(() => document.getElementById('replay').style.display === 'block', null, { timeout: 10000 });
console.log('played to the end: replay shown');
// idle: an idle action starts 1-2 s after the line, the idle layer moves the head while nothing plays
await page.waitForFunction(() => window.hsinStage.state.idleMotion && window.hsinStage.state.playing, null, { timeout: 15000 });
console.log('idle action playing:', await page.evaluate(() => window.hsinStage.state.player.duration.toFixed(1)), 's');
await page.screenshot({ path: path.join(OUT, 'companion_idle.png') });
// a new line interrupts the idle action at once
const r2 = await client.callTool({ name: 'perform', arguments: { say: '欸你看那邊！', emotion: 'surprised', posture: 'stand',
  keyframes: [{ t: 0 }, { t: 0.6, right_arm: { hand: 'shoulder front far' }, hands: { right: 'point' } }, { t: 1.8, right_arm: 'rest' }] } });
await page.waitForFunction(() => !window.hsinStage.state.idleMotion && document.getElementById('say').textContent.includes('那邊'), null, { timeout: 10000 });
console.log('line interrupted the idle action:', r2.content[0].text);
await client.close(); await browser.close(); comp.kill();

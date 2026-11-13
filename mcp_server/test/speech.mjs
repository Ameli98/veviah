// Voice in the desktop companion with a fake TTS (2026-11-05, plan item 10): a tiny HTTP server that answers like
// GPT-SoVITS's api_v2 (/control, POST /tts -> a vowel-like wav), plugged in as a voice with "url". Checks:
// warm-up at settings time, sentences synthesized and played in turn, motion + bubble starting with the first
// sentence's audio, the mouth moving with it, a slow first sentence not holding the motion past 4 s, a new line
// interrupting the old one, voice off = bubble only, and the language line in the perform result.
// Run: node test/speech.mjs
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import http from 'node:http';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
import { chromium } from 'playwright';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = '8798', TTS_PORT = 9899;
let bad = 0;
const check = (ok, what) => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${what}`); if (!ok) bad++; };

// ---- fake api_v2 ----
function wav(seconds, sr = 32000) {         // an "a"-like vowel: 700 + 1200 Hz with a syllable envelope
  const n = Math.round(seconds * sr), b = Buffer.alloc(44 + 2 * n);
  b.write('RIFF', 0); b.writeUInt32LE(36 + 2 * n, 4); b.write('WAVEfmt ', 8); b.writeUInt32LE(16, 16); b.writeUInt16LE(1, 20);
  b.writeUInt16LE(1, 22); b.writeUInt32LE(sr, 24); b.writeUInt32LE(2 * sr, 28); b.writeUInt16LE(2, 32); b.writeUInt16LE(16, 34);
  b.write('data', 36); b.writeUInt32LE(2 * n, 40);
  for (let i = 0; i < n; i++) {
    const t = i / sr, env = Math.max(0, Math.sin(Math.PI * t * 4)) ** 0.5;
    const x = env * (0.5 * Math.sin(2 * Math.PI * 700 * t) + 0.3 * Math.sin(2 * Math.PI * 1200 * t) + 0.1 * Math.sin(2 * Math.PI * 140 * t));
    b.writeInt16LE(Math.round(x * 12000), 44 + 2 * i);
  }
  return b;
}
const asked = [];
const tts = http.createServer((req, res) => {
  let body = ''; req.on('data', (d) => { body += d; });
  req.on('end', () => {
    if (req.url.startsWith('/control')) { res.writeHead(400); res.end('{}'); return; }
    const q = JSON.parse(body || '{}'); asked.push(q);
    const delay = String(q.text).includes('SLOW') ? 6000 : 300;
    setTimeout(() => { res.writeHead(200, { 'content-type': 'audio/wav' }); res.end(wav(0.8)); }, delay);
  });
});
await new Promise((r) => tts.listen(TTS_PORT, '127.0.0.1', r));

const tmp = await fs.mkdtemp(path.join(os.tmpdir(), 'hsin-speech-'));
await fs.writeFile(path.join(tmp, 'refs.list'), ['neutral', 'happy', 'surprised'].map((e) => `${tmp}/${e}/a.wav|fake|ZH|參考句子。`).join('\n'));
await fs.writeFile(path.join(tmp, 'voices.json'), JSON.stringify({ voices: [{ name: 'fake', url: `http://127.0.0.1:${TTS_PORT}`, refs: path.join(tmp, 'refs.list'), lang: 'zh' }] }));
const env = { ...process.env, HSIN_COMPANION_PORT: PORT, HSIN_VOICES: path.join(tmp, 'voices.json') };

const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env, stdio: ['ignore', 'inherit', 'pipe'] });
await new Promise((resolve) => comp.stderr.on('data', (d) => { process.stderr.write(d); if (String(d).includes('open http')) resolve(); }));
const base = `http://127.0.0.1:${PORT}`;
const post = (p, o) => fetch(base + p, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(o) }).then((r) => r.json());
let st = await post('/settings', { voice: true, voiceName: 'fake', voiceLang: 'zh', uiLang: 'zh' });
for (let i = 0; i < 50 && st.state !== 'ready'; i++) { await new Promise((r) => setTimeout(r, 100)); st = await (await fetch(`${base}/settings`)).json(); }
check(st.state === 'ready' && asked.length === 1, `TTS ready after settings, one warm-up sentence (state ${st.state}, ${asked.length} asked)`);

const browser = await chromium.launch({ args: ['--autoplay-policy=no-user-gesture-required'] });
const page = await browser.newPage({ viewport: { width: 480, height: 640 } });
page.on('pageerror', (e) => console.log('[pageerror]', e.message));
await page.goto(`${base}/?idlegap=60,61`);
await page.waitForFunction(() => document.getElementById('status').textContent === '', null, { timeout: 60000 });
// record per frame: bubble text, speaking, mouth open, motion playing
await page.evaluate(() => {
  window.rec = [];
  (function f() { requestAnimationFrame(f); const s = window.hsinStage.state, m = s.face?.mouth ?? {};
    window.rec.push({ t: performance.now(), say: document.getElementById('say').textContent, on: document.getElementById('say').classList.contains('on'),
      mouth: (m.aa ?? 0) + (m.ih ?? 0) + (m.ou ?? 0) + (m.ee ?? 0) + (m.oh ?? 0), playing: s.playing && !s.idleMotion }); })();
});

const client = new Client({ name: 'hsin-test', version: '0' });
await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, '../server.mjs')], env, stderr: 'inherit' }));
const kf = [{ t: 0 }, { t: 0.6, right_arm: { hand: 'shoulder out far' }, hands: { right: 'open' } }, { t: 1.6, right_arm: 'rest' }];
const n0 = asked.length;
const r = await client.callTool({ name: 'perform', arguments: { say: '早安！今天的天氣真的很好耶。我們等一下一起出去走走吧？你想去哪裡我都可以陪你去喔。', emotion: 'happy', posture: 'stand', keyframes: kf } });
check(/Language: `say` is spoken and shown; write it in Chinese/.test(r.content[0].text), `language line in the result: ${r.content[0].text.split('Language:')[1]?.slice(0, 60)}`);
await page.waitForFunction(() => document.getElementById('say').textContent.includes('陪你去喔'), null, { timeout: 10000 }).catch(() => {});
await page.waitForFunction(() => !window.hsinSpeaker.playing, null, { timeout: 10000 }).catch(() => {});
await page.waitForTimeout(400);
let rec = await page.evaluate(() => window.rec.splice(0));
check(asked.length - n0 === 3 && asked.slice(n0).every((q) => q.text_lang === 'zh' && q.ref_audio_path.includes('/happy/')), `3 sentences asked with the happy reference: ${asked.slice(n0).map((q) => q.text).join(' | ')}`);
const firstOn = rec.find((x) => x.on), firstPlay = rec.find((x) => x.playing), firstMouth = rec.find((x) => x.mouth > 0.1);
check(firstOn && firstPlay && Math.abs(firstOn.t - firstPlay.t) < 200, `bubble and motion start together (${firstOn && firstPlay ? Math.round(firstPlay.t - firstOn.t) : '?'} ms apart)`);
check(!!firstMouth && firstMouth.t - firstOn.t < 600, `mouth moves with the voice (opens ${firstMouth ? Math.round(firstMouth.t - firstOn.t) : '?'} ms after the bubble, max ${Math.max(...rec.map((x) => x.mouth)).toFixed(2)})`);
const texts = [...new Set(rec.filter((x) => x.on).map((x) => x.say))];
check(texts.length === 3 && texts[2] === '早安！今天的天氣真的很好耶。我們等一下一起出去走走吧？你想去哪裡我都可以陪你去喔。', `bubble grows sentence by sentence: ${texts.join(' / ')}`);
check(rec.at(-1).mouth < 0.02, `mouth closed after the voice ends (${rec.at(-1).mouth.toFixed(3)})`);

// a slow first sentence: the motion starts after ~4 s without waiting longer
await page.evaluate(() => { window.rec = []; });
const t1 = Date.now();
await client.callTool({ name: 'perform', arguments: { say: 'SLOW 這句很慢。', emotion: 'neutral', posture: 'stand', keyframes: kf } });
await page.waitForFunction(() => window.rec.some((x) => x.playing), null, { timeout: 9000 }).catch(() => {});
const late = Date.now() - t1;
check(late > 3000 && late < 5500, `slow TTS: the motion started after ${late} ms (waits at most 4 s)`);

// lines queue (2026-11-13): a second line sent while the first speaks waits; the first is spoken to its end
await new Promise((r) => setTimeout(r, 2500));
await page.evaluate(() => { window.rec = []; });
await client.callTool({ name: 'perform', arguments: { say: '這是第一句比較長的話。這裡還有第二句比較長的話。最後還有第三句比較長的話。', emotion: 'neutral', posture: 'stand', keyframes: kf } });
await page.waitForFunction(() => document.getElementById('say').textContent.startsWith('這是第一句'), null, { timeout: 6000 }).catch(() => {});
await page.waitForTimeout(300);
await client.callTool({ name: 'perform', arguments: { say: '接著說的第二行話在這裡。', emotion: 'surprised', posture: 'stand', keyframes: kf } });
await page.waitForFunction(() => document.getElementById('say').textContent === '接著說的第二行話在這裡。', null, { timeout: 30000 })
  .then(() => check(true, 'the queued line plays after the first'), () => check(false, 'the queued line plays after the first'));
const said = await page.evaluate(() => window.rec.filter((x) => x.on).map((x) => x.say));
const i2 = said.indexOf('接著說的第二行話在這裡。');
check(i2 > 0 && said.slice(0, i2).some((x) => x.includes('第三句')), 'the first line was spoken to its end before the second');
await page.waitForTimeout(2000);

// voice off: bubble only, at once
await post('/settings', { voice: false });
await page.evaluate(() => { window.rec = []; });
const n1 = asked.length, t2 = Date.now();
const r3 = await client.callTool({ name: 'perform', arguments: { say: '沒有聲音。', emotion: 'neutral', posture: 'stand', keyframes: kf } });
await page.waitForFunction(() => document.getElementById('say').textContent === '沒有聲音。', null, { timeout: 5000 }).catch(() => {});
check(asked.length === n1 && Date.now() - t2 < 2500, `voice off: no TTS call, bubble at once (${Date.now() - t2} ms)`);
check(/voice off/.test(r3.content[0].text), 'language line says voice off');

// voice language differs from the UI language: the result asks for a subtitle, the bubble shows it
await post('/settings', { voice: true, voiceLang: 'ja', uiLang: 'zh' });
for (let i = 0; i < 50 && (await (await fetch(`${base}/settings`)).json()).state !== 'ready'; i++) await new Promise((r) => setTimeout(r, 100));
const r4 = await client.callTool({ name: 'perform', arguments: { say: 'おはよう。', subtitle: '早安。', emotion: 'happy', posture: 'stand', keyframes: kf } });
check(/keep sending both/.test(r4.content[0].text), `voice ja / ui zh with a subtitle: result says it is right (${r4.content[0].text.split('Language:')[1]})`);
await page.waitForFunction(() => document.getElementById('say').textContent === '早安。', null, { timeout: 6000 }).then(() => check(true, 'bubble shows the subtitle'), () => check(false, 'bubble shows the subtitle'));
check(asked.at(-1).text === 'おはよう。' && asked.at(-1).text_lang === 'ja', 'the spoken line is say, in ja');

await client.close(); await browser.close(); comp.kill(); tts.close();
await fs.rm(tmp, { recursive: true, force: true });
console.log(bad ? `${bad} FAILED` : 'all ok');
process.exit(bad ? 1 : 0);

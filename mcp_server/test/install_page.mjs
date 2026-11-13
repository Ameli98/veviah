// The settings page's voice-engine download box (2026-11-13) in headless Chromium with a fake /voice/install.
// Run: node test/install_page.mjs OUTDIR
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] ?? '.';
const { T: ALL } = createRequire(import.meta.url)('../../desktop/i18n.js');
const T = ALL.zh;
const STEPS = ['gpu', 'python', 'pip', 'torch', 'packages', 'standins', 'code', 'weights', 'g2pw', 'lid', 'nltk', 'voice'];
const browser = await chromium.launch();
let fail = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };
try {
  const page = await browser.newPage({ viewport: { width: 640, height: 1100 } });
  page.on('pageerror', (e) => { console.log('[pageerror]', e.message); fail++; });
  await page.addInitScript(({ t, steps }) => {
    // the fake installer: window.IN is what GET /voice/install returns; POSTs are recorded
    window.posts = [];
    window.IN = { supported: true, status: 'idle', step: null, done: [], error: null, bytes: 0, total: 0, detail: '', steps,
                  dir: 'C:\\Users\\x\\AppData\\Local\\veviah\\tts', defaultDir: 'C:\\x', freeGB: 500, needGB: 10, needed: true };
    let speech = { voice: true, voiceName: null, voiceLang: null, releaseMin: 0 };
    window.hsin = {
      chatView: async () => ({ mode: 'mcp', providers: ['anthropic'], provider: 'anthropic', kind: 'anthropic', baseUrl: '', model: '', desc: 'auto', persona: '', hasKey: false, local: false, canEncrypt: true, ready: false, hotkey: 'Control+Alt+Space' }), clients: async () => ({ programs: [], other: [], snippets: { json: '{}', toml: '' } }),
      state: async () => ({ t, uiLang: 'zh', model: '/x.vrm', modelName: 'x', instructions: '', onTop: true, autostart: false, jiggle: 'weak', copied: true,
        claude: { found: true, connected: true, same: true }, speech, speechInfo: { settings: speech, state: 'off', voices: [] } }),
      setSpeech: async (v) => { speech = { ...speech, ...v }; }, setUiLang: async () => {}, setJiggle: async () => {}, setOnTop: async () => {}, setAutostart: async () => {},
      voiceApi: async (m, p, b) => { if (m === 'POST') window.posts.push([p, b]); return p.startsWith('/voice/install') ? window.IN : {}; },
      voiceAudio: async () => null, choosePath: async () => 'D:\\Games',
    };
    window.confirm = () => true;
  }, { t: T, steps: STEPS });
  await page.goto('file://' + path.join(HERE, '../../desktop/settings.html'));
  await page.waitForTimeout(800);
  const vis = (id) => page.evaluate((i) => getComputedStyle(document.getElementById(i)).display !== 'none', id);
  const text = (id) => page.evaluate((i) => document.getElementById(i).textContent, id);
  const reload = () => page.evaluate(() => document.getElementById('voice').dispatchEvent(new Event('change')));

  ok(await vis('installBox'), 'voice on, nothing installed: the download box shows');
  ok((await text('instGo')) === T.installBtn && !(await vis('instCancel')), 'download button, no cancel');
  ok((await text('instDir')).endsWith('veviah\\tts'), 'default folder shown');
  await page.screenshot({ path: path.join(out, 'install_needed.png'), fullPage: true });

  await page.click('#instChange'); await page.waitForTimeout(200);
  ok((await text('instDir')) === 'D:\\Games\\veviah_tts', 'changed folder = an own subfolder of the picked one');

  await page.evaluate(() => Object.assign(window.IN, { status: 'running', step: 'torch', bytes: 1.2e9, total: 2.6e9, done: ['gpu', 'python', 'pip'] }));
  await page.click('#instGo'); await page.waitForTimeout(300);
  const posts = await page.evaluate(() => window.posts);
  ok(posts.some(([p, b]) => p === '/voice/install' && b.dir === 'D:\\Games\\veviah_tts'), 'install started with the chosen folder');
  ok(await vis('instCancel') && !(await vis('instGo')) && await vis('instBar'), 'while running: progress bar + cancel');
  ok((await text('mi')).includes(T.installSteps.torch) && (await text('mi')).includes('1200 / 2600 MB'), 'step name and MB shown: ' + await text('mi'));
  await page.screenshot({ path: path.join(out, 'install_running.png'), fullPage: true });

  await page.evaluate(() => Object.assign(window.IN, { status: 'error', step: 'gpu', error: 'nogpu', done: [] }));
  await page.waitForTimeout(1300);
  ok((await text('mi')) === T.noGpu && (await text('instGo')) === T.installRetry, 'no NVIDIA card: says so, retry button');

  await page.evaluate(() => Object.assign(window.IN, { status: 'idle', error: null, freeGB: 4 }));
  await reload(); await page.waitForTimeout(300);
  ok(await page.evaluate(() => document.getElementById('instGo').disabled) && (await text('mi')).includes('4 GB'), 'too little space: button off, message');

  await page.evaluate(() => Object.assign(window.IN, { status: 'idle', error: null, freeGB: 500, gpu: { ok: false, reason: 'nogpu' } }));
  await reload(); await page.waitForTimeout(300);
  ok(!(await vis('installBox')) && (await text('mv')) === T.noGpu && await page.evaluate(() => document.getElementById('voice').disabled),
    'no NVIDIA card at start: no download box, voice switch off-limits, the reason shown');
  await page.evaluate(() => Object.assign(window.IN, { gpu: { ok: true, name: 'RTX', driver: '580.1' } }));
  await reload(); await page.waitForTimeout(300);
  ok(await vis('installBox') && !(await page.evaluate(() => document.getElementById('voice').disabled)), 'a card found on a later start: the download box is back');
  await page.evaluate(() => Object.assign(window.IN, { status: 'installed', needed: false, freeGB: 500, done: [...window.IN.steps] }));
  await reload(); await page.waitForTimeout(300);
  ok(!(await vis('installBox')) && await vis('instRemoveRow'), 'installed: box hidden, delete button shown');
  await page.click('#instRemove'); await page.waitForTimeout(300);
  ok((await page.evaluate(() => window.posts)).some(([p]) => p === '/voice/install/remove'), 'delete asks the companion to remove');
  await page.screenshot({ path: path.join(out, 'install_done.png'), fullPage: true });
} finally { await browser.close(); }
console.log(fail ? `${fail} failed` : 'all ok'); process.exit(fail ? 1 : 0);

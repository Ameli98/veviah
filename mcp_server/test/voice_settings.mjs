// The settings page's reference-clip table (2026-11-12) in headless Chromium, against a test companion with real voices.
// The Electron bridge (window.hsin) is replaced by one that calls the companion directly (as main.js does).
// Run: HSIN_VOICES=... HSIN_VOICE_REFS=... node test/voice_settings.mjs OUTDIR   (needs a GPU for the TTS)
import { spawn } from 'node:child_process';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] ?? '.';
const { T } = createRequire(import.meta.url)('../../desktop/i18n.js');
const PORT = '8799', BASE = `http://localhost:${PORT}`;
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env: { ...process.env, HSIN_COMPANION_PORT: PORT, HSIN_TTS_PORT: '9896' }, stdio: ['ignore', 'ignore', 'pipe'] });
process.on('exit', () => comp.kill());
await new Promise((r) => comp.stderr.on('data', (d) => { if (String(d).includes('open http')) r(); }));
const post = (p, b) => fetch(BASE + p, { method: 'POST', body: JSON.stringify(b) }).then((r) => r.json());
await post('/settings', { voice: true, voiceName: 'default', voiceLang: 'zh' });
for (let i = 0; i < 90 && (await (await fetch(BASE + '/settings')).json()).state !== 'ready'; i++) await new Promise((r) => setTimeout(r, 1000));
const browser = await chromium.launch();
let fail = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };
try {
  const page = await browser.newPage({ viewport: { width: 640, height: 1400 } });
  page.on('pageerror', (e) => { console.log('[pageerror]', e.message); fail++; });
  await page.exposeFunction('_api', async (m, p, b) => (await fetch(BASE + p, { method: m, body: b ? JSON.stringify(b) : undefined })).json());
  await page.exposeFunction('_audio', async (p) => { const r = await fetch(BASE + p); return r.ok ? [...new Uint8Array(await r.arrayBuffer())] : { error: (await r.json()).error }; });
  await page.exposeFunction('_info', async () => (await fetch(BASE + '/settings')).json());
  await page.addInitScript(({ t }) => {
    window.played = [];
    window.Audio = class { constructor(u) { this.u = u; } play() { window.played.push(this.u); } pause() {} };
    window.hsin = {
      clients: async () => ({ programs: [], snippets: { json: '{}', toml: '' } }),
      chatView: async () => ({ mode: 'mcp', providers: ['anthropic'], provider: 'anthropic', kind: 'anthropic', baseUrl: '', model: '', desc: 'auto', persona: '', hasKey: false, local: false, canEncrypt: true, ready: false, hotkey: 'Control+Alt+V', hotkeyOk: true }),
      state: async () => { const i = await window._info(); return { t, uiLang: 'zh', model: '/x.vrm', modelName: 'x', instructions: '', onTop: true, autostart: false, jiggle: 'weak', copied: true,
        claude: { found: true, connected: true, same: true }, speech: i.settings, speechInfo: i }; },
      setSpeech: async (v) => window._api('POST', '/settings', v), setUiLang: async () => {}, setJiggle: async () => {}, setOnTop: async () => {}, setAutostart: async () => {},
      voiceApi: (m, p, b) => window._api(m, p, b),
      voiceAudio: async (p) => { const r = await window._audio(p); return Array.isArray(r) ? new Uint8Array(r) : r; },
      choosePath: async () => window.nextPath ?? null,
    };
  }, { t: T.zh });
  await page.goto('file://' + path.join(HERE, '../../desktop/settings.html'));
  await page.waitForTimeout(1500);
  await page.evaluate(() => { document.getElementById('refsBox').open = true; });
  await page.waitForSelector('#refs select', { timeout: 10000 });
  ok(await page.locator('#refs tr').count() === 6, 'six emotion rows');
  await page.screenshot({ path: path.join(out, 'voice_settings.png'), fullPage: true });
  // change happy to another clip, then preview it
  const sel = page.locator('#refs select[data-e="happy"]');
  const opts = await sel.locator('option').evaluateAll((os) => os.map((o) => o.value));
  await sel.selectOption(opts[7]); await page.waitForTimeout(800);
  const t1 = await (await fetch(`${BASE}/voice/refs?voice=default`)).json();
  ok(t1.current.happy.wav === decodeURIComponent(opts[7]) && t1.custom.includes('happy'), 'happy now uses the picked clip');
  await page.locator('button[data-v="happy"]').click(); await page.waitForFunction(() => window.played.length > 0, null, { timeout: 60000 });
  ok(true, 'preview played');
  await page.locator('button[data-r="happy"]').click(); await page.waitForTimeout(800);
  const t2 = await (await fetch(`${BASE}/voice/refs?voice=default`)).json();
  ok(!t2.custom.includes('happy'), 'reset back to the voice\'s own clip');
  // add an own recording for shy (a JVNV F1 clip stands in), check its transcript field
  await page.evaluate((f) => { window.nextPath = f; }, path.join(process.env.HOME, 'hsin_runs/tts_bench/data/jvnv/jvnv_v1/F1/sad/regular/F1_sad_regular_18.wav'));
  await page.locator('button[data-a="shy"]').click();
  await page.waitForSelector('#refs input.tx', { timeout: 120000 });
  ok(true, 'own clip added, transcript editable: ' + await page.locator('#refs input.tx').inputValue());
  await page.screenshot({ path: path.join(out, 'voice_settings_added.png'), fullPage: true });
} finally { await browser.close(); await fetch(BASE + '/quit', { method: 'POST' }).catch(() => {}); }
console.log(fail ? `${fail} failed` : 'all ok'); process.exit(fail ? 1 : 0);

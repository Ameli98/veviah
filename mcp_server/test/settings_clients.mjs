// The settings page's step 2 (2026-11-13, Q13): found programs, connect / disconnect, steps, other programs.
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url)), out = process.argv[2] ?? '.';
const T = createRequire(import.meta.url)('../../desktop/i18n.js').T.zh;
const browser = await chromium.launch(); let fail = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };
try {
  const page = await browser.newPage({ viewport: { width: 640, height: 1300 } });
  page.on('pageerror', (e) => { console.log('[pageerror]', e.message); fail++; });
  await page.addInitScript(({ t }) => {
    window.calls = [];
    const progs = [
      { id: 'claude-desktop', name: 'Claude Desktop', installed: true, connected: true, seen: true, how: 'file' },
      { id: 'cursor', name: 'Cursor', installed: true, connected: false, how: 'file' },
      { id: 'zed', name: 'Zed', installed: true, manual: 'zed', how: 'manual' },
      { id: 'codex', name: 'Codex', installed: false, how: 'cli' },
    ];
    window.hsin = {
      chatView: async () => ({ mode: 'mcp', providers: ['anthropic'], provider: 'anthropic', kind: 'anthropic', baseUrl: '', model: '', desc: 'auto', persona: '', hasKey: false, local: false, canEncrypt: true, ready: false, hotkey: 'Control+Alt+Space' }), 
      state: async () => ({ t, uiLang: 'zh', model: '/x.vrm', modelName: 'x', instructions: 'INSTR', onTop: true, autostart: false, jiggle: 'weak', copied: false,
        speech: { voice: false }, speechInfo: { settings: {}, state: 'off', voices: [] } }),
      clients: async () => ({ programs: progs, snippets: { json: '{"mcpServers":{}}', toml: '[mcp_servers.veviah-avatar]' } }),
      connectClient: async (id) => { window.calls.push(['connect', id]); progs.find((p) => p.id === id).connected = true; return { ok: true, restart: false }; },
      disconnectClient: async (id) => { window.calls.push(['disconnect', id]); progs.find((p) => p.id === id).connected = false; return { ok: true }; },
      connectClientFile: async (id) => { window.calls.push(['file', id]); return { ok: true }; },
      copyText: async (x) => { window.calls.push(['copy', x]); return true; }, copyInstructions: async () => true,
      setSpeech: async () => {}, setUiLang: async () => {}, setJiggle: async () => 'weak', setOnTop: async () => {}, setAutostart: async () => false,
      voiceApi: async () => ({ supported: false }), voiceAudio: async () => null, choosePath: async () => null,
    };
  }, { t: T });
  await page.goto('file://' + path.join(HERE, '../../desktop/settings.html'));
  await page.waitForTimeout(800);
  const rows = await page.$$eval('#progs tr', (r) => r.map((x) => x.innerText.replace(/\s+/g, ' ')));
  ok(rows.length === 3 && rows[0].includes(T.stSeen) && rows[1].includes(T.stNot), 'only installed programs, with their state: ' + rows.join(' | '));
  ok(await page.evaluate(() => document.getElementById('s2').classList.contains('done')), 'step 2 done (a program started our server)');
  await page.click('#progs button[data-c="cursor"]'); await page.waitForTimeout(300);
  ok((await page.evaluate(() => window.calls)).some(([a, b]) => a === 'connect' && b === 'cursor'), 'Cursor connected with one click');
  ok(await page.$('#progs button[data-d="cursor"]') !== null, 'then it offers disconnect');
  await page.click('#progs button[data-steps="zed"]'); await page.waitForTimeout(200);
  ok(await page.evaluate(() => document.getElementById('otherBox').open) && (await page.textContent('#manualSteps')) === T.manual_zed, 'Zed: the steps open in "other programs"');
  await page.selectOption('#otherProg', 'codex');
  ok((await page.textContent('#snip')).startsWith('[mcp_servers'), 'Codex shows TOML');
  await page.click('#copyJson');
  ok((await page.evaluate(() => window.calls)).some(([a]) => a === 'copy'), 'copy JSON');
  ok(await page.$('#open') === null && await page.$('#connect') === null, 'no Claude-only buttons left');
  await page.screenshot({ path: path.join(out, 'settings_clients.png'), fullPage: true });
} finally { await browser.close(); }
console.log(fail ? `${fail} failed` : 'all ok'); process.exit(fail ? 1 : 0);

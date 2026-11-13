// The built-in chat's pages (2026-11-13): the settings page's chat card and the chat window, in headless Chromium with
// fake bridges (no Electron, no API). Run: node test/chat_pages.mjs OUTDIR
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const out = process.argv[2] ?? '.';
const { T: ALL } = createRequire(import.meta.url)('../../desktop/i18n.js');
const T = ALL.zh;
let fail = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };
const browser = await chromium.launch();
try {
  // ---- settings page ----
  const page = await browser.newPage({ viewport: { width: 640, height: 1400 } });
  page.on('pageerror', (e) => { console.log('[pageerror]', e.message); fail++; });
  await page.addInitScript(({ t }) => {
    window.sets = [];
    const P = { anthropic: ['anthropic', ''], openrouter: ['openai', 'https://openrouter.ai/api/v1'], ollama: ['openai', 'http://localhost:11434/v1'], custom: ['openai', ''] };
    const CLAUDE = [['claude-opus-5-5', 'Claude Opus 5.5', '$4 / $20'], ['claude-sonnet-5-5', 'Claude Sonnet 5.5', '$2 / $10'], ['claude-haiku-5-5', 'Claude Haiku 5.5', '$0.1 / $0.5']].map(([id, name, price]) => ({ id, name, price }));
    const LISTS = { anthropic: CLAUDE, openrouter: Array.from({ length: 30 }, (_, i) => ({ id: `vendor/model-${i}`, name: i === 7 ? 'Anthropic: Claude Sonnet 5.5' : `Model ${i}`, price: '$1 / $2' })),
                    ollama: [{ id: 'llama3', name: 'llama3', price: null }, { id: 'qwen3', name: 'qwen3', price: null }] };
    window.listCalls = 0;
    let v = { mode: 'mcp', provider: 'anthropic', providers: Object.keys(P), kind: 'anthropic', baseUrl: '', model: '', desc: 'auto', persona: '',
              hasKey: false, local: false, canEncrypt: true, ready: false, hotkey: 'Control+Alt+V', hotkeyOk: true };
    const speech = { voice: false, voiceName: null, voiceLang: null, releaseMin: 0 };
    window.hsin = {
      state: async () => ({ t, uiLang: 'zh', model: '/x.vrm', modelName: 'x', instructions: 'INSTR', onTop: true, autostart: false, jiggle: 'weak', copied: true, speech, speechInfo: { settings: speech, state: 'off', voices: [] } }),
      clients: async () => ({ programs: [], other: [], snippets: { json: '{}', toml: '' } }),
      chatView: async () => v,
      setChat: async (x) => {
        window.sets.push(x);
        if (x.provider) { v = { ...v, provider: x.provider, kind: P[x.provider][0], baseUrl: P[x.provider][1], model: '', local: x.provider === 'ollama' }; }
        if (x.mode) v.mode = x.mode;
        if (x.model !== undefined) v.model = x.model;
        if (x.key) v.hasKey = true;
        if (x.hotkey) { if (x.hotkey.includes('Space')) return { ...v, taken: true }; v.hotkey = x.hotkey; }
        v.ready = !!v.model && (v.hasKey || v.local);
        return v;
      },
      chatTest: async () => ({ ok: false, error: 'key' }), chatModels: async () => { window.listCalls++; if (window.badKey) return { ok: false, error: 'key', models: [] }; await new Promise((r) => setTimeout(r, 50)); return LISTS[v.provider] ? { ok: true, models: LISTS[v.provider] } : { ok: false, error: 'nolist', models: [] }; }, openChat: async () => { window.opened = true; },
      setSpeech: async () => {}, setUiLang: async () => {}, setJiggle: async () => {}, setOnTop: async () => {}, setAutostart: async () => {},
      voiceApi: async () => ({ supported: false }), voiceAudio: async () => null, choosePath: async () => null,
    };
  }, { t: T });
  await page.goto('file://' + path.join(HERE, '../../desktop/settings.html'));
  await page.waitForTimeout(600);
  const vis = (id) => page.evaluate((i) => getComputedStyle(document.getElementById(i)).display !== 'none', id);
  const text = (id) => page.evaluate((i) => document.getElementById(i).textContent, id);
  ok(!(await vis('chatBox')) && await vis('s2') && await vis('s3'), 'MCP mode: programs + instructions shown, no chat box');
  await page.check('input[name=mode][value=builtin]'); await page.waitForTimeout(200);
  ok(await vis('chatBox') && !(await vis('s2')) && !(await vis('s3')), 'built-in mode: chat box shown, MCP steps hidden');
  const opts = () => page.evaluate(() => [...document.querySelectorAll('#modelPick option')].map((o) => o.textContent));
  ok(!(await vis('urlRow')) && await vis('keyRow') && (await opts())[0] === T.keyFirst && await page.isDisabled('#modelPick'), 'Anthropic without a key: no URL, key row, "save the key first"');
  await page.fill('#apiKey', 'sk-test'); await page.click('#saveKey'); await page.waitForTimeout(300);
  ok((await page.evaluate(() => window.sets)).some((x) => x.key === 'sk-test') && (await page.inputValue('#apiKey')) === '' && (await page.getAttribute('#apiKey', 'placeholder')) === T.keySaved,
    'key saved through setChat, field emptied, shows "saved"');
  ok((await opts()).join('|') === 'Claude Opus 5.5（$4 / $20）|Claude Sonnet 5.5（$2 / $10）|Claude Haiku 5.5（$0.1 / $0.5）' && !(await vis('searchRow')),
    'the list comes by itself: names with prices, no search box for a short list');
  ok((await page.evaluate(() => window.sets)).some((x) => x.model === 'claude-opus-5-5') && (await page.inputValue('#modelPick')) === 'claude-opus-5-5', 'nothing chosen yet: the first model is taken');
  await page.selectOption('#modelPick', 'claude-sonnet-5-5'); await page.waitForTimeout(200);
  ok((await page.evaluate(() => window.sets)).some((x) => x.model === 'claude-sonnet-5-5') && (await text('priceNote')).includes('$2 / $10'), 'Sonnet picked: saved, its price shown');
  ok(await page.evaluate(() => document.getElementById('sMode').classList.contains('done')), 'with model + key the step is done');
  const calls = await page.evaluate(() => window.listCalls);
  await page.click('#chatTest'); await page.waitForTimeout(200);
  ok((await text('mc')) === T.chatErr.key && (await page.evaluate(() => window.listCalls)) === calls, 'test with a bad key says so; the list is not fetched again');
  await page.evaluate(() => { window.badKey = true; });
  await page.selectOption('#provider', 'ollama'); await page.waitForTimeout(200); await page.selectOption('#provider', 'anthropic'); await page.waitForTimeout(300);
  ok((await opts()).length === 4 && (await text('mc')) === T.chatErr.key, 'Anthropic list refused (bad key): the four Claude models still offered, the key error shown');
  await page.evaluate(() => { window.badKey = false; });
  await page.selectOption('#provider', 'openrouter'); await page.waitForTimeout(200);
  await page.fill('#apiKey', 'sk-or'); await page.click('#saveKey'); await page.waitForTimeout(300);
  ok(await vis('searchRow') && (await opts()).length === 30 && !(await vis('urlRow')) && !(await vis('urlMore')), 'a cloud service: no URL at all; a long list (30) gets a search box');
  await page.fill('#modelSearch', 'sonnet'); await page.waitForTimeout(100);
  ok((await opts()).length <= 2 && (await opts()).some((o) => o.startsWith('Anthropic: Claude Sonnet 5.5')), 'search narrows it: ' + (await opts()).join(' | '));
  await page.selectOption('#provider', 'custom'); await page.waitForTimeout(200);
  await page.fill('#apiKey', 'x'); await page.click('#saveKey'); await page.waitForTimeout(300);
  ok((await text('mc')) === T.chatErr.nolist && await page.isDisabled('#modelPick') && await vis('urlRow'), "'other': URL shown; a service without a model list: not supported, said so");
  await page.selectOption('#provider', 'ollama'); await page.waitForTimeout(300);
  ok(!(await vis('urlRow')) && await vis('urlMore') && !(await vis('keyRow')) && (await text('priceNote')) === T.localNote && (await opts()).join(',') === 'llama3,qwen3', 'Ollama: URL behind a link, no key, local note, its installed models listed');
  await page.click('#showUrl'); await page.waitForTimeout(100);
  ok(await vis('urlRow') && !(await vis('urlMore')), 'the link opens the URL field');
  await page.fill('#persona', '你叫小薇'); await page.click('#descMode'); await page.waitForTimeout(200);
  ok((await page.evaluate(() => window.sets)).some((x) => x.persona === '你叫小薇'), 'persona saved on leaving the box');
  await page.click('#openChat'); await page.waitForTimeout(100);
  ok(await page.evaluate(() => window.opened) && (await text('openChat')).includes('Ctrl+Alt+V') && (await page.inputValue('#hotkey')) === 'Ctrl+Alt+V', 'open chat button and the hotkey field show Ctrl+Alt+V');
  await page.click('#hotkey'); await page.keyboard.press('Control+Alt+KeyK'); await page.waitForTimeout(200);
  ok((await page.evaluate(() => window.sets)).some((x) => x.hotkey === 'Control+Alt+K') && (await page.inputValue('#hotkey')) === 'Ctrl+Alt+K', 'pressing Ctrl+Alt+K in the field sets it');
  await page.click('#hotkey'); await page.keyboard.press('KeyJ'); await page.waitForTimeout(100);
  ok((await text('hotkeyNote')) === T.hotkeyNeedMod, 'a bare letter is refused (needs Ctrl or Alt)');
  await page.keyboard.press('Control+Alt+Space'); await page.waitForTimeout(200);
  ok((await text('hotkeyNote')).includes('被別的程式占用') && (await page.inputValue('#hotkey')) === 'Ctrl+Alt+K', 'a taken combination: said so, the old one kept');
  await page.screenshot({ path: path.join(out, 'chat_settings.png'), fullPage: true });

  // ---- chat window ----
  const cw = await browser.newPage({ viewport: { width: 400, height: 600 } });
  cw.on('pageerror', (e) => { console.log('[pageerror]', e.message); fail++; });
  await cw.addInitScript(({ t }) => {
    let turns = [{ user: '早安', lines: [{ say: '早安！今天也要加油喔。', emotion: 'happy' }], reply: '' }];
    let n = 0;
    window.hsinChat = {
      state: async () => ({ t, view: { ready: true }, history: { turns, summarized: true, session: { in: 15000, out: 300, cost: 0.0012, costKnown: true } } }),
      send: async (text) => {
        await new Promise((r) => setTimeout(r, 300));
        if (++n === 2) return { error: 'rate' };
        const turn = { user: text, lines: [{ say: `回覆：${text}`, emotion: 'happy' }], reply: '' }; turns = [...turns, turn];
        return { turn, session: { in: 30000, out: 600, cost: 0.0025, costKnown: true } };
      },
      clear: async () => { turns = []; }, settings: async () => {},
    };
    window.confirm = () => true;
  }, { t: T });
  await cw.goto('file://' + path.join(HERE, '../../desktop/chat.html'));
  await cw.waitForTimeout(400);
  const n = (sel) => cw.evaluate((s) => document.querySelectorAll(s).length, sel);
  ok((await n('.me')) === 1 && (await n('.her')) === 1 && (await n('.note')) === 1, 'history shown with the summary note');
  ok((await cw.textContent('#cost')).includes('0.0012') && (await cw.textContent('#cost')).includes('15k'), 'session cost line: ' + await cw.textContent('#cost'));
  await cw.fill('#in', '你好'); await cw.keyboard.press('Enter'); await cw.waitForTimeout(100);
  ok((await n('.wait')) === 1 && await cw.isDisabled('#send'), 'while waiting: dots, send disabled');
  await cw.waitForTimeout(400);
  ok((await cw.evaluate(() => [...document.querySelectorAll('.her')].at(-1).textContent)) === '回覆：你好' && (await cw.textContent('#cost')).includes('0.0025'), 'reply shown, cost updated');
  await cw.fill('#in', '再一次'); await cw.keyboard.press('Enter'); await cw.waitForTimeout(500);
  ok((await cw.evaluate(() => document.querySelector('.err')?.textContent)) === T.chatErr.rate && (await n('.wait')) === 0, 'an error replaces the dots');
  await cw.screenshot({ path: path.join(out, 'chat_window.png') });
  await cw.click('#clear'); await cw.waitForTimeout(300);
  ok((await n('.me')) === 0, 'clear empties the window');
} finally { await browser.close(); }
console.log(fail ? `${fail} failed` : 'all ok'); process.exit(fail ? 1 : 0);

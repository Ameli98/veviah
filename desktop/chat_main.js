// Built-in chat in the app (the API version, 2026-11-13, Q1-Q9): settings (mode, provider, model, persona), the API
// key in Electron safeStorage (Q4: encrypted with the Windows user's key, never in plain text and never sent anywhere
// but the chosen API through our own companion), the chat window and its hotkey. The conversation itself runs in the
// companion (mcp_server/chat.mjs, routes /chat*).
const { BrowserWindow, globalShortcut, ipcMain, safeStorage } = require('electron');
const path = require('node:path');

// presets of the provider select; kind = which adapter (chat.mjs); OpenAI-compatible ones share one
const PROVIDERS = {
  anthropic: { kind: 'anthropic', baseUrl: '', model: 'claude-sonnet-5-5' },   // Sonnet for now (the author decides after trying it)
  openai: { kind: 'openai', baseUrl: 'https://api.openai.com/v1', model: '' },
  openrouter: { kind: 'openai', baseUrl: 'https://openrouter.ai/api/v1', model: '' },
  gemini: { kind: 'openai', baseUrl: 'https://generativelanguage.googleapis.com/v1beta/openai', model: '' },
  deepseek: { kind: 'openai', baseUrl: 'https://api.deepseek.com/v1', model: 'deepseek-chat' },
  ollama: { kind: 'ollama', baseUrl: 'http://localhost:11434', model: '', local: true },   // its own API (context size, chat.mjs)
  lmstudio: { kind: 'openai', baseUrl: 'http://localhost:1234/v1', model: '', local: true },
  custom: { kind: 'openai', baseUrl: '', model: '' },
};
// the chat window's hotkey: Ctrl+Alt+V by default (Ctrl+Alt+Space was taken on the author's PC), user-set in the settings
const HOTKEY = 'Control+Alt+V';
const isLocal = (u) => /^https?:\/\/(localhost|127\.|\[::1\]|0\.0\.0\.0)/i.test(u ?? '');

module.exports = function chatMain({ settings, saveSettings, base, tr, texts, instructions, icon, avatar, log, refreshMenu }) {
  const c = () => (settings.chat ??= { mode: 'mcp', provider: 'anthropic', desc: 'auto', persona: '' });
  const provider = () => PROVIDERS[c().provider] ? c().provider : 'anthropic';
  const baseUrl = () => (c().baseUrl ?? PROVIDERS[provider()].baseUrl);
  const model = () => (c().model ?? PROVIDERS[provider()].model);
  const key = () => {
    const enc = settings.chatKeys?.[provider()];
    if (!enc) return '';
    try { return safeStorage.decryptString(Buffer.from(enc, 'base64')); } catch (e) { log('chat key decrypt failed', e.message); return ''; }
  };
  const cfg = () => ({ kind: PROVIDERS[provider()].kind, baseUrl: baseUrl(), model: model(), key: key(), desc: c().desc ?? 'auto' });
  const builtin = () => c().mode === 'builtin';
  // ready = a model, and a key unless the endpoint is on this computer
  const ready = () => !!(model() && (key() || (PROVIDERS[provider()].kind !== 'anthropic' && isLocal(baseUrl()))));
  const system = () => {
    const p = String(c().persona ?? '').trim();
    return instructions() + (p ? `\n\n${tr('personaHead')}\n${p}` : '');
  };
  const post = async (p, b) => (await fetch(`${base()}${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b ?? {}) })).json();
  const get = async (p) => (await fetch(`${base()}${p}`)).json();

  function view() {
    return { mode: c().mode ?? 'mcp', provider: provider(), providers: Object.keys(PROVIDERS), kind: PROVIDERS[provider()].kind,
             baseUrl: baseUrl(), model: model(), desc: c().desc ?? 'auto', persona: c().persona ?? '', hasKey: !!settings.chatKeys?.[provider()],
             local: isLocal(baseUrl()), canEncrypt: safeStorage.isEncryptionAvailable(), ready: ready(), hotkey: hk(), hotkeyOk: !builtin() || registered === hk() };
  }
  function set(v) {
    const ch = c();
    if (v.provider && PROVIDERS[v.provider] && v.provider !== ch.provider) { ch.provider = v.provider; delete ch.baseUrl; delete ch.model; }
    for (const k of ['mode', 'baseUrl', 'model', 'desc', 'persona']) if (v[k] !== undefined) ch[k] = String(v[k]);
    if (typeof v.key === 'string' && v.key.trim()) {
      if (!safeStorage.isEncryptionAvailable()) throw new Error('safeStorage not available');
      (settings.chatKeys ??= {})[provider()] = safeStorage.encryptString(v.key.trim()).toString('base64');
    }
    if (v.clearKey) delete settings.chatKeys?.[provider()];
    let taken = false;
    if (typeof v.hotkey === 'string' && v.hotkey && v.hotkey !== hk()) {
      // try the new one first; a combination another program holds is refused and the old one kept
      const old = hk(); ch.hotkey = v.hotkey;
      if (builtin() && !hotkey()) { ch.hotkey = old; taken = true; }
    }
    saveSettings(); hotkey(); refreshMenu();
    return { ...view(), taken };
  }

  // ---- the chat window (Q3): history + input; the character's lines also play on the avatar ----
  let chatWin = null;
  function open() {
    if (chatWin) { chatWin.show(); chatWin.focus(); return; }
    chatWin = new BrowserWindow({ width: 400, height: 600, title: tr('chatTitle'), icon, autoHideMenuBar: true,
      webPreferences: { preload: path.join(__dirname, 'preload_chat.js') } });
    chatWin.loadFile(path.join(__dirname, 'chat.html'));
    chatWin.on('closed', () => { chatWin = null; });
  }
  function toggle() { if (chatWin?.isVisible() && chatWin.isFocused()) chatWin.hide(); else open(); }
  let registered = null;
  const hk = () => c().hotkey || HOTKEY;
  function hotkey() {           // (re)register the current hotkey while in built-in mode; false = another program holds it
    if (registered) { globalShortcut.unregister(registered); registered = null; }
    if (!builtin()) return true;
    let ok = false;
    try { ok = globalShortcut.register(hk(), toggle); } catch (e) { log('chat hotkey invalid', hk(), e.message); }
    if (ok) registered = hk(); else log('chat hotkey taken', hk());
    return ok;
  }

  ipcMain.handle('chatView', () => view());
  ipcMain.handle('setChat', (_e, v) => set(v ?? {}));
  ipcMain.handle('chatTest', () => post('/chat/test', { cfg: cfg() }));
  ipcMain.handle('chatModels', () => post('/chat/models', { cfg: cfg() }));
  ipcMain.handle('openChat', () => { open(); return true; });
  ipcMain.handle('chatState', async () => ({ t: texts(), view: view(), history: await get('/chat/history').catch(() => ({ turns: [], session: {} })) }));
  async function send(text) {
    if (!ready()) return { error: 'setup' };
    const r = await post('/chat', { text, cfg: cfg(), system: system() }).catch((e) => ({ error: 'app', message: e.message }));
    // a reply with no line on the avatar still shows in her bubble
    if (!r.error && r.turn && !r.turn.lines?.length && r.turn.reply) avatar()?.webContents.send('hint', r.turn.reply);
    if (r.error) log('chat', r.error, r.message ?? '');
    return r;
  }
  ipcMain.handle('chatSend', (_e, text) => send(text));
  ipcMain.handle('chatClear', () => post('/chat/clear'));

  hotkey();
  // the stored key's shape for the self-test (never the key): length, first 7 characters, whitespace
  const keyShape = () => { const k = key(); return { len: k.length, prefix: k.slice(0, 7), space: /\s/.test(k), stored: (settings.chatKeys?.[provider()] ?? '').length }; };
  return { builtin, ready, open, view, send, keyShape, window: () => chatWin, set, menuItem: () => (builtin() ? [{ label: tr('chatMenu'), accelerator: hk(), click: open }] : []) };
};

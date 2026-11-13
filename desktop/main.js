// HSIN desktop companion (Windows Electron; spec in CLAUDE.md, 2026-11-04). Double-clicking the app:
//   1. starts the companion server in WSL (mcp_server/companion.sh <model>) through wsl.exe and waits for it,
//   2. opens a transparent, frameless, always-on-top window showing http://localhost:<port>/?bg=none&app=1,
//   3. stops the companion server again when the app quits.
// Everything the user sets is done in the app (no files to edit by hand): the VRM model, connecting the MCP server
// to Claude Desktop (writes hsin-avatar into claude_desktop_config.json with the same model, keeping a backup), the
// Project instructions to copy, always on top, start with Windows. Settings: %APPDATA%/hsin-companion/settings.json.
// config.json next to this file (written by deploy.sh) says where the repo is in WSL: {distro, repo, port}.
// Avatar window input (preload_avatar.js <- the page's companion.js): transparent pixels click through, dragging the
// avatar moves the window, the wheel scales the window, right-click opens the menu. Closing hides to the tray.
const { app, BrowserWindow, Menu, Tray, clipboard, dialog, globalShortcut, ipcMain, nativeImage, screen, shell } = require('electron');
const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');
const i18n = require('./i18n.js');
// a test copy with its own settings (release checks next to the user's own install): env VEVIAH_USER_DATA
if (process.env.VEVIAH_USER_DATA) app.setPath('userData', process.env.VEVIAH_USER_DATA);

const CFG = JSON.parse(fs.readFileSync(path.join(__dirname, 'config.json'), 'utf-8'));
const PORT = CFG.port ?? 8790;
const BASE = `http://localhost:${PORT}`;
// Backend (2026-11-12, plan step 4): 'native' = everything in <app folder>/backend (built by build_backend.sh), run with
// this very exe as Node (ELECTRON_RUN_AS_NODE) and the embedded Python; 'wsl' = the repo in WSL as before (development).
const NATIVE = CFG.backend === 'native';
const BACKEND = NATIVE ? path.resolve(path.dirname(process.execPath), CFG.backendDir ?? 'backend') : null;
// the default model: the one config.json names, else any bundled .vrm (2026-11-13, the author: a renamed file must never
// stop the app; a release names the sample 'VRM Sample.vrm')
const DEFAULT_MODEL = NATIVE ? (() => {
  const dir = path.join(BACKEND, 'vrm_player', 'public', 'models');
  for (const n of [CFG.defaultModel, 'VRM Sample.vrm', 'VRM1_Constraint_Twist_Sample.vrm']) if (n && fs.existsSync(path.join(dir, n))) return path.join(dir, n);
  try { const any = fs.readdirSync(dir).filter((f) => f.toLowerCase().endsWith('.vrm')).sort()[0]; if (any) return path.join(dir, any); } catch { /* none */ }
  return path.join(dir, CFG.defaultModel ?? 'VRM Sample.vrm');
})() : `${CFG.repo}/vrm_player/public/models/VRM1_Constraint_Twist_Sample.vrm`;
const NAME = 'VEviaH';                 // the product name (2026-11-12; HSIN before)
const modelPath = (p) => (NATIVE ? p : toWsl(p));        // a path from a Windows dialog, as the backend sees it
const shownPath = (p) => (NATIVE ? p : toWin(p));        // and back, for the dialog / the settings page
// the same file written as a WSL path and as its Windows path (\\\\wsl.localhost\\...) counts as the same
const samePath = (a, b) => !!a && !!b && (a === b || (a.startsWith('/') ? toWin(a) : a).toLowerCase() === (b.startsWith('/') ? toWin(b) : b).toLowerCase());
const baseName = (p) => (NATIVE ? path.win32 : path.posix).basename(p, '.vrm');
const ICON = path.join(__dirname, 'icon.png');
const SELFTEST = (() => { const i = process.argv.indexOf('--selftest'); return i >= 0 ? process.argv[i + 1] : null; })();
// the Project instructions, in the interface language (i18n.js); the voice language reaches Claude through the
// Language line of every perform result (mcp_server/server.mjs)
const instructions = () => i18n.t(uiLang(), 'instructions');

// ---------- log (what the window did, for Claude to read from WSL: %APPDATA%/hsin-companion/log.txt) ----------
const LOG = path.join(app.getPath('userData'), 'log.txt');
function log(...a) {
  try { fs.mkdirSync(path.dirname(LOG), { recursive: true }); fs.appendFileSync(LOG, `${new Date().toISOString()} ${a.join(' ')}\n`); }
  catch { /* nothing */ }
}
process.on('uncaughtException', (e) => log('uncaught', e.stack ?? e));
process.on('unhandledRejection', (e) => log('unhandled', e?.stack ?? e));

// ---------- settings ----------
const SETTINGS = path.join(app.getPath('userData'), 'settings.json');
// renamed HSIN -> VEviaH (2026-11-12): the old settings (%APPDATA%\\hsin-companion) are taken over once; a model path in
// WSL form becomes its \\wsl.localhost path for the native backend
try {
  const old = path.join(app.getPath('appData'), 'hsin-companion', 'settings.json');
  if (CFG.distro && !process.env.VEVIAH_USER_DATA && !fs.existsSync(SETTINGS) && fs.existsSync(old)) {   // dev builds only (a release has no HSIN before it)
    const o = JSON.parse(fs.readFileSync(old, 'utf-8'));
    if (NATIVE && o.model?.startsWith('/')) o.model = toWin(o.model);
    fs.mkdirSync(path.dirname(SETTINGS), { recursive: true }); fs.writeFileSync(SETTINGS, JSON.stringify(o, null, 2));
  }
} catch { /* start fresh */ }
let settings = (() => { try { return JSON.parse(fs.readFileSync(SETTINGS, 'utf-8')); } catch { return {}; } })();
// interface language: the user's choice, else the Windows locale (zh / en / ja)
const uiLang = () => (i18n.LANGS.includes(settings.uiLang) ? settings.uiLang : i18n.fromLocale(app.getLocale()));
const tr = (key, vars) => i18n.t(uiLang(), key, vars);
function saveSettings() {
  fs.mkdirSync(path.dirname(SETTINGS), { recursive: true });
  fs.writeFileSync(SETTINGS, JSON.stringify(settings, null, 2));
}

// A Windows path to the same file as seen from WSL (C:\x -> /mnt/c/x, \\wsl.localhost\<distro>\x -> /x), and back.
function toWsl(p) {
  if (!p || p.startsWith('/')) return p;
  const unc = p.match(/^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)$/i);
  if (unc) return unc[1].replace(/\\/g, '/');
  const drive = p.match(/^([A-Za-z]):\\(.*)$/);
  if (drive) return `/mnt/${drive[1].toLowerCase()}/${drive[2].replace(/\\/g, '/')}`;
  return p;
}
function toWin(p) {
  const mnt = p.match(/^\/mnt\/([a-z])\/(.*)$/);
  if (mnt) return `${mnt[1].toUpperCase()}:\\${mnt[2].replace(/\//g, '\\')}`;
  return `\\\\wsl.localhost\\${CFG.distro}${p.replace(/\//g, '\\')}`;
}

// ---------- AI programs (2026-11-13, Q13: any MCP program alike; clients.js) ----------
const clients = require('./clients');
// the MCP entry every program gets: no model path (Q13a, the server asks the app), so it never needs rewriting when
// the model changes; it changes only when the app moves (statusAll reports such an entry as stale)
const mcpEntry = () => NATIVE
  ? { command: process.execPath, args: [path.join(BACKEND, 'mcp_server', 'server.mjs')], env: { ELECTRON_RUN_AS_NODE: '1' } }
  : { command: 'wsl.exe', args: ['-d', CFG.distro, '-e', `${CFG.repo}/mcp_server/start.sh`] };
async function seenClients() {
  try { const r = await fetch(`${BASE}/clients`, { signal: AbortSignal.timeout(800) }); return r.ok ? await r.json() : {}; }
  catch { return {}; }
}

// ---------- companion server in WSL ----------
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// the avatar page; look = rendering switches (vrm_player/src/look.js; omitted = its defaults, the self-test compares none / all)
const avatarUrl = (look = null) => `${BASE}/?bg=none&app=1&scale=${settings.scale ?? 0.5}&yaw=${settings.yaw ?? 0}&jiggle=${jiggle()}${look ? `&look=${look}` : ''}`;
async function info() {
  try { const r = await fetch(`${BASE}/model/info`, { signal: AbortSignal.timeout(800) }); return r.ok ? await r.json() : null; }
  catch { return null; }
}
async function stopCompanion() {
  try { await fetch(`${BASE}/quit`, { method: 'POST', signal: AbortSignal.timeout(800) }); } catch { /* not running */ }
  for (let i = 0; i < 20 && await info(); i++) await sleep(150);
}
// ---- cloth window (2026-11-13; Q10 / Q11): the solver's cloth-first pass (a skirted model's hand resting on a leg)
// needs the skirt draped on the legs first. The companion relays POST /cloth here over its IPC channel; a hidden
// window runs the player page (vrm_player/dist-player, served at /player/) and its window.hsin.clothSurface, as
// render_server.mjs does through Playwright in WSL. Opened on the first such request, kept until the app quits or
// the model changes; never for models without a skirt (the solver only asks for those).
let clothWin = null, clothReady = null;
function clothWindow() {
  if (clothReady) return clothReady;
  clothWin = new BrowserWindow({ show: false, width: 256, height: 256, webPreferences: { backgroundThrottling: false } });
  clothWin.on('closed', () => { clothWin = null; clothReady = null; });
  const q = new URLSearchParams({ ui: '0', t: '0', model: '/model.vrm', motion: 'none' });
  clothReady = clothWin.loadURL(`${BASE}/player/?${q}`)
    // the page's own redraw loop is useless here (every command poses explicitly): stop it, as render_server does
    .then(() => clothWin.webContents.executeJavaScript('window.hsin.ready.then(() => { window.hsin.state.manual = true; return true; })'))
    .then(() => log('cloth window ready'));
  clothReady.catch((e) => { log('cloth window failed', String(e)); clothWin?.destroy(); clothWin = null; clothReady = null; });
  return clothReady;
}
function closeClothWindow() { clothWin?.destroy(); clothWin = null; clothReady = null; }
async function clothRequest(child, m) {
  try {
    await clothWindow();
    const points = m.motion ? await clothWin.webContents.executeJavaScript(
      `(async (motion, frames) => { await window.hsin.loadMotion(motion); return window.hsin.clothSurface(frames); })(${JSON.stringify(m.motion)}, ${JSON.stringify(m.frames)})`) : [];
    child.send({ type: 'cloth-result', id: m.id, points });
  } catch (e) {
    log('cloth request failed', String(e));
    try { child.send({ type: 'cloth-result', id: m.id, error: String(e.message ?? e) }); } catch { /* companion gone */ }
  }
}

async function startCompanion(model) {
  await stopCompanion();                  // one left over, or running with another model, goes first
  closeClothWindow();                     // it holds the old model
  if (NATIVE) {
    const out = fs.openSync(path.join(app.getPath('userData'), 'companion.log'), 'w');
    const child = spawn(process.execPath, [path.join(BACKEND, 'mcp_server', 'companion.mjs')], { windowsHide: true, detached: false, stdio: ['ignore', out, out, 'ipc'],
      env: { ...process.env, ELECTRON_RUN_AS_NODE: '1', HSIN_COMPANION_PORT: String(PORT), HSIN_VRM_MODEL: model } });
    child.on('message', (m) => { if (m?.type === 'cloth') clothRequest(child, m); });
  } else spawn('wsl.exe', ['-d', CFG.distro, '-e', `${CFG.repo}/mcp_server/companion.sh`, model], { windowsHide: true, detached: false });
  for (let i = 0; i < 100; i++) {         // up to ~30 s (the first WSL start can be slow)
    if (await info()) { await pushSpeech(); return true; }
    await sleep(300);
  }
  return false;
}
// Speech settings live in the app and are sent to the companion (mcp_server/speech.mjs) at start and on every change;
// with voice on the companion starts the TTS at once (the ~20 s model load happens while the app opens).
const speechSettings = () => ({ voice: settings.voice ?? true, voiceName: settings.voiceName ?? null, voiceLang: settings.voiceLang ?? null,
                                uiLang: uiLang(), romaji: true,   // always (2026-11-06, the author: not a user setting)
                                 releaseMin: Number(settings.releaseMin) || 0 });
async function pushSpeech() {
  try {
    const r = await fetch(`${BASE}/settings`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(speechSettings()), signal: AbortSignal.timeout(3000) });
    return r.ok ? await r.json() : null;
  } catch (e) { log('pushSpeech', String(e)); return null; }
}
async function speechInfo() {
  try { const r = await fetch(`${BASE}/settings`, { signal: AbortSignal.timeout(1500) }); return r.ok ? await r.json() : null; }
  catch { return null; }
}

// ---------- windows ----------
let win = null, settingsWin = null, tray = null, quitting = false, chat = null;
let winSize = null;
const place = (x, y) => win.setBounds({ x: Math.round(x), y: Math.round(y), ...winSize });     // the avatar window's size, fixed (setBounds at 150 % scaling grew it by 1 px a time, 2026-11-11)
// --selftest-model <vrm>: a self-test with another model (e.g. a skirted one for the cloth step); never saved
const SELFTEST_MODEL = (() => { const i = process.argv.indexOf('--selftest-model'); return i >= 0 ? process.argv[i + 1] : null; })();
// a saved model that is gone (deleted, moved, or a bundled model renamed by an update: 2026-11-13 test2.vrm -> Vivi.vrm
// made the companion crash at start) falls back to the default model; the setting is cleared once
function savedModel() {
  const m = settings.model;
  if (!m) return null;
  let ok = false;
  try { ok = fs.existsSync(NATIVE || !m.startsWith('/') ? m : toWin(m)); } catch { ok = false; }
  if (ok) return m;
  log('saved model missing, using the default:', m);
  delete settings.model; saveSettings();
  return null;
}
const currentModel = () => (SELFTEST && SELFTEST_MODEL) || savedModel() || DEFAULT_MODEL;
const onTop = () => settings.onTop ?? true;
// MCP mode: connected = some AI program, once, + instructions copied; built-in chat: a model and a key (or a local endpoint)
const setupDone = () => !!(chat?.builtin() ? chat.ready() : settings.connected && settings.copied);   // the default model counts as chosen (2026-11-11)

// The window has a fixed size (the screen's whole work-area height, the avatar's aspect; was 85 %, too small to zoom
// in far, 2026-11-04); the avatar is scaled inside it,
// its feet at the bottom middle, and the transparent rest clicks through. Only the position is kept.
const ASPECT = 0.62;
function windowBounds() {
  const wa = screen.getPrimaryDisplay().workArea;
  const height = wa.height, width = Math.round(height * ASPECT);
  // positions saved before the window was full height (no v: 2) keep only their x
  const sb = settings.bounds;
  // dragged anywhere (the author, 2026-11-04: up and down too); positions saved before that (no v: 3) keep only x
  const p = { x: sb?.x ?? wa.x + wa.width - width - 20, y: sb?.v === 3 ? sb.y : wa.y };
  const d = screen.getDisplayNearestPoint({ x: p.x + Math.round(width / 2), y: p.y + height - 1 }).workArea;
  return { ...keepOnScreen(p.x, p.y, width, height, d), width, height };
}
// at least a quarter of the window (the avatar stands in its middle) stays on the screen; a bottom edge within
// SNAP_PX of the taskbar's top snaps onto it, so she stands on it again without aiming
const SNAP_PX = 12;   // (30 pulled her onto the taskbar from too far, the author 2026-11-04)
function keepOnScreen(x, y, width, height, d) {
  x = Math.min(Math.max(x, d.x - Math.round(width / 4)), d.x + d.width - Math.round(width * 3 / 4));
  y = Math.min(Math.max(y, d.y - Math.round(height * 3 / 4)), d.y + d.height - Math.round(height / 4));
  if (Math.abs(y + height - (d.y + d.height)) < SNAP_PX) y = d.y + d.height - height;
  return { x: Math.round(x), y: Math.round(y) };
}
function homeWindow() {
  if (!win) return;
  const wa = screen.getPrimaryDisplay().workArea, b = win.getBounds();
  place(wa.x + wa.width - b.width - 20, wa.y + wa.height - b.height);
  keepBounds();
}
function createAvatarWindow() {
  const b = windowBounds();
  winSize = { width: b.width, height: b.height };
  win = new BrowserWindow({
    ...b, transparent: true, frame: false, hasShadow: false, resizable: false, alwaysOnTop: onTop(),
    backgroundColor: '#00000000', icon: ICON, title: NAME, skipTaskbar: true, show: false,
    webPreferences: { preload: path.join(__dirname, 'preload_avatar.js'), backgroundThrottling: false,
                      autoplayPolicy: 'no-user-gesture-required' },          // the voice plays without a click
  });
  win.loadURL(avatarUrl());
  win.once('ready-to-show', () => win.show());
  win.on('close', (e) => { if (!quitting) { e.preventDefault(); win.hide(); refreshMenu(); } });   // to the tray
  win.on('closed', () => { win = null; });
}
function keepBounds() { if (win) { const b = win.getBounds(); settings.bounds = { x: b.x, y: b.y, v: 3 }; saveSettings(); } }

function openSettings() {
  if (settingsWin) { settingsWin.show(); settingsWin.focus(); return; }
  settingsWin = new BrowserWindow({
    width: 620, height: 780, title: tr('title'), icon: ICON, autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js') },
  });
  settingsWin.loadFile(path.join(__dirname, 'settings.html'));
  settingsWin.on('closed', () => { settingsWin = null; });
}

function buildMenu() {
  return Menu.buildFromTemplate([
    { label: tr('menuSettings'), click: openSettings },
    ...(chat?.menuItem() ?? []),
    { label: win?.isVisible() ? tr('hide') : tr('show'), click: () => { if (win?.isVisible()) win.hide(); else win?.show(); refreshMenu(); } },
    { label: tr('onTop'), type: 'checkbox', checked: onTop(), click: (m) => setOnTop(m.checked) },
    { label: tr('home'), click: homeWindow },
    { label: tr('voiceMenu'), type: 'checkbox', checked: settings.voice ?? true, click: (m) => setSpeech({ voice: m.checked }) },
    { label: tr('jiggle'), submenu: Object.keys(JIGGLE).map((k) => ({ label: tr('jiggleLevels')[k], type: 'radio', checked: jiggle() === k, click: () => setJiggle(k) })) },
    { type: 'separator' },
    { label: tr('quit'), click: () => app.quit() },
  ]);
}
function refreshMenu() { tray?.setContextMenu(buildMenu()); }
// (the personality sets and their menu were removed 2026-11-11, the author: one set of motions, mcp_server/sets/default.json)
// chest sway (vrm_player/src/jiggle.js), weak by default (2026-11-04, the author)
const JIGGLE = { off: '關', weak: '弱', medium: '中', strong: '強' };
const jiggle = () => (JIGGLE[settings.jiggle] ? settings.jiggle : 'weak');
function setJiggle(level) { if (!JIGGLE[level]) return; settings.jiggle = level; saveSettings(); win?.webContents.send('jiggle', level); refreshMenu(); }
// voice on / off, voice, voice language, romaji, release minutes (any subset); interface language
async function setSpeech(v) {
  for (const k of ['voice', 'voiceName', 'voiceLang', 'releaseMin']) if (k in v) settings[k] = v[k];
  saveSettings(); refreshMenu();
  return pushSpeech();
}
function setUiLang(l) {
  if (!i18n.LANGS.includes(l)) return uiLang();
  settings.uiLang = l; saveSettings(); refreshMenu(); pushSpeech();
  settingsWin?.setTitle(tr('title'));
  return l;
}
function setOnTop(on) { settings.onTop = on; saveSettings(); win?.setAlwaysOnTop(on); refreshMenu(); }

// ---------- avatar window input (preload_avatar.js) ----------
let drag = null;
ipcMain.on('solid', (_e, on) => { win?.setIgnoreMouseEvents(!on, { forward: true }); log('solid', on); });
// dragging keeps the size fixed: setPosition on a transparent window at 150 % scaling re-derived the size on every
// move and the window flickered (the author, 2026-11-04)
ipcMain.on('dragStart', (_e, x, y) => { if (win) { drag = { x, y, b: win.getBounds() }; log('dragStart', x, y, JSON.stringify(drag.b)); } });
// ---------- window snapping (2026-11-11, the author: sit on a window's top edge, lean on its side) ----------
// Other windows come from windows.js (user32 through koffi), in Z order. A drop with her floor row within LEDGE_PX of a
// visible top edge (her middle over the window, the edge not covered there) seats her on it; a drop with one shoulder
// within WALL_PX of a window's left / right side that spans her shoulder height leans her on it. While snapped the
// window follows the target (WATCH_MS); a target that goes away (minimized, closed, maximized, now covered) drops her
// onto the taskbar.
const SEAT_PX = 24, WALL_PX = 30, EDGE_MARGIN = 30, DRAG_LIST_MS = 100, WATCH_MS = 250;
// (2026-11-11, the author: ledge = her bottom on the top edge; wall = the side of her body next to a window's side; round 3
// had wide zones around her middle and snapped from too far)
let wins = null, winsAt = 0, snapped = null;
const firstSeen = new Map();     // window id -> when it was first listed (a window shown < NEW_MS ago does not cover yet)
const NEW_MS = 2000;      // snapped = { kind, side, id, ref: target rect when attached }
// listed once when a drag starts (and again for the drop's decision / the watch): EnumWindows + DWM per window runs on
// the main process and refreshing it every 100 ms during a drag held up the window moves (2026-11-11, the author: she lagged
// behind the mouse)
const winList = () => {
  if (process.platform !== 'win32') return [];
  if (!wins) {
    try {
      const own = new Set([win, settingsWin].filter(Boolean).map((w) => w.getNativeWindowHandle().readBigUInt64LE(0)));
      wins = require('./windows').listWindows(own); winsAt = Date.now();
      for (const w of wins) if (!firstSeen.has(w.id)) firstSeen.set(w.id, winsAt);
    } catch (e) { log('window list failed', e.message); wins = []; }
  }
  return wins;
};
const covered = (list, i, x, y) => list.slice(0, i).some((o) => x >= o.x && x <= o.x + o.width && y >= o.y && y <= o.y + o.height);
// the snap a drop with the window at bounds b would make: { kind: 'ledge' | 'wall', side, id, w, dx, dy } or null
// (dx, dy = how far the window moves to sit / lean there; the page moves the landing shadow by it)
// ledge: her feet up to LEDGE_UP px above a top edge (or LEDGE_DOWN below it), her middle over the window, the edge
// visible there; wall: her middle within WALL_SPAN body widths of a left / right side, on either side of it, the side
// spanning her shoulder height and visible there; maximized / full-screen windows never
function findTarget(b, groundY, box) {
  const list = winList();
  if (!list.length || !box) return null;
  const seat = b.y + box.seat, cx = b.x + box.cx;
  let best = null;
  list.forEach((w, i) => {         // her bottom within SEAT_PX of a visible top edge, her middle over the window
    if (w.maximized || best) return;
    if (Math.abs(seat - w.y) < SEAT_PX && cx > w.x + EDGE_MARGIN && cx < w.x + w.width - EDGE_MARGIN && !covered(list, i, cx, w.y + 2))
      best = { kind: 'ledge', side: null, id: w.id, w, dx: 0, dy: w.y - seat };
  });
  if (best) return best;
  const sy = b.y + box.sy, l = b.x + box.l, r = b.x + box.r;
  let near = WALL_PX;
  list.forEach((w, i) => {         // one side of her body within WALL_PX of a window's side that spans her shoulders
    if (w.maximized || sy < w.y + 10 || sy > w.y + w.height - 10) return;
    // a window to the RIGHT of her on the screen is at her LEFT (she faces the viewer)
    const dl = Math.abs(w.x - r), dr = Math.abs(w.x + w.width - l);
    if (dl < near && !covered(list, i, w.x + 2, sy)) { near = dl; best = { kind: 'wall', side: 'left', id: w.id, w, dx: w.x - r, dy: 0 }; }
    if (dr < near && !covered(list, i, w.x + w.width - 2, sy)) { near = dr; best = { kind: 'wall', side: 'right', id: w.id, w, dx: w.x + w.width - l, dy: 0 }; }
  });
  return best;
}
const plain = (t) => (t ? { kind: t.kind, side: t.side, dx: t.dx, dy: t.dy } : null);
ipcMain.on('dragMove', (_e, x, y, groundY, box) => {
  if (!win || !drag) return;
  const { b } = drag;
  const nx = Math.round(b.x + x - drag.x), ny = Math.round(b.y + y - drag.y);
  place(nx, ny);
  // tell the page whether a drop here snaps onto the taskbar (the same test as dragEnd), for the landing shadow
  const d = screen.getDisplayMatching({ x: nx, y: ny, width: b.width, height: b.height }).workArea;
  const p = keepOnScreen(nx, ny, b.width, b.height, d), bar = d.y + d.height;
  const feet = Number.isFinite(groundY) ? p.y + groundY : null;
  const dy = feet !== null && Math.abs(feet - bar) < SNAP_PX ? bar - feet + (p.y - ny) : null;
  if (dy !== drag.snap) { drag.snap = dy; win.webContents.send('snap', dy); }
  const t = findTarget({ x: nx, y: ny, width: b.width, height: b.height }, groundY, box);
  const key = t ? `${t.kind}:${t.side}:${t.id}` : '';
  if (key !== drag.target) { drag.target = key; win.webContents.send('target', plain(t)); }
});
ipcMain.on('dragStart', () => { clearInterval(watchTimer); watchTimer = null; snapped = null; wins = null; winList(); });   // lifted: off whatever she was on
ipcMain.on('dragEnd', (_e, groundY, box) => {
  let t = null;
  if (win && drag) {
    const b = win.getBounds(), d = screen.getDisplayMatching(b).workArea;
    wins = null;                                        // a fresh list for the decision
    t = findTarget(b, groundY, box);
    let p = keepOnScreen(b.x, b.y, drag.b.width, drag.b.height, d);
    // her feet (the floor under her, groundY px from the window top) within SNAP_PX of the taskbar: onto it
    const feet = Number.isFinite(groundY) ? p.y + groundY : null, bar = d.y + d.height;
    if (!t && feet !== null && Math.abs(feet - bar) < SNAP_PX) p = { x: p.x, y: Math.round(p.y + bar - feet) };
    if (!t) place(p.x, p.y);
    log('dragEnd groundY', groundY, 'feet', feet, 'bar', bar, 'target', JSON.stringify(plain(t)), t?.id ?? '');
  }
  win?.webContents.send('snap', null);
  win?.webContents.send('target', null);
  snapped = t ? { kind: t.kind, side: t.side, id: t.id, ref: { x: t.w.x, y: t.w.y, width: t.w.width, height: t.w.height } } : null;
  win?.webContents.send('attach', plain(t));
  drag = null; keepBounds(); log('dragEnd', JSON.stringify(win?.getBounds())); });
// the page shows the snapped pose: line her up with the edge, then follow the target
ipcMain.on('attached', (_e, groundY, box) => {
  if (!win || !snapped) return;
  const b = win.getBounds(), w = snapped.ref;
  let { x, y } = b;
  if (snapped.kind === 'ledge' && Number.isFinite(groundY)) y = Math.round(w.y - groundY);
  if (snapped.kind === 'wall' && box) x = Math.round(snapped.side === 'left' ? w.x - box.r : w.x + w.width - box.l);
  if (box) { snapped.cx = box.cx; snapped.sy = box.sy; }
  place(x, y);
  keepBounds(); watch();
  log('attached', snapped.kind, snapped.side, JSON.stringify(win.getBounds()));
});
let watchTimer = null;
// Follow the target: its rectangle alone once per frame the page draws ('follow', synced to the display; a 16 ms timer
// here stepped out of time and the movement stuttered, the author 2026-11-11), the full list (covered edge, full screen,
// gone) every WATCH_MS.
function followStep(w) {
  if (!win || !snapped || drag || !w) return;
  const b = win.getBounds(), r = snapped.ref;
  // a wall follows sideways only; once its side no longer spans her shoulders (the window was moved up / down) she
  // stops leaning on air and stands up straight (2026-11-12, the author's screenshot), as when the window goes away
  if (snapped.kind === 'wall' && Number.isFinite(snapped.sy)) {
    const sy = b.y + snapped.sy;
    if (sy < w.y + 10 || sy > w.y + w.height - 10) {
      log('snap target gone: the wall no longer spans her shoulders');
      clearInterval(watchTimer); watchTimer = null; snapped = null; win.webContents.send('detach'); return;
    }
  }
  const dx = snapped.kind === 'wall' && snapped.side === 'right' ? (w.x + w.width) - (r.x + r.width) : w.x - r.x;
  const dy = w.y - r.y;
  if (dx || dy) {
    place(b.x + dx, snapped.kind === 'ledge' ? b.y + dy : b.y);
    snapped.ref = { x: w.x, y: w.y, width: w.width, height: w.height };
    clearTimeout(followStep.save); followStep.save = setTimeout(keepBounds, 400);
  }
}
let cheapOk = true;
ipcMain.on('follow', () => {
  if (!snapped || drag || !cheapOk) return;
  try { followStep(require('./windows').windowRect(snapped.id)); }
  catch (e) { cheapOk = false; log('windowRect failed', e.message); }
});
function watch() {
  clearInterval(watchTimer);
  watchTimer = setInterval(() => {
    if (!win || !snapped || drag) return;
    const b = win.getBounds();
    wins = null;
    const list = winList(), i = list.findIndex((o) => o.id === snapped.id), w = list[i];
    const cov = w && snapped.kind === 'ledge' ? list.slice(0, i).find((o) => { const x = b.x + (snapped.cx ?? b.width / 2); return !o.overlay && Date.now() - (firstSeen.get(o.id) ?? 0) >= NEW_MS && x >= o.x && x <= o.x + o.width && w.y + 2 >= o.y && w.y + 2 <= o.y + o.height; }) : null;
    const why = !w ? 'window gone or minimized' : w.maximized ? (w.full ? 'window full screen' : 'window maximized') : cov ? `edge covered by ${cov.id} ${cov.cls} ${JSON.stringify([cov.x, cov.y, cov.width, cov.height])}` : null;
    if (why) { log('snap target gone:', why); clearInterval(watchTimer); watchTimer = null; snapped = null; win.webContents.send('detach'); return; }
    if (!cheapOk) followStep(w);              // the per-frame call failed: follow here instead
  }, WATCH_MS);
}
// the target went away: down onto the taskbar under her (her floor row, groundY px from the window top)
ipcMain.on('fall', (_e, groundY) => {
  if (!win) return;
  const b = win.getBounds(), disp = screen.getDisplayMatching(b);
  wins = null;
  const full = winList().some((o) => o.full);                    // a full-screen window hides the taskbar: the screen's bottom
  const d = full ? disp.bounds : disp.workArea;
  if (Number.isFinite(groundY)) place(b.x, Math.round(d.y + d.height - groundY));
  keepBounds(); log('fell', JSON.stringify(win.getBounds()));
});
// The wheel scales the avatar inside the window (stage.setScale in the page); the window keeps its size, the app
// only keeps the scale (2026-11-04: resizing the transparent window jumped around and flickered).
ipcMain.on('zoom', (_e, sc) => { settings.scale = sc; clearTimeout(zoomSave); zoomSave = setTimeout(saveSettings, 500); });
let zoomSave = null;
ipcMain.on('turn', (_e, deg) => { settings.yaw = deg; clearTimeout(zoomSave); zoomSave = setTimeout(saveSettings, 500); });
ipcMain.on('menu', () => {
  log('menu');
  if (!settings.menuSeen) { settings.menuSeen = true; saveSettings(); }   // the right-click hint is no longer needed (Q14)
  if (win) buildMenu().popup({ window: win });
});
ipcMain.on('pagelog', (_e, ...a) => log('page', ...a));

// ---------- settings page API (preload.js) ----------
ipcMain.handle('state', async () => {
  const model = currentModel();
  return { model: shownPath(model), modelName: baseName(model), chosen: !!settings.model,
           copied: !!settings.copied, onTop: onTop(), jiggle: jiggle(), autostart: app.getLoginItemSettings().openAtLogin,
           instructions: instructions(), running: !!(await info()),
           uiLang: uiLang(), t: i18n.T[uiLang()], speech: speechSettings(), speechInfo: await speechInfo() };
});
// the AI programs: installed / connected / seen (started our server) / stale; connect / disconnect / pick a file
ipcMain.handle('clients', async () => {
  const st = await clients.statusAll(mcpEntry(), await seenClients());
  if (!settings.connected && st.some((c) => (c.connected && !c.stale) || c.seen)) { settings.connected = true; saveSettings(); }
  return { programs: st, snippets: clients.snippets(mcpEntry()) };
});
ipcMain.handle('connectClient', async (_e, id) => {
  const r = await clients.connect(id, mcpEntry(), (url) => shell.openExternal(url));
  if (r.ok) { settings.connected = true; saveSettings(); }
  log('connect', id, JSON.stringify(r));
  return r;
});
ipcMain.handle('disconnectClient', async (_e, id) => { const r = await clients.disconnect(id); log('disconnect', id, JSON.stringify(r)); return r; });
ipcMain.handle('connectClientFile', async (_e, id) => {
  const r = await dialog.showOpenDialog(settingsWin, { title: tr('pickConfig'), properties: ['openFile'], filters: [{ name: 'JSON', extensions: ['json'] }] });
  if (r.canceled || !r.filePaths[0]) return { ok: false, canceled: true };
  const out = clients.connectFile(id, r.filePaths[0], mcpEntry());
  if (out.ok) { settings.connected = true; saveSettings(); }
  return out;
});
let creditsWin = null;
ipcMain.handle('openCredits', () => {
  if (creditsWin) { creditsWin.show(); creditsWin.focus(); return true; }
  creditsWin = new BrowserWindow({ width: 640, height: 720, title: 'VEviaH', icon: ICON, autoHideMenuBar: true });
  creditsWin.loadFile(path.join(__dirname, 'credits.html'));
  creditsWin.on('closed', () => { creditsWin = null; });
  return true;
});
ipcMain.handle('openSettingsPage', () => { openSettings(); return true; });
ipcMain.handle('copyText', (_e, text) => { clipboard.writeText(String(text ?? '')); return true; });
ipcMain.handle('chooseModel', async () => {
  const r = await dialog.showOpenDialog(settingsWin, { title: tr('chooseVrm'), properties: ['openFile'],
    defaultPath: shownPath((NATIVE ? path.win32 : path.posix).dirname(currentModel())), filters: [{ name: 'VRM', extensions: ['vrm'] }] });
  if (r.canceled || !r.filePaths[0]) return { ok: false, canceled: true };
  settings.model = modelPath(r.filePaths[0]); saveSettings();
  const ok = await startCompanion(settings.model);
  win?.reload();
  return { ok };                         // the MCP server asks the app for the model (Q13a): no config to rewrite
});
ipcMain.handle('useModel', async () => { settings.model = currentModel(); saveSettings(); return true; });
ipcMain.handle('copyInstructions', () => { clipboard.writeText(instructions()); settings.copied = true; saveSettings(); return true; });
ipcMain.handle('setSpeech', (_e, v) => setSpeech(v ?? {}));
ipcMain.handle('setUiLang', (_e, l) => setUiLang(l));
ipcMain.handle('setOnTop', (_e, on) => { setOnTop(!!on); return true; });
ipcMain.handle('setJiggle', (_e, level) => { setJiggle(level); return jiggle(); });
ipcMain.handle('setAutostart', (_e, on) => { app.setLoginItemSettings({ openAtLogin: !!on }); return app.getLoginItemSettings().openAtLogin; });
// reference clips per emotion / the user's own voice (2026-11-12): passed through to the companion's /voice/* routes
ipcMain.handle('voiceApi', async (_e, method, p, b) => {
  try {
    const r = await fetch(`${BASE}${p}`, { method, headers: { 'content-type': 'application/json' }, body: b ? JSON.stringify(b) : undefined,
                                           signal: AbortSignal.timeout(method === 'POST' && p === '/voice/add' ? 900000 : 120000) });
    return await r.json();
  } catch (e) { return { error: String(e.message ?? e) }; }
});
ipcMain.handle('voiceAudio', async (_e, p) => {      // wav bytes for the page (its CSP allows blob: media only)
  try { const r = await fetch(`${BASE}${p}`, { signal: AbortSignal.timeout(120000) }); return r.ok && /audio/.test(r.headers.get('content-type')) ? Buffer.from(await r.arrayBuffer()) : { error: (await r.json().catch(() => ({}))).error ?? r.status }; }
  catch (e) { return { error: String(e.message ?? e) }; }
});
ipcMain.handle('choosePath', async (_e, kind) => {   // kind: wav | ckpt | pth | folder -> a WSL path
  const props = kind === 'folder' ? ['openDirectory'] : ['openFile'];
  const filters = { wav: [{ name: 'WAV', extensions: ['wav'] }], ckpt: [{ name: 'GPT (.ckpt)', extensions: ['ckpt'] }], pth: [{ name: 'SoVITS (.pth)', extensions: ['pth'] }] }[kind];
  const r = await dialog.showOpenDialog(settingsWin, { properties: props, filters });
  return r.canceled || !r.filePaths[0] ? null : modelPath(r.filePaths[0]);
});

// ---------- installer hooks (2026-11-13, the setup exe): run, do one thing, exit; no window, no instance lock ----------
//   --autostart-on        the installer's 'start with Windows' box (the same login item the settings page sets)
//   --uninstall-cleanup   before uninstalling: our entry out of every AI app it was added to (only entries that start
//                         this very exe; other apps' settings untouched), and no login item
const HOOK = ['--autostart-on', '--uninstall-cleanup'].find((f) => process.argv.includes(f));
if (HOOK) {
  app.whenReady().then(async () => {
    try {
      if (HOOK === '--autostart-on') app.setLoginItemSettings({ openAtLogin: true });
      else {
        app.setLoginItemSettings({ openAtLogin: false });
        for (const c of await clients.statusAll(mcpEntry(), {})) {
          if (c.connected && !c.stale) log('uninstall: disconnect', c.id, JSON.stringify(await clients.disconnect(c.id)));
        }
      }
    } catch (e) { log('installer hook failed', HOOK, e.stack ?? e); }
    app.exit(0);
  });
}

// ---------- app ----------
// one instance; a self-test while the app is open does nothing (2026-11-04: one stopped the open app's companion server)
const OTHER = HOOK ? true : !app.requestSingleInstanceLock();
if (OTHER && SELFTEST) {
  try { fs.mkdirSync(SELFTEST, { recursive: true }); fs.writeFileSync(path.join(SELFTEST, 'selftest.json'), JSON.stringify({ error: `${NAME} is open: self-test skipped` })); } catch { /* nothing */ }
  app.exit(0);
} else if (OTHER && !HOOK) app.quit();
app.on('second-instance', () => { win?.show(); win?.focus(); });
app.whenReady().then(async () => {
  if (OTHER) return;
  // tray: the cropped face (the full picture is a purple blob at 16 px), one bitmap per display scale
  const trayImg = nativeImage.createEmpty();
  for (const [s, f] of [[16, 1], [24, 1.5], [32, 2]]) trayImg.addRepresentation({ scaleFactor: f, buffer: require('fs').readFileSync(path.join(__dirname, `tray_${s}.png`)) });
  tray = new Tray(trayImg);
  tray.setToolTip(NAME);
  tray.on('click', () => openSettings());   // a left click opens the settings (Q14: new users look there)
  chat = require('./chat_main.js')({ settings, saveSettings, BASE, tr, texts: () => i18n.T[uiLang()], instructions, icon: ICON, avatar: () => win, log, refreshMenu });
  refreshMenu();
  log('start', app.getVersion(), 'model', currentModel(), 'scale', screen.getPrimaryDisplay().scaleFactor);
  const ok = await startCompanion(currentModel());
  log('companion', ok ? 'up' : 'NOT up');
  createAvatarWindow();
  if (!ok) dialog.showErrorBox(NAME, tr('serverDown'));
  if (!setupDone() && !SELFTEST) openSettings();          // first run: the three setup steps
  // until the right-click menu has been opened once, each start says (in the bubble) where the settings are (Q14)
  if (!settings.menuSeen && !SELFTEST) setTimeout(() => win?.webContents.send('hint', tr('rightClickHint')), 9000);
  if (SELFTEST) selftest(SELFTEST, ok);
});
app.on('will-quit', () => globalShortcut.unregisterAll());
app.on('window-all-closed', () => { /* keeps running in the tray */ });
app.on('before-quit', (e) => {
  if (quitting) return;
  e.preventDefault(); quitting = true;
  stopCompanion().finally(() => app.quit());
});

// --selftest <dir>: checks for Claude (run from WSL): screenshots + state in <dir>, then quit, whatever happens
async function selftest(dir, serverOk) {
  const out = { serverOk };
  const shot = async (w, name) => {
    for (let i = 0; i < 5; i++) {
      try { fs.writeFileSync(path.join(dir, name), (await w.webContents.capturePage()).toPNG()); return true; }
      catch (e) { out[`${name}_error`] = String(e); await sleep(1000); }
    }
    return false;
  };
  try {
    fs.mkdirSync(dir, { recursive: true });
    out.info = await info(); out.settings = settings;
    out.clients = (await clients.statusAll(mcpEntry(), await seenClients())).filter((c) => c.installed).map((c) => ({ id: c.id, connected: c.connected, stale: c.stale, seen: c.seen }));
    // cloth first (2026-11-13): with <dir>/cloth_req.json the Windows solver solves it once through the companion
    // and this app's hidden window (HSIN_CLOTH_URL) and once without (the SKIRT_GAP fallback); motions saved in <dir>
    const reqFile = path.join(dir, 'cloth_req.json');
    if (NATIVE && fs.existsSync(reqFile)) {
      const solveOnce = (env) => new Promise((resolve) => {
        const t0 = Date.now();
        const p = spawn(path.join(BACKEND, 'python', 'python.exe'), ['-m', 'motion_loop.perform', 'solve'],
          { cwd: path.join(BACKEND, 'motion_loop'), env: { ...process.env, ...env }, windowsHide: true });
        let so = '', se = '';
        p.stdout.on('data', (d) => { so += d; }); p.stderr.on('data', (d) => { se += d; });
        p.on('close', () => resolve({ ms: Date.now() - t0, out: so, err: se.slice(-800) }));
        p.stdin.end(fs.readFileSync(reqFile));
      });
      for (const [name, env] of [['cloth', { HSIN_CLOTH_URL: BASE }], ['nocloth', { HSIN_CLOTH_URL: 'http://127.0.0.1:1' }]]) {
        const r = await solveOnce(env);
        fs.writeFileSync(path.join(dir, `${name}_motion.json`), r.out);
        let j = {}; try { j = JSON.parse(r.out); } catch { /* error text below */ }
        out[name] = { ms: r.ms, problems: j.problems ?? null, error: j.error ?? null, stderr: r.err, clothWindow: !!clothWin };
      }
    }
    // the first seconds per frame (2026-11-13: 'the upper body curled toward the belly once at the first start'):
    // torso / neck / head bend from rest, hips height, whether a motion plays
    await win.webContents.executeJavaScript(`(() => { window.__trace = []; const t0 = performance.now();
      const names = ['hips', 'spine', 'chest', 'upperChest', 'neck', 'head', 'leftUpperArm', 'rightUpperArm'];
      const step = () => { const st = window.hsinStage, vrm = st && st.state && st.state.vrm;
        if (vrm) { const row = [((performance.now() - t0) / 1000).toFixed(2)];
          for (const n of names) { const b = vrm.humanoid.getNormalizedBoneNode(n); row.push(b ? (2 * Math.acos(Math.min(1, Math.abs(b.quaternion.w))) * 180 / Math.PI).toFixed(1) : '-'); }
          row.push((vrm.humanoid.getNormalizedBoneNode('hips').position.y).toFixed(3)); row.push(st.hasMotion() ? 'M' : '-');
          window.__trace.push(row.join(' ')); }
        requestAnimationFrame(step); };
      requestAnimationFrame(step); })()`).catch((e) => { out.traceError = String(e); });
    await sleep(9000);                          // model load + first frames
    out.trace = await win.webContents.executeJavaScript('window.__trace').catch((e) => String(e));
    out.bounds = win?.getBounds();
    out.pageStatus = await win?.webContents.executeJavaScript("document.getElementById('status').textContent").catch((e) => String(e));
    out.pageBg = await win?.webContents.executeJavaScript("getComputedStyle(document.documentElement).backgroundColor").catch((e) => String(e));
    await shot(win, 'avatar.png');
    const ids = fs.existsSync(path.join(__dirname, 'selftest_motion.txt')) ? fs.readFileSync(path.join(__dirname, 'selftest_motion.txt'), 'utf-8').trim() : 'rest';
    await fetch(`${BASE}/play`, { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ id: ids, say: '早安！今天天氣好好，一起伸個懶腰吧～', emotion: 'happy', problems: [] }) }).catch(() => {});
    await sleep(1500);
    await shot(win, 'bubble.png');
    await sleep(6000);
    await shot(win, 'after.png');
    out.bubbleOn = await win.webContents.executeJavaScript("document.getElementById('say').className").catch((e) => String(e));
    // window snapping (2026-11-11): the window list, then a ledge, a wall and a fall on the first usable window
    const keep = settings.bounds;
    try {
      wins = null; out.windows = winList().slice(0, 12);
      const b0 = win.getBounds();
      const w = out.windows.find((o) => !o.maximized && o.width > 300 && o.y > 150);
      out.snapWindow = w ?? null;
      if (w) {
        const js = (c) => win.webContents.executeJavaScript(c);
        snapped = { kind: 'ledge', side: null, id: w.id, ref: { x: w.x, y: w.y, width: w.width, height: w.height } };
        place(Math.round(w.x + w.width / 2 - b0.width / 2), Math.round(w.y - b0.height * 0.8));
        await js("hsinSnapTo({kind: 'ledge'})"); await sleep(4000);
        out.ledge = { bounds: win.getBounds(), groundY: await js('hsinStage.groundScreen()'), edgeY: w.y };
        await shot(win, 'ledge.png');
        clearInterval(watchTimer);
        snapped = { kind: 'wall', side: 'left', id: w.id, ref: { x: w.x, y: w.y, width: w.width, height: w.height } };
        place(Math.round(w.x - b0.width * 0.6), Math.round(w.y + 40));
        await js("hsinSnapTo({kind: 'wall', side: 'left'})"); await sleep(4000);
        out.wall = { bounds: win.getBounds(), box: await js('hsinStage.bodyBox()'), edgeX: w.x };
        await shot(win, 'wall.png');
        clearInterval(watchTimer); snapped = null;
        win.webContents.send('detach'); await sleep(2500);
        out.fall = { bounds: win.getBounds(), groundY: await js('hsinStage.groundScreen()'), workArea: screen.getDisplayMatching(win.getBounds()).workArea };
      }
    } catch (e) { out.snapError = String(e.stack ?? e); }
    settings.bounds = keep; saveSettings();
    // render cost with the look switches off / on (2026-11-12): ms per frame on the GPU the window really uses
    try {
      out.bench = {};
      for (const look of ['none', 'all']) {
        win.loadURL(avatarUrl(look));
        await sleep(9000);
        out.bench[look] = await win.webContents.executeJavaScript('hsinStage.bench(120)');
        await shot(win, `look_${look}.png`);
      }
      win.loadURL(avatarUrl());
    } catch (e) { out.benchError = String(e.stack ?? e); }
    openSettings();
    await new Promise((r) => settingsWin.webContents.once('did-finish-load', r));
    await sleep(1500);
    await shot(settingsWin, 'settings.png');
    out.settingsText = await settingsWin.webContents.executeJavaScript('document.body.innerText').catch((e) => String(e));
    // built-in chat (2026-11-13): the settings card in built-in mode, the model list, the chat window; with a saved key
    // and <dir>/chat_line.txt one real line. The user's chat settings are put back afterwards.
    const keepChat = JSON.parse(JSON.stringify(settings.chat ?? null));
    try {
      out.chatKey = chat.keyShape();
      chat.set({ mode: 'builtin' });
      settingsWin.reload(); await new Promise((r) => settingsWin.webContents.once('did-finish-load', r)); await sleep(4000);
      await shot(settingsWin, 'settings_chat.png');
      out.chatView = chat.view();
      out.chatModels = await settingsWin.webContents.executeJavaScript("[...document.querySelectorAll('#modelPick option')].map((o) => o.textContent)").catch((e) => String(e));
      out.chatMsg = await settingsWin.webContents.executeJavaScript("document.getElementById('mc').textContent").catch((e) => String(e));
      chat.open(); await new Promise((r) => chat.window().webContents.once('did-finish-load', r)); await sleep(1500);
      const lineFile = path.join(dir, 'chat_line.txt');
      if (chat.ready() && fs.existsSync(lineFile)) {
        const t0 = Date.now();
        const r = await chat.send(fs.readFileSync(lineFile, 'utf-8').trim());
        out.chatLine = { ms: Date.now() - t0, error: r.error ?? null, lines: r.turn?.lines, usage: r.usage };
        chat.window().reload(); await sleep(2500);
        await shot(win, 'chat_avatar.png');
      }
      await shot(chat.window(), 'chat_window.png');
    } catch (e) { out.chatError = String(e.stack ?? e); }
    settings.chat = keepChat ?? undefined; if (!keepChat) delete settings.chat; saveSettings();
  } catch (e) { out.error = String(e.stack ?? e); }
  try { fs.writeFileSync(path.join(dir, 'selftest.json'), JSON.stringify(out, null, 2)); } catch { /* nothing to do */ }
  app.quit();
}

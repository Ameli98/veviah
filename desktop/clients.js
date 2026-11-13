// AI programs the MCP server can be connected to with one click (2026-11-13, Q13: every program alike, none special).
// Facts per program (paths, formats, CLIs) from their docs, researched 2026-11-13 (CLAUDE.md). Rules (Q13e / Q13i):
//   - a program with its own `mcp add` command (Claude Code, Gemini CLI, Codex) is connected through that command;
//   - otherwise only our entry in its user-level config file is added / replaced, the rest of the file kept, after a
//     backup next to it (<file>.veviah-backup); a file we cannot parse safely is never written;
//   - Zed / Continue / JetBrains get steps instead (comment-preserving or YAML files, or no file at all);
//   - the entry holds no model path (Q13a: the server asks the app which model is in use).
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFile } = require('child_process');

const HOME = os.homedir();
const APPDATA = process.env.APPDATA ?? path.join(HOME, 'AppData', 'Roaming');
const LOCAL = process.env.LOCALAPPDATA ?? path.join(HOME, 'AppData', 'Local');
const exists = (p) => { try { return fs.existsSync(p); } catch { return false; } };

// Claude Desktop keeps its file in different places per install (classic, the packaged Claude-Data\<id>, older packaged
// builds under Packages\Claude_*\LocalCache)
function claudeDesktopFiles() {
  const out = [];
  const classic = path.join(APPDATA, 'Claude', 'claude_desktop_config.json');
  if (exists(classic)) out.push(classic);
  for (const base of [path.join(LOCAL, 'Claude-Data')]) {
    try { for (const d of fs.readdirSync(base)) { const f = path.join(base, d, 'claude_desktop_config.json'); if (exists(f)) out.push(f); } } catch { /* none */ }
  }
  try {
    for (const d of fs.readdirSync(path.join(LOCAL, 'Packages')).filter((x) => /^Claude_/.test(x))) {
      const f = path.join(LOCAL, 'Packages', d, 'LocalCache', 'Roaming', 'Claude', 'claude_desktop_config.json');
      if (exists(f)) out.push(f);
    }
  } catch { /* none */ }
  return out;
}

// how each program names itself when it starts the MCP server (Q13b: 'connected' = it really started us); lower case,
// matched as a substring of the name the server reported
const PROGRAMS = [
  { id: 'claude-desktop', name: 'Claude Desktop', seen: ['claude-ai', 'claude desktop'], files: claudeDesktopFiles,
    detect: () => claudeDesktopFiles().length > 0 || exists(path.join(LOCAL, 'Claude-Data')), restart: true },
  { id: 'claude-code', name: 'Claude Code', seen: ['claude-code'], cli: 'claude',
    add: (e) => ['mcp', 'add', '-s', 'user', ...envArgs(e, '-e'), 'veviah-avatar', '--', e.command, ...e.args], remove: ['mcp', 'remove', '-s', 'user', 'veviah-avatar'] },
  { id: 'antigravity', name: 'Antigravity', seen: ['antigravity'], files: () => [path.join(HOME, '.gemini', 'config', 'mcp_config.json')],
    detect: () => exists(path.join(HOME, '.gemini', 'antigravity')) || exists(path.join(HOME, '.gemini', 'config')), create: true, restart: true },
  { id: 'gemini-cli', name: 'Gemini CLI', seen: ['gemini-cli'], cli: 'gemini',
    add: (e) => ['mcp', 'add', '--scope', 'user', ...envArgs(e, '-e'), 'veviah-avatar', e.command, ...e.args], remove: ['mcp', 'remove', '--scope', 'user', 'veviah-avatar'] },
  { id: 'codex', name: 'Codex', seen: ['codex'], cli: 'codex',
    add: (e) => ['mcp', 'add', ...envArgs(e, '--env'), 'veviah-avatar', '--', e.command, ...e.args], remove: ['mcp', 'remove', 'veviah-avatar'],
    toml: path.join(HOME, '.codex', 'config.toml'), detect: () => exists(path.join(HOME, '.codex')) },
  { id: 'cursor', name: 'Cursor', seen: ['cursor'], files: () => [path.join(HOME, '.cursor', 'mcp.json')], extra: { type: 'stdio' },
    detect: () => exists(path.join(HOME, '.cursor')), create: true },
  { id: 'windsurf', name: 'Windsurf / Devin', seen: ['windsurf', 'devin', 'codeium'],
    files: () => [path.join(APPDATA, 'devin', 'mcp_config.json'), path.join(HOME, '.codeium', 'windsurf', 'mcp_config.json')]
      .filter((f) => exists(path.dirname(f))).slice(0, 1),
    detect: () => exists(path.join(APPDATA, 'devin')) || exists(path.join(HOME, '.codeium', 'windsurf')), create: true },
  { id: 'vscode', name: 'VS Code (Copilot)', seen: ['visual studio code', 'vscode', 'copilot'], files: () => [path.join(HOME, '.copilot', 'mcp-config.json')],
    extra: { type: 'local', tools: ['*'] }, detect: () => exists(path.join(HOME, '.copilot')) || exists(path.join(APPDATA, 'Code', 'User')), create: true },
  { id: 'cline', name: 'Cline', seen: ['cline'], files: () => [path.join(HOME, '.cline', 'data', 'settings', 'cline_mcp_settings.json')],
    detect: () => exists(path.join(HOME, '.cline')), create: true },
  { id: 'lmstudio', name: 'LM Studio', seen: ['lm studio', 'lmstudio'], files: () => [path.join(HOME, '.lmstudio', 'mcp.json')],
    detect: () => exists(path.join(HOME, '.lmstudio')), create: true },
  { id: 'cherry', name: 'Cherry Studio', seen: ['cherry'], link: (e) => `cherrystudio://mcp/install?servers=${Buffer.from(JSON.stringify({ mcpServers: { 'veviah-avatar': e } })).toString('base64')}`,
    detect: () => exists(path.join(HOME, '.cherrystudio')) },
  // steps only (comment-preserving JSONC / YAML / no file)
  { id: 'zed', name: 'Zed', seen: ['zed'], manual: 'zed', detect: () => exists(path.join(APPDATA, 'Zed')) },
  { id: 'continue', name: 'Continue', seen: ['continue'], manual: 'continue', detect: () => exists(path.join(HOME, '.continue')) },
  { id: 'jetbrains', name: 'JetBrains AI Assistant', seen: ['jetbrains'], manual: 'jetbrains', detect: () => exists(path.join(APPDATA, 'JetBrains')) },
];
const KEY = 'veviah-avatar', OLD_KEYS = ['hsin-avatar'];

function envArgs(e, flag) { return Object.entries(e.env ?? {}).flatMap(([k, v]) => [flag, `${k}=${v}`]); }

// --- running a program's own command (found on PATH; npm installs are .cmd shims, so through cmd on Windows) ---
function run(cli, args) {
  return new Promise((resolve) => {
    const win = process.platform === 'win32';
    const [cmd, a] = win ? ['cmd.exe', ['/d', '/s', '/c', [cli, ...args].map(q).join(' ')]] : [cli, args];
    execFile(cmd, a, { windowsHide: true, timeout: 60000, windowsVerbatimArguments: win }, (err, out, errOut) =>
      resolve({ ok: !err, out: String(out ?? ''), err: String(errOut || err?.message || '') }));
  });
}
const q = (s) => (/^[\w./:=\\-]+$/.test(String(s)) ? String(s) : `"${String(s).replace(/"/g, '\\"')}"`);
const onPath = (cli) => run(process.platform === 'win32' ? 'where' : 'which', [cli]).then((r) => r.ok && r.out.trim() !== '');

// --- JSON files: read (strict JSON only), merge our key, back up, write ---
function readJson(f) {
  if (!exists(f)) return { cfg: {}, text: null };
  const text = fs.readFileSync(f, 'utf-8');
  if (!text.trim()) return { cfg: {}, text };
  return { cfg: JSON.parse(text), text };                    // throws on JSONC / broken files: we do not write those
}
function writeJson(f, cfg, text) {
  fs.mkdirSync(path.dirname(f), { recursive: true });
  if (text != null) fs.writeFileSync(`${f}.veviah-backup`, text);
  const tmp = `${f}.veviah-tmp`;
  fs.writeFileSync(tmp, JSON.stringify(cfg, null, 2));
  fs.renameSync(tmp, f);
}
const entryIn = (cfg) => cfg?.mcpServers?.[KEY] ?? OLD_KEYS.map((k) => cfg?.mcpServers?.[k]).find(Boolean) ?? null;

// --- Codex's TOML (used only when its CLI is not on PATH): our [mcp_servers.veviah-avatar] table replaced as text ---
const TOML_HEAD = /^\[mcp_servers\.(?:"?veviah-avatar"?)(?:\.[^\]]*)?\]\s*$/;
function tomlWithout(text) {
  const out = []; let skip = false;
  for (const line of text.split(/\r?\n/)) {
    if (/^\s*\[/.test(line)) skip = TOML_HEAD.test(line.trim());
    if (!skip) out.push(line);
  }
  return out.join('\n').replace(/\n{3,}$/, '\n\n');
}
const tomlStr = (s) => JSON.stringify(String(s));            // a JSON string is a valid TOML basic string
function tomlEntry(e) {
  return [`[mcp_servers.${KEY}]`, `command = ${tomlStr(e.command)}`, `args = [${e.args.map(tomlStr).join(', ')}]`, '',
    `[mcp_servers.${KEY}.env]`, ...Object.entries(e.env ?? {}).map(([k, v]) => `${k} = ${tomlStr(v)}`), ''].join('\n');
}

// ---- the three operations ----
// status of every program: {id, name, installed, connected, stale (an entry with another command: moved app), seen,
// how: 'file' | 'cli' | 'link' | 'manual', manual}
async function statusAll(entry, seenNames) {
  const seen = (p) => Object.keys(seenNames ?? {}).some((n) => p.seen.some((s) => n.toLowerCase().includes(s)));
  const out = [];
  for (const p of PROGRAMS) {
    const s = { id: p.id, name: p.name, connected: false, stale: false, seen: seen(p), manual: p.manual ?? null };
    if (p.cli && await onPath(p.cli)) {
      s.installed = true; s.how = 'cli';
      const r = await run(p.cli, ['mcp', 'list']);
      s.connected = r.ok && r.out.includes(KEY);
      if (p.toml && !s.connected && exists(p.toml)) s.connected = fs.readFileSync(p.toml, 'utf-8').includes(`[mcp_servers.${KEY}]`);
    } else if (p.toml) {
      s.installed = p.detect(); s.how = 'file';
      s.connected = s.installed && exists(p.toml) && fs.readFileSync(p.toml, 'utf-8').includes(`[mcp_servers.${KEY}]`);
    } else if (p.cli) {
      s.installed = false; s.how = 'cli';
    } else if (p.files) {
      s.installed = p.detect(); s.how = 'file';
      for (const f of p.files()) {
        try { const e = entryIn(readJson(f).cfg); if (e) { s.connected = true; s.stale = e.command !== entry.command; } } catch { s.unreadable = true; }
      }
    } else {
      s.installed = p.detect(); s.how = p.link ? 'link' : 'manual';
    }
    out.push(s);
  }
  return out;
}

async function connect(id, entry, openExternal) {
  const p = PROGRAMS.find((x) => x.id === id);
  if (!p) return { ok: false, error: 'unknown program' };
  if (p.manual) return { ok: false, error: 'manual' };
  if (p.link) { await openExternal(p.link(entry)); return { ok: true, link: true }; }
  if (p.cli && await onPath(p.cli)) {
    await run(p.cli, p.remove);                              // replace an old entry (another path, an old name)
    const r = await run(p.cli, p.add(entry));
    return r.ok ? { ok: true } : { ok: false, error: r.err.slice(0, 300) };
  }
  if (p.toml) {
    const text = exists(p.toml) ? fs.readFileSync(p.toml, 'utf-8') : '';
    fs.mkdirSync(path.dirname(p.toml), { recursive: true });
    if (text) fs.writeFileSync(`${p.toml}.veviah-backup`, text);
    fs.writeFileSync(p.toml, `${tomlWithout(text).trimEnd()}\n\n${tomlEntry(entry)}`);
    return { ok: true };
  }
  const files = p.files();
  if (!files.length) return { ok: false, error: 'not found' };
  for (const f of files) {
    if (!exists(f) && !p.create) continue;
    let cur;
    try { cur = readJson(f); } catch { return { ok: false, error: `cannot read ${f} safely (comments or not JSON); add it by hand` }; }
    cur.cfg.mcpServers = cur.cfg.mcpServers ?? {};
    for (const k of OLD_KEYS) delete cur.cfg.mcpServers[k];
    cur.cfg.mcpServers[KEY] = { ...(p.extra ?? {}), ...entry };
    writeJson(f, cur.cfg, cur.text);
  }
  return { ok: true, restart: !!p.restart };
}

async function disconnect(id) {
  const p = PROGRAMS.find((x) => x.id === id);
  if (!p || p.manual || p.link) return { ok: false };
  if (p.cli && await onPath(p.cli)) { const r = await run(p.cli, p.remove); return { ok: r.ok, error: r.ok ? null : r.err.slice(0, 300) }; }
  if (p.toml) {
    if (!exists(p.toml)) return { ok: true };
    const text = fs.readFileSync(p.toml, 'utf-8');
    fs.writeFileSync(`${p.toml}.veviah-backup`, text);
    fs.writeFileSync(p.toml, tomlWithout(text));
    return { ok: true };
  }
  for (const f of p.files()) {
    let cur;
    try { cur = readJson(f); } catch { continue; }
    if (!cur.cfg.mcpServers) continue;
    for (const k of [KEY, ...OLD_KEYS]) delete cur.cfg.mcpServers[k];
    writeJson(f, cur.cfg, cur.text);
  }
  return { ok: true };
}

// a program picked by hand (Q13g: installed somewhere we do not look): our entry into the file the user chose,
// in that program's format
function connectFile(id, file, entry) {
  const p = PROGRAMS.find((x) => x.id === id);
  let cur;
  try { cur = readJson(file); } catch { return { ok: false, error: 'not plain JSON: add it by hand' }; }
  const key = id === 'zed' ? 'context_servers' : 'mcpServers';
  cur.cfg[key] = cur.cfg[key] ?? {};
  cur.cfg[key][KEY] = { ...(p?.extra ?? {}), ...entry };
  writeJson(file, cur.cfg, cur.text);
  return { ok: true };
}

// the entry as text for programs set up by hand: JSON (most programs) and TOML (Codex)
function snippets(entry) {
  return { json: JSON.stringify({ mcpServers: { [KEY]: entry } }, null, 2), toml: tomlEntry(entry) };
}

module.exports = { PROGRAMS, statusAll, connect, disconnect, connectFile, snippets, claudeDesktopFiles, KEY };

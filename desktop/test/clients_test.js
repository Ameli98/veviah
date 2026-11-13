// clients.js against a fake home (run with PATH=/usr/bin:/bin so no real AI CLI is ever called): JSON merge keeps
// other keys and backs up, TOML replace / remove, status, disconnect.
const fs = require('fs'), os = require('os'), path = require('path');
const home = fs.mkdtempSync(path.join(os.tmpdir(), 'veviah-clients-'));
process.env.HOME = home; process.env.APPDATA = path.join(home, 'AppData', 'Roaming'); process.env.LOCALAPPDATA = path.join(home, 'AppData', 'Local');
const C = require('../clients.js');
let fail = 0; const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };
const entry = { command: 'C:\\x\\VEviaH.exe', args: ['C:\\x\\server.mjs'], env: { ELECTRON_RUN_AS_NODE: '1' } };
(async () => {
  // Claude Desktop (packaged path) with other settings and the old key
  const cd = path.join(home, 'AppData', 'Local', 'Claude-Data', 'claude_abc', 'claude_desktop_config.json');
  fs.mkdirSync(path.dirname(cd), { recursive: true });
  fs.writeFileSync(cd, JSON.stringify({ preferences: { a: 1 }, mcpServers: { other: { command: 'x' }, 'hsin-avatar': { command: 'old' } } }));
  fs.mkdirSync(path.join(home, '.cursor')); fs.mkdirSync(path.join(home, '.codex'));
  fs.writeFileSync(path.join(home, '.codex', 'config.toml'), 'model = "gpt"\n\n[mcp_servers.other]\ncommand = "y"\n');
  fs.mkdirSync(path.join(home, '.continue'));
  let st = await C.statusAll(entry, {});
  const S = (id) => st.find((s) => s.id === id);
  ok(S('claude-desktop').installed && S('claude-desktop').connected && S('claude-desktop').stale, 'old hsin-avatar entry seen as connected but stale');
  ok(S('cursor').installed && !S('cursor').connected && S('codex').installed && S('continue').manual === 'continue', 'cursor / codex / continue detected');
  ok(!S('claude-code').installed && !S('antigravity').installed, 'programs not there are not installed');
  for (const id of ['claude-desktop', 'cursor', 'codex']) ok((await C.connect(id, entry, async () => {})).ok, `connect ${id}`);
  const cfg = JSON.parse(fs.readFileSync(cd, 'utf-8'));
  ok(cfg.preferences.a === 1 && cfg.mcpServers.other && !cfg.mcpServers['hsin-avatar'] && cfg.mcpServers['veviah-avatar'].command === entry.command, 'Claude: other keys kept, old key gone, ours in');
  ok(fs.existsSync(cd + '.veviah-backup'), 'Claude: backup written');
  const cur = JSON.parse(fs.readFileSync(path.join(home, '.cursor', 'mcp.json'), 'utf-8'));
  ok(cur.mcpServers['veviah-avatar'].type === 'stdio', 'Cursor: created with type stdio');
  await C.connect('codex', entry, async () => {});        // twice: no duplicate table
  const toml = fs.readFileSync(path.join(home, '.codex', 'config.toml'), 'utf-8');
  ok(toml.includes('model = "gpt"') && toml.includes('[mcp_servers.other]') && toml.split('[mcp_servers.veviah-avatar]').length === 2 && toml.includes('ELECTRON_RUN_AS_NODE = "1"'), 'Codex TOML: kept, one table, env');
  st = await C.statusAll(entry, { 'claude-ai': {}, 'Codex': {} });
  ok(S('claude-desktop').connected && !S('claude-desktop').stale && S('claude-desktop').seen && S('codex').connected && S('codex').seen, 'status after connect, seen names matched');
  ok((await C.connect('continue', entry)).error === 'manual', 'manual programs are not written');
  fs.writeFileSync(path.join(home, '.cursor', 'mcp.json'), '{ // comment\n "mcpServers": {} }');
  ok(!(await C.connect('cursor', entry, async () => {})).ok, 'a JSON file with comments is not written');
  for (const id of ['claude-desktop', 'codex']) await C.disconnect(id);
  ok(!JSON.parse(fs.readFileSync(cd, 'utf-8')).mcpServers['veviah-avatar'] && !fs.readFileSync(path.join(home, '.codex', 'config.toml'), 'utf-8').includes('veviah-avatar'), 'disconnect removes only ours');
  ok(fs.readFileSync(path.join(home, '.codex', 'config.toml'), 'utf-8').includes('[mcp_servers.other]'), 'Codex: other table kept after disconnect');
  const sn = C.snippets(entry);
  ok(JSON.parse(sn.json).mcpServers['veviah-avatar'] && sn.toml.startsWith('[mcp_servers.veviah-avatar]'), 'snippets');
  fs.rmSync(home, { recursive: true, force: true });
  console.log(fail ? `${fail} failed` : 'all ok'); process.exit(fail ? 1 : 0);
})();

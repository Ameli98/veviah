// Built-in chat engine (chat.mjs) through the companion's /chat routes, against a fake OpenAI-compatible API (free):
// tool loop, retry after a rejection, history sent back without keyframes, summary past 20 turns, clear, cost of local.
// Run: node test/chat.mjs        (own ports 8798 / 8799, own config dir: never the user's app or history)
import http from 'node:http';
import os from 'node:os';
import fs from 'node:fs/promises';
import path from 'node:path';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const PORT = 8798, API = 8799;
let fail = 0;
const ok = (c, m) => { console.log(c ? 'ok  ' : 'FAIL', m); if (!c) fail++; };

// ---- the fake API: records every request, answers from a script ----
const reqs = [];
const KF = [{ t: 0, right_arm: 'rest', left_arm: 'rest' }, { t: 0.6, right_arm: { hand: 'head out beside', palm: 'forward' } }, { t: 1.4, right_arm: 'rest' }];
let mode = 'normal';
const api = http.createServer(async (req, res) => {
  let b = ''; for await (const c of req) b += c;
  const j = JSON.parse(b || '{}'); reqs.push(j);
  const last = j.messages.at(-1);
  let msg;
  if (!j.tools) msg = { role: 'assistant', content: 'SUMMARY: the user said hello many times.' };          // summary call
  else if (last.role === 'user') {
    const kf = mode === 'bad' ? [{ t: 0, right_arm: { hand: 'nowhere at all' } }] : KF;
    msg = { role: 'assistant', content: null, tool_calls: [{ id: `c${reqs.length}`, type: 'function', function: { name: 'perform', arguments: JSON.stringify({ say: `你好 ${reqs.length}`, emotion: 'happy', keyframes: kf }) } }] };
    if (mode === 'bad') mode = 'fixed';
  } else if (last.role === 'tool' && mode === 'fixed' && /Not shown yet/.test(last.content)) {
    msg = { role: 'assistant', content: null, tool_calls: [{ id: 'fix', type: 'function', function: { name: 'perform', arguments: JSON.stringify({ say: '修好了', emotion: 'neutral', keyframes: KF }) } }] };
    mode = 'normal';
  } else msg = { role: 'assistant', content: '' };
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ choices: [{ message: msg }], usage: { prompt_tokens: 100, completion_tokens: 10 } }));
});
await new Promise((r) => api.listen(API, '127.0.0.1', r));

const cfgDir = await fs.mkdtemp(path.join(os.tmpdir(), 'veviah-chat-'));
const comp = spawn(process.execPath, [path.join(HERE, '../companion.mjs')], { env: { ...process.env, HSIN_COMPANION_PORT: String(PORT), HSIN_DATA: cfgDir, HSIN_CONFIG_DIR: cfgDir, HSIN_VOICES: path.join(cfgDir, 'none.json') }, stdio: 'ignore' });
const post = async (p, b) => (await fetch(`http://127.0.0.1:${PORT}${p}`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(b ?? {}) })).json();
const get = async (p) => (await fetch(`http://127.0.0.1:${PORT}${p}`)).json();
try {
  for (let i = 0; i < 50; i++) { try { await get('/model/info'); break; } catch { await new Promise((r) => setTimeout(r, 200)); } }
  const cfg = { kind: 'openai', baseUrl: `http://127.0.0.1:${API}/v1`, model: 'fake', key: '' };
  await post('/chat/clear');
  const r1 = await post('/chat', { text: '早安', cfg, system: 'SYS' });
  ok(!r1.error && r1.turn.lines.length === 1 && r1.turn.lines[0].say.startsWith('你好'), 'a line becomes one perform call that is shown: ' + JSON.stringify(r1.turn?.lines ?? r1.error));
  ok(r1.usage.cost === 0 && r1.usage.in === 200, 'local endpoint: cost 0, tokens counted (2 requests)');
  const first = reqs.find((q) => q.tools);
  ok(first.tools[0].function.description.length < 4000, 'local endpoint gets the short tool description (' + first.tools[0].function.description.length + ' chars)');
  ok(first.messages[0].role === 'system' && first.messages[0].content.startsWith('SYS'), 'system prompt first');

  mode = 'bad'; reqs.length = 0;
  const r2 = await post('/chat', { text: '揮手', cfg, system: 'SYS' });
  ok(r2.turn.lines.map((l) => l.say).join('|') === '修好了', 'a rejected motion goes back, the fixed one is shown: ' + JSON.stringify(r2.turn.lines));
  const hist = reqs[0].messages;
  const pastCall = hist.find((m) => m.tool_calls);
  ok(pastCall && JSON.parse(pastCall.tool_calls[0].function.arguments).keyframes === '(omitted)', 'older turns go back without keyframes');
  ok(hist.some((m) => m.role === 'tool' && m.content === 'Shown.'), 'older results are just "Shown."');

  const longCfg = { ...cfg, desc: 'long' }; reqs.length = 0;
  await post('/chat', { text: 'hi', cfg: longCfg, system: 'SYS' });
  const d = reqs[0].tools[0].function.description;
  ok(d.length > 5000 && d.includes('Example, waving hello'), 'desc long = the MCP description + the examples (' + d.length + ' chars)');

  for (let i = 0; i < 19; i++) await post('/chat', { text: `line ${i}`, cfg, system: 'SYS' });
  for (let i = 0; i < 50; i++) { const h = await get('/chat/history'); if (h.summarized) break; await new Promise((r) => setTimeout(r, 200)); }
  const h = await get('/chat/history');
  ok(h.summarized && h.turns.length <= 12, `past 20 turns the older ones are summarized (turns left ${h.turns.length})`);
  reqs.length = 0;
  await post('/chat', { text: 'after', cfg, system: 'SYS' });
  ok(reqs[0].messages[0].content.includes('SUMMARY: the user said hello'), 'the summary rides in the system prompt');
  const c = await post('/chat/clear');
  ok(c.turns.length === 0 && !c.summarized, 'clear empties turns and summary');
  const bad = await post('/chat', { text: 'x', cfg: { ...cfg, baseUrl: 'http://127.0.0.1:1/v1' }, system: 'SYS' });
  ok(bad.error === 'network', 'unreachable endpoint -> error network: ' + bad.error);
  const m = await post('/chat/models', { cfg: { ...cfg, baseUrl: 'http://127.0.0.1:1/v1' } });
  ok(!m.ok, 'model list of an unreachable endpoint: not ok');
} finally {
  try { await post('/quit'); } catch { /* gone */ }
  comp.kill(); api.close();
  await fs.rm(cfgDir, { recursive: true, force: true });
}
console.log(fail ? `${fail} failed` : 'all ok'); process.exit(fail ? 1 : 0);

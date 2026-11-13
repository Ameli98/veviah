// Open-source LLMs writing motions (2026-11-12, plan step 2): each local Ollama model gets the same setup as Claude in
// Claude Desktop: the Project instructions as the system prompt, the real MCP server's perform tool (description +
// schema), one user line per fresh conversation, and the server's real results (a rejected motion goes back once).
// Run: node test/llm_eval.mjs MODEL OUT.jsonl [line ids]
//   MODEL = an Ollama tag (qwen3.5:9b ...). Writes one JSON line per user line: every perform call (arguments, result
//   text, seconds), the final text reply, wall time.
import path from 'node:path';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const { T } = createRequire(import.meta.url)('../../desktop/i18n.js');
const [model, out, only] = process.argv.slice(2);
const OLLAMA = 'http://127.0.0.1:11434/api/chat';

// 10 everyday companion lines + 5 harder ones (sit, turn round, touch the face, jump, stand up again)
export const LINES = [
  ['l01', '早安！'], ['l02', '我今天上班好累喔'], ['l03', '你在幹嘛？'], ['l04', '晚餐吃什麼好呢'],
  ['l05', '我考試考過了！'], ['l06', '外面在下雨欸'], ['l07', '可以安慰我一下嗎'], ['l08', '你覺得我這件衣服好看嗎'],
  ['l09', '我先去洗澡囉'], ['l10', '晚安，明天見'],
  ['h01', '坐下來陪我聊天吧'], ['h02', '轉過去讓我看看你的背'], ['h03', '你害羞的時候會遮臉嗎？做給我看'],
  ['h04', '開心到跳起來！'], ['h05', '好了，站起來伸個懶腰'],
];

const client = new Client({ name: 'llm-eval', version: '0' });
await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, '../server.mjs')],
  env: { ...process.env, HSIN_COMPANION_PORT: '8796' }, stderr: 'ignore' }));   // no companion there: never the user's app
const perform = (await client.listTools()).tools.find((t) => t.name === 'perform');
// variants (2026-11-12, Q51): LLM_SYS_EXTRA = text appended to the system prompt (a: check before sending);
// LLM_TOOL_DESC = a file whose text replaces the tool description (c: a rewritten, shorter description)
const desc = process.env.LLM_TOOL_DESC ? fs.readFileSync(process.env.LLM_TOOL_DESC, 'utf-8') : perform.description;
const tools = [{ type: 'function', function: { name: 'perform', description: desc, parameters: perform.inputSchema } }];
const system = T.zh.instructions + (process.env.LLM_SYS_EXTRA ? '\n' + process.env.LLM_SYS_EXTRA : '');
const REPLAY = model === 'claude' ? JSON.parse(fs.readFileSync(path.join(process.env.HOME, 'hsin_runs/llm_test/claude_args.json'), 'utf-8')) : null;
const think = /gpt-oss/.test(model) ? 'low' : false;    // gpt-oss cannot turn thinking off

async function chat(messages, id) {
  if (REPLAY) {      // Claude's baseline: the motion written by hand for the line, once (a rejection is fixed by hand)
    if (messages.some((m) => m.role === 'tool')) return { message: { role: 'assistant', content: '' } };
    return { message: { role: 'assistant', content: '', tool_calls: [{ function: { name: 'perform', arguments: REPLAY[id] } }] } };
  }
  const r = await fetch(OLLAMA, { method: 'POST', headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ model, messages, tools, stream: false, think, options: { num_ctx: 16384, temperature: 0.7 } }) });
  if (!r.ok) throw new Error(`ollama ${r.status}: ${await r.text()}`);
  return r.json();
}

const rows = [];
for (const [id, line] of LINES) {
  if (only && !only.split(',').includes(id)) continue;
  const t0 = Date.now(), calls = [];
  const messages = [{ role: 'system', content: system }, { role: 'user', content: line }];
  let reply = '', error = null;
  try {
    for (let turn = 0; turn < 4; turn++) {
      const res = await chat(messages, id);
      const msg = res.message;
      messages.push(msg);
      const tc = msg.tool_calls ?? [];
      if (!tc.length) { reply = msg.content ?? ''; break; }
      for (const c of tc) {
        let args = c.function?.arguments;
        if (typeof args === 'string') { try { args = JSON.parse(args); } catch { args = { _unparsed: args }; } }
        const s0 = Date.now();
        let text;
        if (c.function?.name !== 'perform') text = `unknown tool ${c.function?.name}`;
        else {
          try {
            const r = await client.callTool({ name: 'perform', arguments: args });
            text = (r.content ?? []).map((x) => x.text ?? '').join('\n');
            if (r.isError) text = 'ERROR ' + text;
          } catch (e) { text = 'ERROR ' + String(e.message ?? e); }
        }
        calls.push({ args, result: text, s: (Date.now() - s0) / 1000 });
        messages.push({ role: 'tool', content: text, tool_name: 'perform' });
      }
      if (calls.length >= 3) break;
    }
  } catch (e) { error = String(e.message ?? e); }
  const row = { model, id, line, calls, reply, error, s: (Date.now() - t0) / 1000 };
  rows.push(row);
  fs.appendFileSync(out, JSON.stringify(row) + '\n');
  const last = calls.at(-1)?.result ?? '';
  console.log(id, `${row.s.toFixed(0)}s`, `calls ${calls.length}`, error ?? last.replace(/\s+/g, ' ').slice(0, 110));
}
await client.close();

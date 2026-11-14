// The built-in chat with a local Ollama model, exactly as the app runs it (2026-11-14): chat.mjs's Ollama adapter
// (own API, num_ctx 16384) + the real MCP server, the 15 lines of the open-model test (test/llm_eval.mjs), a fresh
// conversation per line. Companion port 8796: no companion there, so it never plays in the user's open app.
// Run: node test/ollama_chat.mjs MODEL OUT.jsonl   (Ollama must be running on localhost:11434)
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createRequire } from 'node:module';
import { Chat } from '../chat.mjs';

const [model, out] = process.argv.slice(2);
// the 15 lines (copied from llm_eval.mjs, which connects its own MCP client at import)
const LINES = [
  ['l01', '早安！'], ['l02', '我今天上班好累喔'], ['l03', '你在幹嘛？'], ['l04', '晚餐吃什麼好呢'],
  ['l05', '我考試考過了！'], ['l06', '外面在下雨欸'], ['l07', '可以安慰我一下嗎'], ['l08', '你覺得我這件衣服好看嗎'],
  ['l09', '我先去洗澡囉'], ['l10', '晚安，明天見'],
  ['h01', '坐下來陪我聊天吧'], ['h02', '轉過去讓我看看你的背'], ['h03', '你害羞的時候會遮臉嗎？做給我看'],
  ['h04', '開心到跳起來！'], ['h05', '好了，站起來伸個懶腰'],
];
process.env.HSIN_COMPANION_PORT = '8796';
const { T } = createRequire(import.meta.url)('../../desktop/i18n.js');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ollama-chat-'));
const chat = new Chat({ port: 8796, configDir: dir });
const cfg = { kind: 'ollama', baseUrl: 'http://localhost:11434', model, key: '', desc: 'auto' };
await chat.tool();
const call = chat.mcp.client.callTool.bind(chat.mcp.client);
let results = [];
chat.mcp.client.callTool = async (req) => { const r = await call(req); results.push((r.content ?? []).map((x) => x.text ?? '').join(' ')); return r; };

const rows = [];
for (const [id, text] of LINES) {
  await chat.clear();
  results = [];
  const t0 = Date.now();
  const r = await chat.send(text, cfg, T.zh.instructions);
  const shown = results.findIndex((x) => /^\[internal[^\]]*\]\s*Shown/.test(x) || /\bShown\b/.test(x.split('.')[0]));
  const simplified = results.some((x) => /simplified/.test(x));
  const kind = r.error ? `error:${r.error}` : shown < 0 ? 'none' : simplified ? 'simplified' : shown === 0 ? 'first' : 'retry';
  rows.push({ id, text, kind, s: (Date.now() - t0) / 1000, calls: results.length, lines: r.turn?.lines ?? [], reply: r.turn?.reply ?? '', results });
  console.log(id, kind, `${((Date.now() - t0) / 1000).toFixed(1)} s`, (r.turn?.lines ?? []).map((l) => l.say).join(' / ').slice(0, 60));
}
fs.writeFileSync(out, rows.map((x) => JSON.stringify(x)).join('\n') + '\n');
const n = (k) => rows.filter((x) => x.kind === k).length;
console.log(`${model}: first ${n('first')} / retry ${n('retry')} / simplified ${n('simplified')} / none ${n('none')} / error ${rows.filter((x) => x.kind.startsWith('error')).length}; ${(rows.reduce((a, x) => a + x.s, 0) / rows.length).toFixed(1)} s per line`);
await chat.close();
fs.rmSync(dir, { recursive: true, force: true });

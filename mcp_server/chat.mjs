// Built-in chat (the API version, 2026-11-13, grilling Q1-Q9): the app's own chat window talks to an LLM API with the
// user's own key and gives it the same `perform` tool an MCP client gets. The tool is the real MCP server (server.mjs)
// over stdio, so solving, the rule filter, the retry / simplified rules and the format fixes are exactly the MCP path.
//
//   provider cfg = {kind: 'anthropic' | 'openai', baseUrl, model, key, desc: 'auto' | 'long' | 'short'}
//     anthropic = the Anthropic SDK; openai = any OpenAI-compatible /chat/completions (OpenAI, OpenRouter, Gemini,
//     DeepSeek, Ollama, LM Studio ...)
//   tool description (Q7): hosted APIs get the MCP description + 4 examples (chat_examples.txt), local endpoints
//     (localhost) the short one (chat_desc_short.txt, from the open-model test of 2026-11-12); 'long' / 'short' force it
//   history (Q5): <config>/chat.json = {summary, turns: [{t, user, lines: [{say, emotion}], reply}]}; older turns are
//     sent back as perform calls without their keyframes (say / emotion only, result 'Shown.'), the current turn's tool
//     round verbatim; past KEEP + SUMMARIZE_AT turns the older ones are summarized into `summary` (kept in the system
//     prompt) and dropped. clear() empties both.
//   cost (Q9): Anthropic list prices below, OpenRouter's usage.cost, localhost = free, else tokens only.
import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Anthropic from '@anthropic-ai/sdk';
import { Client } from '@modelcontextprotocol/client';
import { StdioClientTransport } from '@modelcontextprotocol/client/stdio';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const KEEP = 10, SUMMARIZE_AT = 20;       // past 20 turns, all but the last 10 go into the summary (~40 messages)
const MAX_ROUNDS = 4;                     // tool rounds per user line (perform, maybe one retry, a reply)
// $ per million tokens: input, output, cache read (cache writes = 1.25 x input) — list prices 2026-11
export const PRICES = {
  'claude-fable-5-2': [8, 40, 0.2], 'claude-opus-5-5': [4, 20, 0.2], 'claude-sonnet-5-5': [2, 10, 0.1], 'claude-haiku-5-5': [0.1, 0.5, 0.01],
  'claude-opus-5': [5, 25, 0.5], 'claude-sonnet-5': [2, 10, 0.2], 'claude-haiku-4-5': [1, 5, 0.1],
  'claude-fable-5-1': [10, 50, 1], 'claude-fable-5': [10, 50, 1], 'claude-opus-4-8': [5, 25, 0.5], 'claude-opus-4-7': [5, 25, 0.5],
  'claude-opus-4-6': [5, 25, 0.5], 'claude-opus-4-5': [5, 25, 0.5], 'claude-sonnet-4-6': [3, 15, 0.3],
};
// dated ids (claude-haiku-4-5-20251001) take their base model's price
export const priceOf = (id) => PRICES[id] ?? PRICES[String(id).replace(/-\d{8}$/, '')] ?? null;
export const ANTHROPIC_MODELS = ['claude-opus-5-5', 'claude-sonnet-5-5', 'claude-haiku-5-5', 'claude-fable-5-2'];
export const isLocal = (url) => /^https?:\/\/(localhost|127\.|\[::1\]|0\.0\.0\.0)/i.test(url ?? '');
const SHORT = () => fs.readFile(path.join(HERE, 'chat_desc_short.txt'), 'utf-8');
const EXAMPLES = () => fs.readFile(path.join(HERE, 'chat_examples.txt'), 'utf-8');

export class Chat {
  constructor({ port, configDir, log = () => {}, langs = () => null }) {
    this.port = port; this.langs = langs; this.file = path.join(configDir, 'chat.json'); this.log = log;
    this.queue = Promise.resolve(); this.session = { in: 0, out: 0, cost: 0, costKnown: true };
  }

  // ---- the perform tool: our own MCP server, started once ----
  async tool() {
    if (this.mcp) return this.mcp;
    const env = { ...process.env, HSIN_COMPANION_PORT: String(this.port) };
    if (process.versions.electron) env.ELECTRON_RUN_AS_NODE = '1';
    const client = new Client({ name: 'veviah-chat', version: '1' });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: [path.join(HERE, 'server.mjs')], env, stderr: 'ignore' }));
    const t = (await client.listTools()).tools.find((x) => x.name === 'perform');
    this.mcp = { client, description: t.description, schema: t.inputSchema };
    return this.mcp;
  }
  async close() { try { await this.mcp?.client.close(); } catch { /* gone */ } this.mcp = null; }

  async description(cfg) {
    const { description } = await this.tool();
    const short = cfg.desc === 'short' || (cfg.desc !== 'long' && (cfg.kind === 'ollama' || (cfg.kind === 'openai' && isLocal(cfg.baseUrl))));
    return short ? SHORT() : `${description}\n\n${await EXAMPLES()}`;
  }

  // ---- history ----
  async load() {
    try { return JSON.parse(await fs.readFile(this.file, 'utf-8')); } catch { return { summary: '', turns: [] }; }
  }
  async save(h) { await fs.mkdir(path.dirname(this.file), { recursive: true }); await fs.writeFile(this.file, JSON.stringify(h, null, 1)); }
  async history() { const h = await this.load(); return { turns: h.turns, summarized: !!h.summary, session: this.session }; }
  clear() { return (this.queue = this.queue.then(() => this.save({ summary: '', turns: [] }))); }

  // older turns as neutral messages: perform calls with say / emotion only
  static past(turns) {
    const out = [];
    turns.forEach((tn, i) => {
      out.push({ role: 'user', text: tn.user });
      if (tn.lines?.length) {
        const calls = tn.lines.map((l, k) => ({ id: `past_${i}_${k}`, name: 'perform', args: { say: l.say, ...(l.subtitle ? { subtitle: l.subtitle } : {}), emotion: l.emotion, keyframes: '(omitted)' } }));
        out.push({ role: 'assistant', text: '', calls });
        for (const c of calls) out.push({ role: 'tool', id: c.id, name: 'perform', result: 'Shown.' });
      }
      if (tn.reply) out.push({ role: 'assistant', text: tn.reply, calls: [] });
    });
    return out;
  }

  // ---- one user line (serialized: a second line waits for the first) ----
  send(text, cfg, system) {
    const run = this.queue.then(() => this.turn(String(text ?? '').trim(), cfg, system));
    this.queue = run.catch(() => {});
    return run;
  }

  async turn(text, cfg, system) {
    if (!text) return { error: 'empty' };
    const h = await this.load();
    // the language rule up front (2026-11-13: told only by the first result, the model said its first line again in the
    // voice language and both were spoken); also never the same line twice
    const sys = [system, languageRule(this.langs()), 'Call perform once per reply. A line that was shown is done: never perform it again.',
      h.summary ? `[Earlier conversation, summarized]\n${h.summary}` : ''].filter(Boolean).join('\n\n');
    const tools = [{ name: 'perform', description: await this.description(cfg), schema: (await this.tool()).schema }];
    const msgs = [...Chat.past(h.turns), { role: 'user', text }];
    const turn = { t: Date.now(), user: text, lines: [], reply: '' };
    const usage = { in: 0, out: 0, cost: 0, costKnown: true };
    let error = null;
    try {
      for (let round = 0; round < MAX_ROUNDS; round++) {
        const r = await call(cfg, sys, msgs, tools);
        addUsage(usage, r.usage);
        msgs.push({ role: 'assistant', text: r.text, calls: r.calls, raw: r.raw });
        if (r.refusal) { error = 'refusal'; break; }
        if (!r.calls.length) { turn.reply = r.text.trim(); break; }
        for (const c of r.calls) {
          let result;
          if (c.name !== 'perform') result = `ERROR unknown tool ${c.name}`;
          else {
            try {
              const res = await this.mcp.client.callTool({ name: 'perform', arguments: c.args });
              result = (res.content ?? []).map((x) => x.text ?? '').join('\n');
              if (res.isError) result = 'ERROR ' + result;
              else if (!/Not shown yet/.test(result) && c.args?.say) turn.lines.push({ say: String(c.args.say), emotion: c.args.emotion ?? 'neutral', ...(c.args.subtitle ? { subtitle: String(c.args.subtitle) } : {}) });
            } catch (e) { result = 'ERROR ' + (e.message ?? e); }
          }
          msgs.push({ role: 'tool', id: c.id, name: c.name, result });
        }
      }
    } catch (e) {
      this.log('chat error', e.status ?? '', e.message ?? e);
      // a failure after a line was shown (2026-11-14: Ollama's 500 when it cannot parse gpt-oss's closing reply) ends
      // the turn quietly: she already answered
      if (!turn.lines.length) error = errorCode(e);
    }
    for (const k of ['in', 'out', 'cost']) this.session[k] += usage[k];
    this.session.costKnown &&= usage.costKnown;
    if (turn.lines.length || turn.reply) {
      h.turns.push(turn); await this.save(h);
      if (h.turns.length > SUMMARIZE_AT) this.queue = this.queue.then(() => this.summarize(cfg)).catch((e) => this.log('summary failed', e.message ?? e));
    }
    return { turn, error, usage, session: this.session };
  }

  // older turns -> a short summary kept in the system prompt (Q5)
  async summarize(cfg) {
    const h = await this.load();
    if (h.turns.length <= SUMMARIZE_AT) return;
    const old = h.turns.slice(0, -KEEP);
    const talk = old.map((t) => `User: ${t.user}\nYou: ${[...t.lines.map((l) => l.say), t.reply].filter(Boolean).join(' ')}`).join('\n');
    const ask = 'Summarize this conversation between the user and you (their desktop companion character) as your own memory: '
      + 'facts about the user, what you talked about, promises, the mood. At most 200 words, in the language of the conversation.\n\n'
      + (h.summary ? `Earlier summary:\n${h.summary}\n\n` : '') + `Conversation:\n${talk}`;
    const r = await call(cfg, 'You write short memory notes.', [{ role: 'user', text: ask }], []);
    addUsage(this.session, r.usage);
    if (!r.text.trim()) return;
    const now = await this.load();                     // turns added meanwhile stay
    now.summary = r.text.trim(); now.turns = now.turns.slice(old.length);
    await this.save(now);
  }

  // a tiny request to check key / model / URL (settings page)
  async test(cfg) {
    try { const r = await call(cfg, 'Reply with: ok', [{ role: 'user', text: 'ping' }], [], 16); return { ok: true, text: r.text }; }
    catch (e) { return { ok: false, error: errorCode(e), message: String(e.message ?? e).slice(0, 300) }; }
  }
  // the models this key / endpoint offers, only ones that can chat with tools (2026-11-13, the author: nobody types model
  // names; a service without a list is not supported): [{id, name, price: '$in / $out' | null}]
  async models(cfg) {
    try {
      if (cfg.kind === 'anthropic') {
        const c = new Anthropic({ apiKey: cfg.key });
        const out = [];
        for await (const m of c.models.list()) out.push({ id: m.id, name: m.display_name || m.id, price: priceText(priceOf(m.id)) });
        return { ok: true, models: out };
      }
      if (cfg.kind === 'ollama') return await ollamaModels(cfg);
      const r = await fetch(`${base(cfg)}/models`, { headers: auth(cfg) });
      if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status}`), { status: r.status });
      const data = (await r.json()).data ?? [];
      const models = data
        .filter((m) => m?.id && !NOT_CHAT.test(m.id))
        // OpenRouter says which models take tools; keep those only
        .filter((m) => !Array.isArray(m.supported_parameters) || m.supported_parameters.includes('tools'))
        .map((m) => {
          const id = String(m.id).replace(/^models\//, '');           // Gemini lists 'models/gemini-...'
          const pin = Number(m.pricing?.prompt), pout = Number(m.pricing?.completion);
          return { id, name: m.name || id, price: Number.isFinite(pin) && Number.isFinite(pout) ? priceText([pin * 1e6, pout * 1e6]) : null };
        })
        .sort((x, y) => x.name.localeCompare(y.name));
      if (!models.length) return { ok: false, error: 'nolist', models: [] };
      return { ok: true, models };
    } catch (e) { return { ok: false, error: e.status === 404 ? 'nolist' : errorCode(e), models: [] }; }
  }
}

const LANG = { zh: 'Traditional Chinese', en: 'English', ja: 'Japanese' };
export function languageRule(l) {
  if (!l?.ui) return '';
  const V = LANG[l.voice] ?? l.voice, U = LANG[l.ui] ?? l.ui;
  if (!l.voice) return `Language: the voice is off; \`say\` is shown in the bubble; write it in ${U}.`;
  if (l.voice === l.ui) return `Language: \`say\` is spoken and shown; write it in ${V}, no subtitle.`;
  return `Language: \`say\` is spoken, write it in ${V}; always also give \`subtitle\` = the same line in ${U} (shown in the bubble). Talk to the user in ${U}.`;
}

function addUsage(u, x) {
  if (!x) return;
  u.in += x.in; u.out += x.out;
  if (x.cost == null) u.costKnown = false; else u.cost += x.cost;
}
function errorCode(e) {
  const s = e?.status;
  if (s === 401 || s === 403) return 'key';
  if (s === 404) return 'model';
  if (s === 429) return 'rate';
  if (s === 402) return 'credit';
  if (/credit balance/i.test(e?.message ?? '')) return 'credit';
  if (s >= 500) return 'server';
  if (e?.ollama && /fetch failed|ECONNREFUSED/.test(e?.message ?? e?.cause?.message ?? '')) return 'noollama';
  if (e?.name === 'APIConnectionError' || /fetch failed|ECONNREFUSED|ENOTFOUND/.test(e?.message ?? e?.cause?.message ?? '')) return 'network';
  return 'other';
}
// model ids that cannot chat (embeddings, speech, images, moderation, old completion models)
const NOT_CHAT = /embed|tts|whisper|transcribe|dall-e|image|imagen|moderation|davinci|babbage|audio|realtime|aqa|veo|sora|rerank/i;
const priceText = (p) => (p ? `$${+p[0].toFixed(2)} / $${+p[1].toFixed(2)}` : null);
const base = (cfg) => String(cfg.baseUrl ?? '').replace(/\/+$/, '');
const auth = (cfg) => (cfg.key ? { Authorization: `Bearer ${cfg.key}` } : {});

// ---- the two adapters; both return {text, calls: [{id, name, args}], raw, refusal, usage: {in, out, cost|null}} ----
function call(cfg, system, msgs, tools, maxTokens = 4000) {
  if (cfg.kind === 'anthropic') return callAnthropic(cfg, system, msgs, tools, maxTokens);
  if (cfg.kind === 'ollama') return callOllama(cfg, system, msgs, tools, maxTokens);
  return callOpenAI(cfg, system, msgs, tools, maxTokens);
}

async function callAnthropic(cfg, system, msgs, tools, maxTokens) {
  const client = new Anthropic({ apiKey: cfg.key, maxRetries: 2 });
  const messages = [];
  for (const m of msgs) {
    if (m.role === 'user') messages.push({ role: 'user', content: m.text });
    else if (m.role === 'assistant') {
      // this turn's own replies go back verbatim (thinking blocks included); older turns were rebuilt without any
      const content = m.raw ?? [...(m.text ? [{ type: 'text', text: m.text }] : []), ...m.calls.map((c) => ({ type: 'tool_use', id: c.id, name: c.name, input: c.args }))];
      if (content.length) messages.push({ role: 'assistant', content });
    } else {
      const block = { type: 'tool_result', tool_use_id: m.id, content: m.result, ...(m.result.startsWith('ERROR') ? { is_error: true } : {}) };
      const last = messages.at(-1);
      if (last?.role === 'user' && Array.isArray(last.content) && last.content[0]?.type === 'tool_result') last.content.push(block);
      else messages.push({ role: 'user', content: [block] });
    }
  }
  const req = {
    model: cfg.model, max_tokens: maxTokens, system, messages,
    cache_control: { type: 'ephemeral' },                       // the long tool description + history are reused every line
    ...(tools.length ? { tools: tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.schema })) } : {}),
    // chat needs little reasoning; effort exists on the Claude 5 models (Haiku 4.5 rejects it)
    ...(/^claude-(opus|sonnet|haiku|fable|mythos)-5/.test(cfg.model) ? { output_config: { effort: 'low' } } : {}),
  };
  const r = await client.messages.create(req);
  const text = r.content.filter((b) => b.type === 'text').map((b) => b.text).join('');
  const calls = r.content.filter((b) => b.type === 'tool_use').map((b) => ({ id: b.id, name: b.name, args: b.input }));
  const u = r.usage ?? {};
  const p = priceOf(cfg.model);
  const inTok = (u.input_tokens ?? 0) + (u.cache_creation_input_tokens ?? 0) + (u.cache_read_input_tokens ?? 0);
  const cost = p ? ((u.input_tokens ?? 0) * p[0] + (u.cache_creation_input_tokens ?? 0) * p[0] * 1.25
    + (u.cache_read_input_tokens ?? 0) * p[2] + (u.output_tokens ?? 0) * p[1]) / 1e6 : null;
  return { text, calls, raw: r.content, refusal: r.stop_reason === 'refusal', usage: { in: inTok, out: u.output_tokens ?? 0, cost } };
}

async function callOpenAI(cfg, system, msgs, tools, maxTokens) {
  const messages = [{ role: 'system', content: system }];
  for (const m of msgs) {
    if (m.role === 'user') messages.push({ role: 'user', content: m.text });
    else if (m.role === 'assistant') {
      messages.push({ role: 'assistant', content: m.text || null,
        ...(m.calls.length ? { tool_calls: m.calls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: JSON.stringify(c.args ?? {}) } })) } : {}) });
    } else messages.push({ role: 'tool', tool_call_id: m.id, content: m.result });
  }
  const body = { model: cfg.model, messages, max_tokens: maxTokens,
    ...(tools.length ? { tools: tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } })) } : {}),
    ...(/openrouter\.ai/.test(cfg.baseUrl) ? { usage: { include: true } } : {}) };
  let r;
  for (let a = 0; ; a++) {
    r = await fetch(`${base(cfg)}/chat/completions`, { method: 'POST', headers: { 'content-type': 'application/json', ...auth(cfg) }, body: JSON.stringify(body) });
    if ((r.status === 429 || r.status >= 500) && a < 2) { await new Promise((ok) => setTimeout(ok, 1500 * (a + 1))); continue; }
    break;
  }
  if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`), { status: r.status });
  const j = await r.json();
  const msg = j.choices?.[0]?.message ?? {};
  const calls = (msg.tool_calls ?? []).map((c, i) => {
    let args = c.function?.arguments;
    if (typeof args === 'string') { try { args = JSON.parse(args); } catch { args = { _unparsed: args }; } }
    return { id: c.id || `call_${Date.now()}_${i}`, name: c.function?.name, args };
  });
  const u = j.usage ?? {};
  const cost = isLocal(cfg.baseUrl) ? 0 : typeof u.cost === 'number' ? u.cost : null;
  return { text: msg.content ?? '', calls, raw: null, refusal: false, usage: { in: u.prompt_tokens ?? 0, out: u.completion_tokens ?? 0, cost } };
}

// ---- Ollama through its own API (2026-11-14). Its OpenAI-compatible endpoint cannot set the context size, and
// Ollama's default (4096 tokens) silently cuts the instructions + tool description + history from the front; the
// open-model test of 2026-11-12 ran on /api/chat with num_ctx 16384, so the app does the same. baseUrl = the server
// root (http://localhost:11434). gpt-oss cannot turn thinking off ('low'), others run without it.
const OLLAMA_CTX = 16384;
const ollamaBase = (cfg) => base(cfg).replace(/\/v1$/, '');
async function ollamaFetch(cfg, p, body) {
  let r;
  try {
    r = await fetch(`${ollamaBase(cfg)}${p}`, body ? { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) } : {});
  } catch (e) { throw Object.assign(e, { ollama: true }); }
  if (!r.ok) throw Object.assign(new Error(`HTTP ${r.status}: ${(await r.text()).slice(0, 300)}`), { status: r.status });
  return r.json();
}
// the installed models that can call tools (Ollama's /api/show capabilities), with their size on disk
async function ollamaModels(cfg) {
  const tags = (await ollamaFetch(cfg, '/api/tags')).models ?? [];
  const out = [];
  for (const m of tags) {
    let caps = [];
    try { caps = (await ollamaFetch(cfg, '/api/show', { model: m.name })).capabilities ?? []; } catch { /* unknown: leave it out */ }
    if (!caps.includes('tools')) continue;
    const gb = m.size ? `${(m.size / 1e9).toFixed(1)} GB` : null;
    out.push({ id: m.name, name: m.name, price: gb, size: m.size ?? 0 });
  }
  out.sort((a, b) => a.name.localeCompare(b.name));
  if (!out.length) return { ok: false, error: tags.length ? 'notools' : 'nomodels', models: [] };
  return { ok: true, models: out };
}
async function callOllama(cfg, system, msgs, tools, maxTokens) {
  const messages = [{ role: 'system', content: system }];
  for (const m of msgs) {
    if (m.role === 'user') messages.push({ role: 'user', content: m.text });
    else if (m.role === 'assistant') {
      messages.push({ role: 'assistant', content: m.text || '',
        ...(m.calls.length ? { tool_calls: m.calls.map((c) => ({ function: { name: c.name, arguments: c.args ?? {} } })) } : {}) });
    } else messages.push({ role: 'tool', content: m.result });
  }
  const body = { model: cfg.model, messages, stream: false, think: /gpt-oss/i.test(cfg.model) ? 'low' : false,
    options: { num_ctx: OLLAMA_CTX, num_predict: maxTokens },
    ...(tools.length ? { tools: tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.schema } })) } : {}) };
  let j;
  for (let a = 0; ; a++) {
    try { j = await ollamaFetch(cfg, '/api/chat', body); break; }
    catch (e) {
      // a model that cannot think rejects `think` with 400 on some Ollama versions: once more without it
      if (e.status === 400 && /think/i.test(e.message) && 'think' in body) { delete body.think; continue; }
      // Ollama answers 500 when it cannot parse a model's tool call (seen with gpt-oss): one more try
      if (e.status === 500 && a < 1) continue;
      throw e;
    }
  }
  const msg = j.message ?? {};
  const calls = (msg.tool_calls ?? []).map((c, i) => {
    let args = c.function?.arguments;
    if (typeof args === 'string') { try { args = JSON.parse(args); } catch { args = { _unparsed: args }; } }
    return { id: `call_${Date.now()}_${i}`, name: c.function?.name, args };
  });
  return { text: msg.content ?? '', calls, raw: null, refusal: false, usage: { in: j.prompt_eval_count ?? 0, out: j.eval_count ?? 0, cost: 0 } };
}

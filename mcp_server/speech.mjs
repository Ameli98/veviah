// Speech for the desktop companion (2026-11-05, item (g); plan in CLAUDE.md): the line Claude says is split into
// sentences and spoken by a TTS that talks GPT-SoVITS's api_v2 protocol (POST /tts -> wav). The companion server owns
// it: started when voice is on (and warmed up with one unplayed sentence, so the first real line does not wait for
// the model to load), stopped when voice is turned off, the voice changes, or the companion quits.
//
// Voices: ~/.config/hsin/voices.json (env HSIN_VOICES), never in the repo (a user's own voice stays on their machine):
//   {"voices": [{"name", "engine": "gpt-sovits", "root": GPT-SoVITS checkout, "python": its environment's python,
//                "version": "v2Pro", "gpt": .ckpt, "sovits": .pth, "refs": refs.list, "lang": "ja", "romaji"?: false}]}
//   or {"name", "url": "http://127.0.0.1:9880", "refs", "lang"}: an api_v2 server the user runs (nothing is started;
//   also how the tests plug in a fake TTS).
// refs.list lines are "<wav>|<speaker>|<LANG>|<transcript>" with the wav in a folder named after its emotion
// (refs/<emotion>/x.wav, as an earlier voice project's voice pipeline writes them). The api_v2 server is started through
// tts/gsv_api.py (cudnn.benchmark kept off) with its own config file in ~/.cache/hsin/tts/ (api_v2 writes its config
// back, so the user's checkout is not touched).
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import { existsSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { CACHE_DIR, CONFIG_DIR, freePort } from './model.mjs';
export const VOICES = process.env.HSIN_VOICES || path.join(CONFIG_DIR, 'voices.json');
// The user's choice of reference clip per emotion and the clips they added, per voice (2026-11-12, the author: users
// adjust the reference audio freely): <voice>.json = {emotion: {wav, text, lang}}, <voice>.added.list = refs.list lines
const REF_DIR = process.env.HSIN_VOICE_REFS || path.join(path.dirname(VOICES), 'voice_refs');
const KEEP_LINES = 4;          // lines whose audio is kept (one playing + the page's queue of up to 2 + slack)
export const EMOTIONS = ['neutral', 'happy', 'angry', 'sad', 'shy', 'surprised'];
// sample line for the settings page; long enough that a voice trained in another language reads it well (2026-11-13: the
// 9-letter Chinese line was misread by the default voice)
const PREVIEW = { zh: '嗯，這個聲音聽起來怎麼樣？你喜歡的話就用這個吧。', ja: 'ねえ、こんな感じの声でどうかな？気に入ったら、これにしようね。', en: 'Hey, how does this voice sound to you? If you like it, let us keep this one.' };
const CACHE = path.join(CACHE_DIR, 'tts');
export const LANGS = ['zh', 'ja', 'en'];
export const DEFAULTS = { voice: false, voiceName: null, voiceLang: null, uiLang: null, romaji: true, releaseMin: 0 };
const WARM = { zh: '你好。', ja: 'こんにちは。', en: 'Hello.' };
const FIRST_WAIT_MS = 120000;     // model load + warm-up on a cold start

// Sentences: cut after 。！？!?… and newlines (and ". " in Latin text); a piece with no letters joins its neighbour.
// Then by length (2026-11-12, the author): GPT-SoVITS misreads short lines in a language other than its reference's
// (the Japanese default voice read "今天也要一起加油喔！" alone right 1-3 times in 6, with a sentence after it 6 of 6),
// and long ones drift: a sentence over MAX is cut at commas (or hard), one under MIN joins the next (or the previous)
// while the pair stays within MAX. Length = letters / digits for zh / ja, words for en.
const MIN = { zh: 10, en: 5, ja: 3 };   // measured 2026-11-12 (~/hsin_runs/tts_bench/len_refs): zh <= 9 chars 1-6 of 6 misread, >= 11 none; en <= 4 words, ja 'うん。'
const MAX = { zh: 40, en: 25, ja: 50 };
const unitLen = (s, lang) => lang === 'en' ? (s.match(/[\p{L}\p{N}'’]+/gu) ?? []).length : (s.match(/[\p{L}\p{N}]/gu) ?? []).length;

function cutLong(s, lang) {
  const max = MAX[lang] ?? MAX.zh;
  if (unitLen(s, lang) <= max) return [s];
  const bits = s.split(/(?<=[，、,;；：:—])\s*/u).filter(Boolean);
  const out = [];
  let cur = '';
  for (const b of bits) {
    if (cur && unitLen(cur + b, lang) > max) { out.push(cur); cur = b; } else cur += (cur && lang === 'en' ? ' ' : '') + b;
  }
  if (cur) out.push(cur);
  return out.flatMap((p) => {                       // still too long (no commas): hard cut into equal pieces
    const n = unitLen(p, lang);
    if (n <= max) return [p];
    const k = Math.ceil(n / max);
    if (lang === 'en') { const w = p.split(/\s+/); const per = Math.ceil(w.length / k); return Array.from({ length: k }, (_, i) => w.slice(i * per, (i + 1) * per).join(' ')).filter(Boolean); }
    const ch = [...p]; const per = Math.ceil(ch.length / k);
    return Array.from({ length: k }, (_, i) => ch.slice(i * per, (i + 1) * per).join('')).filter(Boolean);
  });
}

export function sentences(text, lang = 'zh') {
  const parts = String(text ?? '').replace(/\s+/g, ' ').trim()
    .split(/(?<=[。！？!?…]+[」』）)]*)\s*|(?<=[.;]+)\s+|\n+/u).map((s) => s.trim()).filter(Boolean);
  const raw = [];
  for (const p of parts) {
    if (raw.length && !/[\p{L}\p{N}]/u.test(p)) raw[raw.length - 1] += p;
    else raw.push(p);
  }
  const cut = raw.flatMap((p) => cutLong(p, lang));
  const min = MIN[lang] ?? MIN.zh, max = MAX[lang] ?? MAX.zh, sep = lang === 'en' ? ' ' : '';
  const out = [];
  for (let i = 0; i < cut.length; i++) {
    let s = cut[i];
    while (unitLen(s, lang) < min && i + 1 < cut.length && unitLen(s + cut[i + 1], lang) <= max) s += sep + cut[++i];
    if (unitLen(s, lang) < min && out.length && unitLen(out[out.length - 1] + s, lang) <= max) out[out.length - 1] += sep + s;
    else out.push(s);
  }
  return out;
}

export class Speech {
  constructor({ log = () => {}, port = Number(process.env.HSIN_TTS_PORT || 9880), root }) {
    this.log = log; this.port = port; this.root = root;
    this.settings = { ...DEFAULTS };
    this.proc = null; this.state = 'off'; this.ready = null; this.voice = null; this.romajiOf = {};
    this.jobs = new Map();      // serial -> [Promise<Buffer|null>] per sentence (only the newest line is kept)
    this.latest = 0; this.releaseTimer = null; this.error = null;
  }

  async voices() {
    let list = [];
    try { list = JSON.parse(await fs.readFile(VOICES, 'utf-8')).voices ?? []; } catch { return []; }
    // The default voice's clips ship inside the app (backend/tts/default_voice), so a path stored at install time goes
    // stale when the app moves (2026-11-13: a zip unpacked elsewhere, then deleted). A missing path that points into an
    // app's default_voice folder is read from this app's own copy instead.
    const own = path.join(this.root, 'tts', 'default_voice');
    const fix = (p) => {
      if (!p || existsSync(p)) return p;
      const m = String(p).replace(/\\/g, '/').match(/\/tts\/default_voice\/(.+)$/);
      return m && existsSync(path.join(own, m[1])) ? path.join(own, m[1]) : p;
    };
    return list.map((v) => ({ ...v, refs: fix(v.refs), pool: fix(v.pool) }));
  }

  // The settings the app sends (POST /settings). Returns the current state.
  async set(s) {
    const old = this.settings;
    this.settings = { ...this.settings, ...s };
    const voices = await this.voices();
    const v = voices.find((x) => x.name === this.settings.voiceName) ?? voices[0] ?? null;
    if (v && !this.settings.voiceLang) this.settings.voiceLang = v.lang;
    if (!this.settings.voice || !v) await this.stop();
    else if (this.state === 'off' || this.state === 'error' || this.voice?.name !== v.name) { await this.stop(); this.start(v); }
    if (old.releaseMin !== this.settings.releaseMin) this.touch();
    return this.info(voices);
  }

  async info(voices) {
    voices ??= await this.voices();
    const v = voices.find((x) => x.name === this.settings.voiceName) ?? voices[0] ?? null;
    return { settings: this.settings, state: this.state, error: this.error, voice: v?.name ?? null,
             trainedLang: v?.lang ?? null, voices: voices.map((x) => ({ name: x.name, lang: x.lang })) };
  }

  start(v) {
    this.voice = v; this.state = 'loading'; this.error = null;
    this.ready = (async () => {
      this.refs = await this.readRefs(v.refs);
      this.override = await this.readOverride(v.name);
      this.base = v.url ?? `http://127.0.0.1:${this.port}`;
      const t0 = Date.now();
      if (v.url) { await this.warm(v, t0, null); return; }
      // base files this voice's version needs (companion sets prepare = the installer's ensureVersion)
      if (this.prepare) await this.prepare(v);
      await fs.mkdir(CACHE, { recursive: true });
      const cfg = path.join(CACHE, `${v.name}.yaml`);
      // api_v2 reads its weights from the 'custom' section of its config file (and writes the file back)
      await fs.writeFile(cfg, [
        'custom:', `  bert_base_path: GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large`,
        `  cnhuhbert_base_path: GPT_SoVITS/pretrained_models/chinese-hubert-base`, '  device: cuda', '  is_half: true',
        `  t2s_weights_path: ${v.gpt}`, `  version: ${v.version ?? 'v2Pro'}`, `  vits_weights_path: ${v.sovits}`, ''].join('\n'));
      // 9880 is GPT-SoVITS's own default: a user running it already holds it (2026-11-14): the next free one
      this.port = await freePort(this.port);
      this.base = `http://127.0.0.1:${this.port}`;
      const logFile = await fs.open(path.join(CACHE, `${v.name}.log`), 'w');
      // nltk data (English text front end) inside the TTS folder when the installer put it there
      this.proc = condaSpawn(v.python, [path.join(this.root, 'tts/gsv_api.py'), v.root, '-a', '127.0.0.1', '-p', String(this.port), '-c', cfg],
        { cwd: v.root, detached: true, stdio: ['ignore', logFile.fd, logFile.fd], ...(v.nltk ? { env: { ...process.env, NLTK_DATA: v.nltk } } : {}) });
      const proc = this.proc;
      proc.on('exit', (code) => { if (this.proc === proc) { this.proc = null; this.state = 'off'; if (code) this.error = `TTS exited ${code}`; } logFile.close(); });
      // romaji of the reference transcripts (used when the voice speaks another language than its references)
      if (v.lang === 'ja') {
        try { this.romajiOf = JSON.parse(await run(v.python, [path.join(this.root, 'tts/romaji.py'), v.refs])); }
        catch (e) { this.log('romaji', e.message); this.romajiOf = {}; }
      }
      await this.warm(v, t0, proc);
    })().catch(async (e) => { this.error = String(e.message ?? e); this.log('TTS', this.error); await this.stop(); this.state = 'error'; });
  }

  // wait for the server, then synthesize one unplayed sentence (loads the models)
  async warm(v, t0, proc) {
    while (Date.now() - t0 < FIRST_WAIT_MS) {
      if (proc && this.proc !== proc) throw new Error('TTS stopped');
      try { await fetch(`${this.base}/control`, { signal: AbortSignal.timeout(500) }); break; } catch { await sleep(500); }
    }
    await this.synth(WARM[this.settings.voiceLang] ?? WARM.ja, 'neutral');
    this.state = 'ready'; this.log(`TTS ${v.name} ready in ${Date.now() - t0} ms`);
    this.touch();
  }

  async stop() {
    clearTimeout(this.releaseTimer);
    const p = this.proc;
    this.proc = null; this.state = 'off'; this.ready = null;
    if (p) {
      try {
        if (process.platform === 'win32') spawn('taskkill', ['/pid', String(p.pid), '/T', '/F'], { windowsHide: true });
        else process.kill(-p.pid, 'SIGTERM');
      } catch { /* gone */ }
      this.log('TTS stopped');
    }
  }

  // with "release after N minutes" set: the TTS is stopped N minutes after the last line, started again by the next
  touch() {
    clearTimeout(this.releaseTimer);
    const m = Number(this.settings.releaseMin) || 0;
    if (m > 0 && this.proc) this.releaseTimer = setTimeout(() => { this.log('TTS released (idle)'); this.stop(); }, m * 60000);
  }

  async readRefs(file) {
    const refs = {};
    for (const line of (await fs.readFile(file, 'utf-8')).split('\n')) {
      const [w, , lang, text] = line.split('|');
      if (!w || !text) continue;
      const wav = path.resolve(path.dirname(file), w);   // a relative clip path is relative to the list (bundled voice)
      const emo = path.basename(path.dirname(wav));
      (refs[emo] ??= []).push({ wav, lang: (lang ?? '').toLowerCase(), text });
    }
    return refs;
  }

  pickRef(emotion) {
    const o = this.override?.[emotion];
    if (o) return o;
    const list = this.refs?.[emotion]?.length ? this.refs[emotion] : this.refs?.neutral ?? Object.values(this.refs ?? {})[0] ?? [];
    return list[Math.floor(Math.random() * list.length)];
  }

  async synth(text, emotion, ref = this.pickRef(emotion)) {
    if (!ref) throw new Error('no reference clips');
    const lang = this.settings.voiceLang ?? this.voice.lang;
    let prompt = ref.text, promptLang = ref.lang || this.voice.lang;
    // romaji across languages (v2Pro voices); a voice may turn it off ("romaji": false: v5 read the Japanese
    // transcript better for Chinese lines, 2026-11-12 ~/hsin_runs/tts_bench/romaji_test)
    if (this.settings.romaji && this.voice?.romaji !== false && lang !== promptLang && this.romajiOf[ref.wav]) { prompt = this.romajiOf[ref.wav]; promptLang = 'en'; }
    const r = await fetch(`${this.base}/tts`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ text, text_lang: lang, ref_audio_path: ref.wav, prompt_text: prompt, prompt_lang: promptLang,
                             text_split_method: 'cut0', media_type: 'wav', batch_size: 1 }),
    });
    if (!r.ok) throw new Error(`tts ${r.status}: ${(await r.text()).slice(0, 200)}`);
    return Buffer.from(await r.arrayBuffer());
  }

  // A new line: returns {serial, sentences} when it will be spoken, else null (voice off, not ready). Sentences are
  // synthesized one after another (each is faster than real time, so the next is ready before the previous has been
  // played); a newer line stops the older one's synthesis.
  say(serial, text, emotion) {
    if (this.settings.voice && this.state === 'off' && this.voice) this.start(this.voice);   // released after idle
    if (!this.settings.voice || this.state !== 'ready' || !text) return null;
    const parts = sentences(text, this.settings.voiceLang ?? this.voice?.lang);
    if (!parts.length) return null;
    // lines queue (2026-11-13, the author: Gemini sends one reply as two perform calls ~4 s apart and the second cut the
    // first off): the page plays them one after another, so every line is synthesized, in order, on one chain; only
    // the newest KEEP_LINES lines are kept (the page drops older queued ones too)
    this.latest = serial;
    const ref = this.pickRef(emotion), jobs = [];
    this.jobs.set(serial, jobs);
    for (const k of [...this.jobs.keys()].slice(0, -KEEP_LINES)) this.jobs.delete(k);
    for (const s of parts) {
      this.chain = (this.chain ?? Promise.resolve()).then(() => (this.jobs.has(serial) && this.state === 'ready'
        ? this.synth(s, emotion, ref).catch((e) => { this.log('tts', e.message); return null; }) : null));
      jobs.push(this.chain);
    }
    this.touch();
    return { serial, sentences: parts };
  }

  audio(serial, k) { return this.jobs.get(serial)?.[k] ?? null; }

  // ---- reference clips per emotion, adjustable in the settings (2026-11-12) ----
  async readOverride(name) {
    try { return JSON.parse(await fs.readFile(path.join(REF_DIR, `${name}.json`), 'utf-8')); } catch { return {}; }
  }
  // every clip a voice may use: its refs.list, its pool (the default voice: all JVNV F2 clips), the user's own; group =
  // the emotion folder (refs) or the pool's label
  async pool(v) {
    const out = [], seen = new Set();
    const add = async (file, groupFromField, own = false) => {
      if (!file) return;
      let txt; try { txt = await fs.readFile(file, 'utf-8'); } catch { return; }
      for (const line of txt.split('\n')) {
        const [w, grp, lang, text] = line.split('|');
        if (!w || !text) continue;
        const wav = path.resolve(path.dirname(file), w);
        if (seen.has(wav)) continue;
        seen.add(wav);
        out.push({ wav, text: text.trim(), lang: (lang ?? '').toLowerCase(), group: groupFromField ? grp : path.basename(path.dirname(wav)), own });
      }
    };
    await add(v.refs, false); await add(v.pool, true); await add(path.join(REF_DIR, `${v.name}.added.list`), true, true);
    return out;
  }
  async refTable(name) {
    const v = (await this.voices()).find((x) => x.name === name);
    if (!v) return null;
    const refs = await this.readRefs(v.refs), over = await this.readOverride(name);
    const current = Object.fromEntries(EMOTIONS.map((e) => [e, over[e] ?? (refs[e]?.[0] ?? refs.neutral?.[0] ?? null)]));
    // rotate: the voice's own clips per emotion, used in turn (one picked at random per line) while none is chosen
    const rotate = Object.fromEntries(EMOTIONS.map((e) => [e, refs[e]?.length ? refs[e] : (refs.neutral ?? [])]));
    return { voice: name, current, custom: Object.keys(over), rotate, pool: await this.pool(v) };
  }
  async setRef(name, emotion, wav) {             // wav null = back to the voice's own refs
    const v = (await this.voices()).find((x) => x.name === name);
    if (!v || !EMOTIONS.includes(emotion)) throw new Error('unknown voice or emotion');
    const over = await this.readOverride(name);
    if (!wav) delete over[emotion];
    else {
      const c = (await this.pool(v)).find((x) => x.wav === wav);
      if (!c) throw new Error('not one of this voice\'s clips');
      over[emotion] = { wav: c.wav, text: c.text, lang: c.lang || v.lang };
    }
    await fs.mkdir(REF_DIR, { recursive: true });
    await fs.writeFile(path.join(REF_DIR, `${name}.json`), JSON.stringify(over, null, 1));
    if (this.voice?.name === name) this.override = over;
    return this.refTable(name);
  }
  // a clip of the user's own (3-10 s, GPT-SoVITS's limit): copied next to the settings, transcribed by Whisper
  async addClip(name, file, group = 'neutral') {
    const v = (await this.voices()).find((x) => x.name === name);
    if (!v?.python) throw new Error('this voice cannot transcribe (no python)');
    const dir = path.join(REF_DIR, name); await fs.mkdir(dir, { recursive: true });
    const dst = path.join(dir, `${Date.now().toString(36)}_${path.basename(file).replace(/[^\w.-]/g, '_')}`);
    await fs.copyFile(file, dst);
    const r = JSON.parse((await run(v.python, [path.join(this.root, 'tts/voice_tool.py'), 'transcribe', dst])).trim().split('\n').pop());
    if (!(r.dur >= 3 && r.dur <= 10)) { await fs.rm(dst, { force: true }); throw new Error(`clip_length ${r.dur}`); }
    await fs.appendFile(path.join(REF_DIR, `${name}.added.list`), `${dst}|${group}|${r.lang.toUpperCase()}|${r.text}\n`);
    return { wav: dst, text: r.text, lang: r.lang, dur: r.dur };
  }
  async editClipText(name, wav, text) {          // the user corrects Whisper's transcript of an added clip
    const f = path.join(REF_DIR, `${name}.added.list`);
    const lines = (await fs.readFile(f, 'utf-8')).split('\n').map((l) => {
      const a = l.split('|'); return a[0] === wav ? [a[0], a[1], a[2], text.replace(/[|\n]/g, ' ')].join('|') : l; });
    await fs.writeFile(f, lines.join('\n'));
    const over = await this.readOverride(name);
    for (const e of Object.keys(over)) if (over[e].wav === wav) over[e].text = text;
    await fs.writeFile(path.join(REF_DIR, `${name}.json`), JSON.stringify(over, null, 1));
    if (this.voice?.name === name) this.override = over;
  }
  // a short sample with one clip (only the voice that is loaded; the settings page switches to it first)
  async preview(name, wav) {
    if (this.voice?.name !== name || this.state !== 'ready') throw new Error('voice_not_ready');
    const c = (await this.pool(this.voice)).find((x) => x.wav === wav);
    if (!c) throw new Error('not one of this voice\'s clips');
    const lang = this.settings.voiceLang ?? this.voice.lang;
    return this.synth(PREVIEW[lang] ?? PREVIEW.ja, 'neutral', { wav: c.wav, text: c.text, lang: c.lang || this.voice.lang });
  }
  // the user's own fine-tuned GPT-SoVITS voice: its trained weights + a folder of reference wavs. Runs on the bundled
  // GPT-SoVITS (the default voice's checkout and python); the version is read from the weights.
  async addVoice({ name, gpt, sovits, refsDir, lang }) {
    const voices = await this.voices();
    const base = voices.find((x) => x.name === 'default') ?? voices.find((x) => x.python && x.root);
    if (!base) throw new Error('no GPT-SoVITS to run it with');
    if (!/^[\w\- ]{1,40}$/.test(name ?? '') || voices.some((x) => x.name === name)) throw new Error('bad_name');
    const ver = JSON.parse((await run(base.python, [path.join(this.root, 'tts/voice_tool.py'), 'version', base.root, sovits])).trim().split('\n').pop());
    const dir = path.join(REF_DIR, name); await fs.mkdir(dir, { recursive: true });
    const list = path.join(dir, 'refs.list');
    const r = JSON.parse((await run(base.python, [path.join(this.root, 'tts/voice_tool.py'), 'folder', refsDir, list])).trim().split('\n').pop());
    if (!r.count) throw new Error('no_clips');
    const all = JSON.parse(await fs.readFile(VOICES, 'utf-8').catch(() => '{"voices": []}'));
    all.voices.push({ name, engine: 'gpt-sovits', root: base.root, python: base.python, version: ver.version, gpt, sovits,
                      refs: list, lang: lang || 'ja', romaji: /^v[34]|^v5/.test(ver.version) ? false : undefined });
    await fs.writeFile(VOICES, JSON.stringify(all, null, 2));
    return { name, version: ver.version, clips: r.count };
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
// Run a conda environment's python as `conda activate` would: its bin first on PATH, CONDA_PREFIX, and the
// environment's own activate.d scripts (an earlier voice project's adds the NVIDIA libraries to LD_LIBRARY_PATH; without them
// torchcodec, which loads the reference wav, failed: 'Could not load libtorchcodec'). exec keeps the pid.
function condaSpawn(python, args, opts = {}) {
  // Windows (2026-11-12, step 4): the voice's python is a portable one whose packages carry their own CUDA libraries
  // (torch wheels), so it is started directly, without a console window
  if (process.platform === 'win32') return spawn(python, args, { ...opts, env: { ...process.env, ...(opts.env ?? {}), PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8' }, detached: false, windowsHide: true });
  const prefix = path.dirname(path.dirname(python));
  const env = { ...process.env, PATH: `${path.join(prefix, 'bin')}:${process.env.PATH}`, CONDA_PREFIX: prefix };
  const sh = 'for f in "$CONDA_PREFIX"/etc/conda/activate.d/*.sh; do [ -f "$f" ] && . "$f"; done; exec "$@"';
  return spawn('bash', ['-c', sh, 'bash', python, ...args], { ...opts, env });
}
function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = condaSpawn(cmd, args); let out = '', err = '';
    p.stdout.on('data', (d) => { out += d; }); p.stderr.on('data', (d) => { err += d; });
    p.on('error', reject);
    p.on('close', (c) => (c === 0 ? resolve(out) : reject(new Error(`${path.basename(args[0])} exited ${c}: ${err.slice(-200)}`))));
  });
}

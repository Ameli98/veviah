// First-voice-on download of the Windows TTS environment (2026-11-13, plan step 4; decisions in CLAUDE.md
// 'Rest of step 4' / 'Download installer details'). Everything comes from official sources:
//   python.org (embedded Python), bootstrap.pypa.io (pip), download.pytorch.org (torch cu128), PyPI (packages,
//   pinned by tts/constraints_win.txt), GitHub (GPT-SoVITS code at a fixed commit), Hugging Face lj1995/GPT-SoVITS
//   (v5 turbo weights), XXXXRT/GPT-SoVITS-Pretrained (G2PW model), fbaipublicfiles (fasttext language id), nltk data.
// Steps are resumable: each finished step is recorded in <dir>/install.json; file downloads continue from the
// partial file (HTTP Range); each download is tried RETRIES times, then the step stops with its error and a retry
// starts again from that step. The installer never touches anything outside <dir> (and the voices.json it updates).
import fs from 'node:fs/promises';
import fss from 'node:fs';
import path from 'node:path';
import { spawn } from 'node:child_process';

const PY_URL = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip';
const GETPIP = 'https://bootstrap.pypa.io/get-pip.py';
const TORCH = ['torch==2.11.0+cu128', 'torchaudio==2.11.0+cu128'];
const TORCH_INDEX = 'https://download.pytorch.org/whl/cu128';
const GSV_COMMIT = '6d51ffceafbf240145619af4c92408a6e54b7924';
const GSV_ZIP = `https://codeload.github.com/RVC-Boss/GPT-SoVITS/zip/${GSV_COMMIT}`;
const HF = 'https://huggingface.co/lj1995/GPT-SoVITS/resolve/main/';
const WEIGHTS = [            // the v5 turbo inference files only (~2.4 GB with G2PW)
  's1v3.ckpt', 'gsv-v5-pretrained/s2Gv5turbo.pth', 'gsv-v5-pretrained/vocoder.pth',
  'chinese-roberta-wwm-ext-large/config.json', 'chinese-roberta-wwm-ext-large/pytorch_model.bin', 'chinese-roberta-wwm-ext-large/tokenizer.json',
  'chinese-hubert-base/config.json', 'chinese-hubert-base/preprocessor_config.json', 'chinese-hubert-base/pytorch_model.bin',
];
// base files a trained voice of another GPT-SoVITS version needs on top of the v5 install (fetched when such a voice
// is first started; Q4 2026-11-12). Same official repo; paths under GPT_SoVITS/pretrained_models.
export const VERSION_FILES = {
  v2Pro: ['sv/pretrained_eres2netv2w24s4ep4.ckpt'], v2ProPlus: ['sv/pretrained_eres2netv2w24s4ep4.ckpt'],
  v3: ['models--nvidia--bigvgan_v2_24khz_100band_256x/bigvgan_generator.pt', 'models--nvidia--bigvgan_v2_24khz_100band_256x/config.json'],
  v4: ['gsv-v4-pretrained/vocoder.pth'],
};
const G2PW = 'https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip';
const LID = 'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin';
const NLTK = ['averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng', 'cmudict'];
const MIN_DRIVER = 570;      // CUDA 12.8 wheels
const RETRIES = 3;
export const NEED_GB = 10;   // free space asked for before starting (download ~4 GB, ~8-9 GB unpacked)

const STEPS = ['gpu', 'python', 'pip', 'torch', 'packages', 'standins', 'code', 'weights', 'g2pw', 'lid', 'nltk', 'voice'];

export class VoiceInstaller {
  // dir = the TTS folder; root = the app's backend (tts/requirements_win.txt, tts/default_voice);
  // voices = the voices.json to register the default voice in
  constructor({ dir, root, voices, log = () => {} }) {
    Object.assign(this, { dir, root, voices, log });
    this.state = { status: 'idle', step: null, done: [], error: null, bytes: 0, total: 0, detail: '' };
    this.cancelled = false; this.child = null;
  }

  async load() {
    try { this.state.done = JSON.parse(await fs.readFile(path.join(this.dir, 'install.json'), 'utf-8')).done ?? []; } catch {}
    if (STEPS.every((s) => this.state.done.includes(s))) this.state.status = 'installed';
    return this.info();
  }

  info() { return { ...this.state, steps: STEPS, dir: this.dir }; }

  async mark(step) {
    if (!this.state.done.includes(step)) this.state.done.push(step);
    await fs.writeFile(path.join(this.dir, 'install.json'), JSON.stringify({ done: this.state.done, commit: GSV_COMMIT }, null, 1));
  }

  // runs every step not done yet; resolves with info() (status 'installed', 'error' or 'cancelled')
  async run() {
    if (this.state.status === 'running') return this.info();
    this.cancelled = false;
    Object.assign(this.state, { status: 'running', error: null });
    await fs.mkdir(this.dir, { recursive: true });
    await this.load(); this.state.status = 'running';
    try {
      for (const step of STEPS) {
        if (this.state.done.includes(step)) continue;
        if (this.cancelled) throw new Error('cancelled');
        Object.assign(this.state, { step, bytes: 0, total: 0, detail: '' });
        this.log(`voice install: ${step}`);
        await this[`step_${step}`]();
        await this.mark(step);
      }
      Object.assign(this.state, { status: 'installed', step: null });
      this.log('voice install: done');
    } catch (e) {
      const msg = String(e.message ?? e);
      Object.assign(this.state, { status: this.cancelled ? 'cancelled' : 'error', error: this.cancelled ? null : msg });
      this.log(`voice install: ${this.state.status} at ${this.state.step}: ${msg}`);
    }
    return this.info();
  }

  cancel() { this.cancelled = true; this.child?.kill(); this.abort?.abort(); }

  // removes the whole TTS folder (the caller turns the voice off first)
  async remove() {
    this.cancel();
    // only a folder this installer made (install.json in it) is ever deleted
    try { await fs.access(path.join(this.dir, 'install.json')); } catch { throw new Error('not an installed TTS folder'); }
    await fs.rm(this.dir, { recursive: true, force: true });
    Object.assign(this.state, { status: 'idle', done: [], step: null, error: null });
  }

  // ---- steps ----
  // the NVIDIA card and its driver: {ok, name, driver, reason: 'nogpu' | 'driver'}; checked at every app start
  // (Q15: a card added later turns voice on again; nothing is stored)
  async gpu() {
    const out = await this.exec('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader']).catch(() => null);
    if (!out) return { ok: false, reason: 'nogpu' };
    const [name, driver] = out.trim().split('\n')[0].split(',').map((s) => s.trim());
    if (Number(driver.split('.')[0]) < MIN_DRIVER) return { ok: false, reason: 'driver', name, driver };
    return { ok: true, name, driver };
  }

  async step_gpu() {
    const g = await this.gpu();
    if (g.reason === 'nogpu') throw Object.assign(new Error('nogpu'), { code: 'nogpu' });
    if (!g.ok) throw new Error(`driver ${g.driver} < ${MIN_DRIVER}`);
    this.state.detail = `${g.name}, driver ${g.driver}`;
  }

  async step_python() {
    const zip = path.join(this.dir, 'python.zip'), py = path.join(this.dir, 'python');
    await this.download(PY_URL, zip);
    await fs.rm(py, { recursive: true, force: true }); await fs.mkdir(py, { recursive: true });
    await this.exec('tar', ['-xf', zip, '-C', py]);
    // the embedded python ignores PYTHONPATH; site-packages must be listed and 'import site' on
    await fs.writeFile(path.join(py, 'python312._pth'), 'python312.zip\r\n.\r\nLib\\site-packages\r\nimport site\r\n');
    await fs.rm(zip);
  }

  async step_pip() {
    const gp = path.join(this.dir, 'get-pip.py');
    await this.download(GETPIP, gp);
    await this.py([gp, '--no-warn-script-location']);
    await fs.rm(gp);
  }

  async step_torch() { await this.pip(['install', ...TORCH, '--index-url', TORCH_INDEX]); }

  async step_packages() {
    const t = path.join(this.root, 'tts');
    await this.pip(['install', '-r', path.join(t, 'requirements_win.txt'), '-c', path.join(t, 'constraints_win.txt')]);
  }

  // jieba_fast (a C build of jieba GPT-SoVITS imports) has no Windows wheel for 3.12: plain jieba stands in
  async step_standins() {
    const d = path.join(this.dir, 'python', 'Lib', 'site-packages', 'jieba_fast');
    await fs.mkdir(d, { recursive: true });
    await fs.writeFile(path.join(d, '__init__.py'), '# stand-in (VEviaH): jieba_fast has no Windows wheel; jieba has the same API\nfrom jieba import *  # noqa\nfrom jieba import setLogLevel, lcut, cut  # noqa\n');
    await fs.writeFile(path.join(d, 'posseg.py'), 'from jieba.posseg import *\n');
  }

  async step_code() {
    const zip = path.join(this.dir, 'gsv.zip'), tmp = path.join(this.dir, 'gsv_tmp'), dst = path.join(this.dir, 'GPT-SoVITS');
    await this.download(GSV_ZIP, zip);
    await fs.rm(tmp, { recursive: true, force: true }); await fs.mkdir(tmp, { recursive: true });
    await this.exec('tar', ['-xf', zip, '-C', tmp]);
    const [top] = await fs.readdir(tmp);
    await fs.rm(dst, { recursive: true, force: true });
    await fs.rename(path.join(tmp, top), dst);
    await fs.rm(tmp, { recursive: true, force: true }); await fs.rm(zip);
  }

  async step_weights() {
    const pm = path.join(this.dir, 'GPT-SoVITS', 'GPT_SoVITS', 'pretrained_models');
    for (const w of WEIGHTS) await this.download(HF + w, path.join(pm, ...w.split('/')));
  }

  async step_g2pw() {
    const text = path.join(this.dir, 'GPT-SoVITS', 'GPT_SoVITS', 'text'), zip = path.join(this.dir, 'G2PWModel.zip');
    await this.download(G2PW, zip);
    await this.exec('tar', ['-xf', zip, '-C', text]);
    await fs.rm(zip);
  }

  async step_lid() {
    await this.download(LID, path.join(this.dir, 'GPT-SoVITS', 'GPT_SoVITS', 'pretrained_models', 'fast_langdetect', 'lid.176.bin'));
  }

  async step_nltk() {
    const d = path.join(this.dir, 'nltk_data');
    await this.py(['-c', `import nltk,sys\nfor p in ${JSON.stringify(NLTK)}:\n    assert nltk.download(p, download_dir=r"${d}", quiet=True), p`]);
  }

  // the default voice (JVNV F2; refs + pool are bundled with the app in <root>/tts/default_voice) -> voices.json
  async step_voice() {
    const src = path.join(this.root, 'tts', 'default_voice');
    let all = { voices: [] };
    try { all = JSON.parse(await fs.readFile(this.voices, 'utf-8')); } catch {}
    all.voices = (all.voices ?? []).filter((v) => v.name !== 'default');
    all.voices.unshift({
      name: 'default', engine: 'gpt-sovits', root: path.join(this.dir, 'GPT-SoVITS'), python: path.join(this.dir, 'python', 'python.exe'),
      version: 'v5turbo', gpt: 'GPT_SoVITS/pretrained_models/s1v3.ckpt', sovits: 'GPT_SoVITS/pretrained_models/gsv-v5-pretrained/s2Gv5turbo.pth',
      refs: path.join(src, 'refs.list'), pool: path.join(src, 'pool.list'), lang: 'ja', romaji: false, nltk: path.join(this.dir, 'nltk_data'),
    });
    await fs.mkdir(path.dirname(this.voices), { recursive: true });
    await fs.writeFile(this.voices, JSON.stringify(all, null, 2));
  }

  // the extra base files of a voice's version, into that voice's GPT-SoVITS root (a no-op when present)
  async ensureVersion(version, gsvRoot) {
    const files = VERSION_FILES[version] ?? [];
    const pm = path.join(gsvRoot, 'GPT_SoVITS', 'pretrained_models');
    for (const f of files) {
      const dst = path.join(pm, ...f.split('/'));
      if (fss.existsSync(dst)) continue;
      this.log(`voice: fetching ${f} for ${version}`);
      await this.download(HF + f, dst);
    }
  }

  // ---- helpers ----
  py(args) { return this.exec(path.join(this.dir, 'python', 'python.exe'), args, { PYTHONUTF8: '1', PYTHONIOENCODING: 'utf-8', PYTHONNOUSERSITE: '1' }); }
  pip(args) { return this.py(['-m', 'pip', ...args, '--no-warn-script-location', '--disable-pip-version-check', '--progress-bar', 'off']); }

  exec(cmd, args, env = {}) {
    return new Promise((resolve, reject) => {
      const p = this.child = spawn(cmd, args, { env: { ...process.env, ...env }, windowsHide: true });
      let out = '', tail = '';
      const take = (d) => { const s = String(d); out += s; tail = (tail + s).slice(-2000); const last = s.trim().split('\n').pop(); if (last) this.state.detail = last.slice(0, 200); };
      p.stdout.on('data', take); p.stderr.on('data', take);
      p.on('error', reject);
      p.on('close', (code) => { this.child = null; code === 0 ? resolve(out) : reject(new Error(`${path.basename(cmd)} exited ${code}: ${tail.trim().split('\n').slice(-3).join(' | ')}`)); });
    });
  }

  // HTTP download that continues a partial <dst>.part (Range) and is tried RETRIES times
  async download(url, dst) {
    await fs.mkdir(path.dirname(dst), { recursive: true });
    if (fss.existsSync(dst)) return;
    const part = dst + '.part';
    let last;
    for (let k = 0; k < RETRIES; k++) {
      if (this.cancelled) throw new Error('cancelled');
      try { await this.fetchTo(url, part); await fs.rename(part, dst); return; }
      catch (e) { last = e; this.log(`download ${path.basename(dst)} try ${k + 1}: ${e.message}`); await new Promise((r) => setTimeout(r, 2000 * (k + 1))); }
    }
    throw new Error(`download ${url}: ${last?.message}`);
  }

  async fetchTo(url, part) {
    let have = 0;
    try { have = (await fs.stat(part)).size; } catch {}
    this.abort = new AbortController();
    const res = await fetch(url, { headers: have ? { Range: `bytes=${have}-` } : {}, signal: this.abort.signal, redirect: 'follow' });
    if (res.status === 416) return;                                   // already complete
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    if (have && res.status !== 206) have = 0;                         // server ignored Range: start over
    const len = Number(res.headers.get('content-length') || 0);
    this.state.total = len ? have + len : 0; this.state.bytes = have;
    const fh = await fs.open(part, have ? 'a' : 'w');
    try {
      for await (const chunk of res.body) { await fh.write(chunk); this.state.bytes += chunk.length; }
    } finally { await fh.close(); }
    if (len && this.state.bytes !== have + len) throw new Error(`short read ${this.state.bytes}/${have + len}`);
  }
}

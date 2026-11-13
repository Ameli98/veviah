// Lip sync from the audio itself (2026-11-05, plan item 4): no phonemes or timestamps, so it works for any TTS and
// any language. An AnalyserNode on the playing voice gives the level and the spectrum each frame:
//   level  -> how far the mouth opens (RMS in dB between FLOOR_DB and FULL_DB)
//   F1 band (300-900 Hz) share  -> open vowels (aa, oh)
//   F2 high band (1800-3200 Hz) vs low band (900-1800 Hz) -> front vowels (ee, ih) vs back (ou, oh)
// The five VRM mouth shapes (aa ih ou ee oh) are mixed from these two numbers; their sum is MOUTH_MAX x the level.
//   const ls = new LipSync(audioContext); src.connect(ls.input); ls.input.connect(ctx.destination);
//   each frame: face.setMouth(ls.update(dt))
const FLOOR_DB = -50, FULL_DB = -18;
// The loudest syllable opens the mouth only this far (sum of the five shapes; 2026-11-06, the author: it opened too wide;
// in her reference clips (Zenless Zone Zero cutscenes, ~/hsin_runs/ref_lipsync/) lips mostly just part and the widest
// is about our aa 0.25-0.35; at 1 the default model's mouth was ~3x that)
export const MOUTH_MAX = 0.35;
const ATTACK = 0.04, RELEASE = 0.09;          // s: the mouth opens fast and closes a little slower
const BANDS = { f1: [300, 900], low: [900, 1800], high: [1800, 3200] };

export class LipSync {
  constructor(ctx) {
    this.ctx = ctx;
    this.input = ctx.createAnalyser();
    this.input.fftSize = 1024;
    this.input.smoothingTimeConstant = 0.3;
    this.freq = new Float32Array(this.input.frequencyBinCount);
    this.time = new Float32Array(this.input.fftSize);
    this.cur = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    this.active = false;
  }

  band(lo, hi) {
    const hz = this.ctx.sampleRate / this.input.fftSize;
    let s = 0;
    for (let i = Math.floor(lo / hz); i <= Math.ceil(hi / hz) && i < this.freq.length; i++) s += 10 ** (this.freq[i] / 10);
    return s;
  }

  // the mouth weights for this frame (all zero when nothing plays)
  update(dt) {
    let target = { aa: 0, ih: 0, ou: 0, ee: 0, oh: 0 };
    if (this.active) {
      this.input.getFloatTimeDomainData(this.time);
      let e = 0;
      for (const x of this.time) e += x * x;
      const db = 10 * Math.log10(e / this.time.length + 1e-12);
      const level = Math.min(1, Math.max(0, (db - FLOOR_DB) / (FULL_DB - FLOOR_DB)));
      if (level > 0.02) {
        this.input.getFloatFrequencyData(this.freq);
        const f1 = this.band(...BANDS.f1), lo = this.band(...BANDS.low), hi = this.band(...BANDS.high);
        const open = f1 / (f1 + lo + hi + 1e-12);                 // ~0.4 closed vowels .. ~0.8 open ones
        const front = hi / (lo + hi + 1e-12);                    // 0 back .. 1 front
        const o = Math.min(1, Math.max(0, (open - 0.45) / 0.35)), f = Math.min(1, Math.max(0, (front - 0.15) / 0.5));
        const w = { aa: o * (1 - Math.abs(f - 0.5)), oh: o * (1 - f) * 0.8, ee: (1 - o) * f, ih: (1 - o) * f * 0.5 + o * f * 0.3,
                    ou: (1 - o) * (1 - f) };
        const sum = Object.values(w).reduce((a, b) => a + b, 0) || 1;
        for (const k in w) target[k] = MOUTH_MAX * level * w[k] / sum;
      }
    }
    for (const k in this.cur) {
      const tau = target[k] > this.cur[k] ? ATTACK : RELEASE;
      this.cur[k] += (target[k] - this.cur[k]) * Math.min(1, dt / tau);
    }
    return this.cur;
  }
}

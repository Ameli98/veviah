"""Small helpers for the voice settings (2026-11-12), run with a voice's own python (its GPT-SoVITS environment):

  voice_tool.py transcribe WAV [lang]      -> {"text", "lang", "dur"}  (faster-whisper; lang guessed when not given)
  voice_tool.py folder DIR OUT_LIST        -> transcribes every wav under DIR into a refs.list
                                              ("<wav>|<speaker>|<LANG>|<text>"; the emotion = the wav's folder name
                                              when it is one of ours, else the wav is copied under neutral/)
  voice_tool.py version GPT_SOVITS_ROOT SOVITS.pth -> {"version"}  (GPT-SoVITS's own detection of a trained model)

Whisper model: large-v3-turbo from the Hugging Face cache (downloaded on first use)."""
import json, os, shutil, sys

EMOTIONS = ("neutral", "happy", "angry", "sad", "shy", "surprised")
_asr = None


def asr():
    global _asr
    if _asr is None:
        if os.name == "nt":      # CTranslate2 needs cuBLAS / cuDNN: use the DLLs that come with torch
            import importlib.util
            spec = importlib.util.find_spec("torch")
            if spec and spec.origin:
                lib = os.path.join(os.path.dirname(spec.origin), "lib")
                if os.path.isdir(lib):
                    os.add_dll_directory(lib); os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
        from faster_whisper import WhisperModel
        try:
            _asr = WhisperModel("large-v3-turbo", device="cuda", compute_type="float16")
        except Exception:
            _asr = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
    return _asr


def duration(wav):
    import soundfile as sf
    return sf.info(wav).duration


def load16k(wav):
    """16 kHz mono float32 without PyAV (faster-whisper's own decoder breaks on newer av releases)."""
    import numpy as np, soundfile as sf
    x, sr = sf.read(wav, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import resampy
        x = resampy.resample(x, sr, 16000)
    return np.ascontiguousarray(x, dtype=np.float32)


def transcribe(wav, lang=None):
    segs, info = asr().transcribe(load16k(wav), language=lang or None, beam_size=5)
    text = "".join(s.text for s in segs).strip()
    return {"text": text, "lang": (lang or info.language or "ja"), "dur": round(duration(wav), 2)}


def folder(src, out_list):
    lines, base = [], os.path.dirname(os.path.abspath(out_list))
    for root, _, files in os.walk(src):
        for f in sorted(files):
            if not f.lower().endswith(".wav"):
                continue
            wav = os.path.join(root, f)
            emo = os.path.basename(root) if os.path.basename(root) in EMOTIONS else "neutral"
            if os.path.basename(root) != emo:        # speech.mjs reads the emotion from the folder name
                os.makedirs(os.path.join(base, "refs", emo), exist_ok=True)
                dst = os.path.join(base, "refs", emo, f); shutil.copy(wav, dst); wav = dst
            r = transcribe(wav)
            if not 3 <= r["dur"] <= 10:
                print(json.dumps({"skip": wav, "dur": r["dur"]}), file=sys.stderr); continue
            lines.append(f"{wav}|user|{r['lang'].upper()}|{r['text']}")
    open(out_list, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    return {"refs": out_list, "count": len(lines)}


def version(root, sovits):
    sys.path[:0] = [root, os.path.join(root, "GPT_SoVITS")]
    os.chdir(root)
    from process_ckpt import get_sovits_version_from_path_fast
    _, v, _ = get_sovits_version_from_path_fast(sovits)
    return {"version": v}


if __name__ == "__main__":
    cmd, *a = sys.argv[1:]
    out = {"transcribe": lambda: transcribe(*a), "folder": lambda: folder(*a), "version": lambda: version(*a)}[cmd]()
    print(json.dumps(out, ensure_ascii=False))

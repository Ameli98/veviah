"""The default voice bundled with the Windows app (2026-11-13), run from WSL with a python that has soundfile:

  build_tts.py OUT_DIR        e.g. <backend>/tts/default_voice

Writes OUT_DIR/ = refs/<emotion>/*.wav + refs.list (5 clips per emotion), pool/<emotion>/*.flac + pool.list (every
JVNV F2 clip of 3-10 s, 16-bit FLAC: 26 % of the 24-bit wavs), CREDITS.txt. Clip paths in the lists are RELATIVE to
the list (speech.mjs resolves them), so the folder can move with the app. The voice is registered in voices.json by
mcp_server/voice_install.mjs once the TTS environment is installed. Source: ~/hsin_runs/tts_bench/default_voice."""
import os, shutil, sys
import soundfile as sf

SRC = os.path.expanduser(os.environ.get("VEVIAH_VOICE_SRC", "~/hsin_runs/tts_bench/default_voice"))


def main(out):
    os.makedirs(out, exist_ok=True)
    shutil.copy(os.path.join(SRC, "CREDITS.txt"), out)
    for name, sub, flac in (("refs.list", "refs", False), ("pool.list", "pool", True)):
        lines = []
        for ln in open(os.path.join(SRC, name), encoding="utf-8").read().splitlines():
            if not ln.strip():
                continue
            wav, *rest = ln.split("|")
            emo = os.path.basename(os.path.dirname(wav)) if sub == "refs" else rest[0]
            base = os.path.splitext(os.path.basename(wav))[0] + (".flac" if flac else ".wav")
            d = os.path.join(out, sub, emo); os.makedirs(d, exist_ok=True)
            p = os.path.join(d, base)
            if not os.path.exists(p):
                if flac:
                    x, sr = sf.read(wav); sf.write(p, x, sr, subtype="PCM_16", format="FLAC")
                else:
                    shutil.copy(wav, p)
            lines.append("|".join([f"{sub}/{emo}/{base}", *rest]))
        open(os.path.join(out, name), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(name, len(lines))


if __name__ == "__main__":
    main(sys.argv[1])

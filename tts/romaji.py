"""Romaji for Japanese reference transcripts (2026-11-05, the author's trick from CosyVoice: when the voice speaks another
language than its reference clips, give GPT-SoVITS the reference transcript as romaji with prompt_lang "en").

  python tts/romaji.py <refs.list>   ->  JSON {wav path: romaji} on stdout (Japanese lines only)

Uses pyopenjtalk (in GPT-SoVITS's own environment), one word at a time so the romaji has spaces:
"勿論です。医療行為ですから。" -> "mochiron desu. iryoo kooi desu kara."."""
import json
import sys

import pyopenjtalk


def _word(ph):
    s = ""
    for i, p in enumerate(ph):
        if p == "pau":
            s += ", "
        elif p == "cl":                       # small tsu: the next consonant doubles
            s += ph[i + 1][0] if i + 1 < len(ph) and ph[i + 1] not in "aiueo" else ""
        elif p == "N":
            s += "n"
        else:
            s += p.lower()                    # devoiced vowels come as capitals
    return s.strip(", ")


def romaji(text):
    out = []
    for chunk in text.replace("、", "。").replace("？", "。").replace("！", "。").split("。"):
        if not chunk.strip():
            continue
        words = [_word(pyopenjtalk.g2p(n["string"]).split()) for n in pyopenjtalk.run_frontend(chunk) if n["pos"] != "記号"]
        out.append(" ".join(w for w in words if w))
    return ". ".join(out) + "."


if __name__ == "__main__":
    res = {}
    for line in open(sys.argv[1], encoding="utf-8"):
        parts = line.rstrip("\n").split("|")
        if len(parts) == 4 and parts[2].upper() == "JA":
            res[parts[0]] = romaji(parts[3])
    print(json.dumps(res, ensure_ascii=False))

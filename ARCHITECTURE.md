# How VEviaH works

## The idea

The AI never sends raw animation. It calls one tool, `perform(say, emotion, keyframes)`, where the keyframes are
short words and angles ("right hand: face out mid", "touch: cheek", "legs: weight left"). A solver turns those words
into a motion **for the body of the character on screen**: it measures each VRM model once (bone lengths, body and
clothes shape), so a hand that should rest on a knee lands on that model's knee, and arms do not pass through the
body. A rule filter rejects what it still cannot do; the AI gets the reason back and retries once.

```
AI app (Claude Desktop, …) ──MCP stdio──▶ mcp_server/server.mjs ──▶ motion_loop (Python solver + filter)
VEviaH chat window ──▶ mcp_server/chat.mjs ──┘                              │ motion JSON
                                                                            ▼
            desktop/ (Electron window) ◀── mcp_server/companion.mjs (HTTP + events, voice, idle actions)
               └─ vrm_player/app/companion.js + three.js + @pixiv/three-vrm (drawing, physics, lip sync)
```

## Folders

| Folder | What is in it |
|---|---|
| `motion_loop/motion_loop/` | The solver. `keyframes.py` (the keyframe format and the main solve), `arm_words.py` (hand position words, both-arm poses), `contact.py` (touch words: hand on a body part of this model), `depenetrate.py` (pushes arms out of the body along the way), `skeleton.py` (per-model skeleton and body shapes), `filters.py` (the rule filter), `perform.py` (what the MCP server calls), `skeletons/` + `body_fits/` (measured models) |
| `mcp_server/` | `server.mjs` (the MCP tool), `companion.mjs` (local server the window talks to), `chat.mjs` (the built-in chat: Anthropic and OpenAI-compatible APIs), `speech.mjs` (text-to-speech through a GPT-SoVITS `api_v2` server), `voice_install.mjs` (downloads the voice engine), `sets/default.json` (idle actions and touch reactions) |
| `vrm_player/` | The page that draws the character. `src/` shared parts (player, physics: skirts, hair colliders, jiggle, faces, lip sync, measuring a model), `app/` the desktop page, `tools/` probes used during development |
| `desktop/` | The Electron app (`main.js`), settings / chat pages, `i18n.js` (all UI text in English, Chinese, Japanese), `clients.js` (connecting AI apps), build and installer scripts |
| `tts/` | Helpers run inside the voice engine's Python (`gsv_api.py` starts GPT-SoVITS's own `api_v2.py`) |

Research code from earlier stages (a reward model, a keyframe policy, the data pipeline) is still in `motion_loop`;
the app does not use it.

## Building from source

Development happens in WSL 2 (Ubuntu) on Windows; the result is a native Windows app.

1. Install: Node.js 24, Python 3.12, `nsis` (`sudo apt install nsis`).
2. Python: `python3 -m venv .venv && .venv/bin/pip install numpy pillow` (add `torch` to run every test).
3. Node packages:
   ```bash
   (cd vrm_player && npm install)
   (cd mcp_server && npm install && npx playwright install chromium)
   (cd desktop && npm_config_platform=win32 npm install)     # Electron for Windows
   ```
4. The default voice clips are not in the repository. Download a release zip and copy its
   `backend/tts/default_voice` folder to `~/.cache/veviah_build/release/VEviaH/backend/tts/` before building
   (or set `VEVIAH_VOICE_SRC`). Without them the app works, but the default voice is missing.
5. Build: `desktop/build_release.sh` → `VEviaH-<version>-win64.zip` and `VEviaH-<version>-setup.exe` in
   `~/hsin_runs/release/` (set `RELEASE_OUT` to change).

Tests: `cd motion_loop && ../.venv/bin/python -m unittest discover -s tests`; page and server tests are the
`mcp_server/test/*.mjs` scripts (each says how to run it at the top).

## Adding a model by hand

Any `.vrm` works: the app measures it on first load. The bundled models' measurements are in
`motion_loop/motion_loop/skeletons/` and `body_fits/`, keyed by file name and checked by the file's hash.

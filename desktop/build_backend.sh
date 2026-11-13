#!/bin/bash
# Windows-native backend (2026-11-12, plan step 4): everything the app needs next to it, no WSL at run time.
#   build_backend.sh OUT_DIR      e.g. "/mnt/c/Users/<you>/AppData/Local/veviah_build/backend"
# OUT_DIR/
#   python/            embedded Windows Python 3.12 + numpy / Pillow (the solver needs nothing else)
#   motion_loop/       the solver package (motion_loop/motion_loop)
#   mcp_server/        companion.mjs, server.mjs, speech.mjs, model.mjs, sets/ + production node_modules (pure JS)
#   vrm_player/        dist-companion/, dist-player/, public/models/ (the two bundled models only: the default sample + the author's own test2, 2026-11-13)
#   tts/               gsv_api.py, romaji.py, voice_tool.py, requirements / constraints for the TTS download, default_voice/
# Node parts run on Electron's own Node (ELECTRON_RUN_AS_NODE). The TTS environment is not part of this.
set -euo pipefail
OUT="$1"; R="$(cd "$(dirname "$0")/.." && pwd)"
PYV=3.12.10; NUMPY=2.5.3; PILLOW=12.3.0
CACHE="$HOME/.cache/veviah_build"; mkdir -p "$CACHE/wheels" "$OUT"
# python
if [ ! -f "$OUT/python/python.exe" ]; then
  [ -f "$CACHE/py-$PYV.zip" ] || curl -4 -sL -o "$CACHE/py-$PYV.zip" "https://www.python.org/ftp/python/$PYV/python-$PYV-embed-amd64.zip"
  mkdir -p "$OUT/python" && (cd "$OUT/python" && unzip -q -o "$CACHE/py-$PYV.zip")
  "$R/.venv/bin/pip" download -q --only-binary=:all: --platform win_amd64 --python-version 3.12 --implementation cp \
    -d "$CACHE/wheels" "numpy==$NUMPY" "pillow==$PILLOW"
  mkdir -p "$OUT/python/Lib/site-packages"
  for w in "$CACHE"/wheels/numpy-$NUMPY-*win_amd64.whl "$CACHE"/wheels/pillow-$PILLOW-*win_amd64.whl; do (cd "$OUT/python/Lib/site-packages" && unzip -q -o "$w"); done
  # the embedded python ignores PYTHONPATH: its ._pth names the solver folder
  printf 'python312.zip\r\n.\r\nLib\\site-packages\r\n..\\motion_loop\r\nimport site\r\n' > "$OUT/python/python312._pth"
fi
# solver
mkdir -p "$OUT/motion_loop"; rsync -a --delete --exclude __pycache__ "$R/motion_loop/motion_loop" "$OUT/motion_loop/"
# node servers (production dependencies only)
mkdir -p "$OUT/mcp_server"
rsync -a --delete --exclude node_modules --exclude test --exclude '*.sh' "$R/mcp_server/" "$OUT/mcp_server/"
if [ ! -d "$OUT/mcp_server/node_modules/@modelcontextprotocol" ] || [ "$R/mcp_server/package.json" -nt "$OUT/mcp_server/node_modules" ]; then
  ST="$CACHE/npm_prod"; rm -rf "$ST"; mkdir -p "$ST"; cp "$R/mcp_server/package.json" "$R/mcp_server/package-lock.json" "$ST/" 2>/dev/null || cp "$R/mcp_server/package.json" "$ST/"
  (cd "$ST" && npm ci --omit=dev --ignore-scripts -s 2>/dev/null || npm install --omit=dev --ignore-scripts -s)
  rsync -a --delete --exclude .bin "$ST/node_modules/" "$OUT/mcp_server/node_modules/"
fi
# no Linux links in a Windows folder (npm's node_modules/.bin: Windows cannot overwrite them; 2026-11-13)
rm -rf "$OUT/mcp_server/node_modules/.bin"
find "$OUT" -type l -delete
# player pages + bundled models
mkdir -p "$OUT/vrm_player/public/models"
rsync -a --delete "$R/vrm_player/dist-companion" "$R/vrm_player/dist-player" "$OUT/vrm_player/"
rm -rf "$OUT/vrm_player/dist-app"   # the in-chat view (MCP Apps) was removed 2026-11-13
cp "$R/vrm_player/public/models/VRM1_Constraint_Twist_Sample.vrm" "$R/vrm_player/public/models/vivi.vrm" "$OUT/vrm_player/public/models/"
# tts helpers
mkdir -p "$OUT/tts"; cp "$R/tts/gsv_api.py" "$R/tts/romaji.py" "$R/tts/voice_tool.py" "$R/tts/requirements_win.txt" "$R/tts/constraints_win.txt" "$OUT/tts/"
# the default voice (JVNV F2, CC BY-SA 4.0): refs + pool clips, lists with relative paths (desktop/build_tts.py)
# Without the source clips (they are not in the repository): take backend/tts/default_voice from a release zip.
VSRC="${VEVIAH_VOICE_SRC:-$HOME/hsin_runs/tts_bench/default_voice}"
if [ -f "$OUT/tts/default_voice/pool.list" ]; then :
elif [ -f "$VSRC/pool.list" ]; then "${VEVIAH_TTS_PY:-$HOME/miniforge3/envs/tts_gsv5/bin/python}" "$R/desktop/build_tts.py" "$OUT/tts/default_voice"
else echo "default voice clips not found ($VSRC): copy backend/tts/default_voice from a release zip into $OUT/tts/ (see ARCHITECTURE.md)" >&2; fi
du -sh "$OUT" | tail -1

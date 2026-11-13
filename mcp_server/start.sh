#!/bin/bash
# Started by Claude Desktop (Windows) through `wsl.exe -e <this file> [model.vrm]`: no login shell, so nvm's node is
# given by path. The optional argument is the user's VRM (a WSL path, or a Windows path like C:\Users\...\me.vrm).
if [ -n "$1" ]; then
  m="$1"; [[ "$m" =~ ^[A-Za-z]:\\ ]] && m="$(wslpath -u "$m")"
  export HSIN_VRM_MODEL="$m"
fi
cd "$(dirname "$0")" && exec "$HOME/.nvm/versions/node/v24.21.0/bin/node" server.mjs

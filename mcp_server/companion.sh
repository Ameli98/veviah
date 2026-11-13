#!/bin/bash
# The desktop companion window's server: companion.sh [model.vrm] (a WSL path, or a Windows path like C:\Users\...\me.vrm;
# use the same model as the MCP server in claude_desktop_config.json). Then open http://localhost:8790/ in a browser.
if [ -n "$1" ]; then
  m="$1"; [[ "$m" =~ ^[A-Za-z]:\\ ]] && m="$(wslpath -u "$m")"
  export HSIN_VRM_MODEL="$m"
fi
cd "$(dirname "$0")" && exec "$HOME/.nvm/versions/node/v24.21.0/bin/node" companion.mjs

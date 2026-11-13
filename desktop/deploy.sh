#!/bin/bash
# Copies the desktop app to Windows: %LOCALAPPDATA%\hsin-companion (Electron for Windows from node_modules/electron/dist,
# our files in resources/app), writes config.json (where the repo is in WSL). --shortcuts also makes the desktop and
# Start-menu shortcuts. Run from WSL: desktop/deploy.sh [--shortcuts]. The app must not be running.
# --native (2026-11-12, plan step 4): the Windows-native build, renamed VEviaH: %LOCALAPPDATA%\Programs\VEviaH\VEviaH.exe
# with backend\ next to it (build_backend.sh: embedded Python, solver, servers, pages); no WSL at run time.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
LAD="$(wslpath -u "$(cmd.exe /c 'echo %LOCALAPPDATA%' 2>/dev/null | tr -d '\r')")"
NATIVE=0; SHORT=0; for a in "$@"; do [ "$a" = "--native" ] && NATIVE=1; [ "$a" = "--shortcuts" ] && SHORT=1; done
if [ $NATIVE = 1 ]; then DEST="$LAD/Programs/VEviaH"; EXE=VEviaH.exe; NAME=veviah; else DEST="$LAD/hsin-companion"; EXE=HSIN.exe; NAME=hsin-companion; fi
"$HERE/stage_app.sh" "$DEST" "$EXE" "$NAME"
APP="$DEST/resources/app"
[ -f "$HERE/selftest_motion.txt" ] && cp "$HERE/selftest_motion.txt" "$APP/"
if [ $NATIVE = 1 ]; then
  printf '{"backend":"native","backendDir":"backend","port":8790,"distro":"%s","repo":"%s"}\n' "$WSL_DISTRO_NAME" "$REPO" > "$APP/config.json"
  "$HERE/build_backend.sh" "$DEST/backend"
else
  printf '{"distro":"%s","repo":"%s","port":8790}\n' "$WSL_DISTRO_NAME" "$REPO" > "$APP/config.json"
fi
echo "deployed to $DEST"
if [ $SHORT = 1 ]; then
  LNK=$([ $NATIVE = 1 ] && echo VEviaH || echo HSIN)
  W_EXE="$(wslpath -w "$DEST/$EXE")"; W_ICO="$(wslpath -w "$APP/icon.ico")"
  powershell.exe -NoProfile -Command "
    \$s = New-Object -ComObject WScript.Shell
    foreach (\$p in @([Environment]::GetFolderPath('Desktop'), (Join-Path ([Environment]::GetFolderPath('Programs')) ''))) {
      \$l = \$s.CreateShortcut((Join-Path \$p '$LNK.lnk')); \$l.TargetPath = '$W_EXE'; \$l.IconLocation = '$W_ICO'
      \$l.WorkingDirectory = (Split-Path '$W_EXE'); \$l.Description = '$LNK desktop companion'; \$l.Save(); Write-Output (Join-Path \$p '$LNK.lnk')
    }" | tr -d '\r'
fi

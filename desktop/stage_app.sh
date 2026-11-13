#!/bin/bash
# The app's own files (Electron for Windows + resources/app), shared by deploy.sh and build_release.sh.
#   stage_app.sh DEST EXE NAME     e.g. stage_app.sh "$OUT/VEviaH" VEviaH.exe veviah
# DEST/EXE = the renamed electron.exe; DEST/resources/app = our files + koffi's JS and Windows x64 binary.
# config.json is written by the caller.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEST="$1"; EXE="$2"; NAME="$3"
VERSION="$(node -p "require('$HERE/package.json').version")"
DIST="$HERE/node_modules/electron/dist"
[ -f "$DIST/electron.exe" ] || { echo "no Windows Electron in $DIST (npm_config_platform=win32 node node_modules/electron/install.js)"; exit 1; }
if [ ! -f "$DEST/$EXE" ] || [ "$DIST/electron.exe" -nt "$DEST/$EXE" ]; then
  mkdir -p "$DEST"
  cp -r "$DIST/." "$DEST/"
  mv -f "$DEST/electron.exe" "$DEST/$EXE"
  rm -f "$DEST/resources/default_app.asar"
fi
# the exe's own icon and details (2026-11-13: the renamed electron.exe kept Electron's icon and "Electron" in its
# properties / Task Manager): written with electron/rcedit v2.0.0 (MIT, Windows tool, kept in the build cache).
# Done when the exe was copied, the icon or the version changed (stamp per destination in the cache, not in DEST).
RCEDIT="$HOME/.cache/veviah_build/tools/rcedit-x64.exe"
if [ ! -f "$RCEDIT" ]; then
  mkdir -p "$(dirname "$RCEDIT")"
  gh release download v2.0.0 -R electron/rcedit -p rcedit-x64.exe -D "$(dirname "$RCEDIT")" || { echo "could not download rcedit"; exit 1; }
fi
STAMPS="$HOME/.cache/veviah_build/exe_stamps"; mkdir -p "$STAMPS"
ST="$STAMPS/$(printf '%s' "$DEST/$EXE" | md5sum | cut -c1-16)"
WANT="$(md5sum "$HERE/icon.ico" | cut -c1-16) $VERSION $(stat -c %Y "$DEST/$EXE")"
if [ "$(cat "$ST" 2>/dev/null)" != "$WANT" ]; then
  "$RCEDIT" "$(wslpath -w "$DEST/$EXE")" --set-icon "$(wslpath -w "$HERE/icon.ico")" \
    --set-version-string FileDescription VEviaH --set-version-string ProductName VEviaH \
    --set-version-string CompanyName VEviaH --set-version-string OriginalFilename "$EXE" \
    --set-version-string InternalName VEviaH --set-version-string LegalCopyright "MIT License" \
    --set-file-version "$VERSION" --set-product-version "$VERSION" || { echo "rcedit failed (is $EXE running?)"; exit 1; }
  echo "$(md5sum "$HERE/icon.ico" | cut -c1-16) $VERSION $(stat -c %Y "$DEST/$EXE")" > "$ST"
fi
APP="$DEST/resources/app"
mkdir -p "$APP"
cp "$HERE"/{main.js,windows.js,clients.js,chat_main.js,i18n.js,preload.js,preload_avatar.js,preload_chat.js,settings.html,chat.html,credits.html,icon.png,icon.ico,tray_16.png,tray_24.png,tray_32.png} "$APP/"
# koffi (window snapping: other windows' rectangles through user32): its JS and only the Windows x64 binary
# (copied only when changed: a running app holds koffi.node open and Windows refuses to replace it)
K="$APP/node_modules/koffi"; mkdir -p "$K/build/koffi/win32_x64"
cp "$HERE"/node_modules/koffi/{index.js,indirect.js,package.json,LICENSE.txt} "$K/"
for f in "$HERE"/node_modules/koffi/build/koffi/win32_x64/*; do
  cmp -s "$f" "$K/build/koffi/win32_x64/$(basename "$f")" || cp "$f" "$K/build/koffi/win32_x64/"
done
printf '{"name":"%s","version":"%s","main":"main.js"}\n' "$NAME" "$VERSION" > "$APP/package.json"

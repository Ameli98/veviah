#!/bin/bash
# The release (2026-11-13): one staged folder -> a zip and a setup exe, both from the same files.
#   desktop/build_release.sh            -> ~/hsin_runs/release/VEviaH-<version>-win64.zip and VEviaH-<version>-setup.exe
# The staged folder is the native app as deploy.sh --native installs it, minus everything personal: config.json has no
# WSL distro / repo path, the solver keeps only the bundled models' measurements, no self-test files. Needs makensis
# (apt install nsis); the zip is made with Python's zipfile.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VERSION="$(node -p "require('$HERE/package.json').version")"
OUT="${RELEASE_OUT:-$HOME/hsin_runs/release}"
STAGE="$HOME/.cache/veviah_build/release/VEviaH"
mkdir -p "$OUT" "$STAGE"

# One build per finished state (2026-11-13, the author: three rebuilds in 20 minutes because code was still changing).
# A release is built only from committed code, and only once per commit: uncommitted changes stop the build (finish,
# test and commit first), and a release already built from this commit is not built again (--force does).
FORCE=0; for a in "$@"; do [ "$a" = "--force" ] && FORCE=1; done
REPO="$(dirname "$HERE")"
if [ -n "$(git -C "$REPO" status --porcelain --untracked-files=no)" ]; then
  git -C "$REPO" status --short --untracked-files=no
  echo "uncommitted changes (above): finish, test and commit them first, then build once"; exit 1
fi
# files the package copies that git does not know (forgotten `git add`): the build would not match its commit
SHIPPED=(desktop mcp_server motion_loop/motion_loop vrm_player/src vrm_player/app vrm_player/index.html vrm_player/package.json tts)
NEW="$(git -C "$REPO" ls-files --others --exclude-standard -- "${SHIPPED[@]}")"
if [ -n "$NEW" ]; then echo "$NEW"; echo "files not in git (above): git add and commit them (or delete them), then build"; exit 1; fi
# a version already released on GitHub is never built again: raise "version" in desktop/package.json first
GH_REPO="${VEVIAH_GH_REPO:-Ameli98/veviah}"
if command -v gh >/dev/null && gh release view "v$VERSION" -R "$GH_REPO" >/dev/null 2>&1; then
  echo "v$VERSION is already released on GitHub ($GH_REPO): raise the version in desktop/package.json, commit, then build"; exit 1
fi
COMMIT="$(git -C "$REPO" rev-parse --short HEAD)"
STAMP="$OUT/VEviaH-$VERSION.commit"
if [ $FORCE = 0 ] && [ -f "$STAMP" ] && [ "$(cat "$STAMP")" = "$COMMIT" ] && [ -f "$OUT/VEviaH-$VERSION-setup.exe" ] && [ -f "$OUT/VEviaH-$VERSION-win64.zip" ]; then
  echo "release $VERSION is already built from $COMMIT: nothing to do (--force builds it again)"; exit 0
fi

# tests first (the solver's unit tests, ~1 min): a failure stops the build
(cd "$REPO/motion_loop" && ../.venv/bin/python -m unittest discover -s tests -q) || { echo "unit tests failed: not packaging"; exit 1; }
# the pages are built from source every time (a stale vrm_player/dist-* would ship old code under a new commit)
(cd "$REPO/vrm_player" && npm run -s build:companion >/dev/null && npm run -s build:player >/dev/null) || { echo "page build failed"; exit 1; }

"$HERE/stage_app.sh" "$STAGE" VEviaH.exe veviah
printf '{"backend":"native","backendDir":"backend","port":8790,"defaultModel":"VRM Sample.vrm"}\n' > "$STAGE/resources/app/config.json"
rm -f "$STAGE/resources/app/selftest_motion.txt"
"$HERE/build_backend.sh" "$STAGE/backend"

# the bundled models under their shown names (2026-11-13, the author: "VRM Sample" = pixiv's sample, "Vivi" = her own
# test2); the code knows no model by name, so a file name is only a name. Their measurements go along; the default
# sample's are also kept under its original name, the solver's fallback for a model not measured yet.
SK="$STAGE/backend/motion_loop/motion_loop"; MD="$STAGE/backend/vrm_player/public/models"
rename_model() {   # old new
  mv -f "$MD/$1.vrm" "$MD/$2.vrm"
  cp -f "$SK/skeletons/$1.json" "$SK/skeletons/$2.json"; cp -f "$SK/skeletons/$1.sha256" "$SK/skeletons/$2.sha256"
  cp -f "$SK/body_fits/$1.json" "$SK/body_fits/$2.json"
}
rename_model VRM1_Constraint_Twist_Sample "VRM Sample"
rename_model vivi Vivi
KEEP="VRM1_Constraint_Twist_Sample|VRM Sample|Vivi"
for d in skeletons body_fits; do
  for f in "$SK/$d"/*; do
    b="$(basename "$f")"; n="${b%.*}"
    [[ "$n" =~ ^($KEEP)$ ]] || rm -f "$f"
  done
done
# nothing that names this machine may ship
WINUSER="$(cmd.exe /c 'echo %USERNAME%' 2>/dev/null | tr -d '\r' || true)"
if grep -rIl --exclude-dir=node_modules -e "/home/$USER" -e "${WINUSER:-/home/$USER}" "$STAGE/resources/app" "$STAGE/backend/mcp_server" \
     "$STAGE/backend/motion_loop" "$STAGE/backend/tts" 2>/dev/null | head -5 | grep .; then
  echo "personal paths found in the staged files (above): not packaging"; exit 1
fi

ZIP="$OUT/VEviaH-$VERSION-win64.zip"
rm -f "$ZIP"
# the zip and the installer are made from the same staged folder at the same time (each takes 1.5-3 min alone)
python3 -c "import shutil, sys; shutil.make_archive(sys.argv[1][:-4], 'zip', sys.argv[2], 'VEviaH')" "$ZIP" "$(dirname "$STAGE")" & ZPID=$!
makensis -V2 -DVERSION="$VERSION" -DSRC="$STAGE" -DOUTFILE="$OUT/VEviaH-$VERSION-setup.exe" "$HERE/installer.nsi" & NPID=$!
wait $ZPID || { echo "zip failed"; exit 1; }
wait $NPID || { echo "installer failed"; exit 1; }
echo "$COMMIT" > "$STAMP"
ls -la "$ZIP" "$OUT/VEviaH-$VERSION-setup.exe"; echo "built from $COMMIT"

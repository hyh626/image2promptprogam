#!/usr/bin/env bash
# Capture three reference screenshots of view_eval_results.py against the
# demo fixture built by build_demo_fixture.py. Saves PNGs into docs/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FIXTURE="${FIXTURE:-/tmp/demo-fixture}"
OUT_DIR="${OUT_DIR:-$REPO_ROOT/docs/screenshots}"
PORT="${PORT:-8765}"

CHROME="${CHROME:-}"
if [ -z "$CHROME" ]; then
  for c in chromium chromium-browser google-chrome chrome; do
    if command -v "$c" >/dev/null 2>&1; then
      CHROME="$c"
      break
    fi
  done
fi
if [ -z "$CHROME" ]; then
  echo "no chromium / chrome binary found; install one or set CHROME=path" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Build/refresh fixture
python3 "$REPO_ROOT/scripts/build_demo_fixture.py" --out "$FIXTURE"

# Start viewer in the background
python3 "$REPO_ROOT/view_eval_results.py" --root "$FIXTURE" --port "$PORT" \
  > /tmp/viewer-screenshot.log 2>&1 &
VIEWER_PID=$!
trap 'kill $VIEWER_PID 2>/dev/null || true' EXIT
sleep 1.0

# Helper: capture one URL with a fixed viewport.
shoot() {
  local hash="$1" outfile="$2" width="${3:-1280}" height="${4:-1100}"
  local url="http://127.0.0.1:${PORT}/#${hash}"
  echo "capturing ${outfile}"
  "$CHROME" --headless --no-sandbox --disable-gpu --hide-scrollbars \
    --window-size="${width},${height}" \
    --virtual-time-budget=2000 \
    --screenshot="${OUT_DIR}/${outfile}" "$url" >/dev/null 2>&1
}

# 1) Summary table for the experiments folder
shoot "experiments%2Fruns" "01-summary.png" 1400 720

# 2) Run detail for the leader (palette step)
shoot "experiments%2Fruns%2F20260504T112800Z__claude-opus-4-7__add_palette_step" \
  "02-run-detail.png" 1400 1180

# 3) Top-level browser
shoot "" "03-browser.png" 1400 600

echo "saved screenshots to ${OUT_DIR}"

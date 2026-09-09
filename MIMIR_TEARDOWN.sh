#!/bin/bash
# MIMIR_TEARDOWN.sh — wipes everything this session installed. Idempotent.
# Pre-existing items NOT removed: brew ffmpeg, ~/.cache/uv (shared cache).
set -u
ROOT="/Users/prajwalkiran/code/personal/mimir"
echo "🧹 Tearing down Mimir local install..."

# 1. Stop any running app processes (uvicorn backend :8000, vite frontend :5173)
pkill -f "uvicorn main:app" 2>/dev/null && echo "  stopped backend" || true
pkill -f "vite" 2>/dev/null && echo "  stopped frontend" || true

# 2. Project-local install footprint
for p in \
  "$ROOT/backend/venv" \
  "$ROOT/backend/.mimir_cache" \
  "$ROOT/backend/models" \
  "$ROOT/backend/uploads" \
  "$ROOT/backend/temp" \
  "$ROOT/backend/logs" \
  "$ROOT/backend/mimir.db" \
  "$ROOT/backend/mimir.db-shm" \
  "$ROOT/backend/mimir.db-wal" \
  "$ROOT/backend/.mimir_run.env" \
  "$ROOT/backend/.env" \
  "$ROOT/frontend/node_modules" \
  "$ROOT/frontend/dist" ; do
  if [ -e "$p" ]; then rm -rf "$p" && echo "  removed $p"; fi
done

# 3. Verify nothing leaked into HOME caches (these should NOT exist if redirection worked)
for d in ~/.cache/huggingface ~/.cache/torch ~/nltk_data ; do
  [ -e "$d" ] && echo "  ⚠️  unexpected HOME cache present: $d (review before deleting)" || true
done

# 4. Optional: remove the uv-managed CPython 3.12 added this session (shared; uncomment to remove)
# uv python uninstall 3.12

echo "✅ Teardown complete. Pre-existing brew ffmpeg and ~/.cache/uv were left intact."
echo "   Run 'git status' to confirm the working tree is clean."

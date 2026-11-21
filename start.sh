#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"

if [ ! -f ".env" ] && [ -f ".env.example" ]; then
  echo "No .env found — copying .env.example to .env (add OPENROUTER_API_KEY & FAL_KEY for live renders, or use Mock mode)."
  cp .env.example .env
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: 'uv' is required. Install it via: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Syncing LaunchCut Studio environment..."
uv sync

echo ""
echo "Starting LaunchCut Studio on http://${HOST}:${PORT}"
echo "Open http://${HOST}:${PORT} in your browser to use the 6-step HITL Studio."
echo ""

# Guard against bracketed IPv6 NO_PROXY entries that break httpx 0.28
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"

exec uv run gtm-videogen serve --host "${HOST}" --port "${PORT}"

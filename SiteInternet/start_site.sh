#!/usr/bin/env bash
set -euo pipefail
source /workspace/venv/bin/activate

# Load env for the web process (key point!)
if [[ -f /workspace/SiteInternet/env.sh ]]; then
  source /workspace/SiteInternet/env.sh
fi

: "${MISTRAL_API_KEY:?ERROR: MISTRAL_API_KEY missing in /workspace/SiteInternet/env.sh}"

export WEB_HOST="${WEB_HOST:-0.0.0.0}"
export WEB_PORT="${WEB_PORT:-7862}"
export OUT="${OUT:-/workspace/IA/tmp/out}"

exec python /workspace/SiteInternet/server/app.py 2>&1 | tee -a /workspace/SiteInternet/logs/site.log

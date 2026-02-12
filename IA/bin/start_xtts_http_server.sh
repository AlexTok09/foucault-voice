#!/usr/bin/env bash
set -euo pipefail
source /workspace/venv/bin/activate

mkdir -p /workspace/IA/logs

# IMPORTANT: tu dois avoir exporté OUT/ORIG/FTCKPT/SPEAKERS_PTH/SPEAKER_WAV/LANG avant
python /workspace/IA/bin/xtts_http_server.py 2>&1 | tee -a /workspace/IA/logs/xtts_http_server.log

#!/usr/bin/env bash
set -euo pipefail

QUESTION="${1:-}"
if [[ -z "$QUESTION" ]]; then
  echo "Usage: $0 \"ta question\""
  exit 1
fi

# ---- Paths ----
RUN_BASE="/workspace/runs/run_20260210_110029_ft_v2_like_ref_mixed_25ep"
ORIG="$RUN_BASE/run/training/XTTS_v2.0_original_model_files"
FTDIR="$RUN_BASE/run/training/GPT_XTTS_FT-February-10-2026_11+02AM-0000000"
FTCKPT="$FTDIR/best_model_19875.pth"
SPEAKERS_PTH="$ORIG/speakers_xtts.pth"
SPEAKER_WAV="/workspace/test_speaker.wav"
LANG="fr"

CHUNKS_JSON="/workspace/tmp_work/chunks.json"
INFER_SCRIPT="/workspace/tmp_work/xtts_infer_chunked_concat_stream.py"

# ---- Output folder (IA isolated) ----
TS="$(date +%Y%m%d_%H%M%S)"
OUT="/workspace/IA/tmp/out_$TS"
mkdir -p "$OUT"
mkdir -p /workspace/IA/logs

# ---- 1) Text generation (PLACEHOLDER) ----
TEXT="Je vais répondre à ta question de manière claire. ${QUESTION} Il faut regarder comment les dispositifs organisent ce que l’on croit spontané, et comment une parole se forme dans ce cadre."

# ---- 2) Write JSON chunks safely ----
export TEXT CHUNKS_JSON
python - <<'PY'
import os, json
text = os.environ["TEXT"]
chunks_json = os.environ["CHUNKS_JSON"]
obj = {"chunks": [text]}
with open(chunks_json, "w", encoding="utf-8") as f:
    json.dump(obj, f, ensure_ascii=False, indent=2)
print("WROTE", chunks_json)
PY

# ---- 3) Call XTTS inference ----
source /workspace/venv/bin/activate

OUT="$OUT" \
ORIG="$ORIG" \
FTCKPT="$FTCKPT" \
SPEAKERS_PTH="$SPEAKERS_PTH" \
SPEAKER_WAV="$SPEAKER_WAV" \
LANG="$LANG" \
python "$INFER_SCRIPT" --chunks_json "$CHUNKS_JSON" \
  2>&1 | tee "/workspace/IA/logs/ask_${TS}.log"

echo
echo "DONE. Audio output folder:"
echo "  $OUT"
ls -lah "$OUT" | tail -n 20

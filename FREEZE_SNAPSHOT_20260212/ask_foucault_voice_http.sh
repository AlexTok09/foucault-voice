#!/usr/bin/env bash
set -euo pipefail

ORIGINAL_QUESTION="${1:-}"
if [[ -z "$ORIGINAL_QUESTION" ]]; then
  echo "Usage: $0 \"ta question\""
  exit 1
fi

source /workspace/venv/bin/activate

: "${MISTRAL_API_KEY:?ERROR: export MISTRAL_API_KEY=...}"
export SYSTEM_PROMPT_PATH="${SYSTEM_PROMPT_PATH:-/workspace/IA/prompts/system_foucault.txt}"
export STYLE_PACK_PATH="${STYLE_PACK_PATH:-}"
export MIN_CHARS="${MIN_CHARS:-140}"
export MAX_CHARS="${MAX_CHARS:-230}"

mkdir -p /workspace/tmp_work
TS="$(date +%Y%m%d_%H%M%S)"
CHUNKS_JSON="/workspace/tmp_work/chunks_${TS}.json"
export CHUNKS_JSON

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7010}"
BASE_URL="http://${HOST}:${PORT}"

OUT_NAME="${OUT_NAME:-foucault_${TS}.wav}"

STRICT_INSTR="Tu DOIS répondre UNIQUEMENT par un JSON strict et valide, sans markdown ni backticks.
Format exact: {\"chunks\":[\"...\",\"...\",\"...\",\"...\"]}.
4 chunks exactement.
Chaque chunk doit faire entre ${MIN_CHARS} et ${MAX_CHARS} caractères (espaces inclus).
Si un chunk dépasse ${MAX_CHARS}, raccourcis-le.
Si un chunk est trop court, ajoute une phrase simple dans le même chunk.
Ne produis rien d'autre que ce JSON."

ok=0
for attempt in 1 2 3; do
  # 1) Tentative normale
  if python /workspace/IA/bin/mistral_generate.py "${ORIGINAL_QUESTION}" \
    | CHUNKS_JSON="$CHUNKS_JSON" /workspace/IA/bin/mistral_json_to_chunks_json.py >/dev/null 2>/workspace/tmp_work/chunk_err.txt
  then
    ok=1
    break
  fi

  # 2) Retry: on force une réécriture en donnant le JSON invalide
  BAD="/workspace/IA/tmp/bad_chunks.json"
  BAD_JSON=""
  if [[ -f "$BAD" ]]; then
    BAD_JSON="$(cat "$BAD")"
  fi

  if python /workspace/IA/bin/mistral_generate.py \
    "${ORIGINAL_QUESTION}

${STRICT_INSTR}

Ton précédent JSON était invalide. Réécris-le en respectant STRICTEMENT les contraintes ci-dessus.
Voici le JSON invalide à corriger (réécris-le, ne le commente pas) :
${BAD_JSON}" \
    | CHUNKS_JSON="$CHUNKS_JSON" /workspace/IA/bin/mistral_json_to_chunks_json.py >/dev/null 2>/workspace/tmp_work/chunk_err.txt
  then
    ok=1
    break
  fi

  sleep 0.2
done

if [[ "$ok" -ne 1 ]]; then
  echo "ERROR: failed to produce valid chunks after retries" >&2
  cat /workspace/tmp_work/chunk_err.txt >&2
  exit 2
fi

RESP="$(curl -sS "${BASE_URL}/speak" \
  -H 'Content-Type: application/json' \
  -d "{\"chunks_json\":\"${CHUNKS_JSON}\",\"out_name\":\"${OUT_NAME}\"}")"

echo "$RESP"
printf '%s' "$RESP" | python -c 'import sys,json; obj=json.load(sys.stdin); print(obj.get("out_path","")) if obj.get("ok") else exit(2)'

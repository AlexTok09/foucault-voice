#!/usr/bin/env python3
import os, sys, json, re

OUT = os.environ.get("CHUNKS_JSON", "/workspace/tmp_work/chunks.json")
MIN_CHARS = int(os.environ.get("MIN_CHARS", "100"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "220"))

raw = sys.stdin.read()
if not raw.strip():
    print("ERROR: empty model output", file=sys.stderr)
    sys.exit(1)

# Strip ANSI
raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)

# Remove ``` fences if present
raw = re.sub(r"^\s*```[a-zA-Z]*\s*", "", raw)
raw = re.sub(r"\s*```\s*$", "", raw)

# Extract {...}
m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
if not m:
    print("ERROR: no JSON-like object found", file=sys.stderr)
    print(raw, file=sys.stderr)
    sys.exit(2)

blob = m.group(0).strip()

def try_parse_strict(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

obj = try_parse_strict(blob)

# Fallback: parse chunks as lines inside [ ... ] even if not quoted
if obj is None:
    # Find chunks array content
    m2 = re.search(r'"chunks"\s*:\s*\[(.*)\]\s*', blob, flags=re.DOTALL)
    if not m2:
        print("ERROR: could not parse strict JSON and no chunks[] found", file=sys.stderr)
        print("----- BLOB START -----", file=sys.stderr)
        print(blob, file=sys.stderr)
        print("----- BLOB END -----", file=sys.stderr)
        sys.exit(3)

    inside = m2.group(1)

    # Split into non-empty lines, strip commas/quotes
    lines = []
    for line in inside.splitlines():
        t = line.strip()
        if not t:
            continue
        t = t.rstrip(",")
        t = t.strip()
        # remove surrounding quotes if present
        if len(t) >= 2 and ((t[0] == '"' and t[-1] == '"') or (t[0] == "'" and t[-1] == "'")):
            t = t[1:-1].strip()
        # ignore stray brackets
        if t in ("[", "]"):
            continue
        lines.append(t)

    obj = {"chunks": lines}

chunks = obj.get("chunks")
if not isinstance(chunks, list) or not all(isinstance(x, str) for x in chunks) or len(chunks) == 0:
    print("ERROR: JSON must contain {'chunks': [str, ...]}", file=sys.stderr)
    print(json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
    sys.exit(4)

# Enforce length constraints
bad = []
for i, c in enumerate(chunks):
    n = len(c.strip())
    if n < MIN_CHARS or n > MAX_CHARS:
        bad.append((i, n))

if bad:
    print(f"ERROR: chunk length violation (min={MIN_CHARS}, max={MAX_CHARS}): {bad}", file=sys.stderr)
    os.makedirs("/workspace/IA/tmp", exist_ok=True)
    with open("/workspace/IA/tmp/bad_chunks.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print("WROTE /workspace/IA/tmp/bad_chunks.json for inspection", file=sys.stderr)
    sys.exit(5)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"chunks": [c.strip() for c in chunks]}, f, ensure_ascii=False, indent=2)

print(f"WROTE {OUT} (chunks={len(chunks)})")

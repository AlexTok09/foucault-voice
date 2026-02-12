#!/usr/bin/env python3
import os, sys, json, urllib.request

API_KEY = os.environ.get("MISTRAL_API_KEY")
if not API_KEY:
    print("ERROR: MISTRAL_API_KEY not set", file=sys.stderr)
    sys.exit(1)

MODEL = os.environ.get("MISTRAL_MODEL", "mistral-large-latest")
SYSTEM_PROMPT_PATH = os.environ.get("SYSTEM_PROMPT_PATH")

if not SYSTEM_PROMPT_PATH:
    print("ERROR: SYSTEM_PROMPT_PATH not set", file=sys.stderr)
    sys.exit(1)

question = " ".join(sys.argv[1:]).strip()
if not question:
    print("ERROR: no question provided", file=sys.stderr)
    sys.exit(1)

with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
    system_prompt = f.read().strip()

STYLE_PACK_PATH = os.environ.get("STYLE_PACK_PATH")

style_pack = ""
if STYLE_PACK_PATH and os.path.exists(STYLE_PACK_PATH):
    with open(STYLE_PACK_PATH, "r", encoding="utf-8") as f:
        style_pack = f.read().strip()

if style_pack:
    system_prompt = (
        system_prompt
        + "\n\nRéférences internes de style (ne pas citer explicitement) :\n"
        + style_pack
    )

payload = {
    "model": MODEL,
    "messages": [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ],
    "temperature": 0.2,
    "top_p": 0.9,
}

req = urllib.request.Request(
    "https://api.mistral.ai/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    },
    method="POST",
)

with urllib.request.urlopen(req, timeout=60) as resp:
    data = json.loads(resp.read().decode("utf-8"))

print(data["choices"][0]["message"]["content"].strip())

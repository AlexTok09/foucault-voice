#!/usr/bin/env python3
import os, sys, json, time, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import torch
import torchaudio

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ---- Required env ----
OUT          = os.environ.get("OUT", os.getcwd())
ORIG         = os.environ.get("ORIG", os.getcwd())
FTCKPT       = os.environ["FTCKPT"]
SPEAKERS_PTH = os.environ["SPEAKERS_PTH"]
LANG         = os.environ.get("LANG", "fr").strip()
HOST         = os.environ.get("HOST", "127.0.0.1")
PORT         = int(os.environ.get("PORT", "7010"))

SPEAKER_WAVS_ENV = os.environ.get("SPEAKER_WAVS", "").strip()
SPEAKER_WAV      = os.environ.get("SPEAKER_WAV", "").strip()

CONFIG_PATH    = os.path.join(ORIG, "config.json")
TOKENIZER_PATH = os.path.join(ORIG, "vocab.json")

# ---- Audio params ----
SR = int(os.environ.get("SR", "24000"))

# ---- Stitching knobs ----
XFADE_MS        = int(os.environ.get("XFADE_MS", "180"))
EDGE_FADE_MS    = int(os.environ.get("EDGE_FADE_MS", "30"))
PAUSE_MS        = int(os.environ.get("PAUSE_MS", "280"))
TAIL_SILENCE_MS = int(os.environ.get("TAIL_SILENCE_MS", "700"))
FADE_OUT_MS     = int(os.environ.get("FADE_OUT_MS", "80"))
MICRO_PAUSE_MS  = int(os.environ.get("MICRO_PAUSE_MS", "0"))  # optional

def must_exist(p):
    if not p or not os.path.exists(p):
        raise FileNotFoundError(p)

def add_silence(ms: int) -> torch.Tensor:
    n = int(SR * ms / 1000)
    n = max(0, n)
    return torch.zeros((1, n), dtype=torch.float32)

def add_tail_silence(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav
    return torch.cat([wav, add_silence(ms)], dim=1)

def edge_fade(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav
    n = wav.shape[1]
    fade = int(SR * ms / 1000)
    if fade <= 1 or fade*2 >= n:
        return wav
    win = torch.linspace(0, 1, fade, dtype=wav.dtype).unsqueeze(0)
    wav[:, :fade] = wav[:, :fade] * win
    wav[:, -fade:] = wav[:, -fade:] * torch.flip(win, dims=[1])
    return wav

def crossfade(a: torch.Tensor, b: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return torch.cat([a, b], dim=1)
    nfade = int(SR * ms / 1000)
    nfade = max(0, nfade)
    if nfade == 0:
        return torch.cat([a, b], dim=1)
    if a.shape[1] < nfade or b.shape[1] < nfade:
        return torch.cat([a, b], dim=1)

    a_tail = a[:, -nfade:]
    b_head = b[:, :nfade]
    w = torch.linspace(0, 1, nfade, dtype=a.dtype).unsqueeze(0)
    mix = a_tail*(1-w) + b_head*w
    return torch.cat([a[:, :-nfade], mix, b[:, nfade:]], dim=1)

def load_chunks_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "chunks" in obj:
        chunks = obj["chunks"]
    else:
        chunks = obj
    if not isinstance(chunks, list):
        raise ValueError("chunks_json must contain a list under key 'chunks'")
    return [str(x) for x in chunks]

# ---- Model load once ----
def load_model():
    must_exist(CONFIG_PATH)
    must_exist(TOKENIZER_PATH)
    must_exist(FTCKPT)
    must_exist(SPEAKERS_PTH)

    cfg = XttsConfig()
    cfg.load_json(CONFIG_PATH)

    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_path=FTCKPT, vocab_path=TOKENIZER_PATH, speaker_file_path=SPEAKERS_PTH)
    model.cuda()
    model.eval()
    return model

print("LOADING MODEL ONCE...", flush=True)
MODEL = load_model()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Speakers conditioning
SPEAKER_WAVS = []
if SPEAKER_WAVS_ENV:
    SPEAKER_WAVS = [p.strip() for p in SPEAKER_WAVS_ENV.split(",") if p.strip()]
elif SPEAKER_WAV:
    SPEAKER_WAVS = [SPEAKER_WAV]

if not SPEAKER_WAVS:
    print("ERROR: set SPEAKER_WAV or SPEAKER_WAVS", file=sys.stderr)
    sys.exit(2)

for p in SPEAKER_WAVS:
    must_exist(p)

print(f"USING SPEAKER_WAVS: {SPEAKER_WAVS}", flush=True)

# Precompute conditioning once
with torch.no_grad():
    gpt_cond_latent, speaker_embedding = MODEL.get_conditioning_latents(
        audio_path=SPEAKER_WAVS,
        gpt_cond_len=30,
        max_ref_length=60
    )

os.makedirs(OUT, exist_ok=True)
LOCK = threading.Lock()

def synth_one(text: str) -> torch.Tensor:
    # Returns float32 tensor shape [1, T] @ SR
    with torch.no_grad():
        out = MODEL.inference(
            text=text,
            language=LANG,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
            temperature=0.7,
            top_p=0.9,
        )
    wav = out["wav"]
    if isinstance(wav, torch.Tensor):
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        wav = wav.to(torch.float32).cpu()
    else:
        wav = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    return wav

def synth_chunks(chunks):
    pieces = []
    for idx, txt in enumerate(chunks, start=1):
        wav = synth_one(txt)
        wav = edge_fade(wav, EDGE_FADE_MS)
        pieces.append(("wav", wav))
        if MICRO_PAUSE_MS > 0:
            pieces.append(("pause", add_silence(MICRO_PAUSE_MS)))
        if idx < len(chunks) and PAUSE_MS > 0:
            pieces.append(("pause", add_silence(PAUSE_MS)))

    # Merge with crossfade between wav segments (skip pauses)
    final = None
    prev_wav = None
    for kind, item in pieces:
        if kind == "pause":
            if final is None:
                final = item
            else:
                final = torch.cat([final, item], dim=1)
            prev_wav = None
            continue

        wav = item
        if final is None:
            final = wav
        else:
            if prev_wav is None:
                final = torch.cat([final, wav], dim=1)
            else:
                # Crossfade at the boundary
                # final currently ends with prev_wav; apply crossfade with new wav
                final = crossfade(final, wav, XFADE_MS)
        prev_wav = wav

    if final is None:
        final = add_silence(10)
    final = add_tail_silence(final, TAIL_SILENCE_MS)

    if FADE_OUT_MS > 0:
        n = final.shape[1]
        fade = int(SR * FADE_OUT_MS / 1000)
        if fade > 1 and fade < n:
            w = torch.linspace(1, 0, fade, dtype=final.dtype).unsqueeze(0)
            final[:, -fade:] = final[:, -fade:] * w

    return final

def write_wav(path, wav: torch.Tensor):
    torchaudio.save(path, wav, SR, encoding="PCM_S", bits_per_sample=16)

class H(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/health":
            return self._json(200, {"ok": True, "device": DEVICE, "sr": SR, "out": OUT})
        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        if p != "/speak":
            return self._json(404, {"ok": False, "error": "not found"})

        try:
            n = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(n).decode("utf-8", errors="replace")
            req = json.loads(raw) if raw.strip() else {}
            out_name = req.get("out_name") or f"xtts_{int(time.time())}.wav"
            if "/" in out_name or ".." in out_name:
                return self._json(400, {"ok": False, "error": "invalid out_name"})

            chunks = None
            if "chunks_json" in req:
                chunks = load_chunks_from_json(req["chunks_json"])
            elif "chunks" in req:
                if not isinstance(req["chunks"], list):
                    return self._json(400, {"ok": False, "error": "chunks must be a list"})
                chunks = [str(x) for x in req["chunks"]]
            else:
                return self._json(400, {"ok": False, "error": "need chunks_json or chunks[]"})

            if not chunks:
                return self._json(400, {"ok": False, "error": "empty chunks"})

            out_path = os.path.join(OUT, out_name)

            # One request at a time (GPU safety). You can relax later.
            with LOCK:
                final = synth_chunks(chunks)
                write_wav(out_path, final)

            return self._json(200, {"ok": True, "out_path": out_path, "chunks": len(chunks)})

        except Exception as e:
            return self._json(500, {"ok": False, "error": str(e)})

def main():
    httpd = ThreadingHTTPServer((HOST, PORT), H)
    print(f"READY on http://{HOST}:{PORT}  (GET /health, POST /speak)", flush=True)
    httpd.serve_forever()

if __name__ == "__main__":
    main()

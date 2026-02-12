#!/usr/bin/env python3
import os, time, json, re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# ---- env (same as your daemon) ----
OUT          = os.environ.get("OUT", os.getcwd())
ORIG         = os.environ.get("ORIG", os.getcwd())
FTCKPT       = os.environ["FTCKPT"]
SPEAKERS_PTH = os.environ["SPEAKERS_PTH"]

LANG = os.environ.get("LANG", "fr").strip()
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "7010"))

SPEAKER_WAVS_ENV = os.environ.get("SPEAKER_WAVS", "").strip()
SPEAKER_WAV      = os.environ.get("SPEAKER_WAV", "").strip()

CONFIG_PATH    = os.path.join(ORIG, "config.json")
TOKENIZER_PATH = os.path.join(ORIG, "vocab.json")
CKPT_DIR       = os.path.dirname(FTCKPT)

SR = int(os.environ.get("SR", "24000"))
MAX_CHARS = int(os.environ.get("MAX_CHARS", "220"))

# Stitching params (defaults align with your usage)
XFADE_MS        = int(os.environ.get("XFADE_MS", "180"))
EDGE_FADE_MS    = int(os.environ.get("EDGE_FADE_MS", "30"))
PAUSE_MS        = int(os.environ.get("PAUSE_MS", "280"))
TAIL_SILENCE_MS = int(os.environ.get("TAIL_SILENCE_MS", "700"))
FADE_OUT_MS     = int(os.environ.get("FADE_OUT_MS", "80"))
MICRO_PAUSE_MS  = int(os.environ.get("MICRO_PAUSE_MS", "0"))

def must_exist(p):
    if not p or not os.path.exists(p):
        raise FileNotFoundError(p)

def load_chunks_from_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and isinstance(obj.get("chunks"), list):
        return [str(x) for x in obj["chunks"]]
    if isinstance(obj, list):
        return [str(x) for x in obj]
    raise ValueError("chunks_json must be a JSON list or {'chunks':[...]}")

def add_silence(ms: int) -> torch.Tensor:
    n = int(SR * ms / 1000)
    n = max(0, n)
    return torch.zeros((1, n), dtype=torch.float32)

def edge_fade(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav
    n = int(SR * ms / 1000)
    n = min(max(n, 0), wav.shape[1] // 2)
    if n <= 0:
        return wav
    wav = wav.clone()
    win_in  = torch.linspace(0.0, 1.0, steps=n, device=wav.device, dtype=wav.dtype).unsqueeze(0)
    win_out = torch.linspace(1.0, 0.0, steps=n, device=wav.device, dtype=wav.dtype).unsqueeze(0)
    mask = torch.ones_like(wav)
    mask[:, :n] *= win_in
    mask[:, -n:] *= win_out
    return wav * mask

def equal_power_crossfade(a: torch.Tensor, b: torch.Tensor, fade_samples: int) -> torch.Tensor:
    if fade_samples <= 0:
        return torch.cat([a, b], dim=1)
    fade_samples = min(fade_samples, a.shape[1], b.shape[1])
    if fade_samples <= 0:
        return torch.cat([a, b], dim=1)
    a1 = a[:, :-fade_samples]
    a2 = a[:, -fade_samples:]
    b1 = b[:, :fade_samples]
    b2 = b[:, fade_samples:]
    t = torch.linspace(0.0, 1.0, steps=fade_samples, device=a.device, dtype=a.dtype).unsqueeze(0)
    wa = torch.cos(t * 0.5 * torch.pi)
    wb = torch.sin(t * 0.5 * torch.pi)
    mixed = a2 * wa + b1 * wb
    return torch.cat([a1, mixed, b2], dim=1)

def add_tail_silence(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav
    return torch.cat([wav, add_silence(ms).to(wav.device, wav.dtype)], dim=1)

def fade_out(wav: torch.Tensor, ms: int) -> torch.Tensor:
    if ms <= 0:
        return wav
    n = int(SR * ms / 1000)
    n = min(max(n, 0), wav.shape[1])
    if n <= 0:
        return wav
    wav = wav.clone()
    win = torch.linspace(1.0, 0.0, steps=n, device=wav.device, dtype=wav.dtype).unsqueeze(0)
    mask = torch.ones_like(wav)
    mask[:, -n:] *= win
    return wav * mask

# ---- speakers ----
if SPEAKER_WAVS_ENV:
    speaker_wavs = [p.strip() for p in re.split(r"[,\s]+", SPEAKER_WAVS_ENV) if p.strip()]
elif SPEAKER_WAV:
    speaker_wavs = [SPEAKER_WAV]
else:
    speaker_wavs = []

if not speaker_wavs:
    raise RuntimeError("No speaker wav provided (set SPEAKER_WAV or SPEAKER_WAVS)")

for p in [CONFIG_PATH, TOKENIZER_PATH, FTCKPT, SPEAKERS_PTH]:
    must_exist(p)
for p in speaker_wavs:
    must_exist(p)

# ---- load model once ----
print("LOADING MODEL ONCE...", flush=True)
config = XttsConfig()
config.load_json(CONFIG_PATH)
model = Xtts.init_from_config(config)
model.load_checkpoint(
    config,
    checkpoint_dir=CKPT_DIR,
    checkpoint_path=FTCKPT,
    vocab_path=TOKENIZER_PATH,
    speaker_file_path=SPEAKERS_PTH,
    use_deepspeed=False,
)
if torch.cuda.is_available():
    model.cuda()
model.eval()

print("USING SPEAKER_WAVS:", speaker_wavs, flush=True)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wavs)

fade_samples = int(SR * XFADE_MS / 1000)
os.makedirs(OUT, exist_ok=True)

def synth_from_chunks(chunks, out_name=None):
    # basic guard
    chunks = [str(c) for c in chunks if str(c).strip()]
    if not chunks:
        raise ValueError("empty chunks")

    pieces = []
    for idx, txt in enumerate(chunks, start=1):
        if len(txt) > MAX_CHARS * 5:
            # safety: avoid insane single chunk
            txt = txt[: MAX_CHARS * 5]

        with torch.inference_mode():
            out = model.inference(
                text=txt,
                language=LANG,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                temperature=0.65,
                length_penalty=1.0,
                repetition_penalty=2.0,
                top_k=50,
                top_p=0.85,
            )
            wav_raw = out["wav"]
            wav = wav_raw if isinstance(wav_raw, torch.Tensor) else torch.from_numpy(wav_raw).float()
            if wav.ndim == 1:
                wav = wav.unsqueeze(0)

        wav = edge_fade(wav, EDGE_FADE_MS)
        pieces.append(("audio", wav))

        if MICRO_PAUSE_MS > 0:
            pieces.append(("pause", add_silence(MICRO_PAUSE_MS)))

        if idx < len(chunks):
            pieces.append(("pause", add_silence(PAUSE_MS)))

    final = None
    for kind, w in pieces:
        w = w.to(final.device, final.dtype) if final is not None else w
        if final is None:
            final = w
            continue
        if kind == "pause":
            final = torch.cat([final, w], dim=1)
        else:
            final = equal_power_crossfade(final, w, fade_samples)

    final = add_tail_silence(final, TAIL_SILENCE_MS)
    final = fade_out(final, FADE_OUT_MS)

    ts = time.strftime("%Y%m%d_%H%M%S")
    name = out_name or f"xtts_http_{ts}.wav"
    out_path = os.path.join(OUT, name)
    torchaudio.save(out_path, final.cpu(), SR)
    return out_path, len(chunks)

class Handler(BaseHTTPRequestHandler):
    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/health":
            return self._json(200, {
                "ok": True,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "sr": SR,
                "out": OUT,
            })
        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        if self.path != "/speak":
            return self._json(404, {"ok": False, "error": "not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            req = json.loads(raw) if raw else {}
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})

        out_name = req.get("out_name")
        chunks = None

        if req.get("chunks_json"):
            try:
                chunks = load_chunks_from_json(req["chunks_json"])
            except Exception as e:
                return self._json(400, {"ok": False, "error": f"cannot read chunks_json: {e}"})
        elif isinstance(req.get("chunks"), list):
            chunks = [str(x) for x in req["chunks"]]

        if not chunks:
            return self._json(400, {"ok": False, "error": "need chunks_json or chunks[]"})

        try:
            out_path, n = synth_from_chunks(chunks, out_name=out_name)
            return self._json(200, {"ok": True, "out_path": out_path, "chunks": n})
        except Exception as e:
            return self._json(500, {"ok": False, "error": str(e)})

    def log_message(self, *args, **kwargs):
        return

print(f"HTTP SERVER READY on http://{HOST}:{PORT}  (GET /health, POST /speak)", flush=True)
ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()

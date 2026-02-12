#!/usr/bin/env python3
import os, json, subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

SITE_PUBLIC = "/workspace/SiteInternet/public"
ASK_SCRIPT  = "/workspace/IA/bin/ask_foucault_voice_http.sh"

AUDIO_DIR = os.environ.get("OUT", "/workspace/IA/tmp/out")  # doit matcher ton OUT serveur XTTS
HOST = os.environ.get("WEB_HOST", "0.0.0.0")
PORT = int(os.environ.get("WEB_PORT", "8080"))

def read_file(path):
    with open(path, "rb") as f:
        return f.read()

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

        if p == "/" or p == "/index.html":
            data = read_file(os.path.join(SITE_PUBLIC, "index.html"))
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if p.startswith("/audio/"):
            name = p[len("/audio/"):]
            if "/" in name or ".." in name or not name.endswith(".wav"):
                return self._json(400, {"ok": False, "error": "bad audio name"})
            path = os.path.join(AUDIO_DIR, name)
            if not os.path.exists(path):
                return self._json(404, {"ok": False, "error": "audio not found"})
            data = read_file(path)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if p == "/health":
            return self._json(200, {"ok": True, "audio_dir": AUDIO_DIR})

        return self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        p = urlparse(self.path).path
        if p != "/api/ask":
            return self._json(404, {"ok": False, "error": "not found"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            obj = json.loads(raw) if raw else {}
            question = (obj.get("question") or "").strip()
        except Exception as e:
            return self._json(400, {"ok": False, "error": f"bad json: {e}"})

        if not question:
            return self._json(400, {"ok": False, "error": "missing question"})

        if not os.path.exists(ASK_SCRIPT):
            return self._json(500, {"ok": False, "error": f"missing {ASK_SCRIPT}"})

        # Lance le pipeline (Mistral -> chunks -> XTTS HTTP -> wav)
        try:
            cp = subprocess.run(
                [ASK_SCRIPT, question],
                capture_output=True,
                text=True,
                env=os.environ.copy(),
                timeout=180,
            )
        except Exception as e:
            return self._json(500, {"ok": False, "error": f"ask failed: {e}"})

        out = (cp.stdout or "").strip()
        err = (cp.stderr or "").strip()

        if cp.returncode != 0:
            return self._json(500, {"ok": False, "error": "ask script error", "stdout": out[-2000:], "stderr": err[-2000:]})

        # Convention: le script imprime le out_path en dernière ligne
        lines = [l for l in out.splitlines() if l.strip()]
        if not lines:
            return self._json(500, {"ok": False, "error": "no output from ask script", "stdout": out[-2000:], "stderr": err[-2000:]})

        wav_path = lines[-1].strip()
        name = os.path.basename(wav_path)

        # Vérif que c’est dans AUDIO_DIR
        abs_audio_dir = os.path.abspath(AUDIO_DIR)
        abs_wav = os.path.abspath(wav_path)
        if not abs_wav.startswith(abs_audio_dir + os.sep):
            # Si jamais ton script écrit ailleurs, on refuse (sécurité)
            return self._json(500, {"ok": False, "error": "wav path outside audio dir", "wav_path": wav_path, "audio_dir": AUDIO_DIR})

        if not os.path.exists(wav_path):
            return self._json(500, {"ok": False, "error": "wav not found after generation", "wav_path": wav_path})

        return self._json(200, {"ok": True, "audio_url": f"/audio/{name}", "wav_path": wav_path})

    def log_message(self, *args, **kwargs):
        return

print(f"WEB READY on http://{HOST}:{PORT}  (GET /, POST /api/ask, GET /audio/.., GET /health)", flush=True)
ThreadingHTTPServer((HOST, PORT), H).serve_forever()

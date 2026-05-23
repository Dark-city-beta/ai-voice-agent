#!/usr/bin/env python3
"""Small LAN control API for Hermes Voice Bridge.

Endpoints:
- GET  /health
- POST /start
- POST /stop
- POST /mute
- POST /unmute
- POST /toggle-mute

The server controls local_voice_vad_loop.py on the Linux/Hermes host.
Keep this bound to LAN/local network only; do not expose to the internet.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
VOICE_SCRIPT = ROOT / "local_voice_vad_loop.py"
PID_FILE = ROOT / ".voice_loop.pid"
MUTE_FILE = ROOT / ".voice_mute"
RESET_FILE = ROOT / ".voice_reset"
LOG_FILE = ROOT / "voice_control.log"
CONTROL_TOKEN = os.environ.get("VOICE_CONTROL_TOKEN", "")


def build_voice_cmd() -> list[str]:
    return [
        sys.executable,
        str(VOICE_SCRIPT),
        "--input-device", os.environ.get("VOICE_INPUT_DEVICE", "7"),
        "--samplerate", os.environ.get("VOICE_SAMPLERATE", "44100"),
        "--model", os.environ.get("VOICE_STT_MODEL", "base"),
        "--preload-stt",
        "--output-device", os.environ.get("VOICE_OUTPUT_DEVICE", "plughw:0,0"),
        "--level-interval", os.environ.get("VOICE_LEVEL_INTERVAL", "2.0"),
        "--llm-backend", os.environ.get("VOICE_LLM_BACKEND", "direct"),
    ]


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"{ts} {msg}\n")


def read_pid() -> int | None:
    try:
        return int(PID_FILE.read_text().strip())
    except Exception:
        return None


def is_running(pid: int | None = None) -> bool:
    pid = pid or read_pid()
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def status() -> dict[str, Any]:
    pid = read_pid()
    running = is_running(pid)
    if not running and PID_FILE.exists():
        try:
            PID_FILE.unlink()
        except Exception:
            pass
        pid = None
    return {
        "ok": True,
        "running": running,
        "pid": pid if running else None,
        "muted": MUTE_FILE.exists(),
        "host": os.uname().nodename if hasattr(os, "uname") else "unknown",
        "time": time.time(),
    }


def start_voice() -> dict[str, Any]:
    st = status()
    if st["running"]:
        return {**st, "message": "already running"}
    MUTE_FILE.unlink(missing_ok=True)
    RESET_FILE.unlink(missing_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    cmd = build_voice_cmd()
    log("START " + " ".join(cmd))
    out = open(ROOT / "voice_loop_runtime.log", "ab", buffering=0)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=out,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    time.sleep(0.2)
    return {**status(), "message": "started"}


def stop_voice() -> dict[str, Any]:
    pid = read_pid()
    if pid and is_running(pid):
        log(f"STOP pid={pid}")
        try:
            os.killpg(pid, signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
        for _ in range(20):
            if not is_running(pid):
                break
            time.sleep(0.1)
        if is_running(pid):
            try:
                os.killpg(pid, signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    PID_FILE.unlink(missing_ok=True)
    MUTE_FILE.unlink(missing_ok=True)
    RESET_FILE.unlink(missing_ok=True)
    return {**status(), "message": "stopped"}


def set_mute(muted: bool) -> dict[str, Any]:
    if muted:
        MUTE_FILE.write_text("1", encoding="utf-8")
        RESET_FILE.write_text("1", encoding="utf-8")
        msg = "muted"
    else:
        MUTE_FILE.unlink(missing_ok=True)
        RESET_FILE.write_text("1", encoding="utf-8")
        msg = "unmuted"
    log(msg.upper())
    return {**status(), "message": msg}


class Handler(BaseHTTPRequestHandler):
    def _json(self, data: dict[str, Any], code: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: Any) -> None:
        log("HTTP " + fmt % args)

    def do_GET(self) -> None:
        if self.path in ("/", "/health", "/status"):
            self._json(status())
        else:
            self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self) -> None:
        if CONTROL_TOKEN and self.headers.get("X-Voice-Token") != CONTROL_TOKEN:
            self._json({"ok": False, "error": "unauthorized"}, 401)
            return
        try:
            if self.path == "/start":
                self._json(start_voice())
            elif self.path == "/stop":
                self._json(stop_voice())
            elif self.path == "/mute":
                self._json(set_mute(True))
            elif self.path == "/unmute":
                self._json(set_mute(False))
            elif self.path == "/toggle-mute":
                self._json(set_mute(not MUTE_FILE.exists()))
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        except Exception as e:
            log(f"ERROR {e!r}")
            self._json({"ok": False, "error": str(e)}, 500)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8766)
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"voice-control listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

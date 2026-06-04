#!/usr/bin/env python3
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse


APP = FastAPI(title="Hermes Voice Controller Server")
RUN_SCRIPT = Path("/home/dark/.hermes/scripts/run_local_voice.sh")
PID_FILE = Path("/tmp/hermes_voice_loop.pid")
LOG_FILE = Path("/tmp/hermes_voice_loop.log")
CONTROL_DIR = Path("/tmp/hermes_voice_control")
MIC_MUTE_FILE = CONTROL_DIR / "mic_muted"
SPEAKER_MUTE_FILE = CONTROL_DIR / "speaker_muted"
STATE_FILE = CONTROL_DIR / "state.json"
TOKEN = os.environ.get("VOICEHUB_TOKEN", "").strip()
PROC = None


def require_auth(request: Request):
    if not TOKEN:
        return
    if request.headers.get("authorization", "") != f"Bearer {TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def read_pid():
    if PID_FILE.exists():
        try:
            return int(PID_FILE.read_text().strip())
        except Exception:
            return None
    return None


def find_voice_pids():
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "/home/dark/.hermes/scripts/local_voice_vad_loop.py"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    pids = []
    for line in out.splitlines():
        try:
            pid = int(line.strip())
        except Exception:
            continue
        if pid != os.getpid():
            pids.append(pid)
    return pids


def is_running():
    global PROC
    if PROC and PROC.poll() is None:
        return True
    pid = read_pid()
    return bool(pid and Path(f"/proc/{pid}").exists()) or bool(find_voice_pids())


def read_state():
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def write_state(state: str, **extra):
    try:
        CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": True,
            "state": state,
            "pid": read_pid(),
            "mic_muted": MIC_MUTE_FILE.exists(),
            "speaker_muted": SPEAKER_MUTE_FILE.exists(),
            "ts": time.time(),
        }
        payload.update(extra)
        STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def set_flag(path: Path, enabled: bool):
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    if enabled:
        path.write_text(str(time.time()), encoding="utf-8")
    else:
        path.unlink(missing_ok=True)


def status_payload():
    pid = read_pid()
    return {
        "ok": True,
        "running": is_running(),
        "pid": pid,
        "server": "http://192.168.31.200:8766",
        "script": str(RUN_SCRIPT),
        "log": str(LOG_FILE),
        "mic_muted": MIC_MUTE_FILE.exists(),
        "speaker_muted": SPEAKER_MUTE_FILE.exists(),
        "state": read_state(),
    }


@APP.get("/")
@APP.get("/status")
@APP.get("/state")
def status():
    return JSONResponse(status_payload())


@APP.post("/start")
@APP.post("/voice/start")
def start(request: Request):
    require_auth(request)
    global PROC
    if is_running():
        return JSONResponse({**status_payload(), "message": "already running"})
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    write_state("starting")
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    log = LOG_FILE.open("ab", buffering=0)
    PROC = subprocess.Popen(
        ["bash", str(RUN_SCRIPT)],
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    PID_FILE.write_text(str(PROC.pid), encoding="utf-8")
    return JSONResponse({**status_payload(), "message": "started"})


@APP.post("/stop")
@APP.post("/voice/stop")
def stop(request: Request):
    require_auth(request)
    global PROC
    pids = []
    if PROC and PROC.poll() is None:
        pids.append(PROC.pid)
    stored = read_pid()
    if stored:
        pids.append(stored)
    pids.extend(find_voice_pids())
    pids = sorted(set(pid for pid in pids if pid))
    if not pids:
        PID_FILE.unlink(missing_ok=True)
        PROC = None
        write_state("stopped")
        return JSONResponse({**status_payload(), "message": "not running"})
    for pid in pids:
        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(pid, signal.SIGTERM)
            except Exception:
                pass
    time.sleep(0.8)
    for pid in pids:
        if Path(f"/proc/{pid}").exists():
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception:
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception:
                    pass
    PID_FILE.unlink(missing_ok=True)
    PROC = None
    write_state("stopped")
    return JSONResponse({**status_payload(), "message": "stopped"})


@APP.post("/restart")
@APP.post("/voice/restart")
def restart(request: Request):
    require_auth(request)
    stop(request)
    time.sleep(0.5)
    return start(request)


@APP.post("/mic/mute")
@APP.post("/voice/mic/mute")
def mute_mic(request: Request):
    require_auth(request)
    set_flag(MIC_MUTE_FILE, True)
    return JSONResponse({**status_payload(), "message": "mic muted"})


@APP.post("/mic/unmute")
@APP.post("/voice/mic/unmute")
def unmute_mic(request: Request):
    require_auth(request)
    set_flag(MIC_MUTE_FILE, False)
    return JSONResponse({**status_payload(), "message": "mic unmuted"})


@APP.post("/speaker/mute")
@APP.post("/voice/speaker/mute")
def mute_speaker(request: Request):
    require_auth(request)
    set_flag(SPEAKER_MUTE_FILE, True)
    return JSONResponse({**status_payload(), "message": "speaker muted"})


@APP.post("/speaker/unmute")
@APP.post("/voice/speaker/unmute")
def unmute_speaker(request: Request):
    require_auth(request)
    set_flag(SPEAKER_MUTE_FILE, False)
    return JSONResponse({**status_payload(), "message": "speaker unmuted"})

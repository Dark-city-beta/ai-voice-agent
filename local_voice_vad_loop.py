#!/usr/bin/env python3
"""
Hermes local voice VAD loop — variable-length utterances, Russian STT, TTS playback.

Restoration target:
- keep the old Hermes-aware dialogue/session/context behavior;
- preserve timings/voice-flow semantics as much as possible;
- replace only the broken Microsoft/Edge TTS path with local offline male Silero.

Run:
  python3 ~/.hermes/scripts/local_voice_vad_loop.py --list-devices
  python3 ~/.hermes/scripts/local_voice_vad_loop.py
  python3 ~/.hermes/scripts/local_voice_vad_loop.py --input-device 3 --output-device plughw:0,0
"""
import argparse
import audioop
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import json

import requests
import yaml
import threading
import time
import wave
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.signal import resample_poly
from faster_whisper import WhisperModel

VAD_RATE = 16000
CHANNELS = 1
FRAME_MS = 20
VAD_FRAME_SAMPLES = VAD_RATE * FRAME_MS // 1000
FRAME_BYTES = VAD_FRAME_SAMPLES * 2
HERMES_CONFIG = Path.home() / ".hermes" / "config.yaml"
VOICE_SESSION_ID_FILE = Path('/tmp/hermes_voice_session_id.txt')
VOICE_CONTEXT_FILE = Path('/tmp/hermes_voice_shared_context.txt')
SILERO_MODEL_PATH = Path('/home/dark/.cache/torch/hub/snakers4_silero-models_master/src/silero/model/v4_ru.pt')
DEFAULT_TTS_PYTHON = Path("/mnt/city17/Free project/Cat-books/catbooks-2.0/.venv-cpu/bin/python")
DEFAULT_TTS_WORKER = Path("/home/dark/.hermes/scripts/silero_tts_worker.py")
CONTROL_DIR = Path("/tmp/hermes_voice_control")
MIC_MUTE_FILE = CONTROL_DIR / "mic_muted"
SPEAKER_MUTE_FILE = CONTROL_DIR / "speaker_muted"
STATE_FILE = CONTROL_DIR / "state.json"

WHISPER_HALLUCINATIONS = {
    "",
    ".",
    "...",
    "спасибо",
    "спасибо за просмотр",
    "подписывайтесь",
    "подписывайтесь на канал",
    "продолжение следует",
    "субтитры сделал",
    "субтитры создавал",
    "thank you",
    "thanks",
    "thanks for watching",
    "subscribe",
    "bye",
}


def find_shem_boy_input():
    """Prefer the current USB microphone when no input device is specified."""
    preferred_markers = ("SHEM-BOY", "USB Audio", "USB")
    try:
        for i, d in enumerate(sd.query_devices()):
            name = d.get("name", "")
            if d.get("max_input_channels", 0) > 0 and any(marker in name for marker in preferred_markers):
                return i
    except Exception:
        pass
    return None


def db_from_pcm16(pcm: bytes) -> tuple[float, float]:
    if not pcm:
        return -999.0, -999.0
    rms = audioop.rms(pcm, 2)
    mx = audioop.max(pcm, 2)
    maxp = 32768.0
    def db(x):
        return -999.0 if x <= 0 else 20 * math.log10(x / maxp)
    return round(db(rms), 1), round(db(mx), 1)


def write_wav(path: Path, pcm: bytes):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(VAD_RATE)
        w.writeframes(pcm)


def control_flag(path: Path) -> bool:
    try:
        return path.exists()
    except Exception:
        return False


def write_state(state: str, **extra):
    try:
        CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "ok": True,
            "state": state,
            "pid": os.getpid(),
            "mic_muted": control_flag(MIC_MUTE_FILE),
            "speaker_muted": control_flag(SPEAKER_MUTE_FILE),
            "ts": time.time(),
        }
        payload.update(extra)
        STATE_FILE.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def is_whisper_hallucination(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", (text or "").strip().lower())
    normalized = normalized.strip(" .,!?:;«»\"'")
    if normalized in WHISPER_HALLUCINATIONS:
        return True
    if len(normalized) <= 2:
        return True
    return False


def play_tone(output_device: str, freq: int, duration_ms: int = 120, volume: float = 0.18):
    """Fallback short local cue through ALSA."""
    aplay = shutil.which("aplay")
    if not aplay:
        return
    sr = 22050
    n = max(1, int(sr * duration_ms / 1000))
    t = np.arange(n, dtype=np.float32) / sr
    fade_n = min(n // 4, int(sr * 0.015))
    env = np.ones(n, dtype=np.float32)
    if fade_n > 0:
        fade = np.linspace(0.0, 1.0, fade_n, dtype=np.float32)
        env[:fade_n] = fade
        env[-fade_n:] = fade[::-1]
    audio = (np.sin(2 * np.pi * freq * t) * env * volume * 32767).astype(np.int16)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        with wave.open(f.name, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(audio.tobytes())
        subprocess.run([aplay, "-D", output_device, "-q", f.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3)


def play_audio_file(path: str, output_device: str):
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(str(p))
    ffmpeg = shutil.which("ffmpeg")
    aplay = shutil.which("aplay")
    if not (ffmpeg and aplay):
        raise RuntimeError("ffmpeg/aplay missing")
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        r = subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(p), f.name], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout)[-300:])
        subprocess.run([aplay, "-D", output_device, "-q", f.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5)


def clean_text_for_tts(text: str) -> str:
    """Make model text speakable: strip markdown, emoji, code fences, bullets, links."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_~#>]+", " ", text)
    text = re.sub(r"^\s*[-+•]\s+", "", text, flags=re.M)
    text = re.sub(r"[\U00010000-\U0010ffff]", " ", text)
    text = re.sub(r"[\u2600-\u27BF]", " ", text)
    text = text.replace("—", ", ").replace("–", ", ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences_ru(text: str, max_len: int = 110):
    text = clean_text_for_tts(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?…])\s+", text)
    out = []
    buf = ""
    for p in parts:
        if not p:
            continue
        if len(buf) + len(p) + 1 <= max_len:
            buf = (buf + " " + p).strip()
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf:
        out.append(buf)
    final = []
    for chunk in out:
        if len(chunk) <= max_len:
            final.append(chunk)
            continue
        words = chunk.split()
        b = ""
        for w in words:
            if len(b) + len(w) + 1 > max_len:
                if b:
                    final.append(b)
                b = w
            else:
                b = (b + " " + w).strip()
        if b:
            final.append(b)
    return final


class SileroTTSClient:
    def __init__(self, python_path: Path, worker_path: Path):
        self.python_path = Path(python_path)
        self.worker_path = Path(worker_path)
        self.proc = None
        self.lock = threading.Lock()
        self.responses = queue.Queue()
        self.reader = None

    def _reader_loop(self):
        while self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                break
            try:
                self.responses.put(json.loads(line))
            except Exception:
                self.responses.put({"ok": False, "error": line.strip()})

    def _start_locked(self):
        if self.proc and self.proc.poll() is None:
            return True
        if not self.python_path.exists() or not self.worker_path.exists() or not SILERO_MODEL_PATH.exists():
            return False
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("TORCH_NUM_THREADS", "1")
        env.setdefault("SILERO_MODEL_PATH", str(SILERO_MODEL_PATH))
        self.proc = subprocess.Popen(
            [str(self.python_path), str(self.worker_path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        self.reader = threading.Thread(target=self._reader_loop, daemon=True)
        self.reader.start()
        try:
            ready = self.responses.get(timeout=45)
        except queue.Empty:
            self.stop()
            return False
        if not ready.get("ok"):
            print(f"[tts] silero worker not ready: {ready}", flush=True)
            self.stop()
            return False
        print(f"[tts] silero worker ready: {ready.get('model')}", flush=True)
        return True

    def synthesize(self, text: str, speaker: str, wav: Path, timeout: float = 120.0) -> bool:
        with self.lock:
            if not self._start_locked():
                return False
            if not self.proc or not self.proc.stdin:
                return False
            req = {"text": text, "speaker": speaker, "audio_path": str(wav), "sample_rate": 48000}
            try:
                self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
                self.proc.stdin.flush()
                resp = self.responses.get(timeout=timeout)
            except Exception as exc:
                print(f"[tts] silero worker request failed: {exc}", flush=True)
                self.stop()
                return False
            if not resp.get("ok"):
                print(f"[tts] silero worker failed: {resp}", flush=True)
                return False
            return wav.exists()

    def stop(self):
        proc = self.proc
        self.proc = None
        if not proc:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "stop"}) + "\n")
                proc.stdin.flush()
        except Exception:
            pass
        try:
            proc.terminate()
        except Exception:
            pass


class Playback:
    def __init__(self, output_device: str, voice: str, tts_python: Path, tts_worker: Path):
        self.output_device = output_device
        self.voice = voice
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.processes: list[subprocess.Popen] = []
        self.tts = SileroTTSClient(tts_python, tts_worker)

    def stop(self):
        self.stop_event.set()
        with self.lock:
            for p in list(self.processes):
                try:
                    if p.poll() is None:
                        p.terminate()
                except Exception:
                    pass
            self.processes.clear()

    def _track(self, p):
        with self.lock:
            self.processes.append(p)
        return p

    def _untrack(self, p):
        with self.lock:
            if p in self.processes:
                self.processes.remove(p)

    def _play_wav(self, wav: Path):
        p = self._track(subprocess.Popen(["aplay", "-D", self.output_device, "-q", str(wav)]))
        try:
            while p.poll() is None:
                if self.stop_event.is_set():
                    try:
                        p.terminate()
                    except Exception:
                        pass
                    break
                time.sleep(0.03)
        finally:
            self._untrack(p)

    def speak_silero(self, text: str, speaker: str = "aidar") -> bool:
        ffmpeg = shutil.which("ffmpeg")
        aplay = shutil.which("aplay")
        if not (ffmpeg and aplay):
            return False
        self.stop_event.clear()
        text = clean_text_for_tts(text)
        if not text:
            return False
        total_t0 = time.perf_counter()
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "silero.wav"
            if not self.tts.synthesize(text=text, speaker=speaker, wav=wav):
                print("[tts] silero synthesis failed", flush=True)
                return False
            playable = Path(td) / "silero_playable.wav"
            r = subprocess.run(
                [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(wav), "-ac", "2", str(playable)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=60,
            )
            if r.returncode != 0 or not playable.exists():
                print("[tts] silero ffmpeg failed:", (r.stderr or r.stdout)[-500:], flush=True)
                return False
            print(f"[timing] tts.silero.total {time.perf_counter()-total_t0:.2f}s", flush=True)
            self._play_wav(playable)
            return True

    def speak(self, text: str):
        for chunk in split_sentences_ru(text):
            if self.stop_event.is_set():
                break
            ok = self.speak_silero(chunk, speaker=self.voice or "aidar")
            if not ok:
                print(f"[tts] local silero unavailable, text only: {chunk}", flush=True)
                break
        return


class VoiceLoop:
    def __init__(self, args):
        self.args = args
        self.audio_q = queue.Queue(maxsize=200)
        self.utterance_q = queue.Queue()
        self.stop = threading.Event()
        self.playback = Playback(args.output_device, args.tts_voice, Path(args.tts_python), Path(args.tts_worker))
        self.vad = webrtcvad.Vad(args.vad)
        self.stt_lock = threading.Lock()
        self.model = None
        self.state_lock = threading.Lock()
        self.assistant_speaking = False
        self._pcm_tail = b""
        self.listening_muted = False
        self.cue_lock = threading.Lock()

    def cue(self, kind: str):
        if not self.args.cues:
            return
        cue_file = self.args.start_sound if kind == "start" else self.args.stop_sound
        freq = self.args.cue_start_hz if kind == "start" else self.args.cue_stop_hz
        with self.cue_lock:
            try:
                if cue_file:
                    play_audio_file(cue_file, self.args.output_device)
                else:
                    play_tone(self.args.output_device, freq=freq, duration_ms=self.args.cue_ms, volume=self.args.cue_volume)
            except Exception as e:
                print(f"[cue] failed: {e}", flush=True)

    def turn_cue(self, kind: str):
        freq = self.args.accepted_cue_hz if kind == "accepted" else self.args.resume_cue_hz
        with self.cue_lock:
            try:
                play_tone(self.args.output_device, freq=freq, duration_ms=self.args.turn_cue_ms, volume=self.args.turn_cue_volume)
            except Exception as e:
                print(f"[turn-cue] failed: {e}", flush=True)

    def load_model(self):
        if self.model is None:
            print(f"[stt] loading faster-whisper model={self.args.model} compute=int8...", flush=True)
            self.model = WhisperModel(self.args.model, device="cpu", compute_type="int8")

    def audio_callback(self, indata, frames, time_info, status):
        if status and self.args.debug_audio_status:
            print(f"[audio status] {status}", flush=True)
        if self.listening_muted or control_flag(MIC_MUTE_FILE):
            return
        if self.stt_lock.locked() and self.args.drop_mic_during_stt:
            return
        mono = indata[:, 0].astype(np.float32)
        hw_rate = self.args.samplerate
        if hw_rate != VAD_RATE:
            if hw_rate == 48000:
                mono = resample_poly(mono, 1, 3)
            elif hw_rate == 44100:
                mono = resample_poly(mono, 160, 441)
            else:
                mono = resample_poly(mono, VAD_RATE, hw_rate)
        pcm = np.clip(mono * 32768, -32768, 32767).astype(np.int16).tobytes()
        pcm = self._pcm_tail + pcm
        frames_out = []
        while len(pcm) >= FRAME_BYTES:
            frames_out.append(pcm[:FRAME_BYTES])
            pcm = pcm[FRAME_BYTES:]
        self._pcm_tail = pcm
        try:
            for frame in frames_out:
                self.audio_q.put_nowait(frame)
        except queue.Full:
            while True:
                try:
                    self.audio_q.get_nowait()
                except Exception:
                    break

    def vad_worker(self):
        pre_roll_frames = max(1, self.args.pre_roll_ms // FRAME_MS)
        end_silence_frames = max(1, self.args.end_silence_ms // FRAME_MS)
        min_speech_frames = max(1, self.args.min_speech_ms // FRAME_MS)
        max_utterance_frames = max(1, self.args.max_utterance_ms // FRAME_MS)
        start_speech_frames = max(1, self.args.start_speech_ms // FRAME_MS)

        pre = deque(maxlen=pre_roll_frames)
        candidate = []
        speech = []
        triggered = False
        silence_count = 0
        speech_count = 0
        consecutive_start = 0
        last_level_print = 0

        print("[vad] listening... говори свободно, я сам пойму паузу.", flush=True)
        while not self.stop.is_set():
            try:
                frame = self.audio_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if len(frame) != FRAME_BYTES:
                continue
            try:
                is_speech = self.vad.is_speech(frame, VAD_RATE)
            except Exception as e:
                print(f"[vad] bad frame len={len(frame)} err={e}", flush=True)
                continue
            now = time.time()
            if now - last_level_print > self.args.level_interval:
                rms, mx = db_from_pcm16(frame)
                print(f"[level] rms={rms}dB max={mx}dB speech={is_speech}", flush=True)
                last_level_print = now

            if is_speech and self.assistant_speaking and self.args.barge_in:
                print("[barge-in] user speech detected, stopping playback", flush=True)
                self.playback.stop()

            if not triggered:
                pre.append(frame)
                if is_speech:
                    consecutive_start += 1
                    candidate.append(frame)
                else:
                    consecutive_start = 0
                    candidate.clear()
                if consecutive_start >= start_speech_frames:
                    triggered = True
                    speech = list(pre) + candidate
                    pre.clear(); candidate.clear()
                    speech_count = consecutive_start
                    silence_count = 0
                    write_state("recording")
                    print("[vad] speech started", flush=True)
                continue

            speech.append(frame)
            if is_speech:
                speech_count += 1
                silence_count = 0
            else:
                silence_count += 1

            too_quiet = silence_count >= end_silence_frames
            too_long = len(speech) >= max_utterance_frames
            if too_quiet or too_long:
                pcm = b"".join(speech)
                dur = len(speech) * FRAME_MS / 1000.0
                rms, mx = db_from_pcm16(pcm)
                if speech_count >= min_speech_frames:
                    write_state("thinking", utterance_sec=round(dur, 2), rms_db=rms, max_db=mx)
                    print(f"[vad] speech ended dur={dur:.1f}s rms={rms}dB max={mx}dB", flush=True)
                    self.utterance_q.put(pcm)
                else:
                    print(f"[vad] ignored too short dur={dur:.1f}s", flush=True)
                triggered = False
                speech = []
                speech_count = 0
                silence_count = 0
                consecutive_start = 0
                pre.clear(); candidate.clear()

    def transcribe(self, pcm: bytes) -> str:
        with self.stt_lock:
            t0 = time.perf_counter()
            self.load_model()
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segments, info = self.model.transcribe(
                audio,
                language="ru",
                beam_size=1,
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
                no_speech_threshold=0.45,
            )
            text = " ".join(s.text.strip() for s in segments).strip()
            print(f"[stt] lang={info.language} prob={info.language_probability:.3f} text={text or '[NO_SPEECH]'}", flush=True)
            print(f"[timing] stt {time.perf_counter()-t0:.2f}s", flush=True)
            if is_whisper_hallucination(text):
                print(f"[stt] dropped hallucination: {text!r}", flush=True)
                text = ""
            if self.args.save_utterances:
                out = Path(self.args.save_utterances)
                out.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                write_wav(out / f"utt_{ts}.wav", pcm)
            return text

    def _voice_messages(self, text: str):
        return [
            {"role": "system", "content": "Отвечай по-русски естественно и по существу. Дай 1–2 коротких предложения, без Markdown, списков, эмодзи и ссылок. Не молчи и не возвращай пустой ответ."},
            {"role": "user", "content": text},
        ]

    def load_direct_llm_config(self):
        if hasattr(self, "_direct_llm_config"):
            return self._direct_llm_config
        cfg_path = Path(self.args.hermes_config).expanduser()
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        model_cfg = cfg.get("model", {}) or {}
        provider_name = model_cfg.get("provider")
        model = model_cfg.get("default") or model_cfg.get("model")
        providers = cfg.get("custom_providers", []) or []
        provider = next((p for p in providers if p.get("name") == provider_name), None)
        if not provider:
            raise RuntimeError(f"custom provider not found: {provider_name}")
        base_url = (provider.get("base_url") or "").rstrip("/")
        api_key = provider.get("api_key") or os.environ.get(provider.get("api_key_env", ""), "")
        if not (base_url and api_key and model):
            raise RuntimeError("direct LLM config incomplete")
        self._direct_llm_config = {"base_url": base_url, "api_key": api_key, "model": model}
        return self._direct_llm_config

    def ask_direct_provider(self, text: str) -> str:
        conf = self.load_direct_llm_config()
        headers = {"Authorization": f"Bearer {conf['api_key']}", "Content-Type": "application/json"}
        payload = {
            "model": conf["model"],
            "messages": self._voice_messages(text),
            "temperature": self.args.llm_temperature,
            "max_tokens": self.args.llm_max_tokens,
            "stream": False,
        }
        r = requests.post(
            conf["base_url"] + "/chat/completions",
            headers=headers,
            json=payload,
            timeout=(self.args.llm_connect_timeout, self.args.hermes_timeout),
        )
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[-500:]}")
        data = r.json()
        return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

    def load_voice_session_id(self):
        try:
            if VOICE_SESSION_ID_FILE.exists():
                sid = VOICE_SESSION_ID_FILE.read_text(encoding="utf-8").strip()
                return sid or ""
        except Exception:
            return ""
        return ""

    def save_voice_session_id(self, sid: str):
        try:
            if sid:
                VOICE_SESSION_ID_FILE.write_text(sid.strip(), encoding="utf-8")
        except Exception:
            pass

    def load_shared_context(self):
        try:
            if VOICE_CONTEXT_FILE.exists():
                txt = VOICE_CONTEXT_FILE.read_text(encoding="utf-8").strip()
                return txt[:4000]
        except Exception:
            return ""
        return ""

    def ask_cli_hermes(self, text: str) -> str:
        shared = self.load_shared_context()
        if self.args.voice_full_agent:
            agent_rule = "Если задача требует инструментов, реально используй инструменты Hermes. "
        else:
            agent_rule = "Это быстрый голосовой режим: отвечай как собеседник, без инструментов и долгих действий. "
        carry = (
            "Ты работаешь в голосовом контуре Гермеса и должен вести себя как тот же самый помощник, что и в основном чате. "
            + agent_rule +
            "Не забывай предыдущие голосовые шаги этой же сессии. "
            "Если пользователь в текстовом чате заранее прислал важный код или данные, они могут быть добавлены ниже как общий контекст. "
            "КРИТИЧЕСКОЕ ПРАВИЛО: команда DARK 'подключайся к виртуалке' означает подключение к WSL-виртуалке darkcity-wsl через рабочий ноут, а НЕ отключение голосового модуля. "
            "Никогда не отвечай на 'подключайся' как 'отключаюсь' или 'голосовой модуль можно закрывать'. "
            "Ответь по-русски естественно, без Markdown, без списков, без эмодзи, максимум 2 коротких предложения."
        )
        if shared:
            prompt = carry + " Общий текстовый контекст: " + shared + f" Текущая реплика DARK: {text}"
        else:
            prompt = carry + f" Текущая реплика DARK: {text}"
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        env.setdefault("TERM", "dumb")
        env.setdefault("TELEGRAM_PROXY", "socks5h://127.0.0.1:10808")
        env.setdefault("HTTPS_PROXY", "http://127.0.0.1:10809")
        env.setdefault("HTTP_PROXY", "http://127.0.0.1:10809")
        env.setdefault("NO_PROXY", "localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8")
        cmd = ["hermes", "chat", "-q", prompt, "--source", "local-voice", "--quiet"]
        if self.args.voice_full_agent:
            cmd.extend(["--toolsets", "terminal,file,web,browser,vision,skills,memory,session_search,todo"])
        else:
            cmd.extend(["--ignore-rules", "--max-turns", "1"])
        sid = self.load_voice_session_id()
        if sid:
            cmd.extend(["--resume", sid])
        if self.args.voice_full_agent and self.args.voice_profile:
            cmd.extend(["--profile", self.args.voice_profile])
        if self.args.voice_full_agent and self.args.voice_skills:
            cmd.extend(["--skills", self.args.voice_skills])
        try:
            r = subprocess.run(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.args.hermes_timeout,
                env=env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            write_state("error", error="hermes_timeout")
            return "Гермес сейчас слишком долго отвечает. Я тебя услышал, повтори короче или дай одну команду."
        combined = (r.stdout or "") + "\n" + (r.stderr or "")
        m = re.search(r"session_id:\s*([A-Za-z0-9_\-]+)", combined)
        if m:
            self.save_voice_session_id(m.group(1))
        if r.returncode != 0:
            print("[llm] hermes cli failed:", (r.stderr or r.stdout)[-2000:], flush=True)
            write_state("error", error="hermes_cli_failed")
            return "Гермес сейчас не ответил по сети. Голосовой контур живой, попробуй ещё раз через пару секунд."
        cleaned = []
        for line in (r.stdout or "").splitlines():
            t = line.strip()
            if not t:
                continue
            if t.startswith("session_id:"):
                continue
            if t.startswith("session id:"):
                continue
            if t.startswith("↻ Resumed session"):
                continue
            if t.startswith("MEDIA:"):
                continue
            if t.startswith("Commentary to="):
                continue
            if t.startswith("Traceback ") or t.startswith("File \""):
                continue
            if t.startswith("Tool ") or t.startswith("functions.") or t.startswith("image_gen"):
                continue
            if re.match(r"^[A-Za-z](?:\s+[A-Za-z]){3,}$", t):
                continue
            if re.match(r"^(?:[A-Za-zА-Яа-я0-9]+\s+){4,}[A-Za-zА-Яа-я0-9]+$", t) and len(t) < 80 and not re.search(r"[.!?]", t):
                continue
            cleaned.append(line)
        text_out = "\n".join(cleaned).strip()
        candidates = [blk.strip() for blk in re.split(r"\n{2,}", text_out) if blk.strip()]
        if candidates:
            text_out = candidates[-1]
        sentence_lines = []
        for line in text_out.splitlines():
            t = line.strip()
            if re.search(r"[.!?…]", t):
                sentence_lines.append(t)
        if sentence_lines:
            text_out = " ".join(sentence_lines)
        return text_out

    def ask_hermes(self, text: str) -> str:
        print(f"[llm] thinking backend={self.args.llm_backend}...", flush=True)
        t0 = time.perf_counter()
        try:
            if self.args.llm_backend == "direct":
                reply = self.ask_direct_provider(text)
            elif self.args.llm_backend == "cli":
                reply = self.ask_cli_hermes(text)
            else:
                raise RuntimeError(f"unknown llm backend: {self.args.llm_backend}")
        except Exception as e:
            print(f"[llm] {self.args.llm_backend} failed: {e}", flush=True)
            if self.args.llm_backend != "cli" and self.args.llm_fallback_cli:
                reply = self.ask_cli_hermes(text)
            else:
                reply = "Ответ сорвался по сети. Повтори коротко."
        print(f"[timing] llm.{self.args.llm_backend} {time.perf_counter()-t0:.2f}s", flush=True)
        reply = clean_text_for_tts(reply or "")
        return reply or "Пустой ответ модели. Повтори коротко."

    def assistant_worker(self):
        while not self.stop.is_set():
            pcm = self.utterance_q.get()
            self.listening_muted = True
            write_state("thinking")
            print("[mic] accepted phrase, processing", flush=True)
            if self.args.turn_cues:
                self.turn_cue("accepted")
            text = self.transcribe(pcm)
            if not text:
                self.listening_muted = False
                write_state("listening")
                print("[mic] listening resumed", flush=True)
                if self.args.turn_cues:
                    self.turn_cue("resume")
                continue
            print(f"\n[DARK] {text}\n", flush=True)
            reply = self.ask_hermes(text)
            print(f"\n[Максим] {reply}\n", flush=True)
            if self.args.no_tts or control_flag(SPEAKER_MUTE_FILE):
                self.listening_muted = False
                write_state("listening", last_text=text, last_reply=reply, tts_skipped=True)
                print("[mic] listening resumed", flush=True)
                if self.args.turn_cues:
                    self.turn_cue("resume")
                continue
            self.assistant_speaking = True
            write_state("speaking", last_text=text, last_reply=reply)
            try:
                self.playback.speak(reply)
            finally:
                self.assistant_speaking = False
                if self.args.after_tts_pause_ms > 0:
                    time.sleep(self.args.after_tts_pause_ms / 1000.0)
                self.listening_muted = False
                write_state("listening", last_text=text, last_reply=reply)
                print("[mic] listening resumed", flush=True)
                if self.args.turn_cues:
                    self.turn_cue("resume")

    def run(self):
        if self.args.preload_stt:
            self.load_model()
        CONTROL_DIR.mkdir(parents=True, exist_ok=True)
        write_state("starting")
        threading.Thread(target=self.vad_worker, daemon=True).start()
        threading.Thread(target=self.assistant_worker, daemon=True).start()
        print(f"[audio] input_device={self.args.input_device!r} output_device={self.args.output_device!r} samplerate={self.args.samplerate}", flush=True)
        with sd.InputStream(
            samplerate=self.args.samplerate,
            channels=CHANNELS,
            dtype="float32",
            blocksize=max(1, int(self.args.samplerate * FRAME_MS / 1000)),
            device=self.args.input_device,
            callback=self.audio_callback,
        ):
            print("[ready] Voice loop is running. Ctrl+C to stop.", flush=True)
            write_state("listening")
            self.cue("start")
            while not self.stop.is_set():
                time.sleep(0.2)


def list_devices():
    print(sd.query_devices())
    print("\nDefault input/output:", sd.default.device)


def parse_device(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except ValueError:
        return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--input-device", type=parse_device, default=None, help="sounddevice input index/name; omit for default")
    ap.add_argument("--output-device", default="plughw:0,0", help="ALSA aplay device")
    ap.add_argument("--model", default="base", help="faster-whisper model: tiny/base/small")
    ap.add_argument("--vad", type=int, default=2, choices=[0,1,2,3], help="WebRTC VAD aggressiveness")
    ap.add_argument("--pre-roll-ms", type=int, default=400)
    ap.add_argument("--start-speech-ms", type=int, default=160)
    ap.add_argument("--end-silence-ms", type=int, default=2350)
    ap.add_argument("--min-speech-ms", type=int, default=350)
    ap.add_argument("--max-utterance-ms", type=int, default=30000)
    ap.add_argument("--tts-voice", default="aidar")
    ap.add_argument("--tts-python", default=str(DEFAULT_TTS_PYTHON))
    ap.add_argument("--tts-worker", default=str(DEFAULT_TTS_WORKER))
    ap.add_argument("--samplerate", type=int, default=None, help="hardware input sample rate; default=device default or 16000")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--barge-in", action="store_true", default=False)
    ap.add_argument("--no-barge-in", dest="barge_in", action="store_false")
    ap.add_argument("--level-interval", type=float, default=2.5)
    ap.add_argument("--save-utterances", default="/tmp/hermes_voice_utterances")
    ap.add_argument("--hermes-timeout", type=int, default=90)
    ap.add_argument("--llm-backend", choices=["direct", "cli"], default="cli", help="cli restores the old Hermes-aware multi-turn/tool-enabled contour; direct is fallback only")
    ap.add_argument("--llm-fallback-cli", action="store_true", help="fallback to hermes CLI if direct provider fails")
    ap.add_argument("--llm-connect-timeout", type=float, default=8.0)
    ap.add_argument("--llm-temperature", type=float, default=0.4)
    ap.add_argument("--llm-max-tokens", type=int, default=160)
    ap.add_argument("--hermes-config", default=str(HERMES_CONFIG))
    ap.add_argument("--voice-profile", default="", help="Hermes profile for full tool-enabled voice agent")
    ap.add_argument("--voice-skills", default="", help="Comma-separated Hermes skills to preload in voice mode")
    ap.add_argument("--voice-full-agent", action="store_true", help="use full Hermes rules/toolsets instead of fast low-latency voice mode")
    ap.add_argument("--append-context", default="", help="Append shared text context from chat into voice context file and exit")
    ap.add_argument("--show-voice-session", action="store_true", help="Print stored voice session id and exit")
    ap.add_argument("--ask-test", default="", help="Ask Hermes once through the voice adapter, speak the reply, and exit")
    ap.add_argument("--say-test", action="store_true", help="synthesize and play a short Russian phrase, then exit")
    ap.add_argument("--preload-stt", action="store_true", help="load Whisper before opening the mic; default loads on first utterance")
    ap.add_argument("--drop-mic-during-stt", action="store_true", default=True, help="avoid input backlog/overflow while CPU is transcribing")
    ap.add_argument("--keep-mic-during-stt", dest="drop_mic_during_stt", action="store_false")
    ap.add_argument("--debug-audio-status", action="store_true")
    ap.add_argument("--mute-mic-during-tts", action="store_true", default=True, help="prevent speaker echo from re-triggering VAD")
    ap.add_argument("--keep-mic-during-tts", dest="mute_mic_during_tts", action="store_false")
    ap.add_argument("--after-tts-pause-ms", type=int, default=80)
    ap.add_argument("--turn-cues", action="store_true", help="diagnostic beeps: accepted phrase / mic resumed")
    ap.add_argument("--accepted-cue-hz", type=int, default=660)
    ap.add_argument("--resume-cue-hz", type=int, default=1320)
    ap.add_argument("--turn-cue-ms", type=int, default=90)
    ap.add_argument("--turn-cue-volume", type=float, default=0.12)
    ap.add_argument("--cues", action="store_true", default=True, help="play start/stop listening sounds")
    ap.add_argument("--no-cues", dest="cues", action="store_false")
    ap.add_argument("--start-sound", default="/home/dark/.hermes/audio_cues/listen_start.mp3")
    ap.add_argument("--stop-sound", default="/home/dark/.hermes/audio_cues/listen_stop.mp3")
    ap.add_argument("--cue-start-hz", type=int, default=1040)
    ap.add_argument("--cue-stop-hz", type=int, default=520)
    ap.add_argument("--cue-ms", type=int, default=110)
    ap.add_argument("--cue-volume", type=float, default=0.16)
    args = ap.parse_args()
    if args.list_devices:
        list_devices()
        return
    if args.show_voice_session:
        try:
            print(VOICE_SESSION_ID_FILE.read_text(encoding="utf-8").strip())
        except Exception:
            print("")
        return
    if args.append_context:
        existing = ""
        try:
            if VOICE_CONTEXT_FILE.exists():
                existing = VOICE_CONTEXT_FILE.read_text(encoding="utf-8")
        except Exception:
            existing = ""
        merged = (existing + "\n" + args.append_context).strip()[-12000:]
        VOICE_CONTEXT_FILE.write_text(merged, encoding="utf-8")
        print("VOICE_CONTEXT_UPDATED")
        return
    if args.input_device is None:
        preferred = find_shem_boy_input()
        if preferred is not None:
            args.input_device = preferred
            print(f"[audio] auto-selected USB input device {preferred}", flush=True)
    if args.samplerate is None:
        try:
            info = sd.query_devices(args.input_device, 'input')
            args.samplerate = int(info.get('default_samplerate') or VAD_RATE)
        except Exception:
            args.samplerate = VAD_RATE
    if args.say_test:
        if args.start_sound:
            try:
                play_audio_file(args.start_sound, args.output_device)
            except Exception as e:
                print(f"[cue] say-test start sound skipped: {e}", flush=True)
        Playback(args.output_device, args.tts_voice, Path(args.tts_python), Path(args.tts_worker)).speak("Проверка голосового контура. Максим на связи.")
        return
    if args.ask_test:
        loop = VoiceLoop(args)
        reply = loop.ask_hermes(args.ask_test)
        print(reply, flush=True)
        if not args.no_tts:
            loop.playback.speak(reply)
        return
    loop = VoiceLoop(args)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[stop] interrupted", flush=True)
        loop.stop.set()
        loop.playback.stop()
        loop.cue("stop")
        write_state("stopped")

if __name__ == "__main__":
    main()

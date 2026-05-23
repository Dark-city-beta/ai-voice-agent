#!/usr/bin/env python3
"""
Hermes local voice VAD loop — variable-length utterances, Russian STT, TTS playback.

MVP:
- continuous microphone stream via sounddevice
- WebRTC VAD endpointing, no fixed 7-second chunks
- faster-whisper final transcription after utterance endpoint
- Hermes CLI call
- Edge TTS by sentence chunks
- playback via ffmpeg -> aplay
- optional barge-in: if user speaks during playback, stop current playback

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
CONTROL_MUTE_FILE = Path(__file__).resolve().parent / ".voice_mute"
CONTROL_RESET_FILE = Path(__file__).resolve().parent / ".voice_reset"


def find_shem_boy_input():
    """Prefer the known-good USB mic on this host when no input device is specified."""
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0 and "SHEM-BOY" in d.get("name", ""):
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
    # Split remaining too-long chunks on commas/spaces conservatively
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


class Playback:
    def __init__(self, output_device: str, voice: str):
        self.output_device = output_device
        self.voice = voice
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.processes: list[subprocess.Popen] = []

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

    def speak_edge(self, text: str):
        total_t0 = time.perf_counter()
        edge = shutil.which("edge-tts")
        ffmpeg = shutil.which("ffmpeg")
        aplay = shutil.which("aplay")
        if not (edge and ffmpeg and aplay):
            print(f"[tts missing] {text}", flush=True)
            return
        self.stop_event.clear()
        chunks = split_sentences_ru(text)
        if not chunks:
            return
        with tempfile.TemporaryDirectory() as td:
            ready = queue.Queue(maxsize=2)
            sentinel = object()

            def producer():
                try:
                    for i, chunk in enumerate(chunks, 1):
                        if self.stop_event.is_set():
                            break
                        mp3 = Path(td) / f"tts_{i}.mp3"
                        wav = Path(td) / f"tts_{i}.wav"
                        chunk_t0 = time.perf_counter()
                        print(f"[tts] prepare chunk {i}/{len(chunks)}: {chunk[:80]}", flush=True)
                        r = subprocess.run(
                            [edge, "--voice", self.voice, "--text", chunk, "--write-media", str(mp3)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=45,
                        )
                        print(f"[timing] tts.edge chunk={i} {time.perf_counter()-chunk_t0:.2f}s", flush=True)
                        if r.returncode != 0:
                            print("[tts] edge failed:", (r.stderr or r.stdout)[-500:], flush=True)
                            continue
                        ff_t0 = time.perf_counter()
                        r = subprocess.run(
                            [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", str(mp3), str(wav)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=20,
                        )
                        print(f"[timing] tts.ffmpeg chunk={i} {time.perf_counter()-ff_t0:.2f}s", flush=True)
                        if r.returncode != 0:
                            print("[tts] ffmpeg failed:", (r.stderr or r.stdout)[-500:], flush=True)
                            continue
                        while not self.stop_event.is_set():
                            try:
                                ready.put((i, wav), timeout=0.1)
                                break
                            except queue.Full:
                                pass
                finally:
                    while True:
                        try:
                            ready.put(sentinel, timeout=0.1)
                            break
                        except queue.Full:
                            if self.stop_event.is_set():
                                break

            threading.Thread(target=producer, daemon=True).start()
            first_audio = True
            while not self.stop_event.is_set():
                item = ready.get()
                if item is sentinel:
                    break
                i, wav = item
                play_t0 = time.perf_counter()
                if first_audio:
                    print(f"[timing] tts.first_audio {time.perf_counter()-total_t0:.2f}s", flush=True)
                    first_audio = False
                p = self._track(subprocess.Popen([aplay, "-D", self.output_device, "-q", str(wav)]))
                try:
                    while p.poll() is None:
                        if self.stop_event.is_set():
                            try: p.terminate()
                            except Exception: pass
                            break
                        time.sleep(0.03)
                finally:
                    print(f"[timing] tts.play chunk={i} {time.perf_counter()-play_t0:.2f}s", flush=True)
                    self._untrack(p)
        print(f"[timing] tts.total {time.perf_counter()-total_t0:.2f}s", flush=True)


class VoiceLoop:
    def __init__(self, args):
        self.args = args
        self.audio_q = queue.Queue(maxsize=200)
        self.utterance_q = queue.Queue()
        self.stop = threading.Event()
        self.playback = Playback(args.output_device, args.tts_voice)
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
        # diagnostic short beeps inside a dialogue turn:
        # accepted = phrase taken for processing; resume = mic is live again.
        freq = self.args.accepted_cue_hz if kind == "accepted" else self.args.resume_cue_hz
        with self.cue_lock:
            try:
                play_tone(self.args.output_device, freq=freq, duration_ms=self.args.turn_cue_ms, volume=self.args.turn_cue_volume)
            except Exception as e:
                print(f"[turn-cue] failed: {e}", flush=True)

    def set_listening_muted(self, muted: bool, reason: str = ""):
        if self.listening_muted == muted:
            return
        self.listening_muted = muted
        if muted:
            print(f"[mic] stop listening {reason}".strip(), flush=True)
            self.cue("stop")
        else:
            print("[mic] start listening", flush=True)
            self.cue("start")

    def load_model(self):
        if self.model is None:
            print(f"[stt] loading faster-whisper model={self.args.model} compute=int8...", flush=True)
            self.model = WhisperModel(self.args.model, device="cpu", compute_type="int8")

    def audio_callback(self, indata, frames, time_info, status):
        if status and self.args.debug_audio_status:
            print(f"[audio status] {status}", flush=True)
        if self.listening_muted or CONTROL_MUTE_FILE.exists():
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
        # WebRTC VAD accepts only exact 10/20/30 ms frames. Some hardware/resampler
        # combinations can produce +/- samples, so normalize here instead of letting
        # vad_worker explode mid-dialogue.
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
            # Drop backlog rather than increasing live dialogue latency.
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
            if CONTROL_RESET_FILE.exists():
                CONTROL_RESET_FILE.unlink(missing_ok=True)
                while True:
                    try:
                        self.audio_q.get_nowait()
                    except Exception:
                        break
                pre.clear(); candidate.clear(); speech = []
                triggered = False
                speech_count = silence_count = consecutive_start = 0
                print("[vad] reset by control", flush=True)
                continue
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
            if self.args.save_utterances:
                out = Path(self.args.save_utterances)
                out.mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                write_wav(out / f"utt_{ts}.wav", pcm)
            return text

    def _voice_messages(self, text: str):
        return [
            {"role": "system", "content": "Отвечай по-русски устно: 1 короткое предложение, максимум 20 слов, без Markdown, списков, эмодзи и ссылок."},
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

    def ask_cli_hermes(self, text: str) -> str:
        prompt = (
            "Ответь по-русски устно: 1 короткое предложение, максимум 20 слов, без Markdown, списков, эмодзи и ссылок. "
            f"Реплика DARK: {text}"
        )
        env = os.environ.copy()
        env.setdefault("NO_COLOR", "1")
        env.setdefault("TERM", "dumb")
        try:
            r = subprocess.run(
                ["hermes", "chat", "-q", prompt, "--source", "local-voice", "--quiet"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.args.hermes_timeout,
                env=env,
                start_new_session=True,
            )
        except subprocess.TimeoutExpired:
            return "Завис вызов Hermes. Я сбросил этот ход, повтори фразу короче."
        if r.returncode != 0:
            return "Ошибка Hermes: " + (r.stderr or r.stdout)[-700:]
        return r.stdout.strip()

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
            print("[mic] accepted phrase, processing", flush=True)
            if self.args.turn_cues:
                self.turn_cue("accepted")
            text = self.transcribe(pcm)
            if not text:
                self.listening_muted = False
                print("[mic] listening resumed", flush=True)
                if self.args.turn_cues:
                    self.turn_cue("resume")
                continue
            print(f"\n[DARK] {text}\n", flush=True)
            reply = self.ask_hermes(text)
            print(f"\n[Максим] {reply}\n", flush=True)
            if self.args.no_tts:
                self.listening_muted = False
                print("[mic] listening resumed", flush=True)
                if self.args.turn_cues:
                    self.turn_cue("resume")
                continue
            self.assistant_speaking = True
            try:
                self.playback.speak_edge(reply)
            finally:
                self.assistant_speaking = False
                if self.args.after_tts_pause_ms > 0:
                    time.sleep(self.args.after_tts_pause_ms / 1000.0)
                self.listening_muted = False
                print("[mic] listening resumed", flush=True)
                if self.args.turn_cues:
                    self.turn_cue("resume")

    def run(self):
        if self.args.preload_stt:
            self.load_model()
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
    ap.add_argument("--tts-voice", default="ru-RU-DmitryNeural")
    ap.add_argument("--samplerate", type=int, default=None, help="hardware input sample rate; default=device default or 16000")
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--barge-in", action="store_true", default=False)
    ap.add_argument("--no-barge-in", dest="barge_in", action="store_false")
    ap.add_argument("--level-interval", type=float, default=2.5)
    ap.add_argument("--save-utterances", default="/tmp/hermes_voice_utterances")
    ap.add_argument("--hermes-timeout", type=int, default=20)
    ap.add_argument("--llm-backend", choices=["direct", "cli"], default="direct", help="direct provider API avoids spawning hermes CLI each turn")
    ap.add_argument("--llm-fallback-cli", action="store_true", help="fallback to slow hermes CLI if direct provider fails")
    ap.add_argument("--llm-connect-timeout", type=float, default=8.0)
    ap.add_argument("--llm-temperature", type=float, default=0.4)
    ap.add_argument("--llm-max-tokens", type=int, default=64)
    ap.add_argument("--hermes-config", default=str(HERMES_CONFIG))
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
    if args.input_device is None:
        preferred = find_shem_boy_input()
        if preferred is not None:
            args.input_device = preferred
            print(f"[audio] auto-selected SHEM-BOY input device {preferred}", flush=True)
    if args.samplerate is None:
        try:
            info = sd.query_devices(args.input_device, 'input')
            args.samplerate = int(info.get('default_samplerate') or VAD_RATE)
        except Exception:
            args.samplerate = VAD_RATE
    if args.say_test:
        play_audio_file(args.start_sound, args.output_device)
        Playback(args.output_device, args.tts_voice).speak_edge("Проверка голосового контура. Максим на связи.")
        return
    loop = VoiceLoop(args)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[stop] interrupted", flush=True)
        loop.stop.set()
        loop.playback.stop()
        loop.cue("stop")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Long-lived Silero TTS worker for Hermes local voice.

The main voice loop runs in the Hermes venv because it has audio/STT deps.
Torch is available in the Cat-books venv, so this worker stays there, keeps the
Silero model in memory, and accepts one JSON request per line on stdin.
"""
import json
import os
import sys
import traceback
from pathlib import Path

import torch


DEFAULT_MODEL_PATH = "/home/dark/.cache/torch/hub/snakers4_silero-models_master/src/silero/model/v4_ru.pt"


def emit(payload):
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def main():
    model_path = Path(os.environ.get("SILERO_MODEL_PATH", DEFAULT_MODEL_PATH))
    if not model_path.exists():
        emit({"ok": False, "event": "ready", "error": f"missing model: {model_path}"})
        return 2

    try:
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
    except Exception:
        pass

    try:
        importer = torch.package.PackageImporter(str(model_path))
        model = importer.load_pickle("tts_models", "model")
        model.to(torch.device("cpu"))
        emit({"ok": True, "event": "ready", "model": str(model_path)})
    except Exception as exc:
        emit({"ok": False, "event": "ready", "error": repr(exc)})
        traceback.print_exc(file=sys.stderr)
        return 3

    for raw in sys.stdin:
        raw = raw.strip()
        if not raw:
            continue
        try:
            req = json.loads(raw)
            if req.get("cmd") == "stop":
                emit({"ok": True, "event": "stopped"})
                return 0

            text = (req.get("text") or "").strip()
            speaker = req.get("speaker") or "aidar"
            sample_rate = int(req.get("sample_rate") or 48000)
            out = Path(req["audio_path"])
            out.parent.mkdir(parents=True, exist_ok=True)
            if not text:
                emit({"ok": False, "error": "empty text"})
                continue

            model.save_wav(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate,
                audio_path=str(out),
                put_accent=True,
                put_yo=True,
            )
            emit({"ok": True, "path": str(out)})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit({"ok": False, "error": repr(exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

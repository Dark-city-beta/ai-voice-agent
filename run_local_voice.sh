#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec env PYTHONUNBUFFERED=1 python3 ./local_voice_vad_loop.py \
  --input-device 7 \
  --samplerate 44100 \
  --model base \
  --preload-stt \
  --output-device plughw:0,0 \
  --level-interval 2.0 \
  --turn-cues \
  --turn-cue-ms 180 \
  --turn-cue-volume 0.25 \
  --llm-backend direct

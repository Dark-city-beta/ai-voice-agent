#!/usr/bin/env bash
set -euo pipefail
export PATH="/home/dark/.hermes/hermes-agent/venv/bin:/home/dark/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export TELEGRAM_PROXY="${TELEGRAM_PROXY:-socks5h://127.0.0.1:10808}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:10809}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:10809}"
export NO_PROXY="${NO_PROXY:-localhost,127.0.0.1,192.168.0.0/16,10.0.0.0/8}"
exec env PYTHONUNBUFFERED=1 /home/dark/.hermes/hermes-agent/venv/bin/python /home/dark/.hermes/scripts/local_voice_vad_loop.py \
  --input-device 7 \
  --samplerate 48000 \
  --model base \
  --output-device plughw:0,0 \
  --level-interval 1.0 \
  --llm-backend cli \
  --hermes-timeout 55 \
  --tts-python "/mnt/city17/Free project/Cat-books/catbooks-2.0/.venv-cpu/bin/python" \
  --tts-worker /home/dark/.hermes/scripts/silero_tts_worker.py \
  "$@"

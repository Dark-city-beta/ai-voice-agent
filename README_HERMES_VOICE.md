# Hermes Voice Bridge v0.3

Текущая рабочая сборка локальной говорилки Hermes для DARK.

## Что внутри

- `local_voice_vad_loop.py` — основная новая говорилka.
- `voice_bridge_hermes_v0.3.py` — стабильный wrapper entrypoint.
- `mic_calibrate_vad.py` — калибровка микрофона/VAD.
- `run_local_voice.sh` — быстрый запуск на текущей машине.

## Текущая архитектура

- Микрофон: USB `SHEM-BOY`, sounddevice input `7`.
- Захват: 44100 Hz, ресемпл в 16000 Hz для WebRTC VAD.
- Endpointing: переменная длина фразы, без фиксированных чанков.
- STT: `faster-whisper`, русский язык.
- LLM: прямой OpenAI-compatible provider API из `~/.hermes/config.yaml`, без запуска `hermes chat` на каждый ход.
- TTS: Edge TTS, ALSA playback через `aplay -D plughw:0,0`.
- UX: стартовый MP3 при запуске прослушки, финальный MP3 при остановке, без системных звуков между обычными репликами.

## Основные оптимизации v0.3

- Убран тяжёлый per-turn `hermes chat -q` CLI из горячего голосового контура.
- Добавлен `--llm-backend direct` по умолчанию.
- Ответы ограничены для голосового режима: одно короткое предложение.
- Markdown/emoji/спецсимволы очищаются перед TTS.
- TTS начинает первый готовый chunk без ожидания полной склейки ответа.
- Пауза отбивки речи увеличена, чтобы не резать мысли пользователя.

## Быстрый запуск

```bash
./run_local_voice.sh
```

Или явно:

```bash
python3 local_voice_vad_loop.py \
  --input-device 7 \
  --samplerate 44100 \
  --model base \
  --preload-stt \
  --output-device plughw:0,0 \
  --llm-backend direct
```

## Калибровка микрофона

```bash
python3 mic_calibrate_vad.py --input-device 7 --seconds 30 --vad 2
```

## Важные замечания

- Файл читает `~/.hermes/config.yaml` для provider API. Секреты не должны попадать в репозиторий.
- Текущий главный резерв ускорения: заменить Edge TTS на локальный Piper/Silero или streaming TTS.
- Токены GitHub/API не хранить в репозитории.

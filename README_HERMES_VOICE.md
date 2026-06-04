# Hermes Govorilka 0.3

Локальная голосовая говорилка для Hermes Agent: микрофон на Ubuntu-сервере, локальный STT, локальный Silero TTS, колонки и LAN-пульт с Windows.

## Что внутри

- `local_voice_vad_loop.py` - основной голосовой цикл.
- `silero_tts_worker.py` - долгоживущий offline Silero TTS worker.
- `run_local_voice.sh` - запуск голосового цикла на Ubuntu.
- `voice_controller_server.py` - LAN HTTP API для Windows-пульта.
- `voice_control_server.py` - совместимое имя того же control API.
- `windows/GovorilkaTray/GovorilkaTray.exe` - готовый Windows-пульт.

## Архитектура

```text
Windows GovorilkaTray.exe
        |
        | LAN HTTP :8766
        v
Ubuntu voice_controller_server.py
        |
        v
local_voice_vad_loop.py
        |
        +-- USB mic -> WebRTC VAD -> faster-whisper -> Hermes CLI
        +-- Hermes reply -> Silero TTS worker -> ALSA speakers
```

## Основные решения 0.3

- Убран Edge/Microsoft TTS из горячего пути.
- Silero `v4_ru.pt` работает локально и держится в памяти отдельным worker-процессом.
- Основной Python берётся из Hermes venv, где есть `sounddevice`, `webrtcvad`, `faster-whisper`, `fastapi`, `uvicorn`.
- Torch берётся из отдельного Cat-books venv через `silero_tts_worker.py`, чтобы не ломать окружение Hermes.
- USB-микрофон работает на `48000 Hz`; дальше аудио ресемплится в `16000 Hz` для VAD/STT.
- Быстрый голосовой режим Hermes использует `hermes chat --quiet --ignore-rules --max-turns 1`, чтобы бытовые голосовые реплики не ждали полный агентный контур.
- Технические ошибки Hermes пишутся в лог, но не читаются вслух.
- Фильтруются типовые hallucinations Whisper на тишине: `спасибо за просмотр`, `подписывайтесь`, `продолжение следует`, `thanks for watching`.

## Ubuntu запуск

Пример с текущего домашнего сервера:

```bash
./run_local_voice.sh
```

Ручной эквивалент:

```bash
/home/dark/.hermes/hermes-agent/venv/bin/python local_voice_vad_loop.py \
  --input-device 7 \
  --samplerate 48000 \
  --model base \
  --output-device plughw:0,0 \
  --level-interval 1.0 \
  --llm-backend cli \
  --hermes-timeout 55 \
  --tts-python "/mnt/city17/Free project/Cat-books/catbooks-2.0/.venv-cpu/bin/python" \
  --tts-worker /home/dark/.hermes/scripts/silero_tts_worker.py
```

## Control API

Запуск:

```bash
/home/dark/.hermes/hermes-agent/venv/bin/python -m uvicorn voice_controller_server:APP --host 0.0.0.0 --port 8766
```

Основные команды:

```bash
curl http://127.0.0.1:8766/status
curl -X POST http://127.0.0.1:8766/voice/start
curl -X POST http://127.0.0.1:8766/voice/stop
curl -X POST http://127.0.0.1:8766/mic/mute
curl -X POST http://127.0.0.1:8766/mic/unmute
curl -X POST http://127.0.0.1:8766/speaker/mute
curl -X POST http://127.0.0.1:8766/speaker/unmute
```

Не выставляй этот API в интернет. Это LAN-пульт.

## Windows пульт

Готовая сборка:

```text
windows/GovorilkaTray/GovorilkaTray.exe
```

Запуск:

```text
windows/GovorilkaTray/Start-GovorilkaTray.cmd
```

Хоткеи:

- `Ctrl+Alt+G` - включить/выключить говорилку.
- `Ctrl+Alt+M` - mute/unmute микрофон.
- `Ctrl+Alt+S` - mute/unmute колонки.

Адрес сервера меняется в:

```text
windows/GovorilkaTray/config.json
```

## Проверки

Список аудиоустройств:

```bash
/home/dark/.hermes/hermes-agent/venv/bin/python local_voice_vad_loop.py --list-devices
```

Тест локального TTS:

```bash
./run_local_voice.sh --say-test --start-sound ""
```

Тест Hermes без озвучки:

```bash
./run_local_voice.sh --ask-test "проверка связи, ответь коротко" --no-tts
```

## Важно

Секреты, API keys и GitHub tokens не должны попадать в репозиторий.

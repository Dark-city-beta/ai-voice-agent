# AI Voice Agent: Hermes Govorilka 0.3

Говорилка 0.3 - локальный голосовой контур для Hermes Agent без платного TTS:

- Ubuntu-сервер слушает микрофон.
- `faster-whisper` распознаёт русскую речь локально.
- Hermes отвечает через CLI-адаптер.
- Offline Silero озвучивает ответ через колонки.
- Windows-пульт управляет стартом, стопом, микрофоном и колонками по локальной сети.

Подробности:

- [Hermes Govorilka 0.3](README_HERMES_VOICE.md)
- [Windows Govorilka Tray](README_WINDOWS_TRAY.md)

## Быстрый старт Ubuntu

```bash
./run_local_voice.sh
```

## Быстрый старт Windows

```text
windows/GovorilkaTray/Start-GovorilkaTray.cmd
```

## LAN API

```bash
curl http://192.168.31.200:8766/status
curl -X POST http://192.168.31.200:8766/voice/start
curl -X POST http://192.168.31.200:8766/voice/stop
```

## Безопасность

Не публикуй control API в интернет. Это локальный LAN-пульт.

Не коммить токены, API keys и приватные ключи.

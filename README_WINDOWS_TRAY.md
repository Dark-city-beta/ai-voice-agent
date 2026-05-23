# Windows Tray Controller для говорилки

Компактный Windows 11 tray-клиент для управления Linux/Hermes говорилкой по локальной сети.

## Что умеет

- Включить говорилку.
- Выключить говорилку.
- Mute микрофона.
- Unmute микрофона.
- Показывает статус цветом в трее:
  - зелёный: говорилка включена;
  - серый: выключена;
  - жёлтый: mute;
  - красный: нет связи.

## Архитектура

- На Linux/Hermes машине запускается `voice_control_server.py`.
- На Windows запускается `HermesVoiceTray.exe`.
- Windows-клиент ходит в Linux API по LAN:
  - `GET /health`
  - `POST /start`
  - `POST /stop`
  - `POST /mute`
  - `POST /unmute`

Текущий IP Linux-хоста по умолчанию:

```text
http://192.168.31.200:8766
```

Если IP изменится, в трее: `Настройки` → поменять URL.

## Запуск Linux control server

На Linux/Hermes машине из папки проекта:

```bash
python3 voice_control_server.py --host 0.0.0.0 --port 8766
```

Опционально можно задать простой LAN-token:

```bash
VOICE_CONTROL_TOKEN=your-local-token python3 voice_control_server.py --host 0.0.0.0 --port 8766
```

Тогда в Windows tray app надо открыть `Настройки` и вписать этот token.

Проверка:

```bash
curl http://127.0.0.1:8766/health
```

## Сборка EXE на Windows 11

1. Скопировать репозиторий/папку на Windows.
2. Запустить:

```bat
build_windows_tray.bat
```

3. Готовый файл будет здесь:

```text
dist\HermesVoiceTray.exe
```

## Настройки клиента

Файл настроек создаётся автоматически:

```text
%APPDATA%\HermesVoiceTray\config.json
```

Пример:

```json
{
  "server": "http://192.168.31.200:8766"
}
```

## Важно

Не выставлять `voice_control_server.py` в интернет. Это LAN-контроллер, не публичный API.

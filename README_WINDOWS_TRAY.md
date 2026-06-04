# Windows Govorilka Tray

Готовый Windows-пульт для управления Ubuntu-сервером с Hermes Govorilka 0.3.

## Где лежит

```text
windows/GovorilkaTray/GovorilkaTray.exe
windows/GovorilkaTray/Start-GovorilkaTray.cmd
windows/GovorilkaTray/config.json
```

## Запуск

Двойной клик:

```text
Start-GovorilkaTray.cmd
```

Или напрямую:

```text
GovorilkaTray.exe
```

При запуске открывается маленькое окно с кнопками. Окно можно свернуть в трей.

## Кнопки

- `Включить / выключить`
- `Микрофон mute`
- `Колонки mute`
- `Обновить статус`
- `Свернуть в трей`
- `Выход`

## Горячие клавиши

- `Ctrl+Alt+G` - включить/выключить говорилку.
- `Ctrl+Alt+M` - mute/unmute микрофон.
- `Ctrl+Alt+S` - mute/unmute колонки.

## Настройка IP

По умолчанию:

```json
{
  "server": "http://192.168.31.200:8766"
}
```

Если сервер переехал, измени `config.json` рядом с `.exe`.

## Требования

Пульт собран как `.NET Framework` WinForms-приложение без сторонних библиотек. На обычной Windows 10/11 должен запускаться без установки Python.

Если Windows SmartScreen предупреждает о неизвестном приложении, это ожидаемо: файл собран локально и не подписан.

#!/usr/bin/env python3
"""Windows tray controller for Hermes Voice Bridge."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pystray
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import messagebox

APP_DIR = Path.home() / "AppData" / "Roaming" / "HermesVoiceTray"
CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_CONFIG = {"server": "http://192.168.31.200:8766", "token": ""}

state = {"running": False, "muted": False, "last_error": "", "server": DEFAULT_CONFIG["server"], "token": ""}
icon: pystray.Icon | None = None


def load_config() -> dict:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
        return dict(DEFAULT_CONFIG)
    try:
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        return {**DEFAULT_CONFIG, **cfg}
    except Exception:
        return dict(DEFAULT_CONFIG)


def save_config(cfg: dict) -> None:
    APP_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def request(method: str, path: str, timeout: float = 5.0) -> dict:
    url = state["server"].rstrip("/") + path
    req = urllib.request.Request(url, method=method)
    if state.get("token"):
        req.add_header("X-Voice-Token", state["token"])
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def refresh() -> None:
    try:
        data = request("GET", "/health", timeout=2.0)
        state.update({"running": bool(data.get("running")), "muted": bool(data.get("muted")), "last_error": ""})
    except Exception as e:
        state.update({"running": False, "last_error": str(e)})
    update_icon()


def call_action(path: str) -> None:
    try:
        data = request("POST", path, timeout=8.0)
        state.update({"running": bool(data.get("running")), "muted": bool(data.get("muted")), "last_error": ""})
    except Exception as e:
        state["last_error"] = str(e)
        messagebox.showerror("Hermes Voice", f"Ошибка связи с Linux-сервером:\n{e}")
    update_icon()


def make_icon() -> Image.Image:
    color = (50, 180, 80) if state["running"] else (140, 140, 140)
    if state.get("muted"):
        color = (220, 170, 40)
    if state.get("last_error"):
        color = (200, 50, 50)
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((8, 8, 56, 56), fill=color)
    d.rectangle((28, 18, 36, 42), fill=(255, 255, 255))
    d.arc((20, 28, 44, 52), 0, 180, fill=(255, 255, 255), width=4)
    if state.get("muted"):
        d.line((16, 16, 48, 48), fill=(200, 0, 0), width=5)
    return img


def title() -> str:
    if state.get("last_error"):
        return "Hermes Voice: нет связи"
    if state["running"] and state["muted"]:
        return "Hermes Voice: включена, микрофон muted"
    if state["running"]:
        return "Hermes Voice: включена"
    return "Hermes Voice: выключена"


def update_icon() -> None:
    if icon:
        icon.icon = make_icon()
        icon.title = title()
        icon.update_menu()


def open_settings() -> None:
    root = tk.Tk()
    root.title("Hermes Voice settings")
    root.geometry("430x130")
    root.resizable(False, False)
    tk.Label(root, text="Linux control server URL:").pack(anchor="w", padx=10, pady=(10, 0))
    var = tk.StringVar(value=state["server"])
    entry = tk.Entry(root, textvariable=var, width=55)
    entry.pack(padx=10, pady=5)
    tk.Label(root, text="Optional control token:").pack(anchor="w", padx=10)
    token_var = tk.StringVar(value=state.get("token", ""))
    token_entry = tk.Entry(root, textvariable=token_var, width=55, show="*")
    token_entry.pack(padx=10, pady=5)

    def save():
        state["server"] = var.get().strip() or DEFAULT_CONFIG["server"]
        state["token"] = token_var.get().strip()
        save_config({"server": state["server"], "token": state["token"]})
        root.destroy()
        refresh()

    tk.Button(root, text="Сохранить", command=save).pack(pady=8)
    root.mainloop()


def background_poll() -> None:
    while True:
        refresh()
        time.sleep(5)


def menu() -> pystray.Menu:
    return pystray.Menu(
        pystray.MenuItem(lambda item: title(), None, enabled=False),
        pystray.MenuItem("Включить говорилку", lambda: call_action("/start"), enabled=lambda item: not state["running"]),
        pystray.MenuItem("Выключить говорилку", lambda: call_action("/stop"), enabled=lambda item: state["running"]),
        pystray.MenuItem("Mute микрофон", lambda: call_action("/mute"), enabled=lambda item: state["running"] and not state["muted"]),
        pystray.MenuItem("Unmute микрофон", lambda: call_action("/unmute"), enabled=lambda item: state["running"] and state["muted"]),
        pystray.MenuItem("Обновить статус", lambda: refresh()),
        pystray.MenuItem("Настройки", lambda: open_settings()),
        pystray.MenuItem("Выход", lambda: icon.stop() if icon else None),
    )


def main() -> None:
    global icon
    cfg = load_config()
    state["server"] = cfg["server"]
    state["token"] = cfg.get("token", "")
    icon = pystray.Icon("Hermes Voice", make_icon(), title(), menu())
    threading.Thread(target=background_poll, daemon=True).start()
    icon.run()


if __name__ == "__main__":
    main()

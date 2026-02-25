#!/usr/bin/env python3
"""
Voice Bridge — Голосовой мост для общения с OpenClaw агентом.
Микрофон → Vosk STT → OpenClaw → Silero TTS → Колонки

Товарищ Дарк, это твой голосовой интерфейс. ⚙️
"""

import json
import sys
import os
os.environ["USE_NNPACK"] = "0"

import queue
import threading
import subprocess
import time
import argparse
import signal

import sounddevice as sd
import numpy as np
import logging

# --- Логирование Ошибок ---
LOG_FILE = os.path.join(os.path.dirname(__file__), "voice_bridge_errors.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# --- Конфигурация ---
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-ru")
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # 0.5 сек блоки
SILENCE_TIMEOUT = 2.0  # секунд тишины для завершения фразы (уменьшено для быстродействия)
OPENCLAW_CMD = "openclaw"
WAKE_WORDS = ["товарищ", "компьютер", "система", "макс", "максим"]
STOP_WORDS = ["пока", "отбой", "хватит", "стоп"]
FILLERS = ["Угу...", "Секунду...", "Так...", "Понял..."]

# --- Глобальные переменные ---
audio_queue = queue.Queue()
speak_lock = threading.Lock()
speak_count_lock = threading.Lock()
speak_active_count = 0

def is_audio_muted():
    return speak_active_count > 0


def init_vosk():
    """Инициализация модели Vosk для распознавания русской речи."""
    from vosk import Model, KaldiRecognizer
    
    print("📦 Загрузка/поиск модели Vosk (ru)...")
    try:
        model = Model(model_path=VOSK_MODEL_PATH)
    except Exception as e:
        error_msg = f"Ошибка загрузки модели Vosk: {e}"
        print(f"❌ {error_msg}")
        logging.error(error_msg, exc_info=True)
        sys.exit(1)
        
    recognizer = KaldiRecognizer(model, SAMPLE_RATE)
    recognizer.SetWords(True)
    print("✅ Vosk готов к распознаванию")
    return recognizer


def init_silero():
    """Инициализация Silero TTS для синтеза русской речи."""
    import torch

    print("📦 Загрузка модели Silero TTS...")
    model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='ru',
        speaker='v4_ru'  # v4 - стабильная версия
    )
    print("✅ Silero TTS готов")
    return model

def play_beep(freq=440.0, duration=0.2):
    """Воспроизвести короткий звуковой сигнал."""
    try:
        import numpy as np
        import sounddevice as sd
        t = np.linspace(0, duration, int(48000 * duration), endpoint=False)
        beep = 0.5 * np.sin(2 * np.pi * freq * t)
        sd.play(beep.astype(np.float32), samplerate=48000)
    except Exception:
        pass


def play_filler(tts_model):
    """Озвучить короткий филлер ('Так...', 'Секунду...')."""
    import random
    import threading
    if tts_model:
        filler = random.choice(FILLERS)
        # Запускаем в отдельном потоке, чтобы не блокировать отправку запроса
        t = threading.Thread(target=speak, args=(tts_model, filler))
        t.start()

def sanitize_for_tts(text):
    """Очистить текст от символов, которые Silero TTS не умеет читать (эмодзи, латиница)."""
    import re
    # Оставляем только кириллицу, цифры и базовую пунктуацию
    return re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?\-:;—]', ' ', text)


def speak(tts_model, text):
    """Озвучить текст через Silero TTS и воспроизвести через колонки."""
    global speak_active_count
    import torch
    
    if not text.strip():
        return
    
    with speak_count_lock:
        speak_active_count += 1
        
    try:
        with speak_lock:
            # Разбиваем длинный текст на предложения для потокового воспроизведения
            sentences = split_sentences(text)
            
            for sentence in sentences:
                try:
                    sentence = sanitize_for_tts(sentence)
                    if not sentence.strip():
                        continue
                        
                    # Если в предложении нет ни одной русской буквы (только цифры '1.', '2.'), 
                    # Silero выдаст ValueError. Такие строки надо пропустить.
                    import re
                    if not re.search(r'[а-яА-ЯёЁ]', sentence):
                        continue
                    
                    # Генерируем аудио
                    audio = tts_model.apply_tts(
                        text=sentence,
                        speaker='baya',  # Женский голос (можно: aidar, baya, kseniya, xenia, eugene)
                        sample_rate=48000
                    )
                    
                    # Воспроизводим с отступом (padding), чтобы не проглатывать окончания
                    import numpy as np
                    audio_np = audio.numpy()
                    padding = np.zeros(int(48000 * 0.3), dtype=audio_np.dtype)
                    audio_np_padded = np.concatenate((audio_np, padding))
                    sd.play(audio_np_padded, samplerate=48000)
                    sd.wait()
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    error_msg = f"Ошибка TTS на фрагменте '{sentence}': {e}"
                    print(f"⚠️ {error_msg}")
                    logging.error(error_msg, exc_info=True)
                    
            import time
            time.sleep(2.5)  # Увеличенная задержка (грейс-период) после завершения речи для подавления эха

            # Очищаем очередь от аудио, которое могло "просочиться"
            while not audio_queue.empty():
                try:
                    audio_queue.get_nowait()
                except queue.Empty:
                    break

    finally:
        with speak_count_lock:
            speak_active_count -= 1


def split_sentences(text):
    """Разбить текст на предложения для потокового воспроизведения."""
    import re
    # Разбиваем по знакам препинания или переводам строк, чтобы не превысить лимит в 1000 символов
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    
    # Если предложение всё ещё слишком длинное, разбиваем по запятым или длинным пробелам
    final_sentences = []
    for s in sentences:
        if len(s) > 900:
            sub = re.split(r'(?<=[,;:—])\s+', s)
            final_sentences.extend(sub)
        else:
            final_sentences.append(s)
            
    return [s for s in final_sentences if s.strip()]


def send_to_openclaw(text):
    """Отправить текст агенту через OpenClaw CLI и получить ответ."""
    print(f"📤 Отправляю: {text}")
    
    try:
        import os
        env = os.environ.copy()
        env["OPENCLAW_TOKEN"] = "openclaw123"
        result = subprocess.run(
            [OPENCLAW_CMD, "agent", "--session-id", "voice_bridge", "--message", text, "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            env=env
        )
        
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                # Извлекаем текст из payloads
                if "payloads" in data and isinstance(data["payloads"], list) and len(data["payloads"]) > 0:
                    texts = []
                    for p in data["payloads"]:
                        if isinstance(p, dict) and "text" in p and p["text"]:
                            texts.append(p["text"])
                    reply = "\n\n".join(texts)
                else:
                    reply = data.get("reply", data.get("message", data.get("text", result.stdout)))
                return str(reply)
            except json.JSONDecodeError:
                return result.stdout.strip()
        else:
            print(f"⚠️ OpenClaw ошибка: {result.stderr}")
            return "Произошла ошибка при обработке запроса."
    except subprocess.TimeoutExpired:
        return "Превышено время ожидания ответа."
    except FileNotFoundError:
        print(f"❌ Команда {OPENCLAW_CMD} не найдена!")
        return "OpenClaw не установлен или не в PATH."


def audio_callback(indata, frames, time_info, status):
    """Callback для захвата аудио с микрофона."""
    if status:
        print(f"⚠️ Audio: {status}")
    if not is_audio_muted():  # Не слушаем пока говорим
        audio_queue.put(bytes(indata))


def list_devices():
    """Показать доступные аудиоустройства."""
    print("\n🎧 Доступные аудиоустройства:")
    print(sd.query_devices())
    print()


def main():
    parser = argparse.ArgumentParser(description="Voice Bridge — голосовой интерфейс OpenClaw")
    parser.add_argument("--list-devices", action="store_true", help="Показать аудиоустройства")
    parser.add_argument("--device", type=int, default=None, help="ID устройства ввода (микрофон)")
    parser.add_argument("--speaker", type=str, default="baya",
                        help="Голос TTS: aidar, baya, kseniya, xenia, eugene")
    parser.add_argument("--no-tts", action="store_true", help="Только текст, без голоса")
    args = parser.parse_args()
    
    if args.list_devices:
        list_devices()
        return
    
    print("=" * 50)
    print("🎙️  VOICE BRIDGE — Голосовой мост OpenClaw")
    print("=" * 50)
    print()
    
    # Инициализация компонентов
    recognizer = init_vosk()
    
    tts_model = None
    if not args.no_tts:
        tts_model = init_silero()
    
    print()
    print("🟢 Слушаю... Говорите в микрофон!")
    print("   (Ctrl+C для выхода)")
    print()
    
    # Запуск захвата аудио
    try:
        with sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            device=args.device,
            dtype="int16",
            channels=1,
            callback=audio_callback
        ):
            last_partial = ""
            silence_start = None
            accumulated_text = ""
            
            active_session = False
            
            while True:
                try:
                    data = audio_queue.get(timeout=0.1)
                except queue.Empty:
                    # Проверяем таймаут тишины
                    if accumulated_text and silence_start:
                        if time.time() - silence_start > SILENCE_TIMEOUT:
                            # Фраза завершена — обрабатываем
                            final_text = accumulated_text.strip()
                            accumulated_text = ""
                            silence_start = None
                            
                            if final_text and len(final_text) > 1:
                                print(f"\n🗣️  Вы: {final_text}")
                                
                                lower_text = final_text.lower()
                                
                                if not active_session:
                                    # Ждем ключевое слово активации
                                    if not any(w in lower_text for w in WAKE_WORDS):
                                        print(f"   [ Игнорирую: нет обращения по имени ({', '.join(WAKE_WORDS).title()}) ]")
                                        print("\n🟢 Слушаю фон...")
                                        continue
                                    else:
                                        active_session = True
                                        print("\n🔔 Сессия активирована! Теперь можно говорить без имени.")
                                else:
                                    # Сессия активна. Проверяем, не хочет ли пользователь попрощаться
                                    if any(w in lower_text for w in STOP_WORDS):
                                        active_session = False
                                        print("\n💤 Сессия завершена пользователем.")
                                        play_beep(300.0, 0.3)
                                        print("\n🟢 Слушаю фон (назовите имя для старта)...")
                                        continue
                                    
                                print("⏳ Агент обдумывает ответ... (это может занять 5-15 секунд, не выключайте)")
                                
                                # Звуковой сигнал (писк), что мы закончили слушать и начали думать
                                play_beep(440.0, 0.2)
                                
                                # Включаем звук-филлер для сглаживания задержки
                                # if tts_model and active_session:
                                #     play_filler(tts_model)
                                
                                # Получаем ответ от агента
                                reply = send_to_openclaw(final_text)
                                print(f"🤖 Агент: {reply}")
                                
                                # Озвучиваем ответ
                                if tts_model and reply:
                                    speak_thread = threading.Thread(
                                        target=speak,
                                        args=(tts_model, reply)
                                    )
                                    speak_thread.start()
                                
                                # Дожидаемся тишины перед стартом следующего слушания? Нет, просто выводим:
                                if active_session:
                                    print("\n🟢 Слушаю дальше...")
                                else:
                                    print("\n🟢 Слушаю фон (назовите имя для старта)...")
                                    
                                # Сигнал о готовности слушать новую команду
                                play_beep(880.0, 0.1)
                    continue
                
                if is_audio_muted():
                    continue
                
                # Распознаём речь
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "")
                    if text:
                        accumulated_text += " " + text
                        silence_start = time.time()
                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = partial.get("partial", "")
                    if partial_text != last_partial:
                        last_partial = partial_text
                        if partial_text:
                            print(f"\r💬 {partial_text}", end="", flush=True)
                            silence_start = time.time()
                        
    except KeyboardInterrupt:
        print("\n\n🔴 Voice Bridge остановлен. До свидания, товарищ!")
    except Exception as e:
        error_msg = f"Фатальная ошибка захвата аудио: {e}"
        print(f"❌ {error_msg}")
        logging.critical(error_msg, exc_info=True)
        print("Проверьте подключение микрофона: python3 voice_bridge.py --list-devices")
        sys.exit(1)


if __name__ == "__main__":
    main()

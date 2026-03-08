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

# --- Конфигурация ---
VOSK_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "vosk-ru")
SAMPLE_RATE = 16000
BLOCK_SIZE = 8000  # 0.5 сек блоки
SILENCE_TIMEOUT = 2.0  # секунд тишины для завершения фразы
OPENCLAW_CMD = "openclaw"
WAKE_WORDS = ["товарищ", "компьютер", "система", "макс", "максим"]
STOP_WORDS = ["пока", "отбой", "хватит", "стоп"]
TTS_SAMPLE_RATE = 24000  # Снижено с 48000 для ускорения генерации
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache_audio")

# --- Глобальные переменные ---
audio_queue = queue.Queue()
is_speaking = False  # True когда TTS воспроизводит ответ


def init_vosk():
    """Инициализация модели Vosk для распознавания русской речи."""
    from vosk import Model, KaldiRecognizer
    
    print("📦 Загрузка/поиск модели Vosk (ru)...")
    try:
        model = Model(model_path=VOSK_MODEL_PATH)
    except Exception as e:
        print(f"❌ Ошибка загрузки модели Vosk: {e}")
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
        t = np.linspace(0, duration, int(TTS_SAMPLE_RATE * duration), endpoint=False)
        beep = 0.5 * np.sin(2 * np.pi * freq * t)
        sd.play(beep.astype(np.float32), samplerate=TTS_SAMPLE_RATE)
    except Exception:
        pass


def play_cached(name):
    """Воспроизвести заранее записанный WAV файл системной фразы."""
    global is_speaking
    import wave
    path = os.path.join(CACHE_DIR, f"{name}.wav")
    if not os.path.exists(path):
        return False
    try:
        is_speaking = True
        with wave.open(path, 'r') as wf:
            rate = wf.getframerate()
            data = wf.readframes(wf.getnframes())
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32767.0
        sd.play(audio, samplerate=rate)
        sd.wait()
        is_speaking = False
        return True
    except Exception as e:
        print(f"⚠️ Ошибка воспроизведения кэша '{name}': {e}")
        is_speaking = False
        return False

def sanitize_for_tts(text):
    """Очистить текст от символов, которые Silero TTS не умеет читать (эмодзи, латиница)."""
    import re
    # Оставляем только кириллицу, цифры и базовую пунктуацию
    return re.sub(r'[^а-яА-ЯёЁ0-9\s.,!?\-:;—]', ' ', text)


def digits_to_russian_words(text):
    """Конвертировать числа в русские слова для надёжного TTS.
    
    Примеры: '15' → 'пятнадцать', '01:04' → 'ноль один ноль четыре',
             '2026' → 'две тысячи двадцать шесть'
    """
    import re
    
    ones = ['', 'один', 'два', 'три', 'четыре', 'пять', 'шесть', 'семь', 'восемь', 'девять']
    teens = ['десять', 'одиннадцать', 'двенадцать', 'тринадцать', 'четырнадцать',
             'пятнадцать', 'шестнадцать', 'семнадцать', 'восемнадцать', 'девятнадцать']
    tens = ['', '', 'двадцать', 'тридцать', 'сорок', 'пятьдесят',
            'шестьдесят', 'семьдесят', 'восемьдесят', 'девяносто']
    hundreds = ['', 'сто', 'двести', 'триста', 'четыреста', 'пятьсот',
                'шестьсот', 'семьсот', 'восемьсот', 'девятьсот']
    
    def num_to_words(n):
        if n == 0:
            return 'ноль'
        if n < 0:
            return 'минус ' + num_to_words(-n)
        
        parts = []
        if n >= 1000:
            t = n // 1000
            if t == 1:
                parts.append('одна тысяча')
            elif t == 2:
                parts.append('две тысячи')
            elif t in (3, 4):
                parts.append(ones[t] + ' тысячи')
            elif t <= 20:
                parts.append((teens[t - 10] if t >= 10 else ones[t]) + ' тысяч')
            else:
                parts.append(num_to_words(t) + ' тысяч')
            n %= 1000
        if n >= 100:
            parts.append(hundreds[n // 100])
            n %= 100
        if n >= 20:
            parts.append(tens[n // 10])
            n %= 10
        if 10 <= n <= 19:
            parts.append(teens[n - 10])
            n = 0
        if n > 0:
            parts.append(ones[n])
        
        return ' '.join(p for p in parts if p)
    
    def replace_number(match):
        num_str = match.group(0)
        try:
            n = int(num_str)
            if n > 99999:  # Слишком большие числа читаем по цифрам
                return ' '.join(ones[int(d)] if int(d) > 0 else 'ноль' for d in num_str)
            return num_to_words(n)
        except ValueError:
            return num_str
    
    # Обрабатываем время в формате HH:MM
    text = re.sub(r'(\d{1,2}):(\d{2})', lambda m:
                  num_to_words(int(m.group(1))) + ' ' + num_to_words(int(m.group(2))), text)
    
    # Заменяем оставшиеся числа
    text = re.sub(r'\d+', replace_number, text)
    
    return text


def speak(tts_model, text):
    """Озвучить текст через Silero TTS и воспроизвести через колонки."""
    global is_speaking
    import torch
    
    if not text.strip():
        return
    
    is_speaking = True
    # Разбиваем длинный текст на предложения для быстрого начала воспроизведения
    sentences = split_sentences(text)
    
    for sentence in sentences:
        try:
            sentence = sanitize_for_tts(sentence)
            # Конвертируем цифры в русские слова ДО отправки в TTS
            sentence = digits_to_russian_words(sentence)
            if not sentence.strip():
                continue
                
            # Если в предложении нет ни одной русской буквы — пропускаем
            import re
            if not re.search(r'[а-яА-ЯёЁ]', sentence):
                continue
            
            # Генерируем аудио
            audio = tts_model.apply_tts(
                text=sentence,
                speaker='baya',  # Женский голос (можно: aidar, baya, kseniya, xenia, eugene)
                sample_rate=TTS_SAMPLE_RATE
            )
            
            # Воспроизводим с отступом (padding), чтобы не проглатывать окончания
            import numpy as np
            audio_np = audio.numpy()
            padding = np.zeros(int(TTS_SAMPLE_RATE * 0.15), dtype=audio_np.dtype)
            audio_np_padded = np.concatenate((audio_np, padding))
            sd.play(audio_np_padded, samplerate=TTS_SAMPLE_RATE)
            sd.wait()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"⚠️ Ошибка TTS на фрагменте '{sentence}': {e}")
            
    is_speaking = False


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


def _is_rate_limit_response(reply_text):
    """Проверить, является ли ответ сообщением о rate-limit (а не реальным ответом)."""
    if not reply_text:
        return False
    lower = reply_text.lower()
    markers = [
        "rate limit",
        "rate_limit",
        "try again later",
        "превышен",
        "слишком много запросов",
        "too many requests",
        "повторите позже",
        "подождите",
        "quota exceeded",
        "resource exhausted",
    ]
    # Короткий ответ (< 200 символов) с маркером — скорее всего системная ошибка
    if len(reply_text) < 200 and any(m in lower for m in markers):
        return True
    return False


def send_to_openclaw(text):
    """Отправить текст агенту через OpenClaw CLI и получить ответ.
    
    При rate-limit автоматически повторяет запрос до 3 раз с задержкой.
    """
    
    # Фонетические замены: Vosk (RU) распознает английские музыкальные термины русскими буквами
    phonetic_fixes = {
        "ньюретровэйв": "retrowave", "нью ретро вэйв": "retrowave", "нью ретро вейв": "retrowave",
        "синтвейв": "synthwave", "синт вэйв": "synthwave", "синт вейв": "synthwave",
        "киберпанк": "cyberpunk",
        "дарк эмбиент": "dark ambient", "дарк амбиент": "dark ambient",
        "эмбиент": "ambient", "амбиент": "ambient",
        "хэви метал": "heavy metal", "хеви метал": "heavy metal",
        "дэт метал": "death metal", "дез метал": "death metal",
        "блэк метал": "black metal", "блет метал": "black metal", "блек метал": "black metal",
        "метал": "metal", "металл": "metal",
        "индастриал": "industrial", "индустриал": "industrial",
        "хаус": "house", "транс": "trance", "техно": "techno",
        "чилаут": "chillout", "чиллаут": "chillout", "чилхоп": "chillhop",
        "лофай": "lofi", "ло фай": "lofi",
        "инди": "indie", "блюз": "blues", "джаз": "jazz",
        "кантри": "country", "ска": "ska", "дабстеп": "dubstep",
        "драм энд бэйс": "drum and bass", "регги": "reggae",
        "фолк": "folk", "акустик": "acoustic",
        "поп": "pop", "рок": "rock",
        "хард рок": "hard rock", "хардрок": "hard rock",
        "электроника": "edm", "едм": "edm"
    }
    
    # Применяем замены к распознанному тексту
    normalized_text = text.lower()
    for ru, eng in phonetic_fixes.items():
        normalized_text = normalized_text.replace(ru, eng)
    
    print(f"📤 Отправляю: {normalized_text}")
    
    MAX_RETRIES = 3
    RETRY_DELAYS = [10, 20, 30]  # секунды между попытками
    
    for attempt in range(MAX_RETRIES):
        try:
            import os
            env = os.environ.copy()
            env["OPENCLAW_TOKEN"] = "openclaw123"
            t_start = time.time()
            result = subprocess.run(
                [OPENCLAW_CMD, "agent", "--session-id", "voice_bridge", "--message", normalized_text, "--json"],
                capture_output=True,
                text=True,
                timeout=90,
                env=env
            )
            elapsed = time.time() - t_start
            print(f"⏱️  Ответ получен за {elapsed:.1f}с")
            
            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout)
                    # Извлекаем текст из payloads (OpenClaw оборачивает в result)
                    result_data = data.get("result", data)
                    payloads = result_data.get("payloads", data.get("payloads", []))
                    if isinstance(payloads, list) and len(payloads) > 0:
                        texts = []
                        for p in payloads:
                            if isinstance(p, dict) and "text" in p and p["text"]:
                                texts.append(p["text"])
                        reply = "\n\n".join(texts)
                    else:
                        reply = data.get("reply", data.get("message", data.get("text", result.stdout)))
                    reply = str(reply) if reply else "Агент не дал ответа."
                    
                    # Проверяем: не является ли «ответ» на самом деле ошибкой rate-limit
                    if _is_rate_limit_response(reply):
                        if attempt < MAX_RETRIES - 1:
                            delay = RETRY_DELAYS[attempt]
                            print(f"⚠️ Rate-limit обнаружен в ответе (попытка {attempt + 1}/{MAX_RETRIES}). "
                                  f"Жду {delay}с...")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"⚠️ Rate-limit: все {MAX_RETRIES} попытки исчерпаны.")
                            return "API перегружен. Подождите полминуты и попробуйте снова."
                    
                    return reply
                except json.JSONDecodeError:
                    return result.stdout.strip()
            else:
                stderr = result.stderr.lower()
                # Rate-limit в stderr — ретраим
                if "rate" in stderr or "429" in stderr or "too many" in stderr:
                    if attempt < MAX_RETRIES - 1:
                        delay = RETRY_DELAYS[attempt]
                        print(f"⚠️ Rate-limit (stderr, попытка {attempt + 1}/{MAX_RETRIES}). Жду {delay}с...")
                        time.sleep(delay)
                        continue
                    else:
                        return "API перегружен. Подождите полминуты и попробуйте снова."
                
                print(f"⚠️ OpenClaw ошибка: {result.stderr}")
                if "timeout" in stderr or "timed out" in stderr:
                    return "Превышено время ожидания от API."
                return "Произошла ошибка при обработке запроса."
                
        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_DELAYS[attempt]
                print(f"⚠️ Таймаут (попытка {attempt + 1}/{MAX_RETRIES}). Жду {delay}с...")
                time.sleep(delay)
                continue
            return "Превышено локальное время ожидания ответа."
        except FileNotFoundError:
            print(f"❌ Команда {OPENCLAW_CMD} не найдена!")
            return "OpenClaw не установлен или не в PATH."
    
    return "Не удалось получить ответ после нескольких попыток."


RADIO_PLAYER_SCRIPT = os.path.expanduser("~/.openclaw/skills/radio-player/main.py")


def call_radio_player(command, **kwargs):
    """Вызвать radio_player напрямую (без OpenClaw) для мгновенного отклика."""
    args = {"command": command}
    args.update(kwargs)
    try:
        result = subprocess.run(
            [sys.executable, RADIO_PLAYER_SCRIPT, json.dumps(args)],
            capture_output=True, text=True, timeout=10
        )
        if result.stdout.strip():
            data = json.loads(result.stdout.strip())
            return data.get("text", "")
        return ""
    except Exception as e:
        print(f"⚠️ Radio player error: {e}")
        return ""


def _is_mpv_running():
    """Проверить, запущен ли mpv."""
    try:
        result = subprocess.run(["pgrep", "-f", "mpv --no-video"], capture_output=True)
        return result.returncode == 0
    except:
        return False


def handle_music_command(text, tts_model):
    """
    Проверяет, является ли фраза музыкальной командой.
    Если да — выполняет ЛОКАЛЬНО и возвращает True.
    Если нет — возвращает False (фраза уйдёт в OpenClaw).
    """
    import re
    lower = text.lower()
    
    # Фонетические замены жанров
    phonetic_fixes = {
        "ньюретровэйв": "retrowave", "нью ретро вэйв": "retrowave",
        "синтвейв": "synthwave", "синт вэйв": "synthwave",
        "киберпанк": "cyberpunk",
        "дарк эмбиент": "dark ambient", "дарк амбиент": "dark ambient", "дарк эмбиенд": "dark ambient",
        "эмбиент": "ambient", "амбиент": "ambient",
        "хэви метал": "heavy metal", "хеви метал": "heavy metal",
        "дэт метал": "death metal", "дез метал": "death metal",
        "блэк метал": "black metal", "блек метал": "black metal",
        "метал": "metal", "металл": "metal",
        "индастриал": "industrial", "индустриал": "industrial",
        "хаус": "house", "транс": "trance", "техно": "techno",
        "чилаут": "chillout", "чиллаут": "chillout", "чилхоп": "chillhop",
        "лофай": "lofi", "ло фай": "lofi",
        "инди": "indie", "блюз": "blues", "джаз": "jazz",
        "кантри": "country", "ска": "ska", "дабстеп": "dubstep",
        "драм энд бэйс": "drum and bass", "регги": "reggae",
        "фолк": "folk", "акустик": "acoustic",
        "поп": "pop", "рок": "rock",
        "хард рок": "hard rock", "хардрок": "hard rock",
        "электроника": "edm", "едм": "edm"
    }
    
    normalized = lower
    for ru, eng in phonetic_fixes.items():
        normalized = normalized.replace(ru, eng)
    
    # --- СТОП / ВЫКЛЮЧИ МУЗЫКУ ---
    if re.search(r'(выключи|останови|убери)\s*(музыку|радио|стрим|плеер)', normalized) or \
       (normalized.strip() in ["стоп", "хватит"] and _is_mpv_running()):
        print("🎵 [LOCAL] Стоп музыки")
        call_radio_player("stop")
        play_cached("music_stop")
        return True
    
    # --- ПАУЗА ---
    # Ловим: «поставь на паузу», «пауза», «на паузу» (одно слово тоже, если mpv играет)
    if re.search(r'(поставь|постав)\s*(на\s*)?пауз', normalized) or \
       (normalized.strip() == "пауза" and _is_mpv_running()) or \
       re.search(r'^на\s+паузу?$', normalized.strip()):
        print("🎵 [LOCAL] Пауза")
        call_radio_player("pause")
        play_cached("music_pause")
        return True
    
    # --- ПРОДОЛЖАЙ ---
    if re.search(r'(продолж|сними\s*с\s*пауз|играй\s*дальше|возобнови|сними\s*паузу)', normalized):
        print("🎵 [LOCAL] Продолжить")
        call_radio_player("resume")
        play_cached("music_resume")
        return True
    
    # --- СЛЕДУЮЩАЯ / СМЕНИ СТАНЦИЮ ---
    # Vosk часто распознаёт «смени станцию» как «с менее станцию» или «с мне станцию»
    if re.search(r'(следующ|дальше|смени\s*станц|переключ|другую\s*станц|с мн\w*\s*станц|с мен\w*\s*станц|смены\s*станц|менять\s*станц|другая\s*станц|следующая\s*станц)', normalized) or \
       (normalized.strip() in ["дальше", "следующая", "следующую", "переключи"] and _is_mpv_running()):
        print("🎵 [LOCAL] Следующая станция")
        call_radio_player("next")
        play_cached("music_next")
        return True
    
    # --- ГРОМЧЕ ---
    if re.search(r'(сделай|поставь)?\s*(по)?громче', normalized):
        print("🎵 [LOCAL] Громче +10")
        call_radio_player("volume", action="up", value=10)
        play_cached("music_volume_change")
        return True
    
    # --- ТИШЕ ---
    if re.search(r'(сделай|поставь)?\s*(по)?тише', normalized):
        print("🎵 [LOCAL] Тише -10")
        call_radio_player("volume", action="down", value=10)
        play_cached("music_volume_change")
        return True
    
    # --- УСТАНОВИТЬ КОНКРЕТНУЮ ГРОМКОСТЬ ---
    vol_match = re.search(r'(громкость|накрути|поставь)\s*(?:на|громкость)?\s*(\d+)', normalized)
    if vol_match:
        vol = int(vol_match.group(2))
        print(f"🎵 [LOCAL] Громкость = {vol}")
        call_radio_player("volume", action="set", value=vol)
        play_cached("music_volume_set")
        return True
    
    if re.search(r'очень\s*тихо', normalized):
        print("🎵 [LOCAL] Громкость = 10 (фоновый)")
        call_radio_player("volume", action="set", value=10)
        play_cached("music_volume_set")
        return True
    
    # --- ИЗБРАННОЕ: ДОБАВИТЬ ---
    if re.search(r'(добавь|добав|добавить|сохрани|сохранить)\s*(эту\s*)?(станц\w*|радио\w*|музык\w*|её|ее)?\s*(в\s*)?(избранн\w*|любим\w*|сохраненн\w*)', normalized):
        print("🎵 [LOCAL] Добавить в избранное")
        call_radio_player("favorite_add")
        play_cached("music_fav_add")
        return True
    
    # --- ИЗБРАННОЕ: УДАЛИТЬ ---
    if re.search(r'(удали|удалить|убери|убрать)\s*(эту\s*)?(станц\w*|радио\w*|музык\w*|её|ее)?\s*(из\s*)?(избранн\w*|любим\w*|сохраненн\w*)', normalized):
        print("🎵 [LOCAL] Удалить из избранного")
        call_radio_player("favorite_remove")
        play_cached("music_fav_remove")
        return True
    
    # --- ВКЛЮЧИТЬ ИЗ ИЗБРАННОГО ---
    fav_specific = re.search(r'(включи|запусти)\s*(из\s*)?избранн\w*\s+(.+)', normalized)
    if fav_specific:
        station_name = fav_specific.group(3).strip()
        print(f"🎵 [LOCAL] Включить из избранного: {station_name}")
        call_radio_player("play_favorites", query=station_name)
        play_cached("music_fav_play_one")
        return True
    
    if re.search(r'(включи|запусти|играй)\s*(мои\s*)?(любим|избранн)', normalized):
        print("🎵 [LOCAL] Включить все избранные")
        call_radio_player("play_favorites")
        play_cached("music_fav_play_all")
        return True
    
    # --- ВКЛЮЧИТЬ МУЗЫКУ / РАДИО ---
    # Расширенный regex: ловит как прямые команды ("включи джаз"), 
    # так и естественные фразы ("хочу послушать музыку в стиле блюз")
    play_match = re.search(
        r'(включи|запусти|поставь|играй|врубай|врубить|врубь'
        r'|хочу\s+послушать|давай\s+послушаем|можешь\s+(?:включить|поставить|запустить)'
        r'|послушать|послушаем)'
        r'\s*(музыку|музыки|музыка|радио|стрим|станцию)?\s*(.+)?', normalized)
    if play_match:
        query_part = (play_match.group(3) or "").strip()
        # Очистка служебных слов из запроса
        query_part = re.sub(r'^(мне|пожалуйста|плиз|в жанре|жанр|сегодня|сейчас)\s*', '', query_part).strip()
        query_part = re.sub(r'^в\s+стиле\s*', '', query_part).strip()
        # Убираем повторные глаголы-триггеры в начале (Vosk: «хочу послушать музыку включи industrial»)
        query_part = re.sub(r'^(включи|запусти|поставь|играй|врубай|врубь)\s*', '', query_part).strip()
        query_part = re.sub(r'\s+сегодня$|\s+сейчас$|\s+пожалуйста$', '', query_part).strip()
        # Убираем мусорные слова, которые Vosk может добавить после жанра
        query_part = re.sub(r'\s+(включи|запусти|поставь|играй|пожалуйста|плиз|музыку|музыка|музыки|радио).*$', '', query_part).strip()
        
        if query_part and len(query_part) > 1:
            known_genres = [
                "dark ambient", "industrial", "metal", "heavy metal", "death metal", "black metal",
                "blues", "synthwave", "pop", "rock", "hard rock", "jazz", "smooth jazz",
                "country", "classical", "lofi", "chillout", "edm", "techno", "house",
                "trance", "dubstep", "drum and bass", "ambient", "indie", "alternative",
                "punk", "ska", "reggae", "soul", "funk", "disco", "80s", "90s",
                "synthpop", "folk", "acoustic", "downtempo", "psytrance", "gothic", "retrowave"
            ]
            
            is_genre = any(g in query_part for g in known_genres)
            
            if is_genre:
                print(f"🎵 [LOCAL] Включить жанр: {query_part}")
                call_radio_player("play", query=query_part)
                play_cached("music_play_genre")
            else:
                print(f"🎵 [LOCAL] Включить станцию: {query_part}")
                call_radio_player("play", query=query_part, type="name")
                play_cached("music_play_name")
            return True
    
    # Не музыкальная команда
    return False


def audio_callback(indata, frames, time_info, status):
    """Callback для захвата аудио с микрофона."""
    if status:
        print(f"⚠️ Audio: {status}")
    if not is_speaking:  # Не слушаем пока говорим
        audio_queue.put(bytes(indata))


def list_devices():
    """Показать доступные аудиоустройства."""
    print("\n🎧 Доступные аудиоустройства:")
    print(sd.query_devices())
    print()


PID_FILE = "/tmp/voice_bridge.pid"


def _check_pid_lock():
    """Проверить PID-lock: не запущен ли уже другой экземпляр."""
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                old_pid = int(f.read().strip())
            # Проверяем, жив ли процесс
            os.kill(old_pid, 0)  # signal 0 = проверка существования
            print(f"❌ Voice Bridge уже запущен (PID {old_pid}). Выход.")
            print(f"   Для принудительного перезапуска: pkill -f voice_bridge.py")
            sys.exit(1)
        except (ProcessLookupError, ValueError):
            # Процесс мёртв — удаляем stale lock
            os.remove(PID_FILE)
        except PermissionError:
            # Процесс жив, но другого пользователя
            print(f"❌ Voice Bridge уже запущен другим пользователем. Выход.")
            sys.exit(1)
    
    # Записываем наш PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))


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
    
    # Защита от запуска нескольких экземпляров
    _check_pid_lock()
    
    print("=" * 50)
    print("🎙️  VOICE BRIDGE — Голосовой мост OpenClaw")
    print("=" * 50)
    print()
    
    # Инициализация компонентов
    recognizer = init_vosk()
    
    tts_model = None
    if not args.no_tts:
        tts_model = init_silero()
    
    # Очищаем историю сессии и локи при старте — чистый контекст = быстрые ответы
    session_file = os.path.expanduser("~/.openclaw/agents/main/sessions/voice_bridge.jsonl")
    session_lock  = session_file + ".lock"
    # Удаляем lock-файл физически (os.remove), а не обнуляем!
    # OpenClaw проверяет НАЛИЧИЕ файла, обнуление не помогает.
    try:
        if os.path.exists(session_lock):
            os.remove(session_lock)
            print("🔓 Lock-файл удалён")
    except Exception as e:
        print(f"⚠️ Не удалось удалить lock: {e}")
    # Сессию обнуляем (нам нужен пустой файл, а не отсутствие)
    try:
        open(session_file, 'w').close()
    except Exception:
        pass
    print("🗑️  История сессии очищена (временная сессия)")
    # Пауза: даём openclaw-gateway время заметить изменения
    time.sleep(2)

    print()
    print("🟢 Слушаю... Говорите в микрофон!")
    print("   (Ctrl+C для выхода)")
    print()
    
    # Аудио-активация будет после прогрева — чтобы пользователь слышал сигнал
    # только когда система реально готова к диалогу.
    
    # Синхронный прогрев OpenClaw/Gemini — загружаем OAuth, системный промпт и историю сессии
    # ДО начала прослушивания. Первый запрос к Gemini всегда медленный (~30-60с),
    # но так мы убираем это ожидание из реального диалога.
    print("🔥 Прогрев OpenClaw (загрузка системного промпта)...")
    if tts_model:
        play_cached("music_volume_change")  # Короткий звук "Принял" как индикатор
    
    warmup_ok = False
    for warmup_attempt in range(3):
        try:
            # Перед каждой попыткой удаляем lock если он появился
            if os.path.exists(session_lock):
                try:
                    os.remove(session_lock)
                    print(f"🔓 Lock удалён перед попыткой {warmup_attempt + 1}")
                    time.sleep(1)
                except Exception:
                    pass
            
            env = os.environ.copy()
            env["OPENCLAW_TOKEN"] = "openclaw123"
            warmup_result = subprocess.run(
                [OPENCLAW_CMD, "agent", "--session-id", "voice_bridge", "--message", "ping", "--json"],
                capture_output=True, text=True, timeout=90, env=env
            )
            if warmup_result.returncode == 0:
                print("✅ OpenClaw прогрет — система готова к диалогу!")
                warmup_ok = True
                break
            else:
                err = warmup_result.stderr
                if "locked" in err.lower() and warmup_attempt < 2:
                    print(f"⚠️ Прогрев: lock-конфликт, жду 5с (попытка {warmup_attempt + 1}/3)...")
                    time.sleep(5)
                    continue
                print(f"⚠️ Прогрев завершился с ошибкой (попытка {warmup_attempt + 1}/3)")
                if warmup_attempt == 2:
                    print("   Продолжаем без прогрева.")
        except subprocess.TimeoutExpired:
            print(f"⚠️ Прогрев: таймаут 90с (попытка {warmup_attempt + 1}/3)")
            if warmup_attempt == 2:
                print("   Продолжаем без прогрева.")
        except Exception as e:
            print(f"⚠️ Прогрев не удался: {e}")
            break
    
    if not warmup_ok:
        print("⚠️ Прогрев не выполнен — первый запрос может быть медленнее.")
    
    # Сигнал готовности — теперь можно говорить!
    play_beep(880.0, 0.2)
    print("🟢 Система прогрета! Можете говорить.")
    
    # Воспроизводим «activated» ПОСЛЕ прогрева — пользователь слышит сигнал
    # только когда система реально загружена и готова отвечать.
    if tts_model:
        if not play_cached("activated"):
            speak(tts_model, "Голосовое общение активировано")
    
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
                                
                                # Ducking: приостанавливаем радио, если слышим ключевое слово (или если сессия уже активна)
                                if (not active_session and any(w in lower_text for w in WAKE_WORDS)) or active_session:
                                    try:
                                        # Отправляем SIGSTOP плееру mpv (приглушаем музыку)
                                        subprocess.run(["pkill", "-STOP", "mpv"], stderr=subprocess.DEVNULL)
                                    except Exception:
                                        pass
                                
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
                                        # Прощальная голосовая фраза — из кэша (мгновенно)
                                        if tts_model:
                                            if not play_cached("deactivated"):
                                                speak(tts_model, "Голосовое общение отключено")
                                        print("\n🟢 Слушаю фон (назовите имя для старта)...")
                                        continue
                                
                                # Сначала пробуем обработать как музыкальную команду (ЛОКАЛЬНО, без OpenClaw)
                                if handle_music_command(final_text, tts_model):
                                    # Музыкальная команда обработана мгновенно!
                                    # Снимаем mpv с паузы если нужно
                                    try:
                                        subprocess.run(["pkill", "-CONT", "mpv"], stderr=subprocess.DEVNULL)
                                    except Exception:
                                        pass
                                    if active_session:
                                        print("\n🟢 Слушаю дальше...")
                                    else:
                                        print("\n🟢 Слушаю фон (назовите имя для старта)...")
                                    play_beep(880.0, 0.1)
                                    continue
                                
                                # 🎵 Музыкальный режим: если mpv играет — игнорируем всё кроме плеерных команд
                                if _is_mpv_running():
                                    print("   [ 🎵 Музыкальный режим: фраза не является командой плеера — игнорирую ]")
                                    # Снимаем ducking
                                    try:
                                        subprocess.run(["pkill", "-CONT", "mpv"], stderr=subprocess.DEVNULL)
                                    except Exception:
                                        pass
                                    if active_session:
                                        print("\n🟢 Слушаю дальше (🎵 музыкальный режим)...")
                                    else:
                                        print("\n🟢 Слушаю фон (🎵 музыкальный режим)...")
                                    continue
                                
                                print("⏳ Агент обдумывает ответ... (это может занять 5-15 секунд, не выключайте)")
                                
                                # Звуковой сигнал (писк), что мы закончили слушать и начали думать
                                play_beep(440.0, 0.2)
                                
                                # Таймер: если ответ не приходит за 40с — озвучиваем «подожди»
                                waiting_timer = threading.Timer(40.0, lambda: play_cached("waiting_for_model"))
                                waiting_timer.daemon = True
                                waiting_timer.start()
                                
                                # Получаем ответ от агента (только НЕ-музыкальные запросы)
                                reply = send_to_openclaw(final_text)
                                
                                # Отменяем таймер (ответ получен)
                                waiting_timer.cancel()
                                print(f"🤖 Агент: {reply}")
                                
                                # Озвучиваем ответ
                                if tts_model and reply:
                                    def talk_and_resume():
                                        speak(tts_model, reply)
                                        # Снимаем процесс с паузы после окончания речи
                                        try:
                                            subprocess.run(["pkill", "-CONT", "mpv"], stderr=subprocess.DEVNULL)
                                        except Exception:
                                            pass
                                            
                                    speak_thread = threading.Thread(target=talk_and_resume)
                                    speak_thread.start()
                                else:
                                    # Если TTS отключен, снимаем с паузы сразу
                                    try:
                                        subprocess.run(["pkill", "-CONT", "mpv"], stderr=subprocess.DEVNULL)
                                    except Exception:
                                        pass
                                
                                # Дожидаемся тишины перед стартом следующего слушания? Нет, просто выводим:
                                if active_session:
                                    print("\n🟢 Слушаю дальше...")
                                else:
                                    print("\n🟢 Слушаю фон (назовите имя для старта)...")
                                    
                                # Сигнал о готовности слушать новую команду
                                play_beep(880.0, 0.1)
                    continue
                
                if is_speaking:
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
        print(f"\n❌ Ошибка: {e}")
        print("Проверьте подключение микрофона: python3 voice_bridge.py --list-devices")
    finally:
        # Очищаем сессию при выходе — не оставляем историю накапливаться
        try:
            if os.path.exists(session_lock):
                os.remove(session_lock)
        except Exception:
            pass
        try:
            open(session_file, 'w').close()
        except Exception:
            pass
        # Убиваем pending OpenClaw CLI процессы, чтобы gateway не запустил плеер после нас
        subprocess.run(["pkill", "-f", "openclaw agent --session-id voice_bridge"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "mpv --no-video"], stderr=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "while true; do mpv"], stderr=subprocess.DEVNULL)
        # Удаляем PID-lock
        try:
            os.remove(PID_FILE)
        except Exception:
            pass
        print("🗑️  Сессия очищена при выходе.")


if __name__ == "__main__":
    main()

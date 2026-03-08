#!/usr/bin/env python3
"""Скачивание модели Vosk для русского языка."""
import urllib.request
import ssl
import zipfile
import os
import time
import shutil

url = 'https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip'
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
out_zip = os.path.join(out_dir, 'vosk-ru.zip')
final_dir = os.path.join(out_dir, 'vosk-ru')
status_file = os.path.join(out_dir, 'download_status.txt')

os.makedirs(out_dir, exist_ok=True)

def write_status(msg):
    with open(status_file, 'w') as f:
        f.write(msg + '\n')
    print(msg)

ctx = ssl.create_default_context()
req = urllib.request.Request(url)
req.add_header('User-Agent', 'Mozilla/5.0 (X11; Linux x86_64)')
req.add_header('Accept', '*/*')

write_status('DOWNLOADING')

try:
    with urllib.request.urlopen(req, context=ctx, timeout=600) as resp:
        total = int(resp.headers.get('Content-Length', 0))
        downloaded = 0
        start = time.time()
        
        with open(out_zip, 'wb') as f:
            while True:
                chunk = resp.read(32768)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - start
                speed = downloaded / elapsed if elapsed > 0 else 0
                pct = (downloaded / total * 100) if total else 0
                write_status(f'DOWNLOADING {downloaded}/{total} {pct:.0f}% {speed/1024:.0f}KB/s')
    
    actual_size = os.path.getsize(out_zip)
    if actual_size < 1000000:  # < 1MB = failed
        write_status(f'FAILED: file too small ({actual_size} bytes)')
        os.remove(out_zip)
        exit(1)
    
    write_status('EXTRACTING')
    with zipfile.ZipFile(out_zip, 'r') as z:
        z.extractall(out_dir)
    
    src = os.path.join(out_dir, 'vosk-model-small-ru-0.22')
    if os.path.exists(src):
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(src, final_dir)
    
    os.remove(out_zip)
    write_status('DONE')
    print(f'Model ready at {final_dir}')

except Exception as e:
    write_status(f'FAILED: {e}')
    exit(1)

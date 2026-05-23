#!/usr/bin/env python3
"""Calibrate microphone/VAD from continuous external speech (YouTube, speaker, etc.). Supports device sample-rate resampling to 16 kHz for WebRTC VAD."""
import argparse, time, queue, math, audioop
import numpy as np
import sounddevice as sd
import webrtcvad
from scipy.signal import resample_poly

VAD_RATE=16000
FRAME_MS=20
VAD_SAMPLES=VAD_RATE*FRAME_MS//1000

def db_from_pcm(pcm):
    rms=audioop.rms(pcm,2); mx=audioop.max(pcm,2); maxp=32768.0
    db=lambda x: -999.0 if x<=0 else 20*math.log10(x/maxp)
    return db(rms), db(mx)

def parse_device(v):
    if v is None or v=='': return None
    try: return int(v)
    except ValueError: return v

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--list-devices', action='store_true')
    ap.add_argument('--input-device', type=parse_device, default=None)
    ap.add_argument('--seconds', type=int, default=60)
    ap.add_argument('--vad', type=int, default=2, choices=[0,1,2,3])
    ap.add_argument('--interval', type=float, default=1.0)
    ap.add_argument('--samplerate', type=int, default=None, help='hardware input rate; default=device default')
    args=ap.parse_args()
    if args.list_devices:
        print(sd.query_devices()); print('default', sd.default.device); return
    dev_info=sd.query_devices(args.input_device, 'input')
    hw_rate=int(args.samplerate or dev_info.get('default_samplerate') or 44100)
    block=max(1, int(hw_rate*FRAME_MS/1000))
    q=queue.Queue(maxsize=500)
    vad=webrtcvad.Vad(args.vad)
    def cb(indata, frames, time_info, status):
        mono=indata[:,0].astype(np.float32)
        if hw_rate != VAD_RATE:
            # high quality enough for VAD; 44100->16000 or 48000->16000
            if hw_rate == 48000:
                mono16=resample_poly(mono, 1, 3)
            elif hw_rate == 44100:
                mono16=resample_poly(mono, 160, 441)
            else:
                mono16=resample_poly(mono, VAD_RATE, hw_rate)
        else:
            mono16=mono
        # ensure exact 20ms chunks may drift; queue callback-sized converted chunk, then split below
        pcm=np.clip(mono16*32768,-32768,32767).astype(np.int16).tobytes()
        try: q.put_nowait(pcm)
        except queue.Full: pass
    print(f'[calib] input={args.input_device!r} seconds={args.seconds} vad={args.vad} hw_rate={hw_rate} block={block}')
    print('[calib] включи ютубера/речь рядом с микрофоном. Я считаю уровень и долю speech.')
    start=time.time(); last=start
    total=0; speech=0; rms_vals=[]; max_vals=[]; buf=b''
    with sd.InputStream(samplerate=hw_rate, channels=1, dtype='float32', blocksize=block, device=args.input_device, callback=cb):
        while time.time()-start < args.seconds:
            try: buf += q.get(timeout=0.5)
            except queue.Empty: continue
            while len(buf) >= VAD_SAMPLES*2:
                frame=buf[:VAD_SAMPLES*2]; buf=buf[VAD_SAMPLES*2:]
                total += 1
                is_speech=vad.is_speech(frame,VAD_RATE)
                if is_speech: speech += 1
                rms,mx=db_from_pcm(frame); rms_vals.append(rms); max_vals.append(mx)
            if time.time()-last >= args.interval:
                recent_n=max(1, min(len(rms_vals), int(args.interval*1000/FRAME_MS)))
                recent_rms=rms_vals[-recent_n:] or [-999]
                recent_max=max_vals[-recent_n:] or [-999]
                print(f"[calib] t={time.time()-start:5.1f}s speech={speech}/{total} {speech/max(total,1):.0%} rms_avg={sum(recent_rms)/len(recent_rms):.1f}dB max_peak={max(recent_max):.1f}dB", flush=True)
                last=time.time()
    print('[calib] done')
    print(f'[summary] frames={total} speech_ratio={speech/max(total,1):.1%} rms_avg={sum(rms_vals)/max(len(rms_vals),1):.1f}dB max_peak={max(max_vals) if max_vals else -999:.1f}dB')

if __name__=='__main__': main()

#!/usr/bin/env python3
"""Hermes local voice bridge v0.3.

Linux local voice loop for DARK:
- USB mic / sounddevice input
- WebRTC VAD endpointing with variable-length utterances
- faster-whisper Russian STT
- direct OpenAI-compatible provider API from Hermes config by default
- Edge TTS + ALSA playback

This is the current Hermes говорилка checkpoint extracted from
/home/dark/.hermes/scripts/local_voice_vad_loop.py.
"""

# The canonical working copy currently lives in local_voice_vad_loop.py in this
# repository. This wrapper keeps a stable project-facing filename.
from local_voice_vad_loop import main

if __name__ == "__main__":
    main()

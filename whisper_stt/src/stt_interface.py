import sys
from pathlib import Path
import time
import numpy as np
import soundfile as sf

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_stt import HybridSTT
from examples.live_mic_vad_stt import record_with_vad, SAMPLE_RATE

TMP_WAV = "vad_input.wav"

stt = HybridSTT()

def listen_and_transcribe():
    audio = record_with_vad()
    audio = audio / max(1e-6, np.max(np.abs(audio)))
    sf.write(TMP_WAV, audio, SAMPLE_RATE)

    start = time.time()
    result = stt.transcribe(audio_path=TMP_WAV)
    latency = time.time() - start

    return {
        "text": result["text"],
        "language": result["language"],
        "engine": result["engine"],
        "latency": latency,
    }

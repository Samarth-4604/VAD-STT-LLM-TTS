import sys
from pathlib import Path
import sounddevice as sd
import numpy as np
import soundfile as sf
import time
import torch

# --------------------------------------------------
# Path setup
# --------------------------------------------------
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hybrid_stt import HybridSTT

# --------------------------------------------------
# Audio config (Silero-compatible)
# --------------------------------------------------
SAMPLE_RATE = 16000
CHANNELS = 1

VAD_FRAME_SAMPLES = 512        # REQUIRED by Silero for 16 kHz
SILENCE_FRAMES = 15            # ~0.5 sec of silence (15 * 32ms)
TMP_WAV = "vad_input.wav"

# --------------------------------------------------
# Load Silero VAD
# --------------------------------------------------
print("Loading Silero VAD...")
vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    trust_repo=True
)
vad_model.eval()

# --------------------------------------------------
# Init STT
# --------------------------------------------------
stt = HybridSTT()

# --------------------------------------------------


def record_with_vad():
    """
    Records audio until speech ends using Silero VAD.
    """
    print("🎙️ Listening... (start speaking)")

    audio_buffer = []
    silence_counter = 0
    speaking = False

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        blocksize=VAD_FRAME_SAMPLES,
        dtype="float32"
    )

    with stream:
        while True:
            block, _ = stream.read(VAD_FRAME_SAMPLES)
            block = block[:, 0]  # mono

            audio_buffer.append(block)

            audio_tensor = torch.from_numpy(block).unsqueeze(0)

            speech_prob = vad_model(audio_tensor, SAMPLE_RATE).item()

            if speech_prob > 0.5:
                speaking = True
                silence_counter = 0
            else:
                if speaking:
                    silence_counter += 1

            if speaking and silence_counter > SILENCE_FRAMES:
                break

    print("🛑 Speech ended")

    return np.concatenate(audio_buffer)


def main():
    print("\n=== LIVE MIC STT WITH VAD (FAST MODE) ===\n")
    print("Just start speaking. Ctrl+C to exit.\n")

    while True:
        try:
            audio = record_with_vad()

            # Normalize
            audio = audio / max(1e-6, np.max(np.abs(audio)))

            sf.write(TMP_WAV, audio, SAMPLE_RATE)

            start = time.time()
            result = stt.transcribe(audio_path=TMP_WAV)
            end = time.time()

            print("-" * 60)
            print(f"Engine:   {result['engine']}")
            print(f"Language: {result['language'].upper()}")
            print(f"Text:     {result['text']}")
            print(f"Latency:  {end - start:.2f} seconds")
            print("-" * 60 + "\n")

        except KeyboardInterrupt:
            print("\nExiting VAD STT.")
            break


if __name__ == "__main__":
    main()

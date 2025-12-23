import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

fs = 16000
seconds = 5

print("Recording...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
print("Done")

audio = (audio * 32767).astype(np.int16)
write("mic_test.wav", fs, audio)
print("Saved mic_test.wav")

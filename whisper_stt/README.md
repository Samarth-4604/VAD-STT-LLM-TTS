# TeacherBot – Hybrid Speech-to-Text (STT)

This repository contains the **Speech-to-Text (STT) module** of **TeacherBot**, a local, offline multilingual voice interface designed for real-time interaction.

The system combines **OpenAI Whisper** with an **IndicSTT Malayalam model**, using **Voice Activity Detection (Silero VAD)** for automatic speech segmentation.

---

## Features

- 🎙️ Live microphone speech recognition
- ✂️ Automatic Voice Activity Detection (Silero VAD)
- 🌐 Multilingual STT
  - Malayalam (IndicSTT – experimental)
  - English (Whisper)
- 📴 Fully offline processing (no cloud APIs)
- ⚡ Low-latency interaction suitable for assistants and robots
- 🧱 Modular design for future LLM and TTS integration

---

## Architecture

Microphone
↓
Voice Activity Detection (Silero)
↓
Speech-to-Text Routing
├── IndicSTT (Malayalam – experimental)
└── Whisper (English / fallback)



### Design Notes

- VAD automatically starts and stops recording
- Whisper is used for language detection and non-Malayalam speech
- IndicSTT is optimized for clean, close-mic Malayalam speech
- Broadcast or re-recorded audio may reduce accuracy

---

## Prerequisites

- Python 3.9 – 3.12
- Linux (tested on Ubuntu)
- FFmpeg
- GPU optional (CPU-only mode supported)

---

## System Dependencies

Install required system packages:

``` 
sudo apt install ffmpeg portaudio19-dev
``` 

## Installation
Clone the repository and install Python dependencies:

bash
Copy code
``` 
git clone https://github.com/Samarth-4604/TeacherBot.git
cd TeacherBot/whisper-stt
pip install -r requirements.txt
```

## Usage
Live Microphone STT with VAD
Run the live microphone STT pipeline:

bash
Copy code
``` 
python examples/live_mic_vad_stt.py
```

## How it works

Start speaking naturally

Recording starts/stops automatically using VAD

Transcribed text is printed to the console

## Project Structure
css
```
Copy code
src/        Core STT and routing logic
examples/  Live microphone demo (VAD-based)
config/    Configuration files
tests/     Test utilities
```
## Notes on Accuracy
Malayalam STT accuracy depends heavily on microphone quality

Laptop microphones and re-recorded audio reduce performance

IndicSTT is experimental and research-oriented

Whisper-small has limited Malayalam accuracy

## Roadmap
✅ Live STT with VAD

🚧 Text-to-Speech (TTS)

🚧 LLM-based dialogue logic

🚧 Robot / Jetson deployment

## License
MIT License

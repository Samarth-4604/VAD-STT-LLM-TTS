## Architecture Overview

The system follows a sequential pipeline to process voice and generate speech:

```mermaid
graph TD
    A[Microphone] --> B[Silero VAD]
    B --> C[Hybrid STT: Whisper + IndicSTT]
    C --> D[Qwen LLM: Streaming Output]
    D --> E[Piper TTS: Sentence-level Speech]

```

##Clone the files from git
```
git clone https://github.com/Samarth-4604/VAD-STT-LLM-TTS.git
cd VAD-STT-LLM-TTS/
```
# Create a virtual environment
```
python3 -m venv venv
```
# Activate environment
```
source venv/bin/activate
```
# Install dependencies
```
pip install -r requirements.txt
```
## Model Setup (Required)

This repository does not include model weights.

### Qwen LLM
Download from Hugging Face from:
```
https://huggingface.co/Qwen
```
And place it inside the folder named qwen:
```
mkdir -p models/llm/qwen
cd models/llm/qwen
```
After this, your folder should contain files like:
config.json
model.safetensors
tokenizer.json
tokenizer.model


### Whisper STT
Models are downloaded automatically on first run.
Optional cache directory:
models/stt/whisper/

### Piper TTS
Download voice models from:
```
https://huggingface.co/rhasspy/piper-voices
```

Place voices under:
models/tts/piper/






## Configuration
The primary configuration file is located at: whisper_stt/config/config.yaml

This file controls:

Audio Parameters: Sample rate, channels, etc.

Model Selection: Choose specific Whisper or IndicSTT versions.

Language Routing: Logic for handling English vs. Malayalam.

[!IMPORTANT]

Not all configuration fields are active in the current CPU-only mode. Some options are reserved for future GPU-enabled operation.






## CPU vs GPU Execution

TeacherBot runs in **CPU-only mode by default** to ensure stability on
low-VRAM systems.

GPU (CUDA) execution is intentionally **disabled in code**, not just by
configuration. The CPU lock is enforced in the following components:

- `whisper_stt/src/hybrid_stt.py`
- `whisper_stt/src/whisper_stt.py`
- `run_qwen_teacher.py`

To enable GPU execution, the following changes are required:

1. Replace the hard-coded device assignment (`device="cpu"`) with a
   configurable device (e.g. `"cuda"`).
2. Ensure sufficient GPU memory is available for the selected models.
3. Test stability under streaming workloads.

This design prevents accidental CUDA out-of-memory errors and makes
GPU usage an explicit, deliberate choice.




## Installation & Usage

# Create a virtual environment
```
python -m venv venv
```
# Activate environment (Linux/macOS)
```
source venv/bin/activate
```
# Install dependencies
```
pip install -r requirements.txt
```




##Running the Assistant
```
python run_teacherbot_voice.py
```
Interact: Speak naturally into the microphone.

Exit: Press Ctrl+C.

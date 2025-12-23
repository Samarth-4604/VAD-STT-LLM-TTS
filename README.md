## Model Setup (Required)

This repository does not include model weights.

### Qwen LLM
Download from Hugging Face and place in:
models/llm/qwen/

### Whisper STT
Models are downloaded automatically on first run.
Optional cache directory:
models/stt/whisper/

### Piper TTS
Download voice models from:
https://huggingface.co/rhasspy/piper-voices

Place voices under:
models/tts/piper/



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


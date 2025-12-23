import whisper
import torch
import yaml
import numpy as np


class WhisperSTT:
    """
    Speech-to-Text engine using OpenAI Whisper
    FP32-forced (compatible with older whisper versions)
    """

    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        model_config = self.config['model']

        self.device = model_config['device'] if torch.cuda.is_available() else "cpu"

        print(f"Loading Whisper '{model_config['size']}' model on {self.device} (FP32 forced)...")

        # Load model normally (no dtype argument)
        self.model = whisper.load_model(
            model_config['size'],
            device=self.device
        )

        # 🔒 FORCE FP32 AFTER LOADING
        self.model = self.model.float()

        print("Model loaded successfully (FP32, stable mode)!")

    # --------------------------------------------------

    def _audio_sanity_check(self, audio: np.ndarray):
        if audio is None:
            raise ValueError("Audio is None")

        if not np.isfinite(audio).all():
            raise ValueError("Audio contains NaN or Inf")

        if np.max(np.abs(audio)) < 1e-4:
            return False

        return True

    # --------------------------------------------------

    def transcribe_file(self, audio_path, language=None):
        language = language or self.config['model']['language']
        audio_path = str(audio_path)

        audio = whisper.load_audio(audio_path)

        if not self._audio_sanity_check(audio):
            return {
                "text": "",
                "segments": [],
                "language": "unknown"
            }

        result = self.model.transcribe(
            audio_path,
            language=language,
            fp16=False,  # 🔒 HARD DISABLE FP16
            beam_size=self.config['performance']['beam_size'],
            best_of=self.config['performance']['best_of']
        )

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown")
        }

    # --------------------------------------------------

    def transcribe_array(self, audio_array, sample_rate=16000, language=None):
        language = language or self.config['model']['language']
        audio = np.asarray(audio_array, dtype=np.float32)

        if not self._audio_sanity_check(audio):
            return {
                "text": "",
                "segments": [],
                "language": "unknown"
            }

        result = self.model.transcribe(
            audio,
            language=language,
            fp16=False  # 🔒 HARD DISABLE FP16
        )

        return {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", "unknown")
        }

import os
import yaml
import whisper
import torch
from pathlib import Path




from .whisper_stt import WhisperSTT
from .indic_stt import IndicSTT


class HybridSTT:
    """
    FAST Hybrid STT system (CPU-ONLY)

    Pipeline:
    1. Whisper language detection ONLY (no decoding)
    2. If English → Whisper decode
    3. Else → IndicSTT (Malayalam)
    """

    from pathlib import Path
    
    def __init__(self, config_path=None):
        # Resolve config path safely
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "config.yaml"
    
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
    

        print("=== Initializing Hybrid STT System (CPU MODE) ===")

        # 🔒 FORCE CPU MODE
        # GPU/CUDA support is intentionally disabled by default
        # due to limited VRAM. Can be enabled by changing this.
        self.device = "cpu"

        # Load Whisper ONCE (CPU)
        print("\n1. Loading Whisper (language detect + English) [CPU] ...")
        self.whisper = WhisperSTT(config_path)
        self.whisper_model = self.whisper.model.to("cpu")

        # Load IndicSTT (CPU)
        print("\n2. Loading IndicSTT (Malayalam) [CPU] ...")
        indic_cfg = self.config["indic"]
        self.indic = IndicSTT(
            model_path=indic_cfg["model_name"],
            device="cpu"
        )

        self.current_engine = "indic"

    # --------------------------------------------------

    def _detect_language_whisper(self, audio_path):
        """
        FAST language detection using Whisper encoder only (CPU)
        """
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)

        mel = whisper.log_mel_spectrogram(audio).to("cpu")

        _, probs = self.whisper_model.detect_language(mel)
        lang = max(probs, key=probs.get)

        return lang

    # --------------------------------------------------

    def transcribe(self, audio_path=None, audio_array=None, sample_rate=16000):
        """
        Transcription logic:
        - Detect language cheaply
        - Decode ONLY with chosen engine
        """

        if audio_path is None:
            raise ValueError("audio_path is required for fast mode")

        # Step 1: FAST language detection (CPU)
        detected_lang = self._detect_language_whisper(audio_path)

        # Step 2: Routing
        if detected_lang == "en":
            print("Detected English: Using Whisper (CPU).")
            result = self.whisper.transcribe_file(audio_path)
            result["engine"] = "whisper"
            result["language"] = "en"
            self.current_engine = "whisper"
            return result

        # Step 3: Anything else → Malayalam
        print(f"Detected '{detected_lang}' (non-English): Using IndicSTT (CPU).")
        result = self.indic.transcribe(
            audio_path=audio_path,
            audio_array=None,
            sample_rate=sample_rate
        )
        result["engine"] = "indic"
        result["language"] = "ml"
        self.current_engine = "indic"
        return result

    # --------------------------------------------------

    def get_current_engine(self):
        return self.current_engine

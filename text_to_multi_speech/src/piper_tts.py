import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import yaml
import wave
from pathlib import Path
from piper import PiperVoice
import sounddevice as sd
import numpy as np
import tempfile
import os


from pathlib import Path

class PiperTTS:
    def __init__(self, config_path=None):
        # Resolve config path safely
        if config_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            config_path = base_dir / "config" / "config.yaml"

        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        
        base_dir = Path(__file__).resolve().parent.parent
        self.models_dir = base_dir / self.config['paths']['models_dir']
        
        self.models_dir.mkdir(exist_ok=True)
        
        self.current_language = self.config['default_language']
        self.voices = {}
        
        print("Piper TTS initialized")
    
    def load_voice(self, language):
        """Load a voice model for the specified language"""
        if language not in self.config['voices']:
            raise ValueError(f"Language '{language}' not configured")
        
        if language in self.voices:
            return self.voices[language]
        
        voice_config = self.config['voices'][language]
        model_name = voice_config['model']
        lang_code = voice_config['language_code']
        
        # Model path
        model_dir = self.models_dir / lang_code
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"{model_name}.onnx"
        config_path = model_dir / f"{model_name}.onnx.json"
        
        if not model_path.exists():
            print(f"Model for {language} not found. Please download it first.")
            print(f"Download from: https://huggingface.co/rhasspy/piper-voices")
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading {language} voice model...")
        voice = PiperVoice.load(str(model_path), config_path=str(config_path))
        self.voices[language] = voice
        print(f"{language} voice loaded successfully!")
        
        return voice
    
    def speak(self, text, language=None, save_to_file=None):
        """
        Synthesize speech from text
        
        Args:
            text: Text to speak
            language: Language code ('en', 'ml', 'ar')
            save_to_file: Optional path to save audio file
        """
        if language is None:
            language = self.current_language
        
        voice = self.load_voice(language)
        
        # Use synthesize_wav for better quality
        # Create temporary file or use provided filename
        if save_to_file:
            output_file = save_to_file
            temp_file = False
        else:
            fd, output_file = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            temp_file = True
        
        try:
            # Synthesize to WAV file
            with wave.open(output_file, "wb") as wav_file:
                voice.synthesize_wav(text, wav_file)
            
            if save_to_file:
                print(f"Audio saved to {save_to_file}")
            
            # Play the audio
            self._play_wav(output_file)
            
        finally:
            # Clean up temp file if created
            if temp_file and os.path.exists(output_file):
                os.unlink(output_file)
    
    def _play_wav(self, filename):
        """Play a WAV file using sounddevice"""
        with wave.open(filename, 'rb') as wf:
            sample_rate = wf.getframerate()
            n_channels = wf.getnchannels()
            n_frames = wf.getnframes()
            audio = wf.readframes(n_frames)
            
            # Convert byte data to numpy array
            audio_np = np.frombuffer(audio, dtype=np.int16)
            
            # If stereo, reshape for sounddevice
            if n_channels == 2:
                audio_np = audio_np.reshape(-1, 2)
            
            # sounddevice expects float32 in [-1, 1]
            audio_np = audio_np.astype(np.float32) / 32768.0
            
            print(f"Playing audio...")
            sd.play(audio_np, sample_rate)
            sd.wait()
    
    def set_language(self, language):
        """Set default language"""
        if language not in self.config['voices']:
            raise ValueError(f"Language '{language}' not configured")
        self.current_language = language
        print(f"Default language set to: {language}")


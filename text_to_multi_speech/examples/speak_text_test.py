import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.piper_tts import PiperTTS


def main():
    # Initialize TTS
    tts = PiperTTS()
    
    # Test English
    print("\n=== English ===")
    tts.speak("Hello! This is a test of Piper text to speech.", language="en")
    
    # Test Malayalam
    print("\n=== Malayalam ===")
    tts.speak("നമസ്കാരം. ഇത് പൈപ്പർ ടെക്സ്റ്റ് ടു സ്പീച്ച് പരീക്ഷണമാണ്.", language="ml")
    
    # Test Arabic
    print("\n=== Arabic ===")
    tts.speak("مرحبا. هذا اختبار لتحويل النص إلى كلام.", language="ar")


if __name__ == "__main__":
    main()


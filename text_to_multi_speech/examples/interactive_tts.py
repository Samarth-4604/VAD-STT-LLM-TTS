import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.piper_tts import PiperTTS
from langdetect import detect, LangDetectException


class InteractiveTTS:
    """Interactive TTS with automatic language detection"""
    
    def __init__(self):
        self.tts = PiperTTS()
        self.language_map = {
            'en': 'en',
            'ml': 'ml',
            'ar': 'ar',
        }
        print("\n=== Interactive TTS with Auto Language Detection ===")
        print("Supported languages: English, Malayalam, Arabic")
        print("Type 'quit' or 'exit' to stop\n")
    
    def detect_language(self, text):
        """Detect language from text"""
        try:
            detected = detect(text)
            
            # Map detected language to our supported languages
            if detected in self.language_map:
                return self.language_map[detected]
            elif detected.startswith('en'):
                return 'en'
            else:
                # Default to English if unsure
                print(f"Detected language '{detected}' not supported. Using English.")
                return 'en'
        except LangDetectException:
            print("Could not detect language. Using English.")
            return 'en'
    
    def run(self):
        """Run the interactive TTS loop"""
        while True:
            try:
                # Get user input
                text = input("Enter text to speak: ").strip()
                
                # Check for exit commands
                if text.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Skip empty input
                if not text:
                    continue
                
                # Detect language
                language = self.detect_language(text)
                print(f"Detected language: {language}")
                
                # Speak the text
                self.tts.speak(text, language=language)
                print()  # Empty line for readability
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")


def main():
    interactive = InteractiveTTS()
    interactive.run()


if __name__ == "__main__":
    main()


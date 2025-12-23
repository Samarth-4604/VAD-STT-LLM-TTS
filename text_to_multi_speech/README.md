# Multilanguage TTS

> Multi-language text-to-speech using Piper with automatic language detection
A modular Python wrapper for Piper TTS supporting English, Malayalam, and Arabic with automatic language detection and interactive mode.

## Features
- Multi-language support: English, Malayalam (മലയാളം), Arabic (العربية)
- Automatic language detection: Detects input language automatically
- Interactive mode: Real-time keyboard input with TTS
- High quality audio
- Modular design: Easy to extend with more languages
- Offline: Fully local, no API calls required

## Requirements
- piper-tts>=1.2.0
- numpy>=1.21.0
- sounddevice>=0.4.6
- pyyaml>=6.0
- langdetect>=1.0.9


## Installation

Clone the repository
``` 
git clone https://github.com/Harbinger-Bong/piper-tts.git
cd piper-tts
``` 

Create virtual environment
``` 
python3 -m venv penv
source penv/bin/activate
``` 
Install dependencies
``` 
pip install -r requirements.txt
``` 

### Download Voice Models
English
``` 
cd en
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
``` 

Malayalam
``` 
cd ../ml
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/meera/medium/ml_IN-meera-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ml/ml_IN/meera/medium/ml_IN-meera-medium.onnx.json
``` 

Arabic
``` 
cd ../ar
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/ar/ar_JO/kareem/medium/ar_JO-kareem-medium.onnx.json
cd ../..
``` 

## Usage


### Basic Example
``` 
python examples/speak_text.py
``` 
### Interactive Mode (Recommended)
``` 
python examples/interactive_tts.py
``` 

## License

MIT License - see LICENSE file for details.

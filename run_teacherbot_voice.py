import sys
import os
import re
from threading import Thread

from transformers import TextIteratorStreamer

# -------------------------------------------------
# Path setup
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from whisper_stt.src.stt_interface import listen_and_transcribe
from text_to_multi_speech.src.piper_tts import PiperTTS
from run_qwen_teacher import load_model

# -------------------------------------------------
# System Prompt (STRICT LANGUAGE CONTROL)
# -------------------------------------------------
SYSTEM_PROMPT = """You are TeacherBot, a strict bilingual teacher.

MANDATORY RULES:
- If the user speaks English, reply ONLY in English.
- If the user speaks Malayalam, reply ONLY in Malayalam.
- NEVER mix languages.
- NEVER translate unless explicitly asked.
- Keep answers short, clear, and suitable for voice output.
"""

# -------------------------------------------------
# Output language detection (safety net)
# -------------------------------------------------
def detect_output_language(text: str) -> str:
    # Malayalam Unicode range
    for ch in text:
        if '\u0D00' <= ch <= '\u0D7F':
            return "ml"
    return "en"

# -------------------------------------------------
# Main loop
# -------------------------------------------------
def main():
    print("=== TeacherBot Voice Assistant ===")
    print("Speak naturally. Press Ctrl+C to exit.\n")

    # Load LLM (GPU)
    model, tokenizer = load_model()

    # Init TTS (CPU)
    tts = PiperTTS()

    # Conversation memory
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            print("🎙️ Listening...")
            result = listen_and_transcribe()

            if not result or not result["text"].strip():
                continue

            user_text = result["text"]
            lang = result["language"]  # "en" or "ml"

            print(f"\n👤 User ({lang}): {user_text}")

            # ---- HARD language tag for the LLM ----
            messages.append({
                "role": "user",
                "content": f"[LANG={lang.upper()}] {user_text}"
            })

            print("🤖 Teacher: ", end="", flush=True)

            # ---- Build prompt ----
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            streamer = TextIteratorStreamer(
                tokenizer,
                skip_prompt=True,
                skip_special_tokens=True
            )

            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=256,   # 🔥 reduced for voice UX
                temperature=0.7,
                do_sample=True,
            )

            # Run generation in background
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            spoken_buffer = ""
            full_response = ""

            # ---- Stream + speak sentence-by-sentence ----
            for token in streamer:
                print(token, end="", flush=True)
                full_response += token
                spoken_buffer += token

                # Sentence boundary (English + Malayalam)
                if re.search(r"[.!?]|[।]", spoken_buffer):
                    out_lang = detect_output_language(spoken_buffer)
                    tts.speak(spoken_buffer.strip(), language=out_lang)
                    spoken_buffer = ""

            # Speak remaining fragment
            if spoken_buffer.strip():
                out_lang = detect_output_language(spoken_buffer)
                tts.speak(spoken_buffer.strip(), language=out_lang)

            print("\n" + "-" * 50)

            # Save assistant message
            messages.append({
                "role": "assistant",
                "content": full_response
            })

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    main()

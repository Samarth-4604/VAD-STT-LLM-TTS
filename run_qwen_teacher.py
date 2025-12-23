import torch
from threading import Thread
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)
from huggingface_hub import snapshot_download
from colorama import Fore, Style, init
import os

# Initialize colorama
init(autoreset=True)

# ---------------- CONFIG ----------------
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
LOCAL_MODEL_DIR = "./qwen_model_local"

SYSTEM_PROMPT = """You are Qwen Teacher, a helpful AI assistant for children.

Your goal is to explain concepts clearly and simply.

Rules:
1. Answer ONLY the current question.
2. Explain step by step in a friendly way.
3. If the question is in Malayalam, reply in Malayalam.
4. If the question is in English, reply in English.
"""

# ----------------------------------------

def load_model():
    print(f"{Fore.CYAN}Checking model files...{Style.RESET_ALL}")

    if not os.path.exists(LOCAL_MODEL_DIR) or not os.listdir(LOCAL_MODEL_DIR):
        print(f"{Fore.YELLOW}Downloading model (one-time)...{Style.RESET_ALL}")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_MODEL_DIR,
            ignore_patterns=["*.pt", "*.bin"],
        )

    print(f"{Fore.CYAN}Loading model (4-bit NF4)...{Style.RESET_ALL}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LOCAL_MODEL_DIR, trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_MODEL_DIR,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    return model, tokenizer


# ======================================================
# ✅ PROGRAMMATIC API (USED BY VOICE BOT)
# ======================================================
def ask_llm(model, tokenizer, messages, max_new_tokens=256):
    """
    Generate a response from Qwen (non-streaming).
    Used by run_teacherbot_voice.py
    """
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True,
    )

    return response.strip()


# ======================================================
# OPTIONAL: CLI MODE (KEEPING YOUR ORIGINAL FUNCTIONALITY)
# ======================================================
def main():
    print(f"{Fore.GREEN}=== Qwen TeacherBot (Local, Offline) ==={Style.RESET_ALL}")

    model, tokenizer = load_model()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"{Fore.GREEN}Teacher ready. Type 'exit' to quit.{Style.RESET_ALL}")
    print("-" * 50)

    while True:
        try:
            user_input = input(f"{Fore.BLUE}You: {Style.RESET_ALL}")
            if user_input.lower() in ["exit", "quit"]:
                print(f"{Fore.GREEN}Teacher: Bye! Keep learning.{Style.RESET_ALL}")
                break

            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            generation_kwargs = dict(
                inputs,
                streamer=streamer,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print(f"{Fore.YELLOW}Teacher: {Style.RESET_ALL}", end="", flush=True)

            response_text = ""
            for token in streamer:
                print(token, end="", flush=True)
                response_text += token

            print("\n" + "-" * 50)
            messages.append({"role": "assistant", "content": response_text})

        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    main()

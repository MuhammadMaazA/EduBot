"""
A/B comparison: Base Phi-3-mini  vs  Coding-fine-tuned adapter.

Run after finetune_coding.py has produced the adapter at:
    EduBot/checkpoints/phi3-coding-adapter/

Usage:
    source MI2US_Year2s/Interface/venv39/bin/activate
    python scripts/compare_models.py

Prints side-by-side responses for a set of coding questions so you can
judge quality at the next meeting.
"""
import os
import sys
import torch
from pathlib import Path

SCRIPT_DIR  = Path(__file__).resolve().parent
EDUBOT_ROOT = SCRIPT_DIR.parent
CACHE_DIR   = EDUBOT_ROOT / ".cache" / "huggingface"
ADAPTER_DIR = EDUBOT_ROOT / "checkpoints" / "phi3-coding-adapter"

os.environ.setdefault("HF_HOME",           str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR))

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "microsoft/Phi-3-mini-128k-instruct"

TEST_QUESTIONS = [
    "Write a Python function that checks whether a string is a palindrome.",
    "Explain what a linked list is and give a simple Python implementation.",
    "What is the difference between a stack and a queue?",
    "Write a SQL query to find the second-highest salary from an Employees table.",
    "How does binary search work? Give the code in Python.",
]


def load_base():
    print("[INFO] Loading base Phi-3-mini (no adapter)…")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False, cache_dir=str(CACHE_DIR))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=str(CACHE_DIR),
    )
    mdl.eval()
    return tok, mdl


def load_coding():
    print("[INFO] Loading coding-fine-tuned adapter…")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=False, cache_dir=str(CACHE_DIR))
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=str(CACHE_DIR),
    )
    mdl = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
    mdl.eval()
    return tok, mdl


def generate(tok, mdl, question: str, max_new_tokens=256) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": question},
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.2,
            do_sample=True,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    if not ADAPTER_DIR.exists():
        print(f"[ERROR] Adapter not found at {ADAPTER_DIR}")
        print("        Run scripts/finetune_coding.py first.")
        sys.exit(1)

    base_tok, base_mdl = load_base()
    code_tok, code_mdl = load_coding()

    sep = "=" * 70
    for i, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n{sep}")
        print(f"Question {i}: {q}")
        print(sep)

        base_ans = generate(base_tok, base_mdl, q)
        print(f"\n[BASE MODEL]\n{base_ans}")

        code_ans = generate(code_tok, code_mdl, q)
        print(f"\n[CODING ADAPTER]\n{code_ans}")

    print(f"\n{sep}")
    print("[INFO] Comparison complete.")


if __name__ == "__main__":
    main()

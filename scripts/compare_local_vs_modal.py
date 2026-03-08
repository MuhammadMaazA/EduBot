"""
Compare local fine-tuned Phi-3-mini (on blaze GPU) vs Modal deployed endpoint.

Sends the same Socratic Q&A questions to both and prints side-by-side with
timing so you can judge quality + latency.

Usage:
    source app/venv39/bin/activate
    python scripts/compare_local_vs_modal.py

    # Test only Modal (no GPU needed):
    python scripts/compare_local_vs_modal.py --modal-only

    # Test only local (offline):
    python scripts/compare_local_vs_modal.py --local-only

    # Custom Modal URL:
    python scripts/compare_local_vs_modal.py --url https://your-url.modal.run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import textwrap
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
EDUBOT_ROOT = SCRIPT_DIR.parent
CACHE_DIR   = EDUBOT_ROOT / ".cache" / "huggingface"
ADAPTER_DIR = EDUBOT_ROOT / "checkpoints" / "phi3-coding-adapter"

os.environ.setdefault("HF_HOME",            str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE",  str(CACHE_DIR))

BASE_MODEL  = "microsoft/Phi-3-mini-128k-instruct"
MODAL_URL   = "https://muhammadmaaza--edubot-chat.modal.run"

# ── Test questions ────────────────────────────────────────────────────────────
# These match the Socratic tutor use-case (study mode, turn 1)
TEST_QUESTIONS = [
    "What is a for loop?",
    "Explain recursion to me.",
    "What is a linked list?",
    "How does binary search work?",
    "What is Big-O notation?",
]

WIDTH = 80   # console width per column


# ── Helpers ───────────────────────────────────────────────────────────────────
def wrap(text: str, width: int = 38) -> list[str]:
    """Wrap text to `width` chars, return list of lines."""
    lines = []
    for paragraph in text.split("\n"):
        if paragraph.strip():
            lines.extend(textwrap.wrap(paragraph, width=width) or [""])
        else:
            lines.append("")
    return lines or [""]


def side_by_side(left_lines: list[str], right_lines: list[str],
                 left_header: str, right_header: str, col: int = 42):
    """Print two lists of lines side by side."""
    n = max(len(left_lines), len(right_lines))
    left_lines  = left_lines  + [""] * (n - len(left_lines))
    right_lines = right_lines + [""] * (n - len(right_lines))

    sep = "│"
    print(f"\n  {left_header:<{col}}{sep}  {right_header}")
    print("  " + "─" * col + "┼" + "─" * col)
    for l, r in zip(left_lines, right_lines):
        print(f"  {l:<{col}}{sep}  {r}")
    print("  " + "─" * col + "┴" + "─" * col)


def print_header(text: str):
    print("\n" + "═" * (WIDTH + 4))
    print(f"  {text}")
    print("═" * (WIDTH + 4))


# ── Local inference ───────────────────────────────────────────────────────────
def load_local_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[LOCAL] Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=False, cache_dir=str(CACHE_DIR)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    use_cuda = torch.cuda.is_available()
    device   = "cuda:0" if use_cuda else "cpu"
    dtype    = torch.float16 if use_cuda else torch.float32
    print(f"[LOCAL] Loading base model on {device} ({dtype})...")

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=str(CACHE_DIR),
    )

    if ADAPTER_DIR.exists():
        print(f"[LOCAL] Applying coding adapter from {ADAPTER_DIR} ...")
        model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        adapter_loaded = True
    else:
        print(f"[LOCAL] WARNING: adapter not found at {ADAPTER_DIR}, using base weights")
        model = base
        adapter_loaded = False

    model.eval()
    print(f"[LOCAL] Model ready. Adapter loaded: {adapter_loaded}\n")
    return tok, model


def local_ask(tok, model, question: str,
              temperature: float = 0.5, max_tokens: int = 150) -> tuple[str, float]:
    """Run local inference, return (answer, latency_s)."""
    import torch

    # System prompt matches Socratic study mode
    messages = [
        {
            "role": "system",
            "content": (
                "You are EduBot, a friendly AI study buddy. "
                "Guide students with questions and hints rather than giving answers directly. "
                "Ask ONE question per reply. Keep replies to 1-3 short sentences."
            ),
        },
        {"role": "user", "content": question},
    ]

    prompt  = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tok(prompt, return_tensors="pt").to(model.device)

    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tok.eos_token_id,
        )
    latency = round(time.time() - t0, 2)

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = tok.decode(new_tokens, skip_special_tokens=True).strip()
    return answer, latency


# ── Modal inference ───────────────────────────────────────────────────────────
def modal_ask(question: str, session_id: str = "compare-test",
              url: str = MODAL_URL, temperature: float = 0.5,
              max_tokens: int = 150) -> tuple[str, float]:
    """Call Modal endpoint, return (answer, latency_s)."""
    import urllib.request, urllib.error

    payload = json.dumps({
        "question":    question,
        "session_id":  session_id,
        "temperature": temperature,
        "max_tokens":  max_tokens,
        "hint_mode":   True,
        "hint_level":  2,
    }).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode())
        latency = round(time.time() - t0, 2)

        if "error" in data:
            return f"[ERROR] {data['error']}", latency
        return data.get("answer", "[no answer]"), latency

    except urllib.error.HTTPError as e:
        latency = round(time.time() - t0, 2)
        return f"[HTTP {e.code}] {e.reason}", latency
    except Exception as exc:
        latency = round(time.time() - t0, 2)
        return f"[ERROR] {exc}", latency


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compare local vs Modal EduBot")
    parser.add_argument("--modal-only",  action="store_true", help="Skip local model")
    parser.add_argument("--local-only",  action="store_true", help="Skip Modal")
    parser.add_argument("--url",         default=MODAL_URL,   help="Modal chat endpoint URL")
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-tokens",  type=int,   default=150)
    parser.add_argument("--questions",   nargs="+",  help="Custom questions (space-separated)")
    args = parser.parse_args()

    questions = args.questions or TEST_QUESTIONS

    # ── Load local model if needed ────────────────────────────────────────────
    tok = model = None
    if not args.modal_only:
        print_header("Loading local model (this takes ~30s on GPU)...")
        t_load = time.time()
        tok, model = load_local_model()
        print(f"[LOCAL] Load time: {round(time.time() - t_load, 1)}s")

    # ── Warm up Modal ─────────────────────────────────────────────────────────
    if not args.local_only:
        print_header("Warming up Modal endpoint (cold start ~60s)...")
        warmup_ans, warmup_lat = modal_ask("hello", session_id="warmup", url=args.url,
                                           max_tokens=30)
        print(f"[MODAL] Warm-up: {warmup_lat}s  →  {warmup_ans[:80]}")

    # ── Run comparisons ───────────────────────────────────────────────────────
    results = []

    for i, q in enumerate(questions, 1):
        print_header(f"Question {i}/{len(questions)}: {q}")

        local_ans  = local_lat  = None
        modal_ans  = modal_lat  = None

        if not args.modal_only and tok is not None:
            print("[LOCAL] Generating...", end=" ", flush=True)
            local_ans, local_lat = local_ask(tok, model, q,
                                             args.temperature, args.max_tokens)
            print(f"{local_lat}s")

        if not args.local_only:
            print("[MODAL] Sending request...", end=" ", flush=True)
            modal_ans, modal_lat = modal_ask(q, session_id=f"compare-q{i}",
                                             url=args.url,
                                             temperature=args.temperature,
                                             max_tokens=args.max_tokens)
            print(f"{modal_lat}s")

        results.append({
            "question":  q,
            "local":     {"answer": local_ans, "latency": local_lat},
            "modal":     {"answer": modal_ans, "latency": modal_lat},
        })

        # Print side by side
        if local_ans and modal_ans:
            side_by_side(
                wrap(local_ans),
                wrap(modal_ans),
                left_header  = f"LOCAL  ({local_lat}s)",
                right_header = f"MODAL  ({modal_lat}s)",
            )
        elif local_ans:
            print(f"\n  LOCAL ({local_lat}s):\n  {local_ans}\n")
        elif modal_ans:
            print(f"\n  MODAL ({modal_lat}s):\n  {modal_ans}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    print_header("SUMMARY")
    print(f"  {'#':<4} {'Question':<40} {'Local':>8} {'Modal':>8}")
    print("  " + "─" * 64)
    for i, r in enumerate(results, 1):
        ll = f"{r['local']['latency']}s"  if r['local']['latency']  else "—"
        ml = f"{r['modal']['latency']}s"  if r['modal']['latency']  else "—"
        print(f"  {i:<4} {r['question'][:40]:<40} {ll:>8} {ml:>8}")

    if not args.modal_only and not args.local_only:
        local_lats = [r['local']['latency']  for r in results if r['local']['latency']]
        modal_lats = [r['modal']['latency']  for r in results if r['modal']['latency']]
        if local_lats and modal_lats:
            print(f"\n  Avg local latency:  {round(sum(local_lats)/len(local_lats), 2)}s")
            print(f"  Avg Modal latency:  {round(sum(modal_lats)/len(modal_lats), 2)}s")

    # ── Save results to JSON ──────────────────────────────────────────────────
    out_path = EDUBOT_ROOT / "logs" / "compare_local_vs_modal.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")


if __name__ == "__main__":
    main()

"""
Quick test for the Q&A dialogue system (Goal 1).

Tests the QASession multi-turn conversation without a browser or WebSocket.
Run from the Interface directory:

    source venv39/bin/activate
    cd app
    python ../../scripts/test_qa.py
"""
import sys, os
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(HERE))

# Cache dirs
EDUBOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME",             str(EDUBOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE",   str(EDUBOT / ".cache" / "huggingface"))

# Load .env if present
env_file = EDUBOT / "scripts" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from qa import QASession

QUESTIONS = [
    "What is 2 + 2?",
    "Can you give me a simple Python example of that?",  # follow-up to test history
    "Now explain what a variable is in one sentence.",
]

def main():
    session = QASession()
    print("=" * 60)
    print("EduBot Q&A — Multi-turn dialogue test")
    print("=" * 60)

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n[Turn {i}] User: {q}")
        answer = session.chat(q)
        print(f"[Turn {i}] Bot:  {answer}")
        print(f"         (history length: {len(session.get_history())} messages)")

    print("\n--- Resetting session ---")
    session.reset()
    print(f"History after reset: {len(session.get_history())} messages")

    print("\n--- Post-reset question ---")
    q = "Hello, who are you?"
    print(f"User: {q}")
    print(f"Bot:  {session.chat(q)}")

    print("\n[SUCCESS] Q&A test complete.")

if __name__ == "__main__":
    main()

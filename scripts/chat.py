"""
Interactive terminal chat with EduBot.

The robot guides you toward answers with hints and questions
rather than giving direct answers. No code blocks — spoken language only.

Commands:
  /reset   - start a new topic (clears history)
  /quit    - exit
  /history - show the conversation so far

Usage (in bash):
    cd EduBot
    export HF_HOME="$PWD/.cache/huggingface"
    MI2US_Year2s/Interface/venv39/bin/python scripts/chat.py
"""
import sys, os
from pathlib import Path

HERE   = Path(__file__).resolve().parent.parent / "MI2US_Year2s" / "Interface"
EDUBOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

os.environ.setdefault("HF_HOME",           str(EDUBOT / ".cache" / "huggingface"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(EDUBOT / ".cache" / "huggingface"))

env_file = EDUBOT / "scripts" / ".env"
if env_file.exists():
    for line in env_file.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

from qa import QASession

BANNER = """
==================================================
  EduBot
  Ask me anything and we will work through it
  together, step by step.

  /reset    start a new topic
  /history  review the conversation
  /quit     exit
==================================================
"""

def main():
    session = QASession()
    print(BANNER)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye! Keep learning!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit", "exit", "quit"):
            print("Goodbye! Keep learning!")
            break

        if user_input.lower() == "/reset":
            session.reset()
            print("[New topic started. What would you like to learn about?]\n")
            continue

        if user_input.lower() == "/history":
            h = session.get_history()
            if not h:
                print("[No conversation yet.]\n")
            else:
                print()
                for msg in h:
                    label = "You  " if msg["role"] == "user" else "Robot"
                    print(f"  {label}: {msg['content']}")
                print()
            continue

        print("\nRobot: ", end="", flush=True)
        answer = session.chat(user_input)
        print(answer + "\n")

if __name__ == "__main__":
    main()

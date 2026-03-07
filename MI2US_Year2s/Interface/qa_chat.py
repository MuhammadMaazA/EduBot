"""
Simple terminal chat for EduBot using the Q&A backend in qa.py.

Run from this directory with the Interface venv:

    cd /cs/student/projects1/2023/muhamaaz/social-robots/EduBot/MI2US_Year2s/Interface
    source venv39/bin/activate
    export HF_HOME="$PWD/.cache/hf"
    export TRANSFORMERS_CACHE="$PWD/.cache/hf"
    python qa_chat.py
"""

from qa import QASession


def main() -> None:
    session = QASession()
    print("EduBot Q&A terminal chat.")
    print("Type /quit to exit, /reset to clear history.\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q.lower() == "/quit":
            break
        if q.lower() == "/reset":
            session.reset()
            print("Robot: Okay, I've cleared our conversation.\n")
            continue

        ans = session.chat(q)
        print("Robot:", ans, "\n")


if __name__ == "__main__":
    main()


"""
EduBot — Modal deployment script.

Hosts Phi-3-mini + coding LoRA adapter on a cloud GPU (T4 by default).
Supports 4 concurrent sessions with full Socratic hint-ladder logic.

Deploy:   modal deploy modal_deploy.py
Test:     modal run modal_deploy.py
"""
from __future__ import annotations

import re
import time
from typing import Optional

import modal

# ── Modal setup ───────────────────────────────────────────────────────────────
app = modal.App("edubot")

# Container image — installs everything once, cached afterwards
image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch==2.2.2",
        "transformers==4.44.2",
        "peft==0.14.0",
        "accelerate==0.33.0",
        "sentencepiece",
        "protobuf",
        "fastapi[standard]",
    )
)

# Persistent volume to cache the ~7 GB base model between deploys
model_cache = modal.Volume.from_name("edubot-model-cache", create_if_missing=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL      = "microsoft/Phi-3-mini-128k-instruct"
ADAPTER_REPO    = "MuhammadMaazA/phi3-coding-adapter"   # ← your HF repo
CACHE_DIR       = "/cache"

# ── Hint ladder (same as local qa.py) ─────────────────────────────────────────
STAGE_EXPLORE = "explore"
STAGE_NUDGE   = "nudge"
STAGE_STRONG  = "strong"
STAGE_EXPLAIN = "explain"

_LADDER = {
    (1,  3):  (STAGE_EXPLORE, ""),
    (4,  6):  (STAGE_NUDGE, (
        "[NUDGE: The student has been engaging for a few turns. "
        "Break the concept into a smaller piece they might already know. "
        "Give ONE concrete hint using an everyday analogy, then ask ONE follow-up question. "
        "Do NOT give the definition or full answer yet.]"
    )),
    (7,  9):  (STAGE_STRONG, (
        "[STRONG HINT: The student has been trying hard. "
        "Give a very specific hint that almost reveals the answer — fill in most of the picture. "
        "Then ask ONE final guiding question that only requires a small leap. "
        "Still do NOT state the full answer directly.]"
    )),
    (10, 99): (STAGE_EXPLAIN, (
        "[EXPLAIN: The student has made a genuine effort over many turns. "
        "Now give a clear, friendly 2-3 sentence explanation of the answer. "
        "Tie it back to something they said earlier if possible. "
        "End with an encouraging remark and optionally suggest a follow-up topic.]"
    )),
}

SYSTEM_PROMPT = """You are EduBot, a friendly AI study buddy helping students learn computer science. Your job is to make the student THINK and DISCOVER the answer themselves — not to hand it to them.

Your approach (study mode):
1. FIRST, find out what the student already knows. Ask them.
2. Build on what they know using everyday analogies (school, sports, cooking, games).
3. Ask ONE clear question per reply that moves them one small step closer.
4. If they get something right, celebrate it briefly and build on it.
5. If they get something wrong, don't say "wrong" — say something like "interesting thought!" and redirect.
6. If they say they don't know or seem stuck, give a warmer, more concrete hint — but still ask a question.
7. Only after MANY turns of genuine effort should you explain directly.

Strict rules:
- NEVER start with a definition ("A queue is...", "Recursion means...").
- NEVER give the full answer in the first few turns — even if they ask directly.
- Every reply MUST end with a question (except when you've been told to explain).
- Keep replies to 1-3 short sentences. No essays.
- Be warm, casual, and encouraging — like a smart friend, not a textbook.
- If the student says "idk", "I don't know", or seems stuck — do NOT repeat the same hint. Try a COMPLETELY DIFFERENT analogy or break it into a smaller piece. Switch your angle.
- If they say it multiple times, give an even bigger hint each time using a new everyday example — but still ask a question until you've been told to explain.
- Only after MANY turns of genuine effort should you explain directly."""

HINT_PROMPTS = {
    1: "Give a very gentle nudge — just one small hint, then ask a question. Keep it to one or two sentences.",
    2: "Guide with a hint and one follow-up question. Two sentences maximum.",
    3: "Be more direct: give a strong specific hint, but still end with a question. Don't give the full answer yet.",
}

# ── Text processing helpers (matching qa.py) ──────────────────────────────────
_IDK_PATTERNS = re.compile(
    r"(i\s*don.?t\s*know|no\s*idea|no\s*clue|i.?m\s*stuck|i\s*give\s*up|"
    r"just\s*tell\s*me|can\s*you\s*just|please\s*just|i\s*can.?t|help\s*me|"
    r"i\s*have\s*no|confused|lost|clueless|idk|dunno|not\s*sure\s*at\s*all)",
    re.IGNORECASE,
)

_EXPLAIN_RE = re.compile(
    r"^(tell me (about|what|how)|explain|what (is|are|does|do)|describe|define|how does|how do)\b",
    re.IGNORECASE,
)


def _detect_frustration(msg: str) -> bool:
    return bool(_IDK_PATTERNS.search(msg))


def _ladder_stage(turn_count: int) -> str:
    for (lo, hi), (stage, _) in _LADDER.items():
        if lo <= turn_count <= hi:
            return stage
    return STAGE_EXPLAIN


def _ladder_instruction(turn_count: int) -> str:
    for (lo, hi), (_, instr) in _LADDER.items():
        if lo <= turn_count <= hi:
            return instr
    return ""


def _rewrite(student_msg: str, turn_count: int = 1) -> str:
    msg = student_msg.strip()
    stage = _ladder_stage(turn_count)
    if stage == STAGE_EXPLAIN:
        return msg

    frustrated = _detect_frustration(msg)
    prefix = ""
    if frustrated:
        prefix = (
            "[The student is STUCK and does not know the answer. "
            "Do NOT repeat your previous hint or analogy — try a COMPLETELY DIFFERENT approach. "
            "Use a new analogy from a different area (food, sports, games, school, daily life). "
            "Break the concept down into an even smaller piece. "
            "Be extra warm and encouraging — acknowledge their effort. "
            "Give a bigger, more concrete hint than before, but still end with a question. "
            "Do NOT give the full answer yet unless we are in the explain stage.] "
        )

    if stage in (STAGE_EXPLORE, STAGE_NUDGE) and _EXPLAIN_RE.match(msg):
        topic = _EXPLAIN_RE.sub("", msg).strip().strip("?").strip() or msg
        return prefix + (
            f'The student wants to learn about "{topic}". '
            f"Do NOT define or explain it directly. "
            f"Start by asking what they already know or think about it, "
            f"or give ONE short real-world analogy and end with a question. "
            f"Two sentences maximum."
        )
    return prefix + msg if prefix else msg


def _clean(text: str) -> str:
    text = re.sub(r"^\s*(Robot|Tutor|Assistant|AI)\s*:\s*", "", text, flags=re.IGNORECASE)
    for marker in ("Student:", "You:", "User:", "Human:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return re.sub(r"\n{2,}", " ", text).strip()


def _enforce_hint(text: str, student_msg: str = "", turn_count: int = 1) -> str:
    text = text.strip()
    stage = _ladder_stage(turn_count)

    if stage == STAGE_EXPLAIN:
        return text

    if stage == STAGE_STRONG:
        if text.rstrip().endswith("?"):
            return text
        return text.rstrip(".!") + " — what do you think?"

    if text.rstrip().endswith("?"):
        if len(text) > 280:
            sentences = re.split(r"(?<=[.!?])\s+", text)
            kept = []
            for s in reversed(sentences):
                kept.insert(0, s)
                if len(kept) >= 2 and any(x.rstrip().endswith("?") for x in kept):
                    break
            return " ".join(kept)
        return text

    sentences = re.split(r"(?<=[.!])\s+", text)
    core = " ".join(sentences[:2]).rstrip(".!") if sentences else text.rstrip(".!")

    frustrated = _detect_frustration(student_msg)
    lower = student_msg.lower()

    if frustrated:
        return core + ". No worries — what's your best guess, even if you're not sure?"
    if any(w in lower for w in ("explain", "tell me", "what is", "describe", "define")):
        return core + ". Before I say more — what do you already know about this?"
    return core + ". What do you think?"


def _build_system_prompt(hint_mode: bool = True, hint_level: int = 2) -> str:
    if not hint_mode:
        return (
            "You are a helpful assistant. Answer questions clearly and concisely "
            "in spoken language. No code blocks or bullet points."
        )
    hint_instr = HINT_PROMPTS.get(hint_level, HINT_PROMPTS[2])
    return SYSTEM_PROMPT + f"\n\nHint style for this session: {hint_instr}"


# ── Session state ─────────────────────────────────────────────────────────────
# In-memory session storage (per container — fine for a 3-hour experiment)
_sessions: dict[str, dict] = {}


def _get_session(session_id: str) -> dict:
    if session_id not in _sessions:
        _sessions[session_id] = {
            "history": [],
            "turn_count": 0,
            "frustration_bumps": 0,
        }
    return _sessions[session_id]


# ── Modal class ───────────────────────────────────────────────────────────────
@app.cls(
    image=image,
    gpu="T4",                       # 16 GB VRAM — swap to "A10G" for 24 GB
    timeout=600,                    # 10 min max per request
    scaledown_window=300,           # 5 min idle → sleep (stops billing)
    volumes={CACHE_DIR: model_cache},
    retries=0,                      # don't retry on startup failure — surface error fast
)
@modal.concurrent(max_inputs=4)     # 4 simultaneous users on one GPU
class EduBotModel:
    """Singleton Phi-3-mini + LoRA model on a Modal GPU container."""

    @modal.enter()
    def load_model(self):
        """Called once when the container starts — loads model + adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        print("[INFO] Loading Phi-3-mini base model ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL,
            trust_remote_code=False,
            cache_dir=CACHE_DIR,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float16,
            device_map={"": "cuda:0"},
            trust_remote_code=False,
            attn_implementation="eager",
            cache_dir=CACHE_DIR,
        )

        print(f"[INFO] Loading LoRA adapter from {ADAPTER_REPO} ...")
        try:
            self.model = PeftModel.from_pretrained(
                self.model, ADAPTER_REPO, cache_dir=CACHE_DIR
            )
            self.adapter_loaded = True
            print("[INFO] Coding adapter loaded successfully.")
        except Exception as exc:
            print(f"[WARNING] Could not load adapter: {exc}")
            print("[WARNING] Using base model weights only.")
            self.adapter_loaded = False

        self.model.eval()

        # Persist downloaded weights so next cold start is faster
        model_cache.commit()
        print("[INFO] Model ready on cuda:0.")

    def _generate(self, messages: list, temperature: float, max_tokens: int) -> str:
        import torch

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    @modal.fastapi_endpoint(method="POST", label="edubot-chat")
    def chat(self, request: dict) -> dict:
        """
        Main chat endpoint.

        POST JSON body:
        {
            "question":    "What is a for loop?",
            "session_id":  "student-1",       // unique per student
            "temperature":  0.5,              // optional
            "max_tokens":   120,              // optional
            "hint_mode":    true,             // optional
            "hint_level":   2                 // optional (1=gentle, 2=normal, 3=direct)
        }
        """
        question   = request.get("question", "").strip()
        session_id = request.get("session_id", "default")
        temperature = float(request.get("temperature", 0.5))
        max_tokens  = int(request.get("max_tokens", 120))
        hint_mode   = bool(request.get("hint_mode", True))
        hint_level  = int(request.get("hint_level", 2))

        if not question:
            return {"error": "No question provided"}

        # Get or create session
        session = _get_session(session_id)
        session["turn_count"] += 1
        tc = session["turn_count"]

        # Frustration detection
        frustrated = _detect_frustration(question)
        if frustrated:
            session["frustration_bumps"] += 2

        effective_tc = tc + session["frustration_bumps"]
        stage = _ladder_stage(effective_tc)

        print(f"[Q] session={session_id} turn={tc} eff={effective_tc} stage={stage}  {question[:60]}")

        # Build messages
        ladder = _ladder_instruction(effective_tc)
        user_content = (f"{ladder}\n{_rewrite(question, effective_tc)}").strip() if ladder else _rewrite(question, effective_tc)

        messages = [{"role": "system", "content": _build_system_prompt(hint_mode, hint_level)}]
        messages.extend(session["history"])
        messages.append({"role": "user", "content": user_content})

        # Generate
        t0 = time.time()
        try:
            raw_answer = self._generate(messages, temperature, max_tokens)
        except Exception as exc:
            print(f"[ERROR] Inference failed: {exc}")
            return {"error": f"Inference failed: {str(exc)}"}

        answer = _enforce_hint(_clean(raw_answer), question, turn_count=effective_tc)
        latency = round(time.time() - t0, 2)

        # Update session history
        session["history"].append({"role": "user", "content": question})
        session["history"].append({"role": "assistant", "content": answer})

        print(f"[A] session={session_id} turn={tc} stage={stage} {latency}s  {answer[:80]}...")

        return {
            "answer":         answer,
            "stage":          stage,
            "turn":           tc,
            "effective_turn": effective_tc,
            "model":          "phi3-coding-adapter" if self.adapter_loaded else "phi3-base",
            "latency_s":      latency,
        }

    @modal.fastapi_endpoint(method="GET", label="edubot-health")
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "ok",
            "model_loaded": hasattr(self, "model"),
            "adapter_loaded": getattr(self, "adapter_loaded", False),
            "active_sessions": len(_sessions),
        }

    @modal.fastapi_endpoint(method="POST", label="edubot-reset")
    def reset_session(self, request: dict) -> dict:
        """Reset a specific session's conversation history."""
        session_id = request.get("session_id", "default")
        if session_id in _sessions:
            del _sessions[session_id]
        return {"status": "ok", "session_id": session_id}


# ── Local test entrypoint ─────────────────────────────────────────────────────
@app.local_entrypoint()
def main():
    """Fire 4 concurrent test questions to verify the deployment."""
    import concurrent.futures

    model = EduBotModel()

    questions = [
        {"question": "What is a for loop?",       "session_id": "test-1"},
        {"question": "Explain recursion",          "session_id": "test-2"},
        {"question": "What is a linked list?",     "session_id": "test-3"},
        {"question": "How does binary search work?", "session_id": "test-4"},
    ]

    print("=" * 60)
    print("Sending 4 concurrent questions ...")
    print("=" * 60)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(model.chat.remote, q): q["session_id"]
            for q in questions
        }
        for future in concurrent.futures.as_completed(futures):
            sid = futures[future]
            try:
                result = future.result()
                print(f"\n[{sid}] Stage: {result['stage']} | {result['latency_s']}s")
                print(f"  Q: {[q for q in questions if q['session_id']==sid][0]['question']}")
                print(f"  A: {result['answer'][:200]}")
            except Exception as exc:
                print(f"\n[{sid}] ERROR: {exc}")

    print("\n" + "=" * 60)
    print("Done! All 4 responses received.")
    print("=" * 60)

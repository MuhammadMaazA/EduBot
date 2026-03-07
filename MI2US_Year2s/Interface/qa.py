"""
Q&A Dialogue module for EduBot — hint-based Socratic tutoring.

Simple backend:
- One Phi-3-mini model loaded once on cuda:0 (if available).
- Optional coding adapter (phi3-coding-adapter) applied on top.
- No vLLM, no multi-GPU pool — just a single generate() call per request.

"""Hint ladder (enforced per QASession) — study-mode Socratic tutoring:
  turns 1-3  →  explore: ask what the student knows, use analogies, one question per reply
  turns 4-6  →  nudge: give concrete hints, break the problem into smaller pieces
  turns 7-9  →  strong hint: nearly reveal the answer, fill in missing pieces
  turn 10+   →  explain: give a clear summary (only after genuine effort)

  Frustration / "I don't know" signals can bump the stage forward.
"""
import os
import re
from pathlib import Path

import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "microsoft/Phi-3-mini-128k-instruct"
_EDUBOT_ROOT   = Path(__file__).resolve().parents[2]   # EduBot/
CODING_ADAPTER = _EDUBOT_ROOT / "checkpoints" / "phi3-coding-adapter"
CACHE_DIR      = _EDUBOT_ROOT / ".cache" / "huggingface"

# Single global tokenizer/model lazily loaded on first use
_tok = None
_model = None


def _load_model():
    """Load tokenizer + model once, on cuda:0 if available, else CPU."""
    global _tok, _model
    if _model is not None:
        return _tok, _model

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("[INFO] Loading Phi-3-mini for Q&A ...")
    _tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=False, cache_dir=str(CACHE_DIR)
    )
    if _tok.pad_token is None:
        _tok.pad_token = _tok.eos_token

    use_cuda = torch.cuda.is_available()
    device   = "cuda:0" if use_cuda else "cpu"

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        device_map={"": device} if use_cuda else {"": "cpu"},
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=str(CACHE_DIR),
    )

    if CODING_ADAPTER.exists():
        _model = PeftModel.from_pretrained(base, str(CODING_ADAPTER))
        print(f"[INFO]   {device} — coding adapter loaded.")
    else:
        print(f"[WARNING] {device} — coding adapter not found, using base weights.")
        _model = base

    _model.eval()
    print(f"[INFO] Q&A model ready on {device}.")
    return _tok, _model


def _qa_generate(messages: list, temperature: float = 0.5, max_new_tokens: int = 120) -> str:
    """Synchronous inference using the single global model."""
    tok, model = _load_model()
    prompt  = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ── Hint ladder ───────────────────────────────────────────────────────────────
# Injected as an instruction prefix into the user message so the model gets an
# explicit signal rather than relying on it counting turns itself.
#
# Four stages — the session can also jump stages via frustration detection.

STAGE_EXPLORE  = "explore"    # turns 1-3
STAGE_NUDGE    = "nudge"      # turns 4-6
STAGE_STRONG   = "strong"     # turns 7-9
STAGE_EXPLAIN  = "explain"    # turns 10+

_LADDER = {
    (1,  3):  (STAGE_EXPLORE, ""),           # pure Socratic — no extra instruction
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

def _ladder_stage(turn_count: int) -> str:
    """Return the stage name for a given turn count."""
    for (lo, hi), (stage, _instr) in _LADDER.items():
        if lo <= turn_count <= hi:
            return stage
    return STAGE_EXPLAIN

def _ladder_instruction(turn_count: int) -> str:
    for (lo, hi), (_stage, instr) in _LADDER.items():
        if lo <= turn_count <= hi:
            return instr
    return ""


# ── Frustration / "I don't know" detection ────────────────────────────────────
_IDK_PATTERNS = re.compile(
    r"(i\s*don.?t\s*know|no\s*idea|no\s*clue|i.?m\s*stuck|i\s*give\s*up|"
    r"just\s*tell\s*me|can\s*you\s*just|please\s*just|i\s*can.?t|help\s*me|"
    r"i\s*have\s*no|confused|lost|clueless|idk|dunno|not\s*sure\s*at\s*all)",
    re.IGNORECASE,
)

def _detect_frustration(msg: str) -> bool:
    """Return True if the student sounds stuck or frustrated."""
    return bool(_IDK_PATTERNS.search(msg))


# ── Text helpers ──────────────────────────────────────────────────────────────
_EXPLAIN_RE = re.compile(
    r"^(tell me (about|what|how)|explain|what (is|are|does|do)|describe|define|how does|how do)\b",
    re.IGNORECASE,
)

def _rewrite(student_msg: str, turn_count: int = 1) -> str:
    """
    Pre-process the student message:
      - If the student sounds stuck, note it so the model gives a gentler nudge.
      - If it is an "explain X" request and we're still exploring,
        rewrite it so the model gives a hint instead of a definition.
      - In later stages, let the ladder instruction handle it.
    """
    msg = student_msg.strip()
    stage = _ladder_stage(turn_count)

    # In explain stage, let the ladder instruction handle everything
    if stage == STAGE_EXPLAIN:
        return msg

    # If the student sounds stuck, add a compassionate prefix
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

    # Rewrite "explain X" / "what is X" requests during explore/nudge
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
    """Strip leaked role prefixes and hallucinated continuations."""
    text = re.sub(r"^\s*(Robot|Tutor|Assistant|AI)\s*:\s*", "", text, flags=re.IGNORECASE)
    for marker in ("Student:", "You:", "User:", "Human:"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return re.sub(r"\n{2,}", " ", text).strip()


def _enforce_hint(text: str, student_msg: str = "", turn_count: int = 1) -> str:
    """
    Post-processing safety net — ensures the study-mode contract:
      explore / nudge  → must end with a question, keep it short
      strong           → can be longer, should still end with a question
      explain          → full answer allowed, no forced question
    """
    text = text.strip()
    stage = _ladder_stage(turn_count)

    # Explain stage — allow the full answer through
    if stage == STAGE_EXPLAIN:
        return text

    # Strong hint stage — allow longer text, just make sure there's a question
    if stage == STAGE_STRONG:
        if text.rstrip().endswith("?"):
            return text
        return text.rstrip(".!") + " — what do you think?"

    # Explore / nudge — keep it concise and question-driven
    if text.rstrip().endswith("?"):
        if len(text) > 280:
            # Too long — try to keep just the analogy + question
            sentences = re.split(r"(?<=[.!?])\s+", text)
            # Keep last 2-3 sentences that include a question
            kept = []
            for s in reversed(sentences):
                kept.insert(0, s)
                if len(kept) >= 2 and any(x.rstrip().endswith("?") for x in kept):
                    break
            return " ".join(kept)
        return text

    # No question at the end — add a nudge
    sentences = re.split(r"(?<=[.!])\s+", text)
    core = " ".join(sentences[:2]).rstrip(".!") if sentences else text.rstrip(".!")

    frustrated = _detect_frustration(student_msg)
    lower = student_msg.lower()

    if frustrated:
        return core + ". No worries — what\'s your best guess, even if you\'re not sure?"
    if any(w in lower for w in ("explain", "tell me", "what is", "describe", "define")):
        return core + ". Before I say more — what do you already know about this?"
    return core + ". What do you think?"


# ── System prompt ─────────────────────────────────────────────────────────────
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
- Only after MANY turns of genuine effort should you explain directly.

Examples:

Student: what is a queue?
Tutor: Good question! Before I explain — have you ever been in a queue at a shop or cafeteria? Who gets served first?

Student: the person at the front
Tutor: Exactly! So in programming, a queue works the same way. If three items arrive in order A, B, C — which one do you think leaves first?

Student: A?
Tutor: Spot on! That's called FIFO — First In, First Out. Can you think of another real-life example that works like this?

Student: what is recursion?
Tutor: Hmm, have you ever taken a photo of yourself holding a photo of yourself? What happens in that picture?

Student: it goes on forever?
Tutor: Kind of! But what if you said "stop after 3 photos"? In programming, recursion is when a function calls itself. What do you think happens if we forget to tell it when to stop?

Student: I don't know, just tell me
Tutor: No worries, you're doing great! Let me give you a bigger clue: think about standing between two mirrors facing each other. The reflections keep going, right? That's like recursion without a stop condition. What do you think we call the rule that tells it to stop?

Student: idk
Tutor: That's okay! Let me try it differently — imagine you're climbing stairs and you decide "I'll stop when I reach the top." In the same way, a recursive function needs a rule that says "stop here." What do you think that rule might be called?

Student: idk idk
Tutor: No problem at all, let's break it down even more! When you count down from 5 — 5, 4, 3, 2, 1 — you stop at 1, right? You don't keep going to 0, -1, -2. That stopping point is what programmers call the "base case." Does that make sense — can you tell me in your own words what a base case does?

Student: a base case?
Tutor: Yes! Brilliant! So every recursive function needs two things: calling itself AND a base case. Can you think of how you might use recursion to count down from 5 to 1?

Student: explain binary search
Tutor: Let's figure it out together! Imagine you're looking for a word in a dictionary. Do you start at page 1, or do you open it somewhere in the middle? Why?

Student: the middle, because it's faster
Tutor: Right! And then what do you do next — if the word you want comes after the page you opened?

Continue in this style. Guide, don't tell. One step at a time."""


# ── Public API ────────────────────────────────────────────────────────────────
def ask(question: str, history: list = None) -> str:
    """Single-turn Q&A (stateless). Uses turn_count=1 (Socratic mode)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": _rewrite(question, turn_count=1)})
    raw = _qa_generate(messages, temperature=0.5, max_new_tokens=120)
    return _enforce_hint(_clean(raw), question, turn_count=1)


class QASession:
    """
    Multi-turn dialogue with study-mode Socratic escalation.

    The session automatically escalates hint strength as turns increase:
      turns 1-3   → explore  (ask what they know, use analogies, one Q per reply)
      turns 4-6   → nudge    (break problem into smaller pieces, concrete hints)
      turns 7-9   → strong   (nearly reveal, fill in gaps, one final question)
      turn  10+   → explain  (give a clear summary after genuine effort)

    Frustration / "I don't know" signals bump the effective turn count +2,
    so the student gets a bigger hint sooner.
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt
        self.history: list  = []
        self.turn_count: int = 0
        self._frustration_bumps: int = 0  # extra turns added by frustration

    @property
    def effective_turn(self) -> int:
        return self.turn_count + self._frustration_bumps

    @property
    def stage(self) -> str:
        return _ladder_stage(self.effective_turn)

    def chat(self, question: str,
             temperature: float = 0.5,
             max_new_tokens: int = 150) -> str:
        """Send a message, apply hint ladder, return the tutor response."""
        self.turn_count += 1

        # Detect frustration and bump effective turn
        if _detect_frustration(question):
            self._frustration_bumps += 2

        etc = self.effective_turn

        ladder = _ladder_instruction(etc)
        user_content = (f"{ladder}\n{_rewrite(question, etc)}").strip() if ladder else _rewrite(question, etc)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_content})

        raw    = _qa_generate(messages, temperature=temperature,
                              max_new_tokens=max_new_tokens)
        answer = _enforce_hint(_clean(raw), question, turn_count=etc)

        # Store original student message (not the rewritten one) so history reads naturally
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})
        return answer

    def reset(self):
        self.history    = []
        self.turn_count = 0
        self._frustration_bumps = 0

    def get_history(self) -> list:
        return self.history

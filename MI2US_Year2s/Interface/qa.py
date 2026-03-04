"""
Q&A Dialogue module for EduBot — hint-based Socratic tutoring.

Inference backends (selected at import time):
  1. vLLM  — if USE_VLLM=1 env var is set (requires sm_7.0+ GPU, e.g. RTX 3090)
             True continuous batching; all sessions share one engine.
  2. 4-GPU pool — default on the current 4× TITAN X setup.
             One 4-bit quantised model per GPU; requests routed to the
             least-busy GPU so all four run in parallel.

Hint ladder (enforced per QASession):
  turns 1-2  →  pure Socratic: analogy + question, no answer
  turns 3-4  →  stronger hint: nearly reveal the answer, one final question
  turn  5+   →  give the real explanation (2 clear sentences, friendly tone)
"""
from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Optional

import torch

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_MODEL     = "microsoft/Phi-3-mini-128k-instruct"
_EDUBOT_ROOT   = Path(__file__).resolve().parents[2]   # EduBot/
CODING_ADAPTER = _EDUBOT_ROOT / "checkpoints" / "phi3-coding-adapter"
CACHE_DIR      = _EDUBOT_ROOT / ".cache" / "huggingface"

# ── Backend selection ─────────────────────────────────────────────────────────
_USE_VLLM = os.getenv("USE_VLLM", "0") == "1"
_NUM_GPUS = torch.cuda.device_count()           # 0 on CPU-only, 4 on TITAN X

# ── vLLM backend ─────────────────────────────────────────────────────────────
_vllm_engine = None

def _init_vllm():
    """Initialise the vLLM AsyncLLMEngine (call once at server start)."""
    global _vllm_engine
    if _vllm_engine is not None:
        return
    try:
        from vllm import LLM, SamplingParams as _SP  # noqa: F401 – just check import
        from vllm import AsyncLLMEngine, AsyncEngineArgs
    except ImportError:
        raise RuntimeError(
            "USE_VLLM=1 but vllm is not installed. "
            "pip install vllm  (requires sm_7.0+ GPU)"
        )
    adapter = str(CODING_ADAPTER) if CODING_ADAPTER.exists() else None
    args = AsyncEngineArgs(
        model=BASE_MODEL,
        enable_lora=adapter is not None,
        lora_modules=[{"name": "coding", "path": adapter}] if adapter else [],
        quantization="awq" if adapter is None else None,
        gpu_memory_utilization=0.90,
        max_model_len=2048,
        dtype="float16",
        download_dir=str(CACHE_DIR),
    )
    _vllm_engine = AsyncLLMEngine.from_engine_args(args)
    print("[INFO] vLLM AsyncLLMEngine ready.")


async def _vllm_generate(messages: list, temperature: float, max_new_tokens: int) -> str:
    """vLLM inference — async, continuous batching handled internally."""
    from vllm import SamplingParams
    import uuid
    # Build a plain-text prompt from the message list
    tokenizer = await _vllm_engine.get_tokenizer()
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens)
    request_id = str(uuid.uuid4())
    result = ""
    async for output in _vllm_engine.generate(prompt, params, request_id):
        result = output.outputs[0].text
    return result.strip()


# ── 4-GPU pool backend ────────────────────────────────────────────────────────
# Stores (tokenizer, model) for each GPU that has been initialised.
_gpu_pool: dict[int, tuple] = {}
_gpu_locks: dict[int, threading.Lock] = {}   # serialise per-GPU (not global)

def _load_model_on_gpu(gpu_id: int):
    """Load a 4-bit quantised model on a single GPU. Thread-safe, idempotent."""
    if gpu_id in _gpu_pool:
        return _gpu_pool[gpu_id]

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"[INFO] Loading model on cuda:{gpu_id} ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tok = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=False, cache_dir=str(CACHE_DIR)
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=str(CACHE_DIR),
    )
    if CODING_ADAPTER.exists():
        model = PeftModel.from_pretrained(base, str(CODING_ADAPTER))
        print(f"[INFO]   cuda:{gpu_id} — coding adapter loaded.")
    else:
        print(f"[WARNING] cuda:{gpu_id} — coding adapter not found, using base weights.")
        model = base

    model.eval()
    _gpu_pool[gpu_id] = (tok, model)
    _gpu_locks[gpu_id] = threading.Lock()
    free, total = torch.cuda.mem_get_info(gpu_id)
    print(f"[INFO]   cuda:{gpu_id} ready — {(total-free)/1024**3:.1f} GB used / {total/1024**3:.1f} GB total")
    return _gpu_pool[gpu_id]


def init_gpu_pool(gpu_ids: Optional[list[int]] = None):
    """
    Pre-load models on all requested GPUs.
    Called once at server startup.  Blocks until all GPUs are ready.
    """
    if _USE_VLLM:
        return  # vLLM manages its own GPU allocation
    ids = gpu_ids if gpu_ids is not None else list(range(max(_NUM_GPUS, 1)))
    threads = [threading.Thread(target=_load_model_on_gpu, args=(i,), daemon=True) for i in ids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"[INFO] GPU pool ready: {list(_gpu_pool.keys())}")


def _generate_on_gpu(messages: list, temperature: float, max_new_tokens: int, gpu_id: int) -> str:
    """Synchronous inference on a specific GPU (called from a thread)."""
    tok, model = _load_model_on_gpu(gpu_id)      # no-op if already loaded
    with _gpu_locks[gpu_id]:                      # serialise within this GPU
        prompt  = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs  = tok(prompt, return_tensors="pt").to(f"cuda:{gpu_id}")
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


# Legacy single-model path (CPU fallback or if pool not initialised)
_fallback_tokenizer = None
_fallback_model     = None

def _qa_generate(messages: list, temperature: float = 0.5, max_new_tokens: int = 120,
                 gpu_id: int = 0) -> str:
    """
    Dispatch to the appropriate backend.
    Called from a thread (not the async event loop).
    """
    if _USE_VLLM:
        raise RuntimeError("Use _vllm_generate() for vLLM backend (it is async).")

    if _NUM_GPUS > 0:
        return _generate_on_gpu(messages, temperature, max_new_tokens, gpu_id % _NUM_GPUS)

    # CPU / single-device fallback
    global _fallback_tokenizer, _fallback_model
    if _fallback_model is None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        _fallback_tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL, trust_remote_code=False, cache_dir=str(CACHE_DIR)
        )
        if _fallback_tokenizer.pad_token is None:
            _fallback_tokenizer.pad_token = _fallback_tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float32,
            device_map="cpu", trust_remote_code=False,
            attn_implementation="eager", cache_dir=str(CACHE_DIR),
        )
        _fallback_model = (
            PeftModel.from_pretrained(base, str(CODING_ADAPTER))
            if CODING_ADAPTER.exists() else base
        )
        _fallback_model.eval()
    tok, model = _fallback_tokenizer, _fallback_model
    prompt  = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs  = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            temperature=temperature, do_sample=temperature > 0,
            pad_token_id=tok.eos_token_id,
        )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


# ── Hint ladder ───────────────────────────────────────────────────────────────
# Injected as an instruction prefix into the user message so the model gets an
# explicit signal rather than relying on it counting turns itself.

_LADDER = {
    # (min_turn, max_turn): instruction prepended to the user message
    (1, 2): "",           # pure Socratic — no extra instruction needed
    (3, 4): (
        "[ESCALATE: The student has been trying for several turns. "
        "Give a strong, concrete hint that almost reveals the answer, "
        "then ask ONE final guiding question. Do not give the full answer yet.]"
    ),
    (5, 99): (
        "[REVEAL: The student has made a genuine effort over many turns. "
        "Now give a clear, warm 2-sentence explanation of the answer. "
        "End with an encouraging remark, not a question.]"
    ),
}

def _ladder_instruction(turn_count: int) -> str:
    for (lo, hi), instr in _LADDER.items():
        if lo <= turn_count <= hi:
            return instr
    return ""


# ── Text helpers ──────────────────────────────────────────────────────────────
_EXPLAIN_RE = re.compile(
    r"^(tell me (about|what|how)|explain|what (is|are|does|do)|describe|define|how does|how do)\b",
    re.IGNORECASE,
)

def _rewrite(student_msg: str, turn_count: int = 1) -> str:
    """
    Pre-process the student message:
      - If it is an "explain X" request AND we are still in Socratic mode,
        rewrite it so the model gives a hint instead of a definition.
      - If the ladder says to reveal, leave it alone (the instruction prefix
        already tells the model to give the answer).
    """
    msg = student_msg.strip()
    ladder = _ladder_instruction(turn_count)
    if ladder:
        # Ladder instruction is prepended separately in QASession.chat()
        return msg

    if _EXPLAIN_RE.match(msg):
        topic = _EXPLAIN_RE.sub("", msg).strip().strip("?").strip() or msg
        return (
            f'The student wants to learn about "{topic}". '
            f"Do NOT define or explain it. "
            f"Give ONE short real-world analogy as a hint "
            f"and end with a single question that nudges them toward understanding. "
            f"Two sentences maximum."
        )
    return msg


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
    Post-processing safety net.
    - In reveal mode (turn 5+): allow full answer, skip enforcement.
    - In escalate mode (turn 3-4): allow longer text, just ensure it ends properly.
    - In Socratic mode (turn 1-2): trim to first sentence + nudge if no question.
    """
    text = text.strip()
    is_reveal    = turn_count >= 5
    is_escalate  = 3 <= turn_count <= 4

    if is_reveal:
        # Accept full answer — just clean up trailing questions added by habit
        return text

    if text.rstrip().endswith("?"):
        if not is_escalate and len(text) > 200:
            # Too long for Socratic mode — keep only the last question sentence
            sentences = re.split(r"(?<=[.!?])\s+", text)
            for s in reversed(sentences):
                if s.rstrip().endswith("?"):
                    return s
        return text

    # No question at the end
    sentences = re.split(r"(?<=[.!])\s+", text)
    first = sentences[0].rstrip(".!") if sentences else text.rstrip(".!")
    lower = student_msg.lower()
    if any(w in lower for w in ("explain", "tell me", "what is", "describe", "define")):
        return first + ". Before I say more — what do you already think this might mean?"
    return first + ". What do you think about that?"


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a robot tutor helping secondary school students learn computer science fundamentals — topics like data structures, algorithms, programming concepts, and how computers work. You NEVER give a direct definition or explanation. Instead, you guide the student one small step at a time using simple analogies and one question per reply, so they discover the answer themselves.

Rules:
- NEVER start a reply with a definition like "A queue is..." or "A variable is...".
- Every reply must end with a question that moves the student one step closer to understanding.
- Use simple real-world analogies relevant to school life or everyday experience.
- If the student says "explain", "tell me", "what is", or "just tell me" — give a concrete analogy as a hint, then ask a question. Do NOT define.
- Keep every reply to one or two short, clear sentences.
- Be encouraging and patient. Treat wrong answers as good attempts.

Examples:

Student: what is a queue?
Tutor: Think about students lining up to enter a classroom — who gets in first?

Student: the one at the front
Tutor: Right! So if a queue in programming works the same way, which item do you think gets removed first — the one added first or last?

Student: explain queues to me
Tutor: Before I do — imagine a printer with ten documents sent to it at once. In what order do you think it should print them?

Student: what is a variable?
Tutor: Imagine a whiteboard where your teacher writes the current score during a class quiz — what happens to what is written on it as the score changes?

Student: it gets erased and rewritten?
Tutor: Exactly! In programming, that whiteboard has a name so the program can find it again. What do you think we might call a named storage place that can hold changing values?

Student: what is a linked list?
Tutor: Have you ever done a treasure hunt where each clue tells you where the next one is hidden? What do you think happens if one clue goes missing?

Student: you can't find the rest
Tutor: Precisely — so in a linked list, each item points to the next. Can you think of a situation where this is better than storing everything in a row, like seats in a cinema?

Student: what is a stack?
Tutor: Think about a stack of textbooks on your desk — if you need the bottom one, what do you have to do first?

Student: remove the ones on top
Tutor: So which item can you only access last if it was put on first? Does that remind you of any rule you have heard before?

Student: what is an algorithm?
Tutor: Think about a recipe for making toast. What do you think would happen if you put the bread in the toaster before you plugged it in?

Student: it wouldn't work
Tutor: Right — order matters. So what do you think an algorithm might have in common with a recipe?

Student: just tell me the answer
Tutor: One more nudge — think about what you already said and apply it. What is your best guess based on that?

Now continue in exactly this style. One analogy or one question per reply. Never define. Always guide."""


# ── Public API ────────────────────────────────────────────────────────────────
def ask(question: str, history: list = None, gpu_id: int = 0) -> str:
    """Single-turn Q&A (stateless). Uses turn_count=1 (Socratic mode)."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": _rewrite(question, turn_count=1)})
    raw = _qa_generate(messages, temperature=0.5, max_new_tokens=120, gpu_id=gpu_id)
    return _enforce_hint(_clean(raw), question, turn_count=1)


class QASession:
    """
    Multi-turn dialogue with hint ladder escalation.

    The session automatically escalates hint strength as turns increase:
      turns 1-2  → Socratic (pure analogy + question)
      turns 3-4  → Stronger hint (almost the answer + one final question)
      turn  5+   → Full explanation (warm, 2 sentences)
    """

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, gpu_id: int = 0):
        self.system_prompt = system_prompt
        self.history: list  = []
        self.turn_count: int = 0
        self.gpu_id: int     = gpu_id

    def chat(self, question: str,
             temperature: float = 0.5,
             max_new_tokens: int = 120) -> str:
        """Send a message, apply hint ladder, return the tutor response."""
        self.turn_count += 1
        tc = self.turn_count

        ladder = _ladder_instruction(tc)
        user_content = (f"{ladder}\n{_rewrite(question, tc)}").strip() if ladder else _rewrite(question, tc)

        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_content})

        raw    = _qa_generate(messages, temperature=temperature,
                              max_new_tokens=max_new_tokens, gpu_id=self.gpu_id)
        answer = _enforce_hint(_clean(raw), question, turn_count=tc)

        # Store original student message (not the rewritten one) so history reads naturally
        self.history.append({"role": "user",      "content": question})
        self.history.append({"role": "assistant",  "content": answer})
        return answer

    def reset(self):
        self.history    = []
        self.turn_count = 0

    def get_history(self) -> list:
        return self.history

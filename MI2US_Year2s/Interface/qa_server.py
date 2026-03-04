"""
Q&A WebSocket server for EduBot.

Inference backends
──────────────────
  USE_VLLM=1   vLLM AsyncLLMEngine — true continuous batching.
               Requires sm_7.0+ GPU (e.g. RTX 3090).
               All 4 client sessions are served in parallel with no lock.

  USE_VLLM=0   4-GPU worker pool (default, works on current TITAN X setup).
  (default)    One 4-bit model per GPU; each GPU has its own asyncio Queue.
               Requests are routed to the least-busy GPU, so up to 4 clients
               run inference simultaneously.

Environment variables
──────────────────────
  USE_VLLM=1        Switch to vLLM backend
  QA_PORT=10000     WebSocket port (default 10000)
  HTTP_PORT=8080    Static file server port (default 8080)
  GPU_IDS=0,1,2,3   Which GPUs to use (default: all available)

Usage
──────
  # 4-GPU pool (current hardware)
  python qa_server.py

  # vLLM (RTX 3090 or similar)
  USE_VLLM=1 python qa_server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from asyncio import Queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import websockets
import torch
from qa import (
    QASession, SYSTEM_PROMPT,
    _qa_generate, _vllm_generate,
    _clean, _enforce_hint, _rewrite, _ladder_instruction,
    init_gpu_pool, _USE_VLLM,
)

# ── Config ────────────────────────────────────────────────────────────────────
_QA_PORT   = int(os.getenv("QA_PORT",   "10000"))
_HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))
_NUM_GPUS  = torch.cuda.device_count()

_gpu_ids_env = os.getenv("GPU_IDS", "")
GPU_IDS: list[int] = (
    [int(x) for x in _gpu_ids_env.split(",") if x.strip().isdigit()]
    if _gpu_ids_env else list(range(max(_NUM_GPUS, 1)))
)

# ── GPU worker pool ───────────────────────────────────────────────────────────
# One asyncio Queue per GPU. Each queue is drained by a dedicated worker
# coroutine that runs inference in a ThreadPoolExecutor.  Requests go to the
# GPU whose queue is shortest (least-busy routing).

_gpu_queues: dict[int, Queue] = {}       # gpu_id -> asyncio.Queue
_gpu_executor = ThreadPoolExecutor(max_workers=len(GPU_IDS) or 1)


async def _gpu_worker(gpu_id: int):
    """Drain the queue for gpu_id, running inference in a thread."""
    q = _gpu_queues[gpu_id]
    while True:
        future, messages, temperature, max_tokens = await q.get()
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                _gpu_executor,
                lambda: _qa_generate(messages, temperature, max_tokens, gpu_id),
            )
            if not future.done():
                future.set_result(result)
        except Exception as exc:
            if not future.done():
                future.set_exception(exc)
        finally:
            q.task_done()


def _least_busy_gpu() -> int:
    """Return the GPU id whose queue is currently shortest."""
    return min(_gpu_queues, key=lambda g: _gpu_queues[g].qsize())


async def _submit_inference(messages: list, temperature: float, max_tokens: int) -> str:
    """Route one inference request to the best available GPU."""
    loop   = asyncio.get_event_loop()
    future = loop.create_future()
    gpu_id = _least_busy_gpu()
    await _gpu_queues[gpu_id].put((future, messages, temperature, max_tokens))
    return await future


# ── Per-client state ──────────────────────────────────────────────────────────
_sessions: dict[int, QASession] = {}
_settings: dict[int, dict]      = {}

DEFAULT_SETTINGS = {
    "model":       "coding",
    "temperature": 0.5,
    "max_tokens":  120,
    "hint_mode":   True,
    "hint_level":  2,        # 1=gentle 2=normal 3=direct
    "volume":      80,
    "tts":         False,
    "mute":        False,
}

HINT_PROMPTS = {
    1: "Give a very gentle nudge — just one small hint. Keep it to one sentence.",
    2: "Guide with a hint and one follow-up question. Two sentences maximum.",
    3: "Be more direct but still do not give the full answer. Ask a pointed question.",
}


def _build_system_prompt(settings: dict) -> str:
    if not settings.get("hint_mode", True):
        return (
            "You are a helpful assistant. Answer questions clearly and concisely "
            "in spoken language. No code blocks or bullet points."
        )
    level = settings.get("hint_level", 2)
    hint_instr = HINT_PROMPTS.get(level, HINT_PROMPTS[2])
    return SYSTEM_PROMPT + f"\n\nHint style for this session: {hint_instr}"


# ── WebSocket handler ─────────────────────────────────────────────────────────
async def handler(websocket, path):
    client_id = id(websocket)
    settings  = dict(DEFAULT_SETTINGS)

    # Assign this client to the least-busy GPU (only meaningful in pool mode)
    gpu_id = _least_busy_gpu() if not _USE_VLLM else 0
    session = QASession(system_prompt=_build_system_prompt(settings), gpu_id=gpu_id)

    _sessions[client_id] = session
    _settings[client_id] = settings
    print(f"[INFO] Client connected: {websocket.remote_address}  "
          f"(gpu:{gpu_id}, backend:{'vllm' if _USE_VLLM else 'pool'})")

    try:
        async for raw in websocket:
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                msg = {"type": "question", "text": raw}

            msg_type = msg.get("type", "question")

            # ── question ──────────────────────────────────────────────────
            if msg_type == "question":
                question = msg.get("text", "").strip()
                if not question:
                    continue

                s  = _settings[client_id]
                tc = session.turn_count + 1   # what the turn count will be

                print(f"[Q] client={client_id} gpu={gpu_id} turn={tc}  {question[:60]}")

                # Tell the UI we are working (queued or thinking)
                await websocket.send(json.dumps({"type": "thinking", "turn": tc}))

                # Build messages
                ladder      = _ladder_instruction(tc)
                user_content = (f"{ladder}\n{_rewrite(question, tc)}").strip() if ladder else _rewrite(question, tc)
                messages = [{"role": "system", "content": _build_system_prompt(s)}]
                messages.extend(session.get_history())
                messages.append({"role": "user", "content": user_content})

                # Run inference
                try:
                    if _USE_VLLM:
                        raw_answer = await _vllm_generate(messages, s["temperature"], s["max_tokens"])
                    else:
                        raw_answer = await _submit_inference(messages, s["temperature"], s["max_tokens"])
                except Exception as exc:
                    print(f"[ERROR] Inference failed: {exc}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "text": "Sorry, something went wrong. Please try again.",
                    }))
                    continue

                answer = _enforce_hint(_clean(raw_answer), question, turn_count=tc)

                # Update session state
                session.history.append({"role": "user",      "content": question})
                session.history.append({"role": "assistant",  "content": answer})
                session.turn_count = tc

                print(f"[A] turn={tc}  {answer[:80]}...")
                await websocket.send(json.dumps({
                    "type":       "answer",
                    "text":       answer,
                    "model":      s["model"],
                    "turn":       tc,
                    "hint_stage": (
                        "reveal"   if tc >= 5 else
                        "escalate" if tc >= 3 else
                        "socratic"
                    ),
                }))

            # ── settings ──────────────────────────────────────────────────
            elif msg_type == "settings":
                s = _settings[client_id]
                for key in DEFAULT_SETTINGS:
                    if key in msg:
                        s[key] = msg[key]
                # Rebuild session system prompt with new settings
                session.system_prompt = _build_system_prompt(s)
                print(f"[INFO] Settings updated for client {client_id}: "
                      f"model={s['model']} temp={s['temperature']} "
                      f"hint={s['hint_mode']}({s['hint_level']})")
                await websocket.send(json.dumps({"type": "settings_ok"}))

            # ── reset ─────────────────────────────────────────────────────
            elif msg_type == "reset":
                session.reset()
                await websocket.send(json.dumps({
                    "type": "reset_ok",
                    "text": "Conversation cleared.",
                }))
                print(f"[INFO] Session reset — client {client_id}")

            # ── close ─────────────────────────────────────────────────────
            elif msg_type == "close":
                await websocket.close()
                break

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _sessions.pop(client_id, None)
        _settings.pop(client_id, None)
        print(f"[INFO] Client disconnected: {websocket.remote_address}")


# ── Startup ───────────────────────────────────────────────────────────────────
async def _start_pool_workers():
    """Create one asyncio Queue + worker coroutine per GPU."""
    for gpu_id in GPU_IDS:
        _gpu_queues[gpu_id] = Queue()
        asyncio.ensure_future(_gpu_worker(gpu_id))
    print(f"[INFO] GPU worker queues started for GPUs: {GPU_IDS}")


def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    if _USE_VLLM:
        from qa import _init_vllm
        print("[INFO] Initialising vLLM engine ...")
        _init_vllm()
        # vLLM is async internally — no pool workers needed
        # We still create a single dummy queue so _least_busy_gpu() works
        _gpu_queues[0] = Queue()
    else:
        # Pre-load all GPU models, then start worker coroutines
        print(f"[INFO] Loading model on {len(GPU_IDS)} GPU(s): {GPU_IDS} ...")
        load_thread = threading.Thread(target=init_gpu_pool, args=(GPU_IDS,), daemon=True)
        load_thread.start()
        load_thread.join()
        loop.run_until_complete(_start_pool_workers())

    server = websockets.serve(handler, "0.0.0.0", _QA_PORT)
    loop.run_until_complete(server)
    print(f"[INFO] Q&A WebSocket server on port {_QA_PORT}  "
          f"(backend: {'vllm' if _USE_VLLM else f'{len(GPU_IDS)}-gpu-pool'})")
    loop.run_forever()


if __name__ == "__main__":
    base  = Path(__file__).resolve().parents[3]
    cache = base / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME",            str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE",  str(cache))

    # Static file HTTP server
    import subprocess
    subprocess.Popen(
        [sys.executable, "-m", "http.server", str(_HTTP_PORT)],
        cwd=_HERE,
    )
    print(f"[INFO] HTTP server on port {_HTTP_PORT}  "
          f"— open: http://localhost:{_HTTP_PORT}/qa.html")

    run()

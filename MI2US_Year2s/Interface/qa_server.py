"""
Q&A WebSocket server for EduBot (simple backend).

- One global Phi-3-mini model loaded in qa.py on cuda:0 (if available).
- All WebSocket clients share that single model instance.
- Inference is run in a background thread with a global lock so the
  event loop stays responsive, but only one generation runs at a time.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import websockets
from qa import (
    QASession, SYSTEM_PROMPT,
    _qa_generate, _clean, _enforce_hint, _rewrite, _ladder_instruction,
    _load_model, _ladder_stage, _detect_frustration,
    STAGE_EXPLORE, STAGE_NUDGE, STAGE_STRONG, STAGE_EXPLAIN,
)

# ── Config ────────────────────────────────────────────────────────────────────
_PORT = int(os.getenv("PORT", "8080"))

# ── MIME types for static file serving ────────────────────────────────────────
_MIME = {
    '.html': 'text/html; charset=utf-8',
    '.css':  'text/css; charset=utf-8',
    '.js':   'application/javascript; charset=utf-8',
    '.json': 'application/json',
    '.png':  'image/png',
    '.jpg':  'image/jpeg',
    '.jpeg': 'image/jpeg',
    '.ico':  'image/x-icon',
    '.svg':  'image/svg+xml',
    '.mp4':  'video/mp4',
    '.txt':  'text/plain; charset=utf-8',
}

# Single-thread executor + global lock for GPU use
_inference_executor = ThreadPoolExecutor(max_workers=1)
_inference_lock     = asyncio.Lock()

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
    1: "Give a very gentle nudge — just one small hint, then ask a question. Keep it to one or two sentences.",
    2: "Guide with a hint and one follow-up question. Two sentences maximum.",
    3: "Be more direct: give a strong specific hint, but still end with a question. Don't give the full answer yet.",
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
    session   = QASession(system_prompt=_build_system_prompt(settings))

    _sessions[client_id] = session
    _settings[client_id] = settings
    print(f"[INFO] Client connected: {websocket.remote_address}")

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

                # Detect frustration for bump preview
                frustrated = _detect_frustration(question)
                preview_bumps = session._frustration_bumps + (2 if frustrated else 0)
                effective_tc = tc + preview_bumps
                stage = _ladder_stage(effective_tc)

                print(f"[Q] client={client_id} turn={tc} eff={effective_tc} stage={stage}  {question[:60]}")

                # Tell the UI we are working
                await websocket.send(json.dumps({"type": "thinking", "turn": tc}))

                # Build messages
                ladder      = _ladder_instruction(effective_tc)
                user_content = (f"{ladder}\n{_rewrite(question, effective_tc)}").strip() if ladder else _rewrite(question, effective_tc)
                messages = [{"role": "system", "content": _build_system_prompt(s)}]
                messages.extend(session.get_history())
                messages.append({"role": "user", "content": user_content})

                # Run inference in background thread, one at a time
                try:
                    loop = asyncio.get_event_loop()
                    async with _inference_lock:
                        raw_answer = await loop.run_in_executor(
                            _inference_executor,
                            lambda: _qa_generate(messages, s["temperature"], s["max_tokens"]),
                        )
                except Exception as exc:
                    print(f"[ERROR] Inference failed: {exc}")
                    await websocket.send(json.dumps({
                        "type": "error",
                        "text": "Sorry, something went wrong. Please try again.",
                    }))
                    continue

                answer = _enforce_hint(_clean(raw_answer), question, turn_count=effective_tc)

                # Update session state (session.chat does this internally,
                # but we're driving it manually here for the executor pattern)
                session.history.append({"role": "user",      "content": question})
                session.history.append({"role": "assistant",  "content": answer})
                session.turn_count = tc
                if frustrated:
                    session._frustration_bumps += 2

                print(f"[A] turn={tc} stage={stage}  {answer[:80]}...")
                await websocket.send(json.dumps({
                    "type":       "answer",
                    "text":       answer,
                    "model":      s["model"],
                    "turn":       tc,
                    "effective_turn": effective_tc,
                    "stage":      stage,
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


# ── Static file serving via process_request ───────────────────────────────────
async def _serve_static(path, request_headers):
    """Serve static files for plain HTTP; return None for WebSocket upgrades."""
    if request_headers.get("Upgrade", "").lower() == "websocket":
        return None  # hand off to WebSocket handler

    if path == "/":
        path = "/qa.html"

    # Security: prevent path traversal
    safe = path.lstrip("/").replace("..", "")
    file_path = (_HERE / safe).resolve()
    if not str(file_path).startswith(str(_HERE)):
        return (403, [("Content-Type", "text/plain")], b"Forbidden")

    if not file_path.is_file():
        return (404, [("Content-Type", "text/plain")], b"404 Not Found")

    ext = file_path.suffix.lower()
    ctype = _MIME.get(ext, "application/octet-stream")
    body = file_path.read_bytes()
    return (200, [("Content-Type", ctype), ("Cache-Control", "no-cache")], body)


# ── Startup ───────────────────────────────────────────────────────────────────
def run():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Eagerly load the model once at startup so the first
    # question from any client is fast.
    try:
        _load_model()
    except Exception as exc:
        print(f"[ERROR] Failed to load Q&A model: {exc}")
        raise

    server = websockets.serve(
        handler, "0.0.0.0", _PORT,
        process_request=_serve_static,
    )
    loop.run_until_complete(server)
    print(f"[INFO] Combined HTTP + WebSocket server on port {_PORT}")
    print(f"[INFO] Open: http://localhost:{_PORT}/qa.html")
    loop.run_forever()


if __name__ == "__main__":
    base  = Path(__file__).resolve().parents[3]
    cache = base / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME",            str(cache))
    os.environ.setdefault("TRANSFORMERS_CACHE",  str(cache))

    run()

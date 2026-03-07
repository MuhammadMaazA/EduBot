# EduBot — Full Deployment Guide (Modal Cloud GPU)

> **Goal**: Host your fine-tuned Phi-3-mini + coding adapter on a cloud GPU so **4 students
> can use it simultaneously** from any browser — even on a laptop with no GPU.
>
> **Estimated cost for a 3-hour session**: **$0.00** (Modal gives $30 free credits on signup,
> and this uses ~$1.80 total).

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Cost Comparison](#2-cost-comparison)
3. [Prerequisites](#3-prerequisites)
4. [Master Checklist](#4-master-checklist)
5. [Part 1 — Push Your Adapter to HuggingFace (on blaze)](#5-part-1--push-your-adapter-to-huggingface-on-blaze)
6. [Part 2 — Deploy on Modal (on your laptop or any machine)](#6-part-2--deploy-on-modal-on-your-laptop-or-any-machine)
7. [Part 3 — Test It](#7-part-3--test-it)
8. [Part 4 — Connect the Frontend](#8-part-4--connect-the-frontend)
9. [Part 5 — Run the Experiment (4 Users, 3 Hours)](#9-part-5--run-the-experiment-4-users-3-hours)
10. [Troubleshooting](#10-troubleshooting)
11. [Quick Reference](#11-quick-reference)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────┐
│                    YOUR LAPTOP / LAB PCs                  │
│                                                          │
│   Student 1 ─┐                                           │
│   Student 2 ─┤──── Browser (qa_modal.html) ──── HTTPS    │
│   Student 3 ─┤               │                           │
│   Student 4 ─┘               │                           │
└──────────────────────────────│───────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────┐
│              MODAL (Cloud GPU — T4 / A10G)               │
│                                                          │
│   ┌─────────────────────────────────────────────┐        │
│   │  modal_deploy.py                            │        │
│   │                                             │        │
│   │  • Phi-3-mini base model (downloaded once)  │        │
│   │  • Your coding LoRA adapter (from HF)       │        │
│   │  • All hint ladder + Socratic logic         │        │
│   │  • Handles 4 concurrent requests            │        │
│   │  • Auto-scales: 0 → 1 container on demand   │        │
│   └─────────────────────────────────────────────┘        │
│                                                          │
│   URL: https://<your-workspace>--edubot-chat.modal.run   │
└──────────────────────────────────────────────────────────┘
```

**How concurrency works**: Modal loads the model **once** (~8 GB VRAM) on a single T4 (16 GB).
`allow_concurrent_inputs=4` lets it handle 4 requests on that same GPU — **NOT** 4 separate GPUs.
You only pay for 1 GPU.

---

## 2. Cost Comparison

| Platform | GPU | Cost/hr | 3 hrs total | Reliability | Ease |
|---|---|---|---|---|---|
| **Modal** | T4 (16 GB) | ~$0.59 | ~$1.77 | ✅ Very high | ✅ Easiest |
| RunPod | RTX 3090 | ~$0.34 | ~$1.02 | ✅ High | Medium |
| Vast.ai | RTX 3090 | ~$0.15 | ~$0.45 | ⚠️ Variable | Harder |
| HF Endpoints | T4 | ~$0.60 | ~$1.80 | ✅ Very high | Medium |

**Verdict**: Use **Modal** — free tier ($30 credit) covers your experiment many times over.
One-command deploy, one URL to share. If container idles between questions, billing pauses.

---

## 3. Prerequisites

You will need:

| What | Where to get it | Notes |
|---|---|---|
| **HuggingFace account** | https://huggingface.co/join | Free |
| **HuggingFace token** (write access) | https://huggingface.co/settings/tokens → New token → select "Write" | Needed to push adapter |
| **Modal account** | https://modal.com → Sign up (GitHub login works) | $30 free credits |
| **Python 3.9+** on your laptop/lab machine | Already on blaze; install on laptop if needed | For running `modal deploy` |

---

## 4. Master Checklist

Work through this in order. Check off each step as you complete it.

| # | Step | Where | Done? |
|---|---|---|---|
| 1 | Create HuggingFace account (if you don't have one) | Browser | ☐ |
| 2 | Create a HuggingFace token (Write access) | Browser | ☐ |
| 3 | Install `huggingface_hub` on blaze | blaze terminal | ☐ |
| 4 | Login to HuggingFace CLI on blaze | blaze terminal | ☐ |
| 5 | Push adapter to HuggingFace | blaze terminal | ☐ |
| 6 | Verify adapter is on HuggingFace | Browser | ☐ |
| 7 | Create Modal account | Browser | ☐ |
| 8 | Install `modal` on your laptop (or blaze) | terminal | ☐ |
| 9 | Authenticate Modal CLI | terminal | ☐ |
| 10 | Deploy `modal_deploy.py` | terminal | ☐ |
| 11 | Test with single curl request | terminal | ☐ |
| 12 | Test with 4 concurrent requests | terminal | ☐ |
| 13 | Open `qa_modal.html` in browser and test UI | Browser | ☐ |
| 14 | Share URL with 4 students and run experiment | Browser | ☐ |

---

## 5. Part 1 — Push Your Adapter to HuggingFace (on blaze)

Your fine-tuned LoRA adapter lives at:
```
/cs/student/projects1/2023/muhamaaz/social-robots/EduBot/checkpoints/phi3-coding-adapter/
```
It's small (~50–200 MB). You need to push it to HuggingFace so Modal can download it.

### Step 1: Get your HuggingFace token

1. Go to: **https://huggingface.co/settings/tokens**
2. Click **"New token"**
3. Name: `edubot-deploy`
4. Role: **Write**
5. Click **Create**
6. **Copy the token** (starts with `hf_...`) — you will paste it in the next step

### Step 2: SSH into blaze and install tools

```bash
ssh blaze
cd /cs/student/projects1/2023/muhamaaz/social-robots/EduBot
source MI2US_Year2s/Interface/venv39/bin/activate
pip install huggingface_hub
```

### Step 3: Login to HuggingFace

```bash
huggingface-cli login
```

When prompted, paste your token (the one starting with `hf_...`).
It will say: `Login successful`.

### Step 4: Push the adapter

```bash
huggingface-cli upload \
  MuhammadMaazA/phi3-coding-adapter \
  checkpoints/phi3-coding-adapter \
  --repo-type model
```

This will:
- Create a new repo `MuhammadMaazA/phi3-coding-adapter` on HuggingFace
- Upload all files from your adapter folder
- Should take 10–30 seconds

### Step 5: Verify

Go to: **https://huggingface.co/MuhammadMaazA/phi3-coding-adapter**

You should see these files:
- `adapter_config.json`
- `adapter_model.safetensors`
- `tokenizer.json`
- `tokenizer_config.json`
- `special_tokens_map.json`
- `chat_template.jinja`

✅ **Part 1 done!** Your adapter is now publicly available.

---

## 6. Part 2 — Deploy on Modal (on your laptop or any machine)

### Step 6: Create a Modal account

1. Go to: **https://modal.com**
2. Sign up (GitHub login is easiest)
3. You automatically get **$30 free credits**

### Step 7: Install Modal CLI

On your laptop (or on blaze — either works):

```bash
pip install modal
```

### Step 8: Authenticate

```bash
modal setup
```

This opens a browser window. Click "Approve" to link your terminal to your Modal account.

### Step 9: Save the deployment script

The file `modal_deploy.py` should already be in your EduBot folder (I created it for you).
If not, grab it from the repo. Make sure it's at:

```
EduBot/modal_deploy.py
```

### Step 10: Deploy!

```bash
cd /cs/student/projects1/2023/muhamaaz/social-robots/EduBot
modal deploy modal_deploy.py
```

**First deploy takes 3–5 minutes** (it builds a Docker image with all dependencies + downloads the model).
Subsequent deploys are fast (~15 seconds) because everything is cached.

When it finishes, you'll see something like:
```
✓ Created web function chat => https://muhamaaz--edubot-chat.modal.run
✓ Created web function health => https://muhamaaz--edubot-health.modal.run
```

**Copy the `chat` URL** — that's your API endpoint.

---

## 7. Part 3 — Test It

### Test 1: Single request

```bash
curl -X POST https://muhamaaz--edubot-chat.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a for loop?",
    "session_id": "test1",
    "temperature": 0.5,
    "max_tokens": 120,
    "hint_mode": true,
    "hint_level": 2
  }'
```

You should get back a JSON response like:
```json
{
  "answer": "Great question! Before I explain — have you ever had to repeat ...",
  "stage": "explore",
  "turn": 1,
  "model": "phi3-coding-adapter"
}
```

### Test 2: Four concurrent requests

```bash
modal run modal_deploy.py
```

This runs the built-in test function that fires 4 questions simultaneously.
You should see 4 answers come back within a few seconds of each other.

### Test 3: Health check

```bash
curl https://muhamaaz--edubot-health.modal.run
```

Should return:
```json
{"status": "ok", "model_loaded": true}
```

---

## 8. Part 4 — Connect the Frontend

### Option A: Use `qa_modal.html` (recommended)

I've created `qa_modal.html` in your Interface folder. This is a modified version of `qa.html`
that uses `fetch()` POST requests to the Modal URL instead of WebSocket.

1. Open `qa_modal.html`
2. Find this line near the top of the `<script>` section:
   ```javascript
   const API_URL = 'https://YOUR-WORKSPACE--edubot-chat.modal.run';
   ```
3. Replace `YOUR-WORKSPACE--edubot-chat.modal.run` with the actual URL from Step 10
4. Open `qa_modal.html` directly in a browser (no server needed — just double-click it)

### Option B: Serve it from blaze

If you want to serve the HTML from blaze:

```bash
cd /cs/student/projects1/2023/muhamaaz/social-robots/EduBot
bash scripts/start.sh
```

Then open `http://blaze:8080/qa_modal.html` — the HTML file will be served by the existing
server, but inference goes to Modal instead of the local GPU.

### Option C: Serve from anywhere with Python

```bash
cd MI2US_Year2s/Interface
python3 -m http.server 8080
```

Then open `http://localhost:8080/qa_modal.html`

---

## 9. Part 5 — Run the Experiment (4 Users, 3 Hours)

### Before the experiment

1. **Warm up the container**: Send one test request ~5 minutes before students arrive.
   Modal containers auto-sleep after 5 min of inactivity ("cold start" takes ~30-60s on first request).

   ```bash
   curl -X POST https://muhamaaz--edubot-chat.modal.run \
     -H "Content-Type: application/json" \
     -d '{"question": "hello", "session_id": "warmup"}'
   ```

2. **Give each student a different session ID**: The Modal endpoint tracks separate
   conversations per `session_id`. Each student's browser generates a unique one automatically
   via `qa_modal.html`.

3. **Share the URL**: Give students the link to `qa_modal.html` (served from blaze or locally).

### During the experiment

- All 4 students chat simultaneously — Modal handles concurrency
- Monitor from the Modal dashboard: https://modal.com/apps → click your app → see live logs
- Each request costs ~$0.0003 (fraction of a cent)

### After the experiment

Logs are available in the Modal dashboard. You can also add logging to the script.

To **stop billing**:
```bash
modal app stop edubot
```

Or just leave it — the container auto-sleeps after 5 min of no requests and you're not charged
while it's sleeping.

---

## 10. Troubleshooting

### "ModuleNotFoundError: No module named 'modal'"

```bash
pip install modal
```

### "modal setup" doesn't open a browser

On a headless server (like blaze), run:
```bash
modal token set --token-id <id> --token-secret <secret>
```
Get your token from: https://modal.com/settings

### First request is slow (~60s)

This is normal! It's the "cold start" — Modal is spinning up a container + loading the model.
Subsequent requests are fast (~2-5s). Warm up before the experiment.

### "CUDA out of memory"

Upgrade the GPU in `modal_deploy.py`:
```python
gpu="A10G",   # 24 GB VRAM instead of T4's 16 GB
```
Then redeploy: `modal deploy modal_deploy.py`

### Adapter not found

Make sure Step 5 completed successfully. Check:
https://huggingface.co/MuhammadMaazA/phi3-coding-adapter

If the repo is private, add your HF token to Modal:
```bash
modal secret create huggingface HF_TOKEN=hf_your_token_here
```

### "Connection refused" from the frontend

- Check the URL in `qa_modal.html` matches the one from `modal deploy` output
- Make sure you're using `https://` not `http://`
- Check the Modal dashboard for error logs

### Students report slow responses

- Check Modal logs for queue depth
- Consider upgrading to A10G for faster inference
- Reduce `max_tokens` in settings (shorter replies = faster)

---

## 11. Quick Reference

### Commands you'll actually use

| What | Command | Where |
|---|---|---|
| Push adapter to HF | `huggingface-cli upload MuhammadMaazA/phi3-coding-adapter checkpoints/phi3-coding-adapter --repo-type model` | blaze |
| Deploy to Modal | `modal deploy modal_deploy.py` | laptop or blaze |
| Test deployment | `curl -X POST <URL> -H "Content-Type: application/json" -d '{"question":"hello","session_id":"test"}'` | anywhere |
| Run 4-user test | `modal run modal_deploy.py` | laptop or blaze |
| Check status | `curl <health-URL>` | anywhere |
| View live logs | Modal dashboard → Apps → edubot | browser |
| Stop the app | `modal app stop edubot` | laptop or blaze |
| Redeploy after changes | `modal deploy modal_deploy.py` | laptop or blaze |

### Key URLs

| What | URL |
|---|---|
| HuggingFace adapter | `https://huggingface.co/MuhammadMaazA/phi3-coding-adapter` |
| Modal dashboard | `https://modal.com/apps` |
| API endpoint (chat) | `https://<workspace>--edubot-chat.modal.run` |
| API endpoint (health) | `https://<workspace>--edubot-health.modal.run` |
| Frontend | Open `qa_modal.html` in browser |

### Key files

| File | Purpose |
|---|---|
| `modal_deploy.py` | Modal deployment script (the cloud backend) |
| `qa_modal.html` | Frontend that talks to Modal (replaces WebSocket with fetch) |
| `qa.html` | Original frontend (uses WebSocket to local server) |
| `qa_server.py` | Original local server (not needed with Modal) |
| `qa.py` | Core Q&A logic (hint ladder etc. — copied into modal_deploy.py) |

---

## VRAM Budget

| Component | VRAM |
|---|---|
| Phi-3-mini base (float16) | ~7.2 GB |
| Coding LoRA adapter | ~0.1 GB |
| KV-cache (4 concurrent) | ~0.5 GB |
| **Total** | **~7.8 GB** |
| T4 available | 16 GB |
| **Headroom** | **~8.2 GB ✅** |

Plenty of room. No quantisation needed.

---

**You're done!** Start at [Part 1, Step 1](#step-1-get-your-huggingface-token) and work your way down.
Paste any errors into the chat and I'll help you fix them.

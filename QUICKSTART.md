# Quick Start Guide

## 3-Step Setup

### 1. Setup (First time only)
```bash
cd EduBot
bash scripts/setup.sh
```

### 2. Configure API Key
Edit `scripts/.env` and add your OpenAI API key:
```bash
OPENAI_API_KEY=your-api-key-here
```

### 3. Start
```bash
bash scripts/start.sh
```

### Access
Open in browser: `http://your-server:8080/storyGeneration.html`

### Stop
```bash
bash scripts/stop.sh
```

That's it!

## What Gets Installed

- Python virtual environment (`venv39/`)
- All Python dependencies (torch, transformers, etc.)
- Model checkpoints are already included in `.cache/huggingface/` (~7.2GB)
  - Phi-3-mini base model
  - Phi-Ed fine-tuned adapter

## Troubleshooting

**Port in use?**
```bash
bash scripts/stop.sh
bash scripts/start.sh
```

**Check logs:**
```bash
tail -f logs/edubot.log
```

**Reinstall dependencies:**
```bash
cd MI2US_Year2s/Interface
source venv39/bin/activate
pip install -r requirements.txt
```

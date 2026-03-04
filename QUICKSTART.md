# Quick Start Guide

## 3-Step Setup

### 1. Setup (first time only)
```bash
cd EduBot
bash scripts/setup.sh
```

### 2. Configure API Key
Edit `scripts/.env` and add your OpenAI API key (used only as fallback):
```
OPENAI_API_KEY=your-api-key-here
```

### 3. Start
```bash
bash scripts/start.sh          # Q&A dialogue mode (default)
bash scripts/start.sh story    # legacy story-generation mode
```

### Access
- **Q&A mode**: `http://your-server:8080/qa.html`
- **Story mode**: `http://your-server:8080/storyGeneration.html`

### Stop
```bash
bash scripts/stop.sh
```

That's it!

## What Gets Installed

- Python virtual environment (`venv39/`)
- All Python dependencies (torch, transformers, peft, datasets, etc.)
- Model checkpoints already included in `.cache/huggingface/` (~7.2 GB)
  - Phi-3-mini base model (`microsoft/Phi-3-mini-128k-instruct`)
  - Phi-Ed fine-tuned adapter (`Mortadha/Phi-Ed-25072024`)

## Fine-Tuning on Coding Data (A/B Comparison)

```bash
# 1. Fine-tune Phi-3-mini on CodeAlpaca-20k (needs GPU)
source MI2US_Year2s/Interface/venv39/bin/activate
python scripts/finetune_coding.py

# 2. Run A/B comparison: base vs coding-tuned
python scripts/compare_models.py
```

Adapter will be saved to `checkpoints/phi3-coding-adapter/`.

## Troubleshooting

**Port in use?**
```bash
bash scripts/stop.sh
bash scripts/start.sh
```

**Check logs:**
```bash
tail -f logs/qa_server.log    # Q&A mode
tail -f logs/edubot.log       # story mode
```

**Reinstall dependencies:**
```bash
cd MI2US_Year2s/Interface
source venv39/bin/activate
pip install -r requirements.txt
```

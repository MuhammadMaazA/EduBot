# EduBot - Educational AI Assistant

AI-powered educational robot using Phi-3-mini (with fine-tuning support) for conversational Q&A and multicultural classroom integration.

## Features

- **Q&A Dialogue Mode** (default): Back-and-forth conversational Q&A — ask any question and get a clear answer, with full conversation history per session
- **Story Generation Mode** (legacy): Age-appropriate stories for different educational levels
- **Fallback Logic**: Automatic fallback to OpenAI API if local model fails
- **Coding Fine-tune Pipeline**: Scripts to fine-tune on CodeAlpaca-20k and A/B compare vs base model
- **Multi-language Support**: English, French, German
- **Image Generation**: DALL-E 3 integration for story illustrations
- **Web Interface**: Modern web-based UI — no physical robot required for demo

## Quick Start

### 1. Setup

```bash
cd EduBot
bash scripts/setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Create necessary directories
- Set up configuration files

### 2. Configure API Key

Edit `scripts/.env` and add your OpenAI API key (used only as fallback):

```
OPENAI_API_KEY=your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Start EduBot (Q&A mode — default)

```bash
bash scripts/start.sh
```

Then open: `http://your-server:8080/qa.html`

### 4. Start in story mode (legacy)

```bash
bash scripts/start.sh story
```

Then open: `http://your-server:8080/storyGeneration.html`

### 5. Stop EduBot

```bash
bash scripts/stop.sh
```

## Project Structure

```
EduBot/
├── MI2US_Year2s/
│   └── Interface/
│       ├── qa.py              # Q&A dialogue module (multi-turn)
│       ├── qa_server.py       # Standalone Q&A WebSocket + HTTP server
│       ├── qa.html            # Q&A chat web interface
│       ├── storytelling.py    # LLM core (Phi-3 local + OpenAI fallback)
│       ├── main.py            # Legacy story-generation entry point
│       ├── server.py          # WebSocket server (legacy)
│       ├── imageGeneration.py # DALL-E image generation
│       └── *.html             # Web interface files
├── scripts/
│   ├── setup.sh               # Initial setup
│   ├── start.sh               # Start (qa or story mode)
│   ├── stop.sh                # Stop all processes
│   ├── finetune_coding.py     # Fine-tune on CodeAlpaca-20k dataset
│   ├── compare_models.py      # A/B compare base vs coding-tuned model
│   └── .env                   # API keys (template included)
├── checkpoints/
│   └── phi3-coding-adapter/   # Produced by finetune_coding.py
├── logs/                      # Application logs
├── .cache/
│   └── huggingface/           # Model checkpoints (~7.2 GB, included)
└── README.md
```

## How It Works

### Story Generation Flow

1. **Primary Method**: Uses Phi-3-mini with Phi-Ed fine-tuning (local, free)
2. **Fallback Method**: If Phi-3 fails, automatically uses OpenAI GPT-3.5-turbo
3. **Error Handling**: Clear error messages if both methods fail

### Age Groups Supported

- Toddlers
- Preschoolers
- Early Elementary
- Late Elementary
- Preteens

## Troubleshooting

### Model Loading

Model checkpoints are included in `.cache/huggingface/` directory. On first run, models will load from cache automatically. If cache is missing or corrupted, models will download automatically from HuggingFace (~7GB download).

### Port Already in Use

If ports 8080 or 10000 are in use:
```bash
bash scripts/stop.sh
# Wait a few seconds
bash scripts/start.sh
```

### Check Logs

```bash
tail -f logs/edubot.log        # Backend logs
tail -f logs/http_server.log   # HTTP server logs
```

### Virtual Environment Issues

If you encounter import errors:
```bash
cd MI2US_Year2s/Interface
source venv39/bin/activate
pip install -r requirements.txt
```

## Configuration

### Environment Variables

Set in `scripts/.env`:
- `OPENAI_API_KEY`: Required for fallback and image generation

### Cache Directories

All caches are stored in `EduBot/.cache/`:
- HuggingFace models: `.cache/huggingface/`
- Pip packages: `.cache/pip/`

## Development

### Testing Story Generation

```bash
cd MI2US_Year2s/Interface
source venv39/bin/activate
python ../../scripts/test_story_generation.py
```

### Testing Fallback Logic

```bash
cd MI2US_Year2s/Interface
source venv39/bin/activate
python ../../scripts/test_fallback.py
```

## Models & References

### AI Models Used

- **Phi-3-mini-128k-instruct**: Base language model by Microsoft
  - Source: https://huggingface.co/microsoft/Phi-3-mini-128k-instruct
  - License: MIT
  
- **Phi-Ed-25072024**: Fine-tuned educational adapter by Mortadha
  - Source: https://huggingface.co/Mortadha/Phi-Ed-25072024
  - Purpose: Educational storytelling fine-tuning

### Original Repository

This project is based on:
- **MI2US_Year2s**: https://github.com/dtozadore/MI2US_Year2s
- Original implementation for Alpha Mini robot with LLM storytelling

### Model Storage

Model checkpoints (~7.2GB) are stored in `.cache/huggingface/`:
- Base model: `models--microsoft--Phi-3-mini-128k-instruct/`
- Adapter: `models--Mortadha--Phi-Ed-25072024/`

Models are included in the repository and will load automatically on first run.

## License

See LICENSE files in subdirectories.

## Support

For issues or questions, check the logs in `logs/` directory.

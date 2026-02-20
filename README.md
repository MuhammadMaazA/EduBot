# EduBot - Educational Robot Storytelling System

AI-powered educational robot using Phi-3-mini (with Phi-Ed fine-tuning) for interactive storytelling and multicultural integration in classrooms.

## Features

- **Story Generation**: Age-appropriate stories for different educational levels
- **Multi-language Support**: English, French, German
- **Question Generation**: AI-generated questions with answers
- **Fallback Logic**: Automatic fallback to OpenAI API if local model fails
- **Image Generation**: DALL-E 3 integration for story illustrations
- **Web Interface**: Modern web-based UI for easy access

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

Edit `scripts/.env` and add your OpenAI API key:

```bash
OPENAI_API_KEY=your-api-key-here
```

Get your API key from: https://platform.openai.com/api-keys

### 3. Start EduBot

```bash
bash scripts/start.sh
```

### 4. Access Web Interface

Open in your browser:
- **Main Interface**: `http://your-server:8080/storyGeneration.html`
- **Lecture Generation**: `http://your-server:8080/lectureGeneration.html`

### 5. Stop EduBot

```bash
bash scripts/stop.sh
```

## Project Structure

```
EduBot/
├── MI2US_Year2s/
│   └── Interface/          # Main application code
│       ├── main.py        # Main application entry point
│       ├── storytelling.py # AI story generation (Phi-3 + fallback)
│       ├── server.py      # WebSocket server
│       ├── imageGeneration.py # DALL-E image generation
│       └── *.html         # Web interface files
├── scripts/
│   ├── setup.sh          # Initial setup script
│   ├── start.sh          # Start EduBot
│   ├── stop.sh           # Stop EduBot
│   └── .env              # Configuration (API keys - template included)
├── logs/                 # Application logs
├── .cache/               # Model checkpoints (~7.2GB, included)
│   └── huggingface/      # Phi-3-mini and Phi-Ed models
└── README.md            # This file
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

### AI Levels

- Level 0: Use story as-is
- Level 1: Simple AI enhancement
- Level 2: Creative AI enhancement
- Level 3: Complete story continuation
- Level 4: Generate story from questions
- Level 5-7: Lecture content generation

## Requirements

- **Python**: 3.9 or higher
- **RAM**: 16GB+ recommended (for local Phi-3 model)
- **GPU**: Optional but recommended (8GB+ VRAM)
- **Disk Space**: Models are included in `.cache/huggingface/` (~7.2GB). Additional ~1GB for dependencies.

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

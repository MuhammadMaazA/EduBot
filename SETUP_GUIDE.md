# Development Environment Setup Guide

Quick guide to get your development environment ready for the educational robot project.

---

## Prerequisites

- **Python:** 3.9 or higher
- **Git:** For version control
- **GPU:** Recommended (8GB+ VRAM) but not required for initial setup
- **OS:** Windows, macOS, or Linux

---

## Step 1: Python Environment

### Option A: Using venv (Standard)

```bash
# Navigate to project folder
cd "c:\Users\mmaaz\University\Miscellaneous\Outreach Project"

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### Option B: Using conda (Alternative)

```bash
# Create conda environment
conda create -n robot-edu python=3.10

# Activate
conda activate robot-edu
```

---

## Step 2: Install Basic Dependencies

### From Existing Project

```bash
cd MI2US_Year2s/Interface

# Install existing requirements
pip install -r requirements.txt
```

**Note:** The requirements.txt file has encoding issues. If it fails, install manually:

```bash
pip install openai pillow requests websockets pydantic tqdm
```

### For New Development (LLM + Robot)

```bash
# Core ML libraries
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
# OR for CPU only:
# pip install torch torchvision torchaudio

# Hugging Face ecosystem
pip install transformers
pip install peft
pip install accelerate
pip install bitsandbytes  # For QLoRA (requires CUDA)
pip install datasets

# Reachy Mini robot
pip install reachy-mini

# Optional: Simulation
pip install reachy-mini[simulation]

# Web server
pip install websockets

# Utilities
pip install python-dotenv  # For environment variables
```

---

## Step 3: Test Installations

### Test 1: PyTorch & CUDA

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

**Expected Output (with GPU):**
```
PyTorch version: 2.1.0
CUDA available: True
CUDA version: 11.8
GPU: NVIDIA GeForce RTX 3060
GPU Memory: 12.00 GB
```

**Expected Output (CPU only):**
```
PyTorch version: 2.1.0
CUDA available: False
```

### Test 2: Transformers & Model Loading

```python
from transformers import AutoTokenizer

# Test tokenizer (lightweight)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")
print("✓ Transformers working!")

# Test encoding
text = "Hello, Reachy Mini!"
tokens = tokenizer(text)
print(f"Tokenized: {tokens}")
```

### Test 3: Reachy Mini SDK

```python
try:
    from reachy_mini import ReachyMini
    from reachy_mini.utils import create_head_pose
    print("✓ Reachy Mini SDK installed!")
except ImportError as e:
    print(f"✗ Reachy Mini SDK not found: {e}")
```

---

## Step 4: Download Phi-Ed Model

### Option A: Automatic Download (Recommended)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# This will download the model (first time only)
model_name = "Mortadha/Phi-Ed-25072024"

print("Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model... (this may take a while)")
# For testing, use CPU first (no GPU required)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cpu",  # Change to "auto" when using GPU
    trust_remote_code=True
)

print("✓ Model downloaded and loaded successfully!")
print(f"Model size: {model.num_parameters() / 1e9:.2f}B parameters")
```

**Note:** Model is ~2-4GB. Download may take 5-30 minutes depending on internet speed.

### Option B: Manual Download

1. Go to https://huggingface.co/Mortadha/Phi-Ed-25072024
2. Click "Files and versions"
3. Download files to local folder
4. Load from local path:

```python
model = AutoModelForCausalLM.from_pretrained("./local/path/to/model")
```

---

## Step 5: Test Phi-Ed Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Load model and tokenizer
model_name = "Mortadha/Phi-Ed-25072024"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  # Use GPU if available
    trust_remote_code=True
)

# Create text generation pipeline
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Test generation
prompt = "Generate a short story about a robot for 5-year-old children:"
output = generator(
    prompt,
    max_length=200,
    num_return_sequences=1,
    temperature=0.7
)

print(output[0]['generated_text'])
```

**Expected:** Should generate an age-appropriate story about robots.

---

## Step 6: Test Existing Code (Optional)

### Test storytelling.py

```bash
cd MI2US_Year2s/Interface

# Edit storytelling.py and add your OpenAI API key
# Lines 4-6: OpenAI.api_key = "YOUR_KEY"

# Test at bottom of file
python storytelling.py
```

**Note:** This requires OpenAI API key. Skip if you don't have one.

### Test WebSocket Server

```python
# In one terminal:
python server.py

# In another terminal or script:
import asyncio
import websockets

async def test_client():
    uri = "ws://localhost:10000"
    async with websockets.connect(uri) as websocket:
        await websocket.send("Hello, server!")
        response = await websocket.recv()
        print(f"Received: {response}")

asyncio.run(test_client())
```

---

## Step 7: Set Up IDE (Optional but Recommended)

### VS Code
1. Install VS Code
2. Install Python extension
3. Install Jupyter extension (for notebooks)
4. Select your virtual environment as Python interpreter

### PyCharm
1. Install PyCharm Community Edition
2. Open project folder
3. Configure Python interpreter (select venv)

---

## Step 8: Create .env File (For API Keys)

```bash
# In project root
touch .env  # Mac/Linux
# OR manually create .env file on Windows
```

**Contents of .env:**
```
# OpenAI (if still using for comparison)
OPENAI_API_KEY=your_key_here

# Hugging Face (for private models)
HF_TOKEN=your_token_here

# Other settings
DEVICE=cuda  # or cpu
MODEL_NAME=Mortadha/Phi-Ed-25072024
```

**In your Python code:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
```

**Important:** .env is already in .gitignore (won't be committed)

---

## Common Issues & Solutions

### Issue 1: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Use 4-bit quantization
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
```

### Issue 2: bitsandbytes Installation Failed (Windows)
```
ERROR: Could not build wheels for bitsandbytes
```

**Solution:**
```bash
# Use pre-built Windows version
pip install bitsandbytes-windows
```

### Issue 3: Slow Model Loading
**Solution:** Models cache in `~/.cache/huggingface/`. First load is slow, subsequent loads are fast.

### Issue 4: Import Errors
```
ModuleNotFoundError: No module named 'transformers'
```

**Solution:**
```bash
# Make sure virtual environment is activated!
# You should see (venv) in your terminal prompt

# Verify Python location
which python  # Mac/Linux
where python  # Windows

# Should point to venv folder
```

---

## Verification Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created and activated
- [ ] PyTorch installed and CUDA detected (if GPU available)
- [ ] Transformers, PEFT, accelerate installed
- [ ] Reachy Mini SDK installed
- [ ] Phi-Ed model downloaded (2-4GB)
- [ ] Test inference runs successfully
- [ ] IDE configured (optional)
- [ ] .env file created (optional)

---

## Next Steps

Once setup is complete:

1. Read CODEBASE_ANALYSIS.md to understand existing code
2. Read REACHY_MINI_GUIDE.md to learn robot API
3. Read QLORA_GUIDE.md if fine-tuning needed
4. Start with simple examples in `/examples` folder
5. Begin migrating OpenAI calls to Phi-Ed

---

## Getting Help

### Resources
- **Transformers:** https://huggingface.co/docs/transformers
- **PEFT/LoRA:** https://huggingface.co/docs/peft
- **Reachy Mini:** https://huggingface.co/docs/reachy_mini
- **PyTorch:** https://pytorch.org/docs

### Community
- Hugging Face Discord: https://discord.gg/hugging-face
- Reachy Mini Discord: https://discord.gg/Y7FgMqHsub
- Professor: daniel.tozadore@epfl.ch

---

## Estimated Setup Time

- **Basic setup** (Python, libraries): 30-60 minutes
- **Model download**: 10-30 minutes (depending on internet)
- **Testing**: 15-30 minutes
- **Total**: 1-2 hours

---

*Last updated: February 10, 2026*

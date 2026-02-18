# How to Run the Project & Why We Can't Stream the Model

---

## Key Terms (Plain English)

**Model**
A file (or set of files) that contains a trained AI brain. You give it text, it gives text back. Think of it like a very advanced autocomplete.

**Standalone Model**
A model that is complete on its own — it has everything it needs to run by itself. Example: GPT-3.5 is a standalone model. You can point at it and say "run this" and it works.

**PEFT Adapter (what Phi-Ed is)**
PEFT stands for *Parameter-Efficient Fine-Tuning*. Instead of retraining an entire model from scratch (which takes weeks and costs thousands), you train a small "patch" that sits on top of an existing model and changes how it behaves. Phi-Ed is one of these patches — it teaches the base model to talk like an educational assistant for children. On its own, the adapter file is useless. It *needs* the base model underneath it to function. Like a DLC pack for a video game — it won't run without the original game installed.

**Base Model (Phi-3-mini)**
The full standalone model that Phi-Ed is built on top of. Made by Microsoft. About 3.8 billion parameters (a measure of how complex/capable it is). This is the "original game" in the analogy above.

**HuggingFace**
A website (huggingface.co) that hosts AI models, like a GitHub for AI. You can download models from it or, for some models, use their servers to run the model remotely without downloading it.

**HuggingFace Inference API**
A service HuggingFace offers where *they* run the model on *their* servers, and you just send text and get a response back over the internet. This is what "streaming" or "calling remotely" means. You don't need the model on your computer at all.

**Streaming**
Instead of waiting for the full response to be generated before seeing anything, streaming sends you the response word by word as it's being generated. Faster feeling UX. Both local and remote models can stream — the question is whether you *can* use a remote server at all.

**Local Inference**
Running the model on your own computer. The model files are downloaded to your machine, and all computation happens on your CPU or GPU. No internet needed after the first download.

**venv (Virtual Environment)**
An isolated Python environment. Keeps the libraries for this project separate from everything else on your computer so they don't conflict. Think of it like a separate clean room just for this project.

**transformers**
A Python library by HuggingFace that lets you load and run AI models locally on your computer.

**peft**
A Python library that lets you load PEFT adapters (like Phi-Ed) on top of a base model.

**accelerate**
A helper library that figures out whether to run the model on your GPU or CPU and sets it up automatically.

**torch (PyTorch)**
The underlying engine that actually does the math when the model generates text. The other libraries (transformers, peft, accelerate) all run on top of this.

---

## Why We Can't Stream Phi-Ed from HuggingFace's Servers

HuggingFace's Inference API only works with **standalone models**. It cannot load a PEFT adapter because an adapter is not a complete model — it's a patch on top of one.

When you call HuggingFace's servers, you say "run model X on this input". Their servers have model X ready to go. But Phi-Ed isn't a model — it's an add-on. To run Phi-Ed, the server would need to:

1. Load Phi-3-mini (the full base model, ~7GB)
2. Load the Phi-Ed adapter on top of it
3. Then run your input

HuggingFace's free Inference API does not support this setup for arbitrary adapters. To do this on HuggingFace's servers, you'd need to pay for a **Dedicated Inference Endpoint** (~$0.60/hour), which deploys a custom server just for your model combination.

**The bottom line:** To use Phi-Ed (the educational fine-tune), you must run it locally.

---

## What Happens When You Run It

The first time `main.py` runs and the AI is called, it will:

1. Download `Phi-3-mini` from HuggingFace (~7GB, saved to your HuggingFace cache)
2. Download the `Phi-Ed` adapter (~small, a few hundred MB)
3. Load both into memory
4. Start generating responses

Every run after the first is fast to load because the files are already cached on your machine.

---

## Requirements

- **RAM:** At least 16GB recommended (model runs on CPU if no GPU)
- **GPU (optional but faster):** Any NVIDIA GPU with 8GB+ VRAM — model will auto-detect and use it
- **Disk space:** ~8GB for the model files (downloaded once, stored in `C:\Users\mmaaz\.cache\huggingface`)
- **Python:** 3.10 or higher

---

## How to Run

### 1. Activate the virtual environment
Open a terminal in the `MI2US_Year2s/Interface` folder and run:
```bash
venv\Scripts\activate
```
You'll see `(venv)` appear at the start of your terminal prompt. This means you're inside the isolated environment.

### 2. Run the app
```bash
python main.py
```

The first run will take several minutes while the model downloads. After that it's much quicker.

### 3. To deactivate the venv when you're done
```bash
deactivate
```

---

## If You Want to Use a Remote Model Instead (No Phi-Ed Fine-tune)

If you don't have enough RAM/GPU and want to test quickly, you can swap to HuggingFace's hosted version of the *base* Phi-3-mini model. You'd lose the educational fine-tuning but gain the ability to run without downloading anything.

That would require changing `storytelling.py` to use `huggingface_hub.InferenceClient` instead of loading locally — ask for this if needed.

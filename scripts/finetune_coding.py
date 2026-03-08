"""
Fine-tune Phi-3-mini on a Python coding dataset using LoRA / QLoRA.

Dataset: iamtarun/python_code_instructions_18k_alpaca  (~18 600 Python instruction-response pairs)
  Source: https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca
  Columns: instruction, input, output, prompt

Streams directly from HuggingFace — no full dataset download needed.

Adapter saved to: EduBot/checkpoints/phi3-coding-adapter/

Usage (from EduBot root, in bash):
    export HF_HOME="$PWD/.cache/huggingface"
    nohup app/venv39/bin/python scripts/finetune_coding.py \
        > logs/finetune.log 2>&1 &
    echo $! > logs/finetune.pid
    tail -f logs/finetune.log   # watch progress

GPU memory guidance:
    - 8 GB VRAM : USE_4BIT=True  (QLoRA, default)
    - 16 GB VRAM: USE_8BIT=True
    - 24+ GB    : set both False for full float16
"""
import os
import sys
import time
import logging
from pathlib import Path

# --- Path / env setup ---
SCRIPT_DIR  = Path(__file__).resolve().parent
EDUBOT_ROOT = SCRIPT_DIR.parent
CACHE_DIR   = EDUBOT_ROOT / ".cache" / "huggingface"
OUTPUT_DIR  = EDUBOT_ROOT / "checkpoints" / "phi3-coding-adapter"
LOG_DIR     = EDUBOT_ROOT / "logs"

for d in (CACHE_DIR, OUTPUT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("HF_HOME",                   str(CACHE_DIR))
os.environ.setdefault("TRANSFORMERS_CACHE",         str(CACHE_DIR))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF",    "expandable_segments:True")

# Flush stdout/stderr immediately (important for nohup log tailing)
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# --- Imports ---
import torch
from datasets import load_dataset, IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType

# ============================================================
# Config
# ============================================================
BASE_MODEL   = "microsoft/Phi-3-mini-128k-instruct"
DATASET_NAME = "iamtarun/python_code_instructions_18k_alpaca"
MAX_SEQ_LEN  = 256   # halved to cut activation memory
NUM_EPOCHS   = 1
BATCH_SIZE   = 1    # minimum per-device batch
GRAD_ACCUM   = 16   # effective batch = 16, same training signal
LR           = 2e-4

USE_4BIT     = True
USE_8BIT     = False

LORA_R       = 8    # smaller rank = fewer trainable params
LORA_ALPHA   = 16
LORA_DROPOUT = 0.05
LORA_TARGETS = ["qkv_proj", "o_proj"]  # Phi-3-mini uses combined qkv_proj

# How many examples to buffer from the stream before training starts
STREAM_BUFFER = 18_600
# ============================================================


def format_example(row: dict) -> str:
    """Use the pre-built prompt column, or assemble from parts."""
    if row.get("prompt"):
        return row["prompt"].strip()
    instruction = row.get("instruction", "").strip()
    inp         = row.get("input", "").strip()
    output      = row.get("output", "").strip()
    text = f"### Instruction:\n{instruction}"
    if inp:
        text += f"\n\n### Input:\n{inp}"
    text += f"\n\n### Response:\n{output}"
    return text


def main():
    log.info("=== EduBot coding fine-tune ===")
    log.info(f"Dataset : {DATASET_NAME} (streaming)")
    log.info(f"Output  : {OUTPUT_DIR}")

    # ---- Stream dataset from HuggingFace ----
    log.info("Connecting to HuggingFace dataset stream…")
    stream = load_dataset(DATASET_NAME, split="train", streaming=True)

    log.info(f"Buffering up to {STREAM_BUFFER} examples…")
    examples = []
    for i, row in enumerate(stream):
        examples.append(row)
        if (i + 1) % 2000 == 0:
            log.info(f"  buffered {i+1} rows…")
        if i + 1 >= STREAM_BUFFER:
            break
    log.info(f"Dataset ready: {len(examples)} examples")

    # ---- Tokenizer ----
    log.info(f"Loading tokenizer: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=False,
        cache_dir=str(CACHE_DIR),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Tokenise ----
    log.info("Tokenising…")
    tokenised = []
    for row in examples:
        text = format_example(row)
        enc  = tokenizer(
            text,
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        tokenised.append({
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels":         enc["input_ids"].squeeze().clone(),
        })
    log.info(f"Tokenised {len(tokenised)} examples")

    # Wrap as a simple torch Dataset
    class ListDataset(torch.utils.data.Dataset):
        def __init__(self, data): self.data = data
        def __len__(self):        return len(self.data)
        def __getitem__(self, i): return self.data[i]

    train_dataset = ListDataset(tokenised)

    # ---- Model ----
    bnb_config = None
    if USE_4BIT and not USE_8BIT:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif USE_8BIT:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    log.info(f"Loading model: {BASE_MODEL} (4bit={USE_4BIT}, 8bit={USE_8BIT})")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        dtype=torch.float16 if not (USE_4BIT or USE_8BIT) else None,
        device_map="auto",
        trust_remote_code=False,
        attn_implementation="eager",
        cache_dir=str(CACHE_DIR),
    )

    # ---- LoRA ----
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()  # required for QLoRA + gradient checkpointing
    model.print_trainable_parameters()

    # ---- Train ----
    train_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,
        gradient_checkpointing=True,       # trade compute for memory
        optim="paged_adamw_8bit",          # 8-bit optimiser uses ~4x less memory
        logging_steps=25,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=False,       # reduces peak host memory
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    log.info("Starting training…")
    start = time.time()
    trainer.train()
    elapsed = (time.time() - start) / 60
    log.info(f"Training complete in {elapsed:.1f} min")

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    log.info(f"[SUCCESS] Adapter saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

# QLoRA Fine-Tuning Guide

## What is QLoRA?

**QLoRA (Quantized Low-Rank Adaptation)** is an efficient method for fine-tuning large language models on limited hardware resources.

### The Problem
- Fine-tuning a 7B parameter model typically requires 112GB VRAM (two 80GB A100 GPUs)
- Most researchers don't have access to expensive GPU clusters
- Full fine-tuning is slow and costly

### The Solution
QLoRA reduces memory requirements by:
1. **Quantization:** Storing the base model in 4-bit precision (instead of 16/32-bit)
2. **Low-Rank Adaptation (LoRA):** Only training small "adapter" layers
3. **Result:** Fine-tune large models on consumer GPUs (even a single 24GB GPU!)

---

## How QLoRA Works

### 1. Model Quantization
- Base model frozen in 4-bit precision
- Reduces memory footprint by ~75%
- Minimal performance loss

### 2. LoRA Adapters
- Small trainable matrices inserted into model layers
- Learn task-specific adaptations
- Only a few million parameters vs billions in base model

### 3. Training
- Only adapter weights updated during training
- Base model remains frozen
- Fast training (hours instead of days)

---

## Benefits for Your Project

1. **Affordable:** Fine-tune Phi-3-Mini (3.8B params) on laptop GPU
2. **Fast:** Training takes hours, not days
3. **Flexible:** Create multiple adapters for different tasks
4. **Reversible:** Keep base model, swap adapters as needed

---

## Your Use Case: Educational Robot

### What You'll Fine-Tune

**Base Model:** Microsoft Phi-3-Mini-128k-instruct (already fine-tuned by professor as Phi-Ed)

**Your Adapters (from paper):**

1. **Pedagogical Strategy Adapter**
   - Determines appropriate teaching strategies
   - Input: Student profile, topic, context
   - Output: Strategy recommendation (Socratic, direct instruction, etc.)

2. **Response Generation Adapter**
   - Generates responses aligned with chosen strategy
   - Input: Strategy + context + question
   - Output: Age-appropriate, personality-aligned response

3. **Personality Shaping** (via prompts)
   - Extraversion/introversion control
   - Achieved through prompt engineering, not separate adapter

---

## Technical Setup

### Required Libraries

```bash
pip install transformers>=4.40.0
pip install peft>=0.10.0
pip install torch>=2.1.0
pip install accelerate>=0.27.0
pip install bitsandbytes>=0.43.0
pip install datasets
```

### Hardware Requirements

**Minimum (Inference):**
- GPU: 8GB VRAM (GTX 1080, RTX 3060)
- RAM: 16GB
- Storage: 10GB

**Recommended (Fine-tuning):**
- GPU: 24GB VRAM (RTX 3090, RTX 4090, A100)
- RAM: 32GB
- Storage: 50GB

**Your Laptop:** Check GPU specs!

---

## Basic QLoRA Code Structure

### 1. Load Base Model in 4-bit

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Load Phi-Ed model
model = AutoModelForCausalLM.from_pretrained(
    "Mortadha/Phi-Ed-25072024",
    quantization_config=bnb_config,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("Mortadha/Phi-Ed-25072024")
```

### 2. Configure LoRA

```python
from peft import LoraConfig, get_peft_model

# LoRA configuration
lora_config = LoraConfig(
    r=16,  # Rank of adapter matrices
    lora_alpha=32,  # Scaling parameter
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Which layers to adapt
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# Check trainable parameters
model.print_trainable_parameters()
# Output: trainable params: 2.3M || all params: 3,821M || trainable%: 0.06%
```

### 3. Prepare Training Data

```python
from datasets import Dataset

# Your educational dialogue dataset
data = [
    {
        "input": "Student age: 7, Topic: Photosynthesis. Generate story.",
        "output": "Once upon a time, there was a little plant named Lily..."
    },
    {
        "input": "Strategy: Socratic method. Ask about water cycle.",
        "output": "What do you think happens when the sun heats the water?"
    },
    # ... more examples
]

dataset = Dataset.from_list(data)

def preprocess(examples):
    return tokenizer(
        examples["input"] + " " + examples["output"],
        truncation=True,
        max_length=512
    )

dataset = dataset.map(preprocess, batched=True)
```

### 4. Train

```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./phi-ed-story-adapter",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=100,
    evaluation_strategy="steps",
    eval_steps=100
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start training
trainer.train()

# Save adapter
model.save_pretrained("./phi-ed-story-adapter")
```

### 5. Inference

```python
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Mortadha/Phi-Ed-25072024",
    quantization_config=bnb_config,
    device_map="auto"
)

# Load your trained adapter
model = PeftModel.from_pretrained(base_model, "./phi-ed-story-adapter")

# Generate
prompt = "Generate a story about space for 5-year-olds:"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_length=200)
print(tokenizer.decode(outputs[0]))
```

---

## Training Data Preparation

### What You Need

1. **Story Generation Examples**
   - Input: Topic + age group
   - Output: Generated story

2. **Question Generation Examples**
   - Input: Story + age group
   - Output: 3 questions with answers

3. **Personality-Aligned Responses**
   - Input: Context + personality trait
   - Output: Response matching personality

4. **Multilingual Examples**
   - English, French, German versions
   - Translation pairs

### Data Sources

- Reuse prompts from existing `storytelling.py`
- Children's books (public domain)
- Educational content from previous study
- Synthetic data from GPT-4 (bootstrap)
- Professor's existing Phi-Ed training data

### Format

```json
{
  "instruction": "Generate an educational story about [topic] for [age group]",
  "input": "Topic: Rainbows, Age: Preschoolers (3-5 years)",
  "output": "Once upon a time, in a sunny meadow after a big rain...",
  "personality": "neutral",
  "language": "en"
}
```

---

## From the Research Papers

### Paper Findings

**Model:** Phi-3-Mini-128k-instruct → Fine-tuned with QLoRA
**Training:** Two QLoRA adapters
1. Pedagogical strategy selection
2. Personality-aligned response generation

**Results:**
- Consistent personality expression (validated via LIWC analysis)
- Students perceived personality shifts
- Personality affected engagement and learning outcomes

### Your Implementation

You'll be **using** the existing Phi-Ed model, not training from scratch. However, you may want to:

1. **Fine-tune for your specific content** (optional)
2. **Create adapters for new languages** (if needed)
3. **Adapt for Reachy Mini specific interactions**

---

## Step-by-Step Plan

### Week 1-2: Setup & Learning
1. Install all required libraries
2. Download Phi-Ed model from Hugging Face
3. Test basic inference (no fine-tuning yet)
4. Study existing code prompts

### Week 3-4: Data Preparation
1. Collect/create training examples
2. Format data properly
3. Split into train/validation sets
4. Quality check

### Week 5-6: Training (if needed)
1. Start with small dataset (test setup)
2. Monitor training metrics
3. Evaluate on validation set
4. Iterate and improve

### Week 7+: Integration
1. Integrate with Reachy Mini
2. Test end-to-end system
3. Refine based on testing
4. Prepare for user studies

---

## Resources

### Official Tutorials
1. **PyTorch torchtune:** https://docs.pytorch.org/torchtune/stable/tutorials/qlora_finetune.html
2. **MLflow + PEFT:** https://mlflow.org/docs/latest/llms/transformers/tutorials/fine-tuning/transformers-peft.html
3. **Towards AI Guide:** https://towardsai.net/p/machine-learning/fine-tuning-a-small-llm-with-qlora-a-complete-practical-guide-even-on-a-single-gpu

### Papers
1. QLoRA Paper: "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
2. LoRA Paper: "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
3. Your Project Papers (in your folder, not on GitHub!)

### Community
- Hugging Face Forums: https://discuss.huggingface.co/
- PEFT GitHub: https://github.com/huggingface/peft

---

## Common Issues & Solutions

### Issue 1: Out of Memory (OOM)
**Solution:**
- Reduce batch size
- Increase gradient accumulation steps
- Use smaller max_length
- Enable gradient checkpointing

### Issue 2: Slow Training
**Solution:**
- Use fp16 or bf16
- Increase batch size (if memory allows)
- Use larger GPU or distributed training

### Issue 3: Poor Quality Outputs
**Solution:**
- More training data
- Better data quality
- Adjust learning rate
- Longer training

### Issue 4: Model Not Loading
**Solution:**
- Check CUDA/PyTorch versions
- Update bitsandbytes
- Try CPU first (slow but works)

---

## Questions for Professor

1. Do we need to fine-tune or just use existing Phi-Ed?
2. Can we access the original training data?
3. What new capabilities do we need to add?
4. Hardware resources available at UCL?
5. Timeline for model development?

---

## Next Steps

1. [ ] Test loading Phi-Ed model locally
2. [ ] Run basic inference
3. [ ] Measure memory usage
4. [ ] Compare outputs to OpenAI GPT-3.5
5. [ ] Decide if additional fine-tuning needed
6. [ ] Prepare training data (if fine-tuning)
7. [ ] Set up training pipeline
8. [ ] Monitor and iterate

---

*Document created: February 10, 2026*
*Based on 2026 QLoRA tutorials and research papers*

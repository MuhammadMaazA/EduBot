# PhinetunEd
Originally implemented by Mortadha Abderrahim and Daniel Tozodore   
Adapters available at: https://huggingface.co/Mortadha/Phi-Ed-25072024

## Fine-tuning & Evaluation

To accommodate limited computing resources and ensure fast response times, we focus on Small Language Models (SLMs) with ~3B parameters or less. Our main model is **Phi-3-Mini-128k-instruct** (3.8B params, 128K context) — a leading open-source model in its size class at the time of fine-tuning.

We fine-tuned the model using **QLoRA** on a consumer GPU, leveraging 4-bit quantization and low-rank adapters for efficient training with minimal memory usage. Our training set consisted of 2,500 samples from **Cosmopedia**, a synthetic dataset with educational and commonsense-focused instruction tasks.

### Evaluation

We benchmarked the models on the following tasks:

- **MMLU**: General knowledge and reasoning
- **HellaSwag**: Story completion from multiple-choice options
- **WritingPrompts**: Creative story generation, judged by GPT-4-turbo

| Model            | MMLU  | HellaSwag | WritingPrompts |
|------------------|-------|-----------|----------------|
| GPT-3.5-turbo    | **70%** | **85.5%**   | 6.87           |
| Phi-3 base       | 62%   | 59%      | 7.05           |
| **PhinetunEd** (ours) | 60%   | 58%      | **7.46**         |


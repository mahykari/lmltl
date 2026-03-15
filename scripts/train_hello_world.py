"""
Phase 0 hello-world SFT job.

Goal: verify that Qwen 2.5-1.5B loads, LoRA attaches, the GPU has enough VRAM,
and TRL's SFTTrainer completes 100 steps without crashing.

No research content. Dataset is a tiny hardcoded set of instruction-response pairs.
"""

import os
import argparse
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig


# ---------------------------------------------------------------------------
# Tiny synthetic dataset — just enough to exercise the training loop.
# Replace with real data in Phase 3.
# ---------------------------------------------------------------------------
EXAMPLES = [
    {"instruction": "What is the capital of France?",    "response": "The capital of France is Paris."},
    {"instruction": "Explain what a prime number is.",   "response": "A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself."},
    {"instruction": "Write a haiku about autumn.",       "response": "Leaves fall silently / Crimson and gold fill the air / Winter waits ahead"},
    {"instruction": "What is 17 multiplied by 6?",      "response": "17 multiplied by 6 equals 102."},
    {"instruction": "Name three planets in our solar system.", "response": "Three planets in our solar system are Mars, Jupiter, and Saturn."},
    {"instruction": "What is the speed of light?",      "response": "The speed of light in a vacuum is approximately 299,792,458 metres per second."},
    {"instruction": "Translate 'hello' into Spanish.",   "response": "The Spanish translation of 'hello' is 'hola'."},
    {"instruction": "What does DNA stand for?",          "response": "DNA stands for deoxyribonucleic acid."},
    {"instruction": "Who wrote Romeo and Juliet?",       "response": "Romeo and Juliet was written by William Shakespeare."},
    {"instruction": "Describe the water cycle briefly.", "response": "Water evaporates from oceans and lakes, rises as vapour, condenses into clouds, and falls as precipitation before flowing back to water bodies."},
] * 20   # 200 examples total; enough for 100 steps at batch_size=2


def make_dataset() -> Dataset:
    """Format examples as a single 'text' field using the Qwen chat template."""
    texts = []
    for ex in EXAMPLES:
        text = (
            f"<|im_start|>user\n{ex['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{ex['response']}<|im_end|>"
        )
        texts.append({"text": text})
    return Dataset.from_list(texts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--output_dir", default="checkpoints/hello_world")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    print(f"PyTorch version:  {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU:              {torch.cuda.get_device_name(0)}")
        print(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # -----------------------------------------------------------------------
    # Model + tokenizer
    # -----------------------------------------------------------------------
    print(f"\nLoading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M")

    # -----------------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------------
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters (LoRA): {trainable / 1e6:.2f}M")

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    dataset = make_dataset()
    print(f"Dataset size: {len(dataset)} examples")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=10,
        bf16=True,
        logging_steps=10,
        save_steps=args.max_steps,   # save once at end for hello-world
        save_total_limit=1,
        report_to="none",            # no W&B for hello-world
        max_seq_length=256,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    print("\nStarting training...")
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nCheckpoint saved to: {args.output_dir}")
    print("Phase 0 complete — pipeline verified.")


if __name__ == "__main__":
    main()

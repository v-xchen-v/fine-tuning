# pip install transformers datasets peft bitsandbytes matplotlib tensorboard hf-cli
# hf auth login

import torch
import requests
import os

from PIL import Image
from datasets import load_dataset, load_from_disk
from peft import get_peft_model, LoraConfig

from transformers import Trainer
from transformers import TrainingArguments
from transformers import PaliGemmaProcessor
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

device = "cuda"
model_id = "google/paligemma-3b-pt-224"
cache_dir = "vqav2_split_cache"

# Load dataset - using trust_remote_code=True for dataset scripts
# Check if cached split exists
if os.path.exists(cache_dir):
    print(f"Loading cached split dataset from {cache_dir}...")
    split_ds = load_from_disk(cache_dir)
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]
    print("Cached dataset loaded successfully!")
else:
    print("Loading dataset from HuggingFace...")
    try:
        ds = load_dataset("HuggingFaceM4/VQAv2", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative: loading from parquet revision...")
        try:
            ds = load_dataset("HuggingFaceM4/VQAv2", split="validation", revision="refs/convert/parquet")
        except Exception as e2:
            print(f"Parquet version also failed: {e2}")
            print("Please install datasets<4.0: pip install 'datasets<4.0'")
            raise
    
    print("Splitting dataset (this may take a while)...")
    split_ds = ds.train_test_split(test_size=0.05, seed=42)  # Fixed seed for reproducibility
    train_ds = split_ds["train"]
    test_ds = split_ds["test"]
    
    # Save split to disk for future use
    print(f"Saving split dataset to {cache_dir} for future use...")
    split_ds.save_to_disk(cache_dir)
    print("Split dataset cached successfully!")

processor = PaliGemmaProcessor.from_pretrained(model_id)

def collate_fn(examples):
  texts = [f"<image> <bos> answer {example['question']}" for example in examples]
  labels= [example['multiple_choice_answer'] for example in examples]
  images = [example["image"].convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")

  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

# Freeze vision encoder and projector
for param in model.model.vision_tower.parameters():
    param.requires_grad = False

for param in model.model.multi_modal_projector.parameters():
    param.requires_grad = False

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

args=TrainingArguments(
        num_train_epochs=2,
        remove_unused_columns=False,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=100,
        optim="adamw_torch",
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=1,
        output_dir="finetuned_paligemma_vqav2_small",
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False
    )

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        data_collator=collate_fn,
        args=args
    )

trainer.train()

# Save the model
trainer.save_model()
print("Model saved to finetuned_paligemma_vqav2_small")

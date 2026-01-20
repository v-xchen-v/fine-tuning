
# pip install -q -U git+https://github.com/huggingface/transformers.git datasets peft bitsandbytes
# huggingface-cli login

import torch
import requests

from PIL import Image
from datasets import load_dataset
from peft import get_peft_model, LoraConfig

from transformers import Trainer
from transformers import TrainingArguments
from transformers import PaliGemmaProcessor
from transformers import BitsAndBytesConfig
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

device = "cuda"
model_id = "google/paligemma-3b-pt-224"

split_ds = ds["validation"].train_test_split(test_size=0.05)
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

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
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
        optim="adamw_hf",
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

finetuned_model_id = "pyimagesearch/finetuned_paligemma_vqav2_small"
model = PaliGemmaForConditionalGeneration.from_pretrained(finetuned_model_id)
processor = AutoProcessor.from_pretrained(model_id)

prompt = "Describe the image."
image_file = test_ds[9]["image"]

inputs = processor(image_file, prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])

prompt = "Describe the image."
image_file = test_ds[99]["image"]

inputs = processor(image_file, prompt, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=20)

print(processor.decode(output[0], skip_special_tokens=True)[len(prompt):])
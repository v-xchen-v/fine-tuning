# pip install -q -U git+https://github.com/huggingface/transformers.git datasets peft bitsandbytes
# huggingface-cli login

import torch
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from datasets import load_dataset, load_from_disk
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

device = "cuda"
model_id = "google/paligemma-3b-pt-224"
finetuned_model_path = "finetuned_paligemma_vqav2_small"
cache_dir = "vqav2_split_cache"

# Load dataset for testing - use cached split if available
if os.path.exists(cache_dir):
    print(f"Loading cached split dataset from {cache_dir}...")
    split_ds = load_from_disk(cache_dir)
    test_ds = split_ds["test"]
    print("Cached dataset loaded successfully!")
else:
    print("Cache not found. Loading dataset from HuggingFace...")
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
    test_ds = split_ds["test"]
    # Save split to disk for future use
    print(f"Saving split dataset to {cache_dir} for future use...")
    split_ds.save_to_disk(cache_dir)
    print("Split dataset cached successfully!")

# Load the fine-tuned model
print(f"Loading fine-tuned model from {finetuned_model_path}...")
model = PaliGemmaForConditionalGeneration.from_pretrained(
    finetuned_model_path,
    torch_dtype=torch.bfloat16
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# add root of repo as module search path
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from utils.visualization.vis import plot_and_save

# Test on first example
prompt = "Describe the image."
image_file = test_ds[9]["image"]

inputs = processor(image_file, prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=20)

output_text = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
print("Test 1:")
print(output_text)

# Plot and save first test
plot_and_save(image_file, prompt, output_text, "finetuned_model_output_1.png")

# Test on second example
prompt = "Describe the image."
image_file = test_ds[99]["image"]

inputs = processor(image_file, prompt, return_tensors="pt").to(device)
output = model.generate(**inputs, max_new_tokens=20)

output_text = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
print("\nTest 2:")
print(output_text)

# Plot and save second test
plot_and_save(image_file, prompt, output_text, "finetuned_model_output_2.png")

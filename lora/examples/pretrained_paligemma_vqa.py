# pip install transformers

# add root of repo as module search path
import os
import sys
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from utils.visualization.vis import plot_and_save

import torch
import requests
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

pretrained_model_id = "google/paligemma-3b-pt-224"
preprocessor = AutoProcessor.from_pretrained(pretrained_model_id)
pretrained_model = PaliGemmaForConditionalGeneration.from_pretrained(
    pretrained_model_id)

prompt = "What is behind the cat?"
image_file = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cat.png?download=true"

raw_image = Image.open(requests.get(image_file, stream=True).raw)
inputs = preprocessor(raw_image.convert("RGB"), prompt, return_tensors="pt")
outputs = pretrained_model.generate(**inputs, max_new_tokens=20)
answer = preprocessor.decode(outputs[0], skip_special_tokens=True)[len(prompt):]
print("Pretrained model answer:", answer)

# plot and save pretrained model result
plot_and_save(raw_image, prompt, answer, "pretrained_model_output.png")
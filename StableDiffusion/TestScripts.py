import model_loader
import pipeline
from PIL import Image
from pathlib import Path
from transformers import CLIPTokenizer
import torch
from datetime import datetime
import os

DEVICE = "cpu"

ALLOW_CUDA = True
ALLOW_MPS = False
GENERATE_DIR = '../../SD-Generate'

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")

tokenizer = CLIPTokenizer("./data/vocab.json", merges_file="./data/merges.txt")
model_file = "/data-shared/NAS/RBE306TC/Tools/StableDiffusionModels/"


models = model_loader.preload_models_from_standard_weights(model_file, DEVICE)

## TEXT TO IMAGE

prompt = "a fish swimming in the water, fighting against a submarine, 4k resolution."


uncond_prompt = ""  # Also known as negative prompt
do_cfg = True
cfg_scale = 8  # min: 1, max: 14

## IMAGE TO IMAGE

input_image = None
# Comment to disable image to image
# image_path = "./images/dog.jpg"
# input_image = Image.open(image_path)
# Higher values means more noise will be added to the input image, so the result will further from the input image.
# Lower values means less noise is added to the input image, so output will be closer to the input image.
strength = 0.999

## SAMPLER
sampler = "ddpm"
num_inference_steps = 100
seed = 42

output_image = pipeline.generate(
    prompt=prompt,
    uncond_prompt=uncond_prompt,
    input_image=input_image,
    strength=strength,
    do_cfg=do_cfg,
    cfg_scale=cfg_scale,
    sampler_name=sampler,
    n_inference_steps=num_inference_steps,
    seed=seed,
    models=models,
    device=DEVICE,
    idle_device="cpu",
    tokenizer=tokenizer,
)

# Combine the input image and the output image into a single image.
# Image.fromarray(output_image)
fname = f"{datetime.now():%Y%m%d-%H%M%S}.png"
fname = os.path.join(GENERATE_DIR, fname)
os.makedirs(GENERATE_DIR, exist_ok=True)
Image.fromarray(output_image).save(fname)
print("Saved:", fname)
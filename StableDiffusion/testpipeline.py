import os
import torch
import time
from PIL import Image
from datetime import datetime
from transformers import CLIPTokenizer
import model_loader
import pipeline

# ================= 全局配置 =================
GENERATE_DIR = '/data-shared/NAS/RBE306TC/AY2526/GeneratedImages/'
MODEL_FILE = "/data-shared/NAS/RBE306TC/Tools/StableDiffusionModels/"
VOCAB_FILE = "./data/vocab.json"
MERGES_FILE = "./data/merges.txt"

DEVICE = "cpu"
ALLOW_CUDA = True
ALLOW_MPS = False

if torch.cuda.is_available() and ALLOW_CUDA:
    DEVICE = "cuda"
elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
    DEVICE = "mps"
print(f"Using device: {DEVICE}")
# ============================================

# 加载模型函数
def load_models():
    print(f"Loading Tokenizer from {VOCAB_FILE}...")
    tokenizer = CLIPTokenizer(VOCAB_FILE, merges_file=MERGES_FILE)
    
    print(f"Preloading Models from {MODEL_FILE}...")
    models = model_loader.preload_models_from_standard_weights(MODEL_FILE, DEVICE)
    return models, tokenizer

# 生成函数
def generate_image(models, tokenizer, prompt, steps=50, cfg_scale=8.0, seed=None):
    now = datetime.now()
    seed = seed or int(now.timestamp())
    print(f"Generating image with seed: {seed}, prompt: {prompt}, steps: {steps}, cfg_scale: {cfg_scale}")
    
    # 生成图像
    output_image_np = pipeline.generate(
        prompt=prompt,
        uncond_prompt="",
        input_image=None,
        strength=0.8,
        do_cfg=True,
        cfg_scale=cfg_scale,
        sampler_name="ddpm",
        n_inference_steps=steps,
        seed=seed,
        models=models,
        device=DEVICE,
        idle_device="cpu",
        tokenizer=tokenizer,
    )

    # 保存图像
    result_image = Image.fromarray(output_image_np)
    os.makedirs(GENERATE_DIR, exist_ok=True)
    fname = now.strftime("%Y%m%d-%H%M%S.png")
    save_path = os.path.join(GENERATE_DIR, "SD-" + fname)
    result_image.save(save_path)
    print(f"Image saved to {save_path}")

# 主函数：预加载模型并等待用户输入
def main():
    # 1. 载入模型和tokenizer
    models, tokenizer = load_models()

    # 固定宽度
    width = 512  # 固定宽度，无需用户输入
    
    while True:
        # 2. 等待用户输入
        prompt = input("Enter your prompt (or 'exit' to quit): ").strip()
        
        if prompt.lower() == "exit":
            print("Exiting...")
            break
        
        # 3. 检查token数量是否超过77
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        token_count = len(input_ids[0])

        if token_count > 77:
            print(f"Prompt contains {token_count} tokens, which exceeds the limit of 77 tokens. Please input a shorter prompt.")
            continue
        
        # 4. 调用生成函数
        generate_image(models, tokenizer, prompt)
        
        # 5. 继续等待下一次输入
        continue

if __name__ == "__main__":
    main()
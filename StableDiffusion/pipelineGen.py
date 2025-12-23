# import model_loader
# import pipeline
# import argparse
# import torch
# import os
# import textwrap
# import math
# import time
# from PIL import Image, ImageDraw, ImageFont
# from pathlib import Path
# from transformers import CLIPTokenizer
# from datetime import datetime

# # ================= 全局配置 =================
# GENERATE_DIR = '/data-shared/NAS/RBE306TC/AY2526/GeneratedImages/'
# MODEL_FILE = "/data-shared/NAS/RBE306TC/Tools/StableDiffusionModels/"
# VOCAB_FILE = "./data/vocab.json"
# MERGES_FILE = "./data/merges.txt"

# DEVICE = "cpu"
# ALLOW_CUDA = True
# ALLOW_MPS = False

# if torch.cuda.is_available() and ALLOW_CUDA:
#     DEVICE = "cuda"
# elif (torch.has_mps or torch.backends.mps.is_available()) and ALLOW_MPS:
#     DEVICE = "mps"
# print(f"Using device: {DEVICE}")
# # ============================================

# # ----------------------------------------------------------------------
# # 辅助函数：给图片加底部文字
# # ----------------------------------------------------------------------
# def add_footer(original_img, text_content):
#     width, height = original_img.size
    
#     # 动态字号 (宽度 / 40)
#     font_size = max(20, int(width / 40)) 
#     try:
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
#     except IOError:
#         try:
#             font = ImageFont.truetype("arial.ttf", font_size)
#         except IOError:
#             font = ImageFont.load_default()

#     draw = ImageDraw.Draw(original_img)
#     lines = []
#     # 估算每行字符数
#     chars_per_line = int(width / (font_size * 0.6))
    
#     for para in text_content.split('\n'):
#         lines.extend(textwrap.wrap(para, width=chars_per_line))

#     # 计算底部高度
#     bbox = font.getbbox("Tg") 
#     line_height = bbox[3] - bbox[1] + int(font_size * 0.5)
#     padding = 20
#     footer_height = (len(lines) * line_height) + (padding * 2)

#     # 创建新画布
#     new_height = height + footer_height
#     new_img = Image.new('RGB', (width, new_height), (0, 0, 0)) # 黑色背景
#     new_img.paste(original_img, (0, 0))
    
#     draw = ImageDraw.Draw(new_img)
#     y_text = height + padding
    
#     for line in lines:
#         draw.text((padding, y_text), line, font=font, fill=(255, 255, 255)) # 白色文字
#         y_text += line_height
        
#     return new_img

# def parse_args():
#     parser = argparse.ArgumentParser(description="Stable Diffusion Generator")
    
#     # 必需参数
#     parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    
#     # 可选参数
#     parser.add_argument("--uncond_prompt", type=str, default="", help="Negative prompt")
#     parser.add_argument("--input_image", type=str, default=None, help="Path to input image (Img2Img)")
#     parser.add_argument("--strength", type=float, default=0.8, help="Img2Img Strength (0-1)")
#     parser.add_argument("--cfg_scale", type=float, default=8.0, help="CFG Scale")
#     parser.add_argument("--steps", type=int, default=50, help="Inference steps")
#     parser.add_argument("--seed", type=int, default=None, help="Manual seed (default: System Time)")
    
#     return parser.parse_args()

# def main():
#     args = parse_args()

#     # --- 1. 设置 Seed (当前系统时间) ---
#     now = datetime.now()
#     if args.seed is None:
#         # 使用当前时间戳作为随机种子
#         seed = int(now.timestamp())
#     else:
#         seed = args.seed
        
#     print(f"Loading Tokenizer from {VOCAB_FILE}...")
#     tokenizer = CLIPTokenizer(VOCAB_FILE, merges_file=MERGES_FILE)
    
#     print(f"Preloading Models from {MODEL_FILE}...")
#     models = model_loader.preload_models_from_standard_weights(MODEL_FILE, DEVICE)

#     # --- 2. 处理 Img2Img 输入 ---
#     input_image_pil = None
#     if args.input_image:
#         if os.path.exists(args.input_image):
#             print(f"Loading Input Image: {args.input_image}")
#             input_image_pil = Image.open(args.input_image)
#         else:
#             print(f"Warning: Input image {args.input_image} not found. Switching to Text-to-Image mode.")

#     print(f"Starting Generation...")
#     print(f"  Prompt: {args.prompt}")
#     print(f"  Seed:   {seed}")
    
#     # --- 3. 执行生成 ---
#     output_image_np = pipeline.generate(
#         prompt=args.prompt,
#         uncond_prompt=args.uncond_prompt,
#         input_image=input_image_pil,
#         strength=args.strength,
#         do_cfg=True,
#         cfg_scale=args.cfg_scale,
#         sampler_name="ddpm",
#         n_inference_steps=args.steps,
#         seed=seed,
#         models=models,
#         device=DEVICE,
#         idle_device="cpu",
#         tokenizer=tokenizer,
#     )

#     # --- 4. 后处理 (添加注脚 + 保存) ---
#     result_image = Image.fromarray(output_image_np)
    
#     # # 构造注脚文字
#     # info_text = (
#     #     f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')} | Seed: {seed} | Steps: {args.steps} | CFG: {args.cfg_scale}\n"
#     #     f"Prompt: {args.prompt}"
#     # )
#     # if input_image_pil:
#     #     info_text += f"\nBase Img: {os.path.basename(args.input_image)} (Str: {args.strength})"

#     # # 添加黑底白字注脚
#     # final_image = add_footer(result_image, info_text)

#     # 确保目录存在并保存
#     os.makedirs(GENERATE_DIR, exist_ok=True)
#     fname = now.strftime("%Y%m%d-%H%M%S.png")
#     save_path = os.path.join(GENERATE_DIR, "SD-" + fname)
    
#     result_image.save(save_path)
#     print(f"Done! Image saved to: {save_path}")

# if __name__ == "__main__":
#     main()


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
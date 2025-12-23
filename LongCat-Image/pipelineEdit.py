# import torch
# import argparse
# import os
# import math
# import time
# import textwrap
# from datetime import datetime
# from PIL import Image, ImageDraw, ImageFont
# from transformers import AutoProcessor
# from longcat_image.models import LongCatImageTransformer2DModel
# from longcat_image.pipelines import LongCatImageEditPipeline

# # ================= 配置区域 =================
# CHECKPOINT_DIR = '/data-shared/NAS/NewModels/LongCat-Image/LongCat-Image-Edit/'
# STEPS = 50
# GUIDANCE = 4.5
# OUTPUT_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages"
# # ============================================

# def get_multiple_of_16(value):
#     """确保数值是 16 的倍数，防止模型报错"""
#     value = int(math.ceil(value))
#     if value % 16 == 0:
#         return value
#     return ((value // 16) + 1) * 16

# def add_footer(original_img, text_content):
#     """给图片增加黑底白字的底部信息栏"""
#     width, height = original_img.size
    
#     # 动态计算字号
#     font_size = int(height / 35)
#     if font_size < 18: font_size = 18
    
#     try:
#         font = ImageFont.truetype("Arial.ttf", font_size)
#     except:
#         font = ImageFont.load_default()

#     draw = ImageDraw.Draw(original_img)
#     lines = []
#     chars_per_line = int(width / (font_size * 0.6))
    
#     for para in text_content.split('\n'):
#         lines.extend(textwrap.wrap(para, width=chars_per_line))

#     bbox = font.getbbox("Tg") 
#     line_height = bbox[3] - bbox[1] + int(font_size * 0.5)
#     padding = 20
#     footer_height = (len(lines) * line_height) + (padding * 2)

#     new_height = height + footer_height
#     new_img = Image.new('RGB', (width, new_height), (0, 0, 0))
#     new_img.paste(original_img, (0, 0))
    
#     draw = ImageDraw.Draw(new_img)
#     y_text = height + padding
    
#     for line in lines:
#         draw.text((padding, y_text), line, font=font, fill=(255, 255, 255))
#         y_text += line_height
        
#     return new_img

# def parse_args():
#     parser = argparse.ArgumentParser(description="LongCat Image Editing")
#     parser.add_argument("--input", type=str, required=True, help="原始图片路径")
#     parser.add_argument("--prompt", type=str, required=True, help="编辑指令")
#     # 已删除 --width 参数
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     # --- 1. 准备文件名和 Seed ---
#     now = datetime.now()
#     filename_base = now.strftime("%Y%m%d-%H%M%S.png")
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     output_path = os.path.join(OUTPUT_DIR, "LongCatEdit-" + filename_base)
#     seed = int(now.timestamp())

#     if not os.path.exists(args.input):
#         print(f"Error: 找不到输入图片 {args.input}")
#         return

#     # --- 2. 图片尺寸处理 (关键修改) ---
#     original_pil = Image.open(args.input).convert("RGB")
#     orig_w, orig_h = original_pil.size
    
#     # 为了模型能跑，必须调整到 16 的倍数
#     model_w = get_multiple_of_16(orig_w)
#     model_h = get_multiple_of_16(orig_h)
    
#     # 临时缩放给模型用
#     input_image = original_pil.resize((model_w, model_h), Image.LANCZOS)

#     print(f"================ 编辑配置 ================")
#     print(f"  Input:  {args.input}")
#     print(f"  Prompt: {args.prompt}")
#     print(f"  Orig Size:  {orig_w} x {orig_h}")
#     print(f"  Model Size: {model_w} x {model_h} (临时调整)")
#     print(f"  Seed:   {seed}")
#     print(f"  Output: {output_path}")
#     print(f"==========================================")

#     # --- 3. 加载模型 ---
#     if torch.cuda.is_available():
#         dtype = torch.bfloat16
#     elif torch.backends.mps.is_available():
#         dtype = torch.float16
#     else:
#         dtype = torch.float32

#     text_processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, subfolder='tokenizer')
#     transformer = LongCatImageTransformer2DModel.from_pretrained(
#         CHECKPOINT_DIR, subfolder='transformer', torch_dtype=dtype, use_safetensors=True
#     )
#     pipe = LongCatImageEditPipeline.from_pretrained(
#         CHECKPOINT_DIR, transformer=transformer, text_processor=text_processor, torch_dtype=dtype
#     )
#     pipe.enable_model_cpu_offload()

#     # --- 4. 执行编辑 ---
#     print("开始进行AI编辑...")
#     result_image = pipe(
#         input_image,
#         args.prompt,
#         negative_prompt='',
#         guidance_scale=GUIDANCE,
#         num_inference_steps=STEPS,
#         num_images_per_prompt=1,
#         generator=torch.Generator("cpu").manual_seed(seed)
#     ).images[0]

#     # --- 5. 后处理：恢复原始尺寸并添加注脚 ---
#     # 关键步骤：把模型输出的图（可能是16倍数）缩放回用户原始输入的尺寸
#     if result_image.size != (orig_w, orig_h):
#         result_image = result_image.resize((orig_w, orig_h), Image.LANCZOS)

#     print("正在添加元数据注脚...")
#     info_text = (
#         f"File: {filename_base} | Seed: {seed} | Size: {orig_w}x{orig_h}\n"
#         f"Source: {os.path.basename(args.input)}\n"
#         f"Prompt: {args.prompt}"
#     )
    
#     final_image = add_footer(result_image, info_text)

#     # 6. 保存
#     final_image.save(output_path)
#     print(f"处理完成！图片已保存至: {output_path}")

# if __name__ == "__main__":
#     main()



import os
import torch
import argparse
import time
import textwrap
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImageEditPipeline
import math

# ================= 配置区域 =================
CHECKPOINT_DIR = '/data-shared/NAS/NewModels/LongCat-Image/LongCat-Image-Edit/'
STEPS = 50
GUIDANCE = 4.5
OUTPUT_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages"
# ============================================

def get_multiple_of_16(value):
    """确保数值是 16 的倍数，防止模型报错"""
    value = int(math.ceil(value))
    if value % 16 == 0:
        return value
    return ((value // 16) + 1) * 16

def add_footer(original_img, text_content):
    """给图片增加黑底白字的底部信息栏"""
    width, height = original_img.size
    
    # 动态计算字号
    font_size = int(height / 35)
    if font_size < 18: font_size = 18
    
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(original_img)
    lines = []
    chars_per_line = int(width / (font_size * 0.6))
    
    for para in text_content.split('\n'):
        lines.extend(textwrap.wrap(para, width=chars_per_line))

    bbox = font.getbbox("Tg") 
    line_height = bbox[3] - bbox[1] + int(font_size * 0.5)
    padding = 20
    footer_height = (len(lines) * line_height) + (padding * 2)

    new_height = height + footer_height
    new_img = Image.new('RGB', (width, new_height), (0, 0, 0))
    new_img.paste(original_img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    y_text = height + padding
    
    for line in lines:
        draw.text((padding, y_text), line, font=font, fill=(255, 255, 255))
        y_text += line_height
        
    return new_img

def parse_args():
    parser = argparse.ArgumentParser(description="LongCat Image Editing")
    parser.add_argument("--input", type=str, required=True, help="原始图片路径")
    parser.add_argument("--prompt", type=str, required=True, help="编辑指令")
    return parser.parse_args()

def load_model():
    """加载模型"""
    print("📦 加载 LongCat 模型...")

    try:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16 if torch.backends.mps.is_available() else torch.float32

        text_processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, subfolder='tokenizer')
        transformer = LongCatImageTransformer2DModel.from_pretrained(
            CHECKPOINT_DIR, subfolder='transformer', torch_dtype=dtype, use_safetensors=True
        )
        pipe = LongCatImageEditPipeline.from_pretrained(
            CHECKPOINT_DIR, transformer=transformer, text_processor=text_processor, torch_dtype=dtype
        )
        pipe.enable_model_cpu_offload()
        return pipe
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None

def generate_image(pipe, input_image, prompt, seed):
    """生成图片并添加水印"""
    print("开始进行AI编辑...")
    result_image = pipe(
        input_image,
        prompt,
        negative_prompt='',
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

    return result_image

def main():
    # 加载模型
    pipe = load_model()
    if not pipe:
        return

    print("模型加载完成，等待用户输入...")

    while True:
        # 1. 等待用户输入图片路径和 prompt
        input_image_path = input("Enter the path to your input image (or 'exit' to quit): ").strip()
        if input_image_path.lower() == 'exit':
            print("Exiting the program.")
            break

        if not os.path.exists(input_image_path):
            print(f"Error: 找不到输入图片 {input_image_path}")
            continue

        prompt = input("Enter the prompt for image editing: ").strip()
        
        # 2. 处理图片尺寸
        original_pil = Image.open(input_image_path).convert("RGB")
        orig_w, orig_h = original_pil.size
        model_w = get_multiple_of_16(orig_w)
        model_h = get_multiple_of_16(orig_h)
        
        # 临时缩放给模型用
        input_image = original_pil.resize((model_w, model_h), Image.LANCZOS)
        
        # 3. 生成图像
        seed = int(time.time())
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"LongCatEdit-{timestamp}.png"
        save_path = os.path.join(OUTPUT_DIR, filename)

        print(f"🚀 任务启动 | Time: {timestamp}")
        print(f"⚙️  参数: Width={orig_w}, Height={orig_h} (16:9), Seed={seed}")
        print(f"📝 Prompt: {prompt[:50]}...")

        # 生成图像
        result_image = generate_image(pipe, input_image, prompt, seed)

        # 处理后保存
        print("正在添加元数据注脚...")
        info_text = (
            f"File: {filename} | Seed: {seed} | Size: {orig_w}x{orig_h}\n"
            f"Source: {os.path.basename(input_image_path)}\n"
            f"Prompt: {prompt}"
        )
        
        final_image = add_footer(result_image, info_text)

        # 保存
        final_image.save(save_path)
        print(f"\n🎉 任务完成！图片已保存至: {save_path}")

if __name__ == "__main__":
    main()
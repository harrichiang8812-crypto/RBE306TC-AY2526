# import torch
# import argparse
# import os
# import math
# import time
# import textwrap # 用于文字换行
# from datetime import datetime
# from PIL import Image, ImageDraw, ImageFont # 用于图片处理
# from transformers import AutoProcessor
# from longcat_image.models import LongCatImageTransformer2DModel
# from longcat_image.pipelines import LongCatImagePipeline

# # ================= 配置区域 =================
# CHECKPOINT_DIR = '/data-shared/NAS/NewModels/LongCat-Image/LongCat-Image/'
# STEPS = 50
# GUIDANCE = 4.5
# OUTPUT_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages"

# DEFAULT_PROMPT = 'A young Asian woman wearing a yellow knitted sweater paired with a white necklace. Her hands rest on her knees, and her expression is serene. The background is a rough brick wall; warm afternoon sunlight spills over her, creating a tranquil and cozy atmosphere. The shot is framed at a medium distance, highlighting her demeanor and the details of her attire. Soft light falls on her face, emphasizing her facial features and the texture of her accessories, adding depth and approachability to the image. The overall composition is clean, with the texture of the brick wall complementing the interplay of light and shadow, accentuating the subject\'s elegance and composure.'
# # ============================================

# def get_multiple_of_16(value):
#     value = int(math.ceil(value))
#     if value % 16 == 0:
#         return value
#     return ((value // 16) + 1) * 16

# def add_footer(original_img, text_content):
#     """
#     给图片增加黑底白字的底部信息栏
#     """
#     width, height = original_img.size
    
#     # 1. 尝试加载字体 (为了清晰度，尽量加载 TrueType 字体)
#     # 字体大小设为宽度的 1/40，保证在不同分辨率下都清晰
#     font_size = int(height / 35)
#     if font_size < 18: font_size = 18
    
#     try:
#         font = ImageFont.truetype("Arial.ttf", font_size)
#     except:
#         font = ImageFont.load_default()


#     # try:
#     #     # Linux 常见字体路径
#     #     font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
#     # except IOError:
#     #     try:
#     #         # Mac/Win 常见字体
#     #         font = ImageFont.truetype("arial.ttf", font_size)
#     #     except IOError:
#     #         # 最后的保底，但默认字体不支持调整大小，会很小
#     #         font = ImageFont.load_default()
#     #         print("Warning: 未找到系统字体，使用默认字体，可能较小。")

#     # 2. 准备文字内容并自动换行
#     draw = ImageDraw.Draw(original_img) # 仅用于计算文字大小
    
#     lines = []
#     # 使用 textwrap 自动根据图片宽度换行
#     # 平均字符宽度估计：width / (font_size * 0.6) 这是一个经验值
#     chars_per_line = int(width / (font_size * 0.6))
    
#     # 将 text_content 里的每一段（比如 Prompt 是一大段）再进行折行处理
#     for para in text_content.split('\n'):
#         lines.extend(textwrap.wrap(para, width=chars_per_line))

#     # 3. 计算底部需要的高度
#     # 获取单行文字高度 (bbox: left, top, right, bottom)
#     bbox = font.getbbox("Tg") 
#     line_height = bbox[3] - bbox[1] + int(font_size * 0.5) # 增加行间距
    
#     padding = 20
#     footer_height = (len(lines) * line_height) + (padding * 2)

#     # 4. 创建新画布 (黑色背景)
#     new_height = height + footer_height
#     new_img = Image.new('RGB', (width, new_height), (0, 0, 0)) # (0,0,0) is Black

#     # 5. 粘贴原图和绘制文字
#     new_img.paste(original_img, (0, 0))
    
#     draw = ImageDraw.Draw(new_img)
#     y_text = height + padding
    
#     for line in lines:
#         draw.text((padding, y_text), line, font=font, fill=(255, 255, 255)) # White text
#         y_text += line_height
        
#     return new_img

# def parse_args():
#     parser = argparse.ArgumentParser(description="LongCat Image Generation")
#     parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="输入的提示词")
#     parser.add_argument("--width", type=int, default=1344, help="图片宽度")
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     # --- 1. 准备文件名和参数 ---
#     now = datetime.now()
#     filename_base = now.strftime("%Y%m%d-%H%M%S.png")
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     output_path = os.path.join(OUTPUT_DIR, "LongCatGen-" + filename_base)
#     seed = int(now.timestamp())
    
#     # --- 2. 分辨率计算 ---
#     target_width = get_multiple_of_16(args.width)
#     raw_height = target_width * 9 / 16
#     target_height = get_multiple_of_16(raw_height)

#     print(f"================ 生成配置 ================")
#     print(f"  时间: {now.strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"  Seed: {seed}")
#     print(f"  分辨率: {target_width} x {target_height}")
#     print(f"  保存路径: {output_path}")
#     print(f"==========================================")

#     # 3. 加载模型
#     if torch.cuda.is_available():
#         dtype = torch.bfloat16
#         device_str = "cuda"
#     elif torch.backends.mps.is_available():
#         dtype = torch.float16
#         device_str = "mps"
#     else:
#         dtype = torch.float32
#         device_str = "cpu"

#     if not os.path.exists(CHECKPOINT_DIR):
#         print(f"Error: 模型路径 {CHECKPOINT_DIR} 不存在。")
#         return

#     text_processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, subfolder='tokenizer')
#     transformer = LongCatImageTransformer2DModel.from_pretrained(
#         CHECKPOINT_DIR, subfolder='transformer', torch_dtype=dtype, use_safetensors=True
#     )
#     pipe = LongCatImagePipeline.from_pretrained(
#         CHECKPOINT_DIR, transformer=transformer, text_processor=text_processor, torch_dtype=dtype
#     )
#     pipe.enable_model_cpu_offload()

#     # 4. 生成图片
#     print("开始生成...")
#     image = pipe(
#         args.prompt,
#         height=target_height,
#         width=target_width,
#         guidance_scale=GUIDANCE,
#         num_inference_steps=STEPS,
#         num_images_per_prompt=1,
#         generator=torch.Generator("cpu").manual_seed(seed),
#         enable_cfg_renorm=True,
#         enable_prompt_rewrite=True 
#     ).images[0]

#     # --- 5. 处理底部文字信息 ---
#     print("正在添加图片元数据注脚...")
    
#     # 构造要显示的文本内容
#     info_text = (
#         f"File: {filename_base} | Seed: {seed} | Size: {target_width}x{target_height}\n"
#         f"Prompt: {args.prompt}"
#     )
    
#     # 调用函数添加页脚
#     final_image = add_footer(image, info_text)

#     # 6. 保存
#     final_image.save(output_path)
#     print(f"生成完成！带注脚的图片已保存至: {output_path}")

# if __name__ == "__main__":
#     main()


import torch
import argparse
import os
import math
import time
import textwrap  # 用于文字换行
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont  # 用于图片处理
from transformers import AutoProcessor
from longcat_image.models import LongCatImageTransformer2DModel
from longcat_image.pipelines import LongCatImagePipeline

# ================= 配置区域 =================
CHECKPOINT_DIR = '/data-shared/NAS/NewModels/LongCat-Image/LongCat-Image/'
STEPS = 50
GUIDANCE = 4.5
OUTPUT_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages"

DEFAULT_PROMPT = 'A young Asian woman wearing a yellow knitted sweater paired with a white necklace. Her hands rest on her knees, and her expression is serene. The background is a rough brick wall; warm afternoon sunlight spills over her, creating a tranquil and cozy atmosphere. The shot is framed at a medium distance, highlighting her demeanor and the details of her attire. Soft light falls on her face, emphasizing her facial features and the texture of her accessories, adding depth and approachability to the image. The overall composition is clean, with the texture of the brick wall complementing the interplay of light and shadow, accentuating the subject\'s elegance and composure.'
# ============================================

def get_multiple_of_16(value):
    value = int(math.ceil(value))
    if value % 16 == 0:
        return value
    return ((value // 16) + 1) * 16

def add_footer(original_img, text_content):
    """
    给图片增加黑底白字的底部信息栏
    """
    width, height = original_img.size
    
    # 1. 尝试加载字体 (为了清晰度，尽量加载 TrueType 字体)
    # 字体大小设为宽度的 1/40，保证在不同分辨率下都清晰
    font_size = int(height / 35)
    if font_size < 18: font_size = 18
    
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # 2. 准备文字内容并自动换行
    draw = ImageDraw.Draw(original_img)  # 仅用于计算文字大小
    
    lines = []
    # 使用 textwrap 自动根据图片宽度换行
    # 平均字符宽度估计：width / (font_size * 0.6) 这是一个经验值
    chars_per_line = int(width / (font_size * 0.6))
    
    # 将 text_content 里的每一段（比如 Prompt 是一大段）再进行折行处理
    for para in text_content.split('\n'):
        lines.extend(textwrap.wrap(para, width=chars_per_line))

    # 3. 计算底部需要的高度
    # 获取单行文字高度 (bbox: left, top, right, bottom)
    bbox = font.getbbox("Tg") 
    line_height = bbox[3] - bbox[1] + int(font_size * 0.5)  # 增加行间距
    
    padding = 20
    footer_height = (len(lines) * line_height) + (padding * 2)

    # 4. 创建新画布 (黑色背景)
    new_height = height + footer_height
    new_img = Image.new('RGB', (width, new_height), (0, 0, 0))  # (0,0,0) is Black

    # 5. 粘贴原图和绘制文字
    new_img.paste(original_img, (0, 0))
    
    draw = ImageDraw.Draw(new_img)
    y_text = height + padding
    
    for line in lines:
        draw.text((padding, y_text), line, font=font, fill=(255, 255, 255))  # White text
        y_text += line_height
        
    return new_img

def parse_args():
    parser = argparse.ArgumentParser(description="LongCat Image Generation")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="输入的提示词")
    parser.add_argument("--width", type=int, default=1344, help="图片宽度")
    return parser.parse_args()

# 生成图像函数
def generate_image(prompt, width, pipe, seed):
    # --- 1. 分辨率计算 ---
    target_width = get_multiple_of_16(width)
    raw_height = target_width * 9 / 16
    target_height = get_multiple_of_16(raw_height)

    print(f"================ 生成配置 ================")
    print(f"  Seed: {seed}")
    print(f"  分辨率: {target_width} x {target_height}")
    print(f"==========================================")

    # 2. 生成图片
    image = pipe(
        prompt,
        height=target_height,
        width=target_width,
        guidance_scale=GUIDANCE,
        num_inference_steps=STEPS,
        num_images_per_prompt=1,
        generator=torch.Generator("cpu").manual_seed(seed),
        enable_cfg_renorm=True,
        enable_prompt_rewrite=True 
    ).images[0]

    # 3. 处理底部文字信息
    print("正在添加图片元数据注脚...")
    
    # 构造要显示的文本内容
    filename_base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    info_text = (
        f"File: {filename_base} | Seed: {seed} | Size: {target_width}x{target_height}\n"
        f"Prompt: {prompt}"
    )
    
    # 调用函数添加页脚
    final_image = add_footer(image, info_text)

    # 4. 保存
    output_path = os.path.join(OUTPUT_DIR, "LongCatGen-" + filename_base)
    final_image.save(output_path)
    print(f"生成完成！带注脚的图片已保存至: {output_path}")

# 主函数，加载模型并等待用户输入
def main():
    # 加载模型
    print("正在加载模型...")
    
    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        dtype = torch.float16
        device_str = "mps"
    else:
        dtype = torch.float32
        device_str = "cpu"

    if not os.path.exists(CHECKPOINT_DIR):
        print(f"Error: 模型路径 {CHECKPOINT_DIR} 不存在。")
        return

    text_processor = AutoProcessor.from_pretrained(CHECKPOINT_DIR, subfolder='tokenizer')
    transformer = LongCatImageTransformer2DModel.from_pretrained(
        CHECKPOINT_DIR, subfolder='transformer', torch_dtype=dtype, use_safetensors=True
    )
    pipe = LongCatImagePipeline.from_pretrained(
        CHECKPOINT_DIR, transformer=transformer, text_processor=text_processor, torch_dtype=dtype
    )
    pipe.enable_model_cpu_offload()

    print("模型加载完成。开始等待用户输入...\n")
    
    while True:
        # 1. 等待用户输入 prompt 和 width
        prompt = input("Enter your prompt (or 'exit' to quit): ").strip()
        if prompt.lower() == 'exit':
            print("Exiting the program.")
            break
        
        try:
            width = int(input("Enter the image width (e.g., 1920): "))
        except ValueError:
            print("Invalid input. Please enter an integer for width.")
            continue
        
        # 2. 使用当前时间戳生成种子
        seed = int(time.time())

        # 3. 调用生成函数
        generate_image(prompt, width, pipe, seed)

if __name__ == "__main__":
    main()
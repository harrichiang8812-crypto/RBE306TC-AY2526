# import torch
# from diffusers import ZImagePipeline
# import time
# import argparse
# from datetime import datetime
# import math
# import os
# # --- 【修改】导入 Image 以创建新画布 ---
# from PIL import Image, ImageDraw, ImageFont
# import textwrap

# MODEL_PATH_MAC =  "/Users/harric/Downloads/Z-Image-Turbo"
# MODEL_PATH_LINUX = "/data-shared/NAS/NewModels/Z-Image-Turbo/"
# GENERATE_DIR_LINUX = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages"
# GENERATE_DIR_MAC = "./"



# # --- 检查设备并在 macOS 上使用 MPS ---
# def get_device():
#     """动态检查并返回可用的加速设备 (mps, cuda, or cpu)"""
#     if torch.backends.mps.is_available():
#         return "mps",MODEL_PATH_MAC,GENERATE_DIR_MAC
#     elif torch.cuda.is_available():
#         return "cuda",MODEL_PATH_LINUX,GENERATE_DIR_LINUX
#     else:
#         return "cpu",0,0

# device, MODEL_PATH, GENERATE_DIR = get_device()
# print(f"Using device: {device}")

# # --- 模型加载 ---
# try:
#     pipe = ZImagePipeline.from_pretrained(MODEL_PATH, 
#                                           torch_dtype=torch.bfloat16,
#                                           low_cpu_mem_usage=False)
#     pipe.to(device)
#     print(f"Model loaded and moved to device: {device}")

# except Exception as e:
#     print(f"\n❌ ERROR: Failed to load ZImagePipeline from '../Z-Image-Turbo'.")
#     print(f"Please check the path and dependencies. Details: {e}")
#     pipe = None
#     exit(1)


# # --- 辅助函数：确保尺寸是 16 的倍数 ---
# def round_to_16(x):
#     """将数字向上取整到最接近的 16 的倍数。"""
#     if x <= 0:
#         return 16
#     return math.ceil(x / 16) * 16

# # --- 主执行函数 ---
# def generate_image():
#     # 1. 设置参数解析器
#     parser = argparse.ArgumentParser(description="Z-Image Generation Script")
#     parser.add_argument("--prompt", type=str, required=True, help="The prompt to guide image generation.")
#     parser.add_argument("--width", type=int, default=1920, help="The desired width of the output image. (Default: 1920)")
    
#     args = parser.parse_args()
    
#     # 2. 自动计算 Height 并确保尺寸为 16 的倍数
#     aspect_ratio_h_w = 9 / 16
#     final_width = round_to_16(args.width) 
#     calculated_height = final_width * aspect_ratio_h_w
#     final_height = round_to_16(calculated_height)

#     # 3. 生成基于系统时间的随机种子
#     current_seed = int(time.time())

#     # 4. 生成基于系统时间的文件名
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     filename = f"{timestamp}.png"

#     print("-" * 50)
#     print(f"Starting image generation...")
#     print(f"  Width x Height: {final_width}x{final_height}")
#     print(f"  Seed: {current_seed}")
#     print(f"  Prompt: {args.prompt}")
#     print("-" * 50)
    
#     # 5. 执行图像生成
#     image = pipe(
#         prompt=args.prompt,
#         height=final_height,
#         width=final_width,
#         num_inference_steps=50,
#         guidance_scale=0.0,
#         generator=torch.Generator(device).manual_seed(current_seed), 
#     ).images[0]
    
#     # -----------------------------------------------------------
#     # 【修改功能】在底部增加黑色区域并写入文件名和Prompt
#     # -----------------------------------------------------------
    
#     # 1. 字体设置 (根据图片高度自动调整字号)
#     font_size = int(image.height / 35)
#     if font_size < 18: font_size = 18
    
#     try:
#         font = ImageFont.truetype("Arial.ttf", font_size)
#     except:
#         font = ImageFont.load_default()
    
#     # 2. 准备文本内容
#     # 估算每行字符数 (图片宽度 / 字宽系数)
#     chars_per_line = int(image.width / (font_size * 0.55))
#     wrapped_prompt = textwrap.fill(f"Prompt: {args.prompt}", width=chars_per_line)
    
#     final_text = f"File: {filename}\nSeed: {current_seed}\n{wrapped_prompt}"
    
#     # 3. 计算文本区域所需高度
#     margin = 20
#     line_spacing = 10
    
#     # 创建一个临时 Draw 对象来计算高度
#     temp_draw = ImageDraw.Draw(image)
#     try:
#         bbox = temp_draw.multiline_textbbox((0, 0), final_text, font=font, spacing=line_spacing)
#         text_height = bbox[3] - bbox[1]
#     except AttributeError:
#         # 旧版 PIL 兼容
#         _, text_height = temp_draw.multiline_textsize(final_text, font=font, spacing=line_spacing)
    
#     footer_height = text_height + (margin * 2)
    
#     # 4. 创建新画布 (黑色背景，高度 = 原图 + 底部区域)
#     new_height = image.height + footer_height
#     combined_image = Image.new("RGB", (image.width, new_height), "black")
    
#     # 5. 拼接图片
#     combined_image.paste(image, (0, 0))
    
#     # 6. 绘制文字 (白色，写在底部区域)
#     draw = ImageDraw.Draw(combined_image)
#     text_x = margin
#     text_y = image.height + margin
    
#     draw.multiline_text((text_x, text_y), final_text, font=font, fill="white", spacing=line_spacing)
    
#     # -----------------------------------------------------------

#     # 7. 保存最终拼接好的图像
#     os.makedirs(GENERATE_DIR, exist_ok=True)
#     combined_image.save(os.path.join(GENERATE_DIR,"Z-Image-" + filename))
#     print(f"✅ Image successfully generated and saved to: {filename}")
#     print("-" * 50)


# if __name__ == "__main__":
#     if pipe is not None:
#         generate_image()



import torch
from diffusers import ZImagePipeline
import time
import argparse
from datetime import datetime
import math
import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

# --- 模型路径和生成目录设置 ---
MODEL_PATH_MAC =  "/Users/harric/Downloads/Z-Image-Turbo"
MODEL_PATH_LINUX = "/data-shared/NAS/NewModels/Z-Image-Turbo/"
GENERATE_DIR_LINUX = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages"
GENERATE_DIR_MAC = "./"

# --- 检查设备并在 macOS 上使用 MPS ---
def get_device():
    """动态检查并返回可用的加速设备 (mps, cuda, or cpu)"""
    if torch.backends.mps.is_available():
        return "mps", MODEL_PATH_MAC, GENERATE_DIR_MAC
    elif torch.cuda.is_available():
        return "cuda", MODEL_PATH_LINUX, GENERATE_DIR_LINUX
    else:
        return "cpu", 0, 0

device, MODEL_PATH, GENERATE_DIR = get_device()
print(f"Using device: {device}")

# --- 模型加载 ---
try:
    pipe = ZImagePipeline.from_pretrained(MODEL_PATH, 
                                          torch_dtype=torch.bfloat16,
                                          low_cpu_mem_usage=False)
    pipe.to(device)
    print(f"Model loaded and moved to device: {device}")

except Exception as e:
    print(f"\n❌ ERROR: Failed to load ZImagePipeline from '{MODEL_PATH}'.")
    print(f"Please check the path and dependencies. Details: {e}")
    pipe = None
    exit(1)

# --- 辅助函数：确保尺寸是 16 的倍数 ---
def round_to_16(x):
    """将数字向上取整到最接近的 16 的倍数。"""
    if x <= 0:
        return 16
    return math.ceil(x / 16) * 16

# --- 生成图像的函数 ---
def generate_image(prompt, width):
    # 2. 自动计算 Height 并确保尺寸为 16 的倍数
    aspect_ratio_h_w = 9 / 16  # 使用 16:9 比例
    final_width = round_to_16(width)  # 将输入的宽度转为 16 的倍数
    calculated_height = final_width * aspect_ratio_h_w  # 根据宽度计算高度
    final_height = round_to_16(calculated_height)  # 确保高度是 16 的倍数

    # 3. 生成基于系统时间的随机种子
    current_seed = int(time.time())

    # 4. 生成基于系统时间的文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"{timestamp}.png"

    print("-" * 50)
    print(f"Starting image generation...")
    print(f"  Width x Height: {final_width}x{final_height}")
    print(f"  Seed: {current_seed}")
    print(f"  Prompt: {prompt}")
    print("-" * 50)
    
    # 5. 执行图像生成
    image = pipe(
        prompt=prompt,
        height=final_height,
        width=final_width,
        num_inference_steps=50,
        guidance_scale=0.0,
        generator=torch.Generator(device).manual_seed(current_seed), 
    ).images[0]
    
    # -----------------------------------------------------------
    # 【修改功能】在底部增加黑色区域并写入文件名和Prompt
    # -----------------------------------------------------------
    
    # 1. 字体设置 (根据图片高度自动调整字号)
    font_size = int(image.height / 35)
    if font_size < 18: font_size = 18
    
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # 2. 准备文本内容
    # 估算每行字符数 (图片宽度 / 字宽系数)
    chars_per_line = int(image.width / (font_size * 0.55))
    wrapped_prompt = textwrap.fill(f"Prompt: {prompt}", width=chars_per_line)
    
    final_text = f"File: {filename}\nSeed: {current_seed}\n{wrapped_prompt}"
    
    # 3. 计算文本区域所需高度
    margin = 20
    line_spacing = 10
    
    # 创建一个临时 Draw 对象来计算高度
    temp_draw = ImageDraw.Draw(image)
    try:
        bbox = temp_draw.multiline_textbbox((0, 0), final_text, font=font, spacing=line_spacing)
        text_height = bbox[3] - bbox[1]
    except AttributeError:
        # 旧版 PIL 兼容
        _, text_height = temp_draw.multiline_textsize(final_text, font=font, spacing=line_spacing)
    
    footer_height = text_height + (margin * 2)
    
    # 4. 创建新画布 (黑色背景，高度 = 原图 + 底部区域)
    new_height = image.height + footer_height
    combined_image = Image.new("RGB", (image.width, new_height), "black")
    
    # 5. 拼接图片
    combined_image.paste(image, (0, 0))
    
    # 6. 绘制文字 (白色，写在底部区域)
    draw = ImageDraw.Draw(combined_image)
    text_x = margin
    text_y = image.height + margin
    
    draw.multiline_text((text_x, text_y), final_text, font=font, fill="white", spacing=line_spacing)
    
    # -----------------------------------------------------------

    # 7. 保存最终拼接好的图像
    os.makedirs(GENERATE_DIR, exist_ok=True)
    combined_image.save(os.path.join(GENERATE_DIR,"Z-Image-" + filename))
    print(f"✅ Image successfully generated and saved to: {filename}")
    print("-" * 50)

# --- 主函数：预加载模型并持续等待用户输入 ---
def main():
    # 程序启动时，加载模型（已在全局变量中完成）
    print(f"Waiting for your inputs...\n")

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

        # 2. 调用生成函数
        generate_image(prompt, width)

if __name__ == "__main__":
    if pipe is not None:
        main()
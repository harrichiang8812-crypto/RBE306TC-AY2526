# import os
# # 1. 显存碎片优化 (核心配置，保持不动)
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# import torch
# import argparse
# import time
# import textwrap
# from datetime import datetime
# from PIL import Image, ImageDraw, ImageFont
# from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKL
# from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig
# import shutil

# # ================= 核心配置区域 =================
# MODEL_PATH = "/data-shared/NAS/NewModels/Flux2"
# SAVE_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages/"
# OFFLOAD_FOLDER = "./flux_offload_prod"
# NUM_STEPS=50
# # ===============================================

# def print_memory(gpu_id, tag):
#     if torch.cuda.is_available():
#         mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
#         print(f"   📊 [{tag}] GPU {gpu_id} 占用: \033[1;33m{mem:.2f} GB\033[0m")

# def add_watermark(image, prompt, filename, seed):
#     """在图片底部添加黑底白字的信息"""
#     # 字体设置 (尝试加载系统字体，失败则用默认)
#     try:
#         # 尝试常见 Linux 字体路径
#         font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
#     except:
#         font = ImageFont.load_default()

#     # 准备文本内容
#     info_text = f"File: {filename} | Seed: {seed}\nPrompt: {prompt}"
    
#     # 文本换行处理 (每行约80字符)
#     wrapper = textwrap.TextWrapper(width=80)
#     lines = wrapper.wrap(text=info_text)
    
#     # 计算文字区域高度
#     line_height = 30 # 每行预留高度
#     footer_height = (len(lines) * line_height) + 20 # 加上一点边距
    
#     # 创建新画布 (原图高度 + 底部高度)
#     new_height = image.height + footer_height
#     new_image = Image.new("RGB", (image.width, new_height), "black")
    
#     # 贴上原图
#     new_image.paste(image, (0, 0))
    
#     # 写字
#     draw = ImageDraw.Draw(new_image)
#     y_text = image.height + 10
#     for line in lines:
#         draw.text((20, y_text), line, font=font, fill="white")
#         y_text += line_height
        
#     return new_image

# def parse_args():
#     parser = argparse.ArgumentParser(description="Flux.2 Production Generator")
#     parser.add_argument("--prompt", type=str, required=True, help="生成的提示词")
#     parser.add_argument("--width", type=int, default=1024, help="图片宽度 (默认 1024)")
#     # height 不需要输入，自动计算
#     return parser.parse_args()

# def main():
#     args = parse_args()
    
#     # 1. 准备参数
#     # 计算 16:9 高度，并确保是 16 的倍数 (Diffusers 对尺寸敏感)
#     calc_height = int(args.width * 9 / 16)
#     height = (calc_height // 16) * 16 
    
#     seed = int(time.time())
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     filename = f"Flux2Gen-{timestamp}.png"
#     save_path = os.path.join(SAVE_DIR, filename)
    
#     print(f"🚀 任务启动 | Time: {timestamp}")
#     print(f"⚙️  参数: Width={args.width}, Height={height} (16:9), Seed={seed}")
#     print(f"📝 Prompt: {args.prompt[:50]}...")

#     # 2. 检查输出目录
#     os.makedirs(SAVE_DIR, exist_ok=True)
    
#     # 3. 清理环境
#     torch.cuda.empty_cache()
#     if os.path.exists(OFFLOAD_FOLDER):
#         shutil.rmtree(OFFLOAD_FOLDER)
#     os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

#     # 4. 量化配置
#     quant_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#         bnb_4bit_use_double_quant=True,
#     )

#     # =========================================================
#     # 核心加载逻辑 (保持之前的成功版本不变)
#     # =========================================================
    
#     # --- Part 1: Mistral (GPU 0) ---
#     print("\n📦 [Step 1] 加载 Mistral...")
#     try:
#         config = AutoConfig.from_pretrained(MODEL_PATH, subfolder="text_encoder")
#         if hasattr(config, "use_cache"): config.use_cache = False
        
#         text_encoder = AutoModel.from_pretrained(
#             MODEL_PATH, subfolder="text_encoder", config=config,
#             quantization_config=quant_config, torch_dtype=torch.bfloat16,
#             device_map={"": 0} 
#         )
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
#     except Exception as e:
#         print(f"❌ Mistral 加载失败: {e}"); return

#     # --- Part 2: VAE (GPU 0) ---
#     print("\n📦 [Step 2] 加载 VAE...")
#     try:
#         vae = AutoencoderKL.from_pretrained(
#             MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16,
#         ).to("cuda:0")
#     except Exception as e:
#         print(f"❌ VAE 加载失败: {e}"); return

#     # --- Part 3: Transformer (Load Balanced) ---
#     print("\n🏗️ [Step 3] 加载 Transformer...")
#     max_memory_transformer = {
#         0: "9GiB",   1: "23GiB",  2: "23GiB", 3: "23GiB"
#     }
#     try:
#         transformer = Flux2Transformer2DModel.from_pretrained(
#             MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
#             device_map="auto", max_memory=max_memory_transformer,
#             offload_folder=OFFLOAD_FOLDER
#         )
#     except Exception as e:
#         print(f"❌ Transformer 加载失败: {e}"); return

#     # --- Part 4: 组装 Pipeline ---
#     print("\n🔧 [Step 4] 组装 Pipeline...")
#     pipe = Flux2Pipeline(
#         vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
#         transformer=transformer, scheduler=None
#     )
#     from diffusers import FlowMatchEulerDiscreteScheduler
#     pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

#     # --- Part 4.5: VAE 补丁 (Patch) ---
#     if not hasattr(pipe.vae, "bn"):
#         class MockBN(torch.nn.Module):
#             def __init__(self, channels):
#                 super().__init__()
#                 self.register_buffer("running_mean", torch.zeros(channels))
#                 self.register_buffer("running_var", torch.ones(channels))
        
#         # 强制 128 通道
#         pipe.vae.bn = MockBN(128).to(pipe.vae.device).to(dtype=torch.bfloat16)
#         print("   ✅ VAE Patch (128ch) injected.")

#     # =========================================================
#     # 生成与保存逻辑
#     # =========================================================
    
#     print("\n🎨 开始生成图片...")
#     try:
#         image = pipe(
#             prompt=args.prompt, 
#             num_inference_steps=NUM_STEPS,
#             guidance_scale=3.5,
#             height=height,
#             width=args.width,
#             generator=torch.Generator("cpu").manual_seed(seed)
#         ).images[0]

#         # 添加水印元数据
#         print("   🖌️ 正在处理水印和元数据...")
#         final_image = add_watermark(image, args.prompt, filename, seed)
        
#         # 保存
#         final_image.save(save_path)
#         print(f"\n🎉 任务完成！")
#         print(f"   📂 保存路径: {save_path}")
        
#     except RuntimeError as e:
#         print(f"\n❌ 生成报错: {e}")
#     finally:
#         shutil.rmtree(OFFLOAD_FOLDER, ignore_errors=True)

# if __name__ == "__main__":
#     main()



import os
import torch
import argparse
import time
import textwrap
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKL
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import shutil

# ================= 核心配置区域 =================
MODEL_PATH = "/data-shared/NAS/NewModels/Flux2"
SAVE_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages/"
OFFLOAD_FOLDER = "./flux_offload_prod"
NUM_STEPS = 50
# ===============================================

def print_memory(gpu_id, tag):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"   📊 [{tag}] GPU {gpu_id} 占用: \033[1;33m{mem:.2f} GB\033[0m")

def add_watermark(image, prompt, filename, seed):
    """在图片底部添加黑底白字的信息"""
    # 字体设置 (尝试加载系统字体，失败则用默认)
    try:
        # 尝试常见 Linux 字体路径
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
    except:
        font = ImageFont.load_default()

    # 准备文本内容
    info_text = f"File: {filename} | Seed: {seed}\nPrompt: {prompt}"
    
    # 文本换行处理 (每行约80字符)
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=info_text)
    
    # 计算文字区域高度
    line_height = 30 # 每行预留高度
    footer_height = (len(lines) * line_height) + 20 # 加上一点边距
    
    # 创建新画布 (原图高度 + 底部高度)
    new_height = image.height + footer_height
    new_image = Image.new("RGB", (image.width, new_height), "black")
    
    # 贴上原图
    new_image.paste(image, (0, 0))
    
    # 写字
    draw = ImageDraw.Draw(new_image)
    y_text = image.height + 10
    for line in lines:
        draw.text((20, y_text), line, font=font, fill="white")
        y_text += line_height
        
    return new_image

def parse_args():
    parser = argparse.ArgumentParser(description="Flux.2 Production Generator")
    parser.add_argument("--prompt", type=str, required=True, help="生成的提示词")
    parser.add_argument("--width", type=int, default=1024, help="图片宽度 (默认 1024)")
    # height 不需要输入，自动计算
    return parser.parse_args()

def load_models():
    """加载模型"""
    print("📦 [Step 1] 加载模型...")

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        # Part 1: Mistral (GPU 0)
        config = AutoConfig.from_pretrained(MODEL_PATH, subfolder="text_encoder")
        if hasattr(config, "use_cache"): config.use_cache = False
        
        text_encoder = AutoModel.from_pretrained(
            MODEL_PATH, subfolder="text_encoder", config=config,
            quantization_config=quant_config, torch_dtype=torch.bfloat16,
            device_map={"": 0} 
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")

        # Part 2: VAE (GPU 0)
        vae = AutoencoderKL.from_pretrained(
            MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16,
        ).to("cuda:0")

        # Part 3: Transformer (Load Balanced)
        max_memory_transformer = {
            0: "9GiB",   1: "23GiB",  2: "23GiB", 3: "23GiB"
        }
        transformer = Flux2Transformer2DModel.from_pretrained(
            MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16,
            device_map="auto", max_memory=max_memory_transformer,
            offload_folder=OFFLOAD_FOLDER
        )

        # Part 4: 组装 Pipeline
        pipe = Flux2Pipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer,
            transformer=transformer, scheduler=None
        )
        from diffusers import FlowMatchEulerDiscreteScheduler
        pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

        # Part 4.5: VAE 补丁 (Patch)
        if not hasattr(pipe.vae, "bn"):
            class MockBN(torch.nn.Module):
                def __init__(self, channels):
                    super().__init__()
                    self.register_buffer("running_mean", torch.zeros(channels))
                    self.register_buffer("running_var", torch.ones(channels))
            
            # 强制 128 通道
            pipe.vae.bn = MockBN(128).to(pipe.vae.device).to(dtype=torch.bfloat16)
            print("   ✅ VAE Patch (128ch) injected.")

        return pipe
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return None

def generate_image(pipe, prompt, width):
    """生成图片并添加水印"""
    # 1. 计算 16:9 高度
    calc_height = int(width * 9 / 16)
    height = (calc_height // 16) * 16 
    
    seed = int(time.time())
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filename = f"Flux2Gen-{timestamp}.png"
    save_path = os.path.join(SAVE_DIR, filename)
    
    print(f"🚀 任务启动 | Time: {timestamp}")
    print(f"⚙️  参数: Width={width}, Height={height} (16:9), Seed={seed}")
    print(f"📝 Prompt: {prompt[:50]}...")

    # 2. 生成图片
    try:
        image = pipe(
            prompt=prompt, 
            num_inference_steps=NUM_STEPS,
            guidance_scale=3.5,
            height=height,
            width=width,
            generator=torch.Generator("cpu").manual_seed(seed)
        ).images[0]

        # 添加水印元数据
        print("   🖌️ 正在处理水印和元数据...")
        final_image = add_watermark(image, prompt, filename, seed)
        
        # 保存
        final_image.save(save_path)
        print(f"\n🎉 任务完成！")
        print(f"   📂 保存路径: {save_path}")
        
    except RuntimeError as e:
        print(f"\n❌ 生成报错: {e}")

def main():
    # 加载模型
    pipe = load_models()
    if not pipe:
        return

    print("模型加载完成，等待用户输入...")

    while True:
        # 1. 等待用户输入 prompt 和 width
        prompt = input("Enter your prompt (or 'exit' to quit): ").strip()
        if prompt.lower() == 'exit':
            print("Exiting the program.")
            break
        
        try:
            width = int(input("Enter the image width (e.g., 1024): "))
        except ValueError:
            print("Invalid input. Please enter an integer for width.")
            continue

        # 2. 使用当前时间戳生成种子
        generate_image(pipe, prompt, width)

if __name__ == "__main__":
    main()
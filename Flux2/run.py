import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from diffusers import Flux2Pipeline, Flux2Transformer2DModel, AutoencoderKL
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig, AutoConfig
import shutil
import gc

# ================= 配置区域 =================
MODEL_PATH = "/data-shared/NAS/NewModels/Flux2"
OUTPUT_FILE = "flux_result_final.png"
OFFLOAD_FOLDER = "./flux_offload_final_v2"
# ===========================================

def print_memory(gpu_id, tag):
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated(gpu_id) / 1024**3
        print(f"   📊 [{tag}] GPU {gpu_id} 占用: \033[1;33m{mem:.2f} GB\033[0m")

def main():
    print(f"🚀 FLUX.2 (完美修正版) 启动...")
    
    torch.cuda.empty_cache()
    if os.path.exists(OFFLOAD_FOLDER):
        shutil.rmtree(OFFLOAD_FOLDER)
    os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

    # 量化配置
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ---------------------------------------------------------
    # Part 1: 加载 Mistral (GPU 0)
    # ---------------------------------------------------------
    print("\n📦 [Step 1] 加载 Mistral (Text Encoder)...")
    
    text_encoder = None
    tokenizer = None
    
    try:
        config = AutoConfig.from_pretrained(MODEL_PATH, subfolder="text_encoder")
        if hasattr(config, "use_cache"):
            config.use_cache = False
        
        text_encoder = AutoModel.from_pretrained(
            MODEL_PATH, 
            subfolder="text_encoder", 
            config=config,
            quantization_config=quant_config,
            torch_dtype=torch.bfloat16,
            device_map={"": 0} 
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")
        print_memory(0, "Mistral 就位")
        
    except Exception as e:
        print(f"❌ Text Encoder 加载失败: {e}")
        return

    # ---------------------------------------------------------
    # Part 2: 加载 VAE (GPU 0)
    # ---------------------------------------------------------
    print("\n📦 [Step 2] 加载 VAE...")
    try:
        vae = AutoencoderKL.from_pretrained(
            MODEL_PATH, 
            subfolder="vae", 
            torch_dtype=torch.bfloat16,
        ).to("cuda:0")
        print_memory(0, "VAE 就位")
    except Exception as e:
        print(f"❌ VAE 加载失败: {e}")
        return

    # ---------------------------------------------------------
    # Part 3: 加载 Transformer (全卡均摊)
    # ---------------------------------------------------------
    print("\n🏗️ [Step 3] 加载 Transformer (目标: 所有 GPU)...")
    
    max_memory_transformer = {
        0: "9GiB",   
        1: "23GiB",  
        2: "23GiB", 
        3: "23GiB"
    }
    
    try:
        transformer = Flux2Transformer2DModel.from_pretrained(
            MODEL_PATH,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory=max_memory_transformer,
            offload_folder=OFFLOAD_FOLDER
        )
        print("   ✅ Transformer 加载完毕！")
    except Exception as e:
        print(f"❌ Transformer 加载失败: {e}")
        return

    # ---------------------------------------------------------
    # Part 4: 组装 Pipeline
    # ---------------------------------------------------------
    print("\n🔧 [Step 4] 组装 Pipeline...")
    
    pipe = Flux2Pipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        scheduler=None
    )
    
    from diffusers import FlowMatchEulerDiscreteScheduler
    pipe.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")

    # ---------------------------------------------------------
    # Part 4.5: 【核心修复 V2】VAE 补丁 (强制 128 通道)
    # ---------------------------------------------------------
    print("\n🩹 [Step 4.5] 注入 VAE 补丁 (通道修正版)...")
    
    if not hasattr(pipe.vae, "bn"):
        print("   ⚠️ 检测到 VAE 缺失 'bn' 层，正在手动创建 Mock 层...")
        
        class MockBN(torch.nn.Module):
            def __init__(self, channels):
                super().__init__()
                self.register_buffer("running_mean", torch.zeros(channels))
                self.register_buffer("running_var", torch.ones(channels))

        # 【核心修改点】
        # 之前的代码读取到的是 32 (来自 config)，但报错说是 128
        # 所以我们直接硬编码为 128
        correct_channels = 128
        print(f"   🔧 强制设定 BN 通道数为: {correct_channels} (解决 Dimension Mismatch)")
        
        # 创建补丁
        pipe.vae.bn = MockBN(correct_channels).to(pipe.vae.device)
        
        # 确保数据类型匹配
        pipe.vae.bn.running_mean = pipe.vae.bn.running_mean.to(dtype=torch.bfloat16)
        pipe.vae.bn.running_var = pipe.vae.bn.running_var.to(dtype=torch.bfloat16)
        
        print("   ✅ 补丁注入成功！")

    # ---------------------------------------------------------
    # Part 5: 生成
    # ---------------------------------------------------------
    print("\n🎨 开始生成图片...")
    print(f"   - GPU 0 最终占用: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
    
    prompt = "A cinematic shot of a cyberpunk city, neon lights, rain, high detail, 8k, photorealistic, shallow depth of field"
    
    try:
        image = pipe(
            prompt=prompt, 
            num_inference_steps=28,
            guidance_scale=3.5,
            height=1024,
            width=1024,
            generator=torch.Generator("cpu").manual_seed(42)
        ).images[0]

        image.save(OUTPUT_FILE)
        print(f"\n🎉🎉🎉 终于！！！图片生成成功！已保存至: {os.path.abspath(OUTPUT_FILE)}")
        
    except RuntimeError as e:
        print(f"\n❌ 生成报错: {e}")

    shutil.rmtree(OFFLOAD_FOLDER, ignore_errors=True)

if __name__ == "__main__":
    main()

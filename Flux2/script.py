import torch
import os
import json
from datetime import datetime
from diffusers import FluxPipeline, FluxTransformer2DModel, AutoencoderKL, FlowMatchEulerDiscreteScheduler
from transformers import AutoModel, AutoConfig, AutoTokenizer, CLIPTokenizer, CLIPTextModel
from safetensors.torch import load_file
# 【新增】引入底层转换函数，用于手动修补权重
from diffusers.loaders.single_file_utils import convert_flux_transformer_checkpoint_to_diffusers

# ==========================================
# 【配置区域】
# ==========================================
unet_path = "/data-shared/NAS/NewModels/Flux2/flux2_dev_fp8mixed.safetensors"
vae_path = "/data-shared/NAS/NewModels/Flux2/flux2-vae.safetensors"
mistral_path = "/data-shared/NAS/NewModels/Flux2/mistral_3_small_flux2_bf16.safetensors"
local_mistral_config = "./mistral_config"
clip_path = "/data-shared/NAS/NewModels/Flux2/clip_l.safetensors"
local_clip_config = "./clip_config"
transformer_config_path = "./transformer_config.json" # 你的 UNet 配置文件
# ==========================================

print(">>> 1. 正在检查所有文件...")
path_list = [unet_path, vae_path, mistral_path, local_mistral_config, clip_path, local_clip_config, transformer_config_path]
for p in path_list:
    if not os.path.exists(p):
        print(f"❌ 找不到文件: {p}")
        exit()

dtype = torch.bfloat16
print("✅ 文件就位。")

# ---------------------------------------------------------
# 1. 加载 Mistral
# ---------------------------------------------------------
print(f"2. 加载 Mistral...")
config_m = AutoConfig.from_pretrained(local_mistral_config, local_files_only=True)
with torch.device("meta"):
    text_encoder_2 = AutoModel.from_config(config_m)
text_encoder_2 = text_encoder_2.to_empty(device="cpu")
sd_m = load_file(mistral_path)
text_encoder_2.load_state_dict(sd_m, strict=False)
text_encoder_2.to(dtype=dtype)
tokenizer_2 = AutoTokenizer.from_pretrained(local_mistral_config, use_fast=False, local_files_only=True)

# ---------------------------------------------------------
# 2. 加载 CLIP
# ---------------------------------------------------------
print(f"3. 加载 CLIP...")
full_config = AutoConfig.from_pretrained(local_clip_config, local_files_only=True)
text_config = getattr(full_config, "text_config", full_config)

with torch.device("meta"):
    text_encoder = CLIPTextModel(text_config)
text_encoder = text_encoder.to_empty(device="cpu")

sd_c = load_file(clip_path)
new_sd_c = {}
for k, v in sd_c.items():
    if k.startswith("text_model.") or k.startswith("logit_scale"):
        new_sd_c[k] = v
    else:
        new_sd_c[f"text_model.{k}"] = v

text_encoder.load_state_dict(new_sd_c, strict=False)
text_encoder.to(dtype=dtype)
tokenizer = CLIPTokenizer.from_pretrained(local_clip_config, local_files_only=True)

# ---------------------------------------------------------
# 3. 加载 VAE
# ---------------------------------------------------------
print(f"4. 加载 VAE (离线模式)...")
vae_config_path = "./vae_config.json"
if not os.path.exists(vae_config_path):
    print("❌ 错误：找不到 vae_config.json")
    exit()

vae = AutoencoderKL.from_single_file(
    vae_path, 
    config=vae_config_path, 
    torch_dtype=dtype,
    local_files_only=True
)

# ---------------------------------------------------------
# 5. 加载 UNet (手动修补版 - 彻底解决 KeyError)
# ---------------------------------------------------------
print(f"5. 加载 UNet (手动 Patch 模式)...")

# 1. 读取 Config 为字典
with open(transformer_config_path, "r") as f:
    transformer_config_dict = json.load(f)

# 2. 初始化空模型 (根据 config)
print("   初始化 UNet 骨架...")
with torch.device("meta"):
    transformer = FluxTransformer2DModel.from_config(transformer_config_dict)
transformer = transformer.to_empty(device="cpu").to(dtype=dtype)

# 3. 读取原始权重
print(f"   读取 Safetensors 权重: {os.path.basename(unet_path)}...")
original_state_dict = load_file(unet_path)

# 4. 【核心修复】检测并补全缺失的 bias
# FP8 模型常把 bias 优化掉了，但 diffusers 转换脚本必须要有
missing_bias_keys = [
    "time_in.in_layer.bias",
    "time_in.out_layer.bias",
    "vector_in.in_layer.bias",
    "vector_in.out_layer.bias",
    "guidance_in.in_layer.bias",
    "guidance_in.out_layer.bias"
]

print("   正在检查并修补缺失的 bias 参数...")
for key in missing_bias_keys:
    if key not in original_state_dict:
        # 找到对应的 weight 来确定 shape
        weight_key = key.replace(".bias", ".weight")
        if weight_key in original_state_dict:
            # bias 的长度通常等于 weight 的第一个维度 (out_channels)
            bias_shape = original_state_dict[weight_key].shape[0]
            print(f"   ⚠️ 补全缺失参数: {key} (全0填充)")
            # 创建全 0 的 bias
            original_state_dict[key] = torch.zeros(bias_shape, dtype=dtype)

# 5. 执行转换 (Original -> Diffusers)
print("   正在转换权重格式...")
converted_state_dict = convert_flux_transformer_checkpoint_to_diffusers(
    original_state_dict, 
    config=transformer_config_dict
)

# 6. 载入模型
print("   载入权重到模型...")
# strict=False 允许我们忽略掉一些无关紧要的不匹配
m, u = transformer.load_state_dict(converted_state_dict, strict=False)
print(f"   UNet 加载完毕 (忽略: {len(u)} 个, 丢失: {len(m)} 个)")


# ---------------------------------------------------------
# 6. 生成
# ---------------------------------------------------------
print(">>> 组装 Pipeline...")
scheduler = FlowMatchEulerDiscreteScheduler()
pipe = FluxPipeline(
    transformer=transformer,
    vae=vae,
    text_encoder=text_encoder,
    text_encoder_2=text_encoder_2,
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    scheduler=scheduler
)

prompt = "A futuristic city with neon lights, realistic, 8k, masterpiece"
print(f"\n>>> 生成中: {prompt}")

# 显存优化 (如果 24G 显存爆了，把下面这行取消注释)
# pipe.enable_model_cpu_offload()

with torch.inference_mode():
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=20,
        generator=torch.Generator("cpu").manual_seed(42)
    ).images[0]

filename = f"flux_result_{datetime.now().strftime('%H%M%S')}.png"
image.save(filename)
print(f"✅ 图片保存在: {os.path.abspath(filename)}")
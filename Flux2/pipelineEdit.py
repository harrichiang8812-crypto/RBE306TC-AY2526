import os
import torch
import argparse
import time
import textwrap
import gc
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

from diffusers import FluxImg2ImgPipeline, AutoencoderKL, Flux2Transformer2DModel
from transformers import AutoTokenizer, AutoModel
from diffusers.utils import load_image

# ================= 配置 =================
MODEL_PATH = "/data-shared/NAS/NewModels/Flux2"
SAVE_DIR = "/data-shared/NAS/RBE306TC/AY2526/GeneratedImages/"
NUM_STEPS = 28
MAX_LONG_SIDE = 2048

# ================= 工具 =================
def add_watermark(image, prompt, filename, seed):
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 24)
    except:
        font = ImageFont.load_default()
    info_text = f"File: {filename} | Seed: {seed}\nPrompt: {prompt}"
    wrapper = textwrap.TextWrapper(width=80)
    lines = wrapper.wrap(text=info_text)
    line_height = 28
    footer_height = len(lines)*line_height + 20

    new_image = Image.new("RGB", (image.width, image.height + footer_height), "black")
    new_image.paste(image, (0,0))
    draw = ImageDraw.Draw(new_image)
    y = image.height + 10
    for line in lines:
        draw.text((20, y), line, font=font, fill="white")
        y += line_height
    return new_image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--strength", type=float, default=0.75)
    parser.add_argument("--guidance_scale", type=float, default=3.5)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()

# ================= 主程序 =================
def main():
    args = parse_args()
    seed = args.seed if args.seed is not None else int(time.time())

    print(f"\n📥 Loading image: {args.image}")
    try:
        raw_image = load_image(args.image).convert("RGB")
    except Exception as e:
        print(f"❌ Failed load image: {e}")
        return

    # Resize long side <= MAX_LONG_SIDE
    ow, oh = raw_image.size
    longest = max(ow, oh)
    if longest > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / longest
        raw_image = raw_image.resize((int(ow*scale), int(oh*scale)), Image.LANCZOS)

    w, h = raw_image.size
    w = (w // 32) * 32
    h = (h // 32) * 32
    if (w, h) != raw_image.size:
        raw_image = raw_image.resize((w,h), Image.LANCZOS)
    print(f"📏 Adjusted size: {w}x{h}")

    # ---- Text Encoder ----
    print("📦 Loading Text Encoder (fp32, multi‑GPU)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, subfolder="tokenizer")

    # 分配 text encoder 到 GPU0/GPU1
    text_encoder = AutoModel.from_pretrained(
        MODEL_PATH,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        device_map={
            # 你可以按层更细化，但这是示例
            "transformer.h.0": 0, "transformer.h.1": 0,
            "transformer.h.2": 1, "transformer.h.3": 1,
            "transformer.h.4": 0, "transformer.h.5": 0,
            "transformer.h.6": 1, "transformer.h.7": 1,
            "transformer.h.8": 2, "transformer.h.9": 2,
            "transformer.h.10": 3, "transformer.h.11": 3,
        }
    )

    text_inputs = tokenizer(
        args.prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    text_inputs = {k: v.to(text_encoder.device) for k,v in text_inputs.items()}

    with torch.no_grad():
        enc_out = text_encoder(**text_inputs)
        prompt_embeds = enc_out.last_hidden_state
        if hasattr(enc_out, "pooler_output") and enc_out.pooler_output is not None:
            pooled_prompt_embeds = enc_out.pooler_output
        else:
            pooled_prompt_embeds = enc_out.last_hidden_state.mean(dim=1)

    del text_encoder, enc_out
    torch.cuda.empty_cache(); gc.collect()

    # ---- VAE ----
    print("📦 Loading VAE (fp32 all GPU0)...")
    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH,
        subfolder="vae",
        torch_dtype=torch.float32
    ).to("cuda:0")

    # ---- Transformer ----
    print("📦 Loading Transformer (fp32 multi GPU)...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        MODEL_PATH,
        subfolder="transformer",
        torch_dtype=torch.float32,
        device_map={
            # 你可以按层进一步优化
            "encoder.layer.0": 0, "encoder.layer.1": 0,
            "encoder.layer.2": 1, "encoder.layer.3": 1,
            "encoder.layer.4": 2, "encoder.layer.5": 2,
            "encoder.layer.6": 3, "encoder.layer.7": 3,
            "encoder.layer.8": 0, "encoder.layer.9": 0,
            "encoder.layer.10":1, "encoder.layer.11":1,
        }
    )

    from diffusers import FlowMatchEulerDiscreteScheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )

    pipe = FluxImg2ImgPipeline(
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
        transformer=transformer,
        scheduler=scheduler
    )

    print("\n🎨 Generating image...")
    prompt_embeds = prompt_embeds.to("cuda:0")
    pooled_prompt_embeds = pooled_prompt_embeds.to("cuda:0")

    try:
        output = pipe(
            image=raw_image,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            strength=args.strength,
            num_inference_steps=NUM_STEPS,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator("cuda:0").manual_seed(seed),
            output_type="pil"
        )
    except Exception as e:
        print(f"\n❌ Failed to generate:\n{e}")
        return

    final_img = output.images[0]
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    fname = f"Flux2-Edit-{timestamp}.png"
    save_path = os.path.join(SAVE_DIR, fname)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print("📥 Adding watermark and saving...")
    wm = add_watermark(final_img, args.prompt, fname, seed)
    wm.save(save_path)
    print(f"\n✅ Saved to: {save_path}")

if __name__ == "__main__":
    main()
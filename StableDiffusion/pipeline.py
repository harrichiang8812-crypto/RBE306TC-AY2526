import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler
from PIL import Image, ImageDraw, ImageFont 
import os
from datetime import datetime
import textwrap


WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

GIF_FRAMES = 100
GIF_PLAY = 10000  # <--- 修正了这里的语法错误


INTERMEDIATE_DIR = '/data-shared/NAS/RBE306TC/AY2526/SD-Intermediate'



# ----------------------------------------------------------------------
# 辅助函数：给图片加底部文字
# ----------------------------------------------------------------------
def add_caption_footer(pil_image, step_text, prompt_text):
    """
    在图片底部增加黑色区域并写入 Step 和 Prompt
    """
    # 1. 字体设置 (尝试加载 Arial，失败则用默认)
    try:
        font_step = ImageFont.truetype("arial.ttf", 30) 
        font_prompt = ImageFont.truetype("arial.ttf", 16)
    except IOError:
        font_step = ImageFont.load_default()
        font_prompt = ImageFont.load_default()

    margin = 10
    line_spacing = 4
    bg_color = "black"
    text_color = "white"

    # 2. 处理文字换行
    # width=60 表示大约60个字符换一行，根据图片宽度512px调整
    wrapped_prompt = textwrap.wrap(f"Prompt: {prompt_text}", width=60)
    
    # 3. 计算底部所需高度
    try:
        # PIL 9.2.0+ 使用 getbbox
        _, _, _, h_step = font_step.getbbox(step_text)
        # 取单行高度作为参考
        _, _, _, h_prompt = font_prompt.getbbox("Tg") 
    except AttributeError:
        # 旧版 PIL 兼容
        _, h_step = font_step.getsize(step_text)
        _, h_prompt = font_prompt.getsize("Tg")

    # 总高度计算
    footer_height = margin + h_step + margin + (len(wrapped_prompt) * (h_prompt + line_spacing)) + margin
    
    # 4. 创建新画布
    new_width = pil_image.width
    new_height = pil_image.height + footer_height
    combined_image = Image.new("RGB", (new_width, new_height), bg_color)
    
    # 5. 拼接和绘制
    combined_image.paste(pil_image, (0, 0))
    draw = ImageDraw.Draw(combined_image)
    
    # 写 Step
    draw.text((margin, pil_image.height + margin), step_text, font=font_step, fill=text_color)
    
    # 写 Prompt
    current_y = pil_image.height + margin + h_step + margin
    for line in wrapped_prompt:
        draw.text((margin, current_y), line, font=font_prompt, fill=text_color)
        current_y += h_prompt + line_spacing
        
    return combined_image

    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
    # Shape: (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # Shape: (1, 160 * 2)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)


def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8,
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")

        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        # --- 文本编码部分 (保持不变) ---
        if do_cfg:
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.batch_encode_plus(
                [uncond_prompt], padding="max_length", max_length=77
            ).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            uncond_context = clip(uncond_tokens)
            context = torch.cat([cond_context, uncond_context])
        else:
            tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            context = clip(tokens)
        to_idle(clip)

        # --- Sampler 初始化 (保持不变) ---
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        # --- Img2Img 或 Text2Img 初始化 (保持不变) ---
        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32, device=device)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            input_image_tensor = input_image_tensor.unsqueeze(0)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)

        diffusion = models["diffusion"]
        diffusion.to(device)

        # --- 创建中间文件保存目录 ---
        timestamp = f"{datetime.now():%Y%m%d-%H%M%S}"
        output_dir = INTERMEDIATE_DIR+f"/{timestamp}"
        gif_dir = INTERMEDIATE_DIR
        os.makedirs(output_dir, exist_ok=True)
        print(f"Intermediate images will be saved to: {output_dir}")

        
        # --- 初始化 GIF 帧列表 ---
        intermediate_frames = []

        # 防止除零错误，计算采样间隔
        gif_interval = max(1, int(round(len(sampler.timesteps)/ GIF_FRAMES)))
        
        # --- 核心去噪循环 ---
        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            time_embedding = get_time_embedding(timestep).to(device)

            model_input = latents
            if do_cfg:
                model_input = model_input.repeat(2, 1, 1, 1)

            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # 执行一步去噪
            latents = sampler.step(timestep, latents, model_output)

            
            if i==0 or i ==len(sampler.timesteps)-1 or i % gif_interval == 0:
            
                # ----------------------------------------------------
                # >>>>> 保存当前 Latent 状态为图像 (带 Prompt 底部) <<<<<
                # ----------------------------------------------------
                
                # 1. 临时加载 Decoder 模型
                decoder = models["decoder"]
                decoder.to(device)
                
                # 2. 解码并后处理
                temp_latents = latents / 0.18215 
                images = decoder(temp_latents)
                images = rescale(images, (-1, 1), (0, 255), clamp=True)
                images = images.permute(0, 2, 3, 1)
                temp_output_image_np = images.to("cpu", torch.uint8).numpy()[0]
                
                # 3. 创建 PIL Image 对象
                current_frame = Image.fromarray(temp_output_image_np)

                # 4. 【关键修改】调用 add_caption_footer，在底部添加黑色区域和文字
                current_step_num = i + 1
                step_text = f"Step: {current_step_num:03d}/{n_inference_steps:03d}"
                current_frame = add_caption_footer(current_frame, step_text, prompt)
                
                # 5. 保存帧并构造文件名
                intermediate_frames.append(current_frame)
                filename_template = f"{current_step_num:03d}-{n_inference_steps:03d}"
                filename = os.path.join(output_dir, filename_template)
                
                # print(f"Intermediate Sample Saved: {filename}")
                current_frame.save(filename+".png")
                
                # 6. 将 Decoder 移回 idle device
                to_idle(decoder)

        to_idle(diffusion)

        # ----------------------------------------------------
        # >>>>> 生成 GIF 动画 (修改版：最后一帧停顿1秒) <<<<<
        # ----------------------------------------------------
        if intermediate_frames:
            # 1. 计算基础帧的持续时间 (毫秒)
            base_duration = int(GIF_PLAY / len(intermediate_frames))
            
            # 2. 创建持续时间列表：所有帧默认使用 base_duration
            # 列表长度必须等于总帧数
            durations = [base_duration] * len(intermediate_frames)
            
            # 3. 将最后一帧的持续时间修改为 1000ms (1秒)
            durations[-1] = 1000
            
            # 第一张图片作为 GIF 的底图
            first_frame = intermediate_frames[0]
            # 后续所有图片作为帧
            other_frames = intermediate_frames[1:]
            
            gif_filename = os.path.join(gif_dir, f"{timestamp}.gif")
            
            first_frame.save(
                gif_filename, 
                save_all=True, 
                append_images=other_frames, 
                duration=durations, # <--- 这里传入列表
                loop=0 # 0 means loop infinitely
            )
            print(f"\n✨ GIF Animation Generated: {gif_filename} (Total Frames: {len(intermediate_frames)})")


        # --- 最终结果的解码 ---
        decoder = models["decoder"]
        decoder.to(device)
        
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents / 0.18215)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        
        # 7. 【关键修改】给最终结果也加上底部文字
        final_pil = Image.fromarray(images[0])
        final_step_text = f"Result: {n_inference_steps} Steps"
        final_captioned = add_caption_footer(final_pil, final_step_text, prompt)
        
        # 返回最终图像 (转回 numpy 以保持格式一致)
        return np.array(final_captioned)
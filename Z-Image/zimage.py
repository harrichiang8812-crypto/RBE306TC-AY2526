import torch
from diffusers import ZImagePipeline
import time

current_seed = int(time.time())



# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "../Z-Image-Turbo",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

#prompt="""深蓝色科技感背景，中央居中放置一个现代扁平化风格的播放器图标，图标右方清晰显示文字："zwplayer"，使用无衬线科技字体，白色发光效果。图标正上方横向排列大号加粗文字："让视频播放更简单"，其中"让视频播放"的字体为蓝色，“更”为黄色字体，“简单”为蓝色字体，轻微投影增强可读性。图标周围以极简线条和微光粒子构成抽象数据流动视觉，象征“全协议、易集成、多功能、低延时、零成本”五大特性，每个特性以小型标签形式环绕图标分布，分别标注："全协议"、"易集成"、"多功能"、"低延时"、"零成本"，字体为浅灰色半透明白色描边。页面底部右侧角落放置公众号标识，文字内容为："公众号：zwplayer"，使用较小字号，置于半透明黑色圆角矩形底板上，确保清晰可辨。整体构图对称，光影柔和，突出中央主体，无多余装饰元素。"""


#prompt = "A Chinese couple is sitting casually on a living room rug, surrounded by six cats of different coat patterns. All six cats must be sitting upright on the ground and looking directly forward. The cats include: an orange and white cat, an odd-eyed white cat, a black cat, a calico cat, an American Shorthair Scottish Fold, and a tuxedo cat. The scene should convey a warm and peaceful domestic atmosphere."

prompt = "A Chinese couple is sitting casually on a living room rug, surrounded by six cats of different coat patterns. The image must clearly depict the full bodies of the couple and all six cats, ensuring none are cropped. All six cats must be sitting upright on the ground and looking directly forward. The cats include: an orange and white cat, an odd-eyed white cat, a black cat, a calico cat, an American Shorthair Scottish Fold, and a tuxedo cat. The scene should convey a warm and peaceful domestic atmosphere."


# 2. 生成图片,记得修改参数里面的宽高
image = pipe(
    prompt=prompt,
    height=1088,
    width=1920,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(current_seed),
).images[0]

image.save("test1.png")


## 官网给的例子
#prompt2 = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."

#image = pipe(
#    prompt=prompt2,
#    height=720,
#    width=1280,
#    num_inference_steps=9,  # This actually results in 8 DiT forwards
#    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
#    generator=torch.Generator("cuda").manual_seed(42),
#).images[0]

#image.save("girl.png")

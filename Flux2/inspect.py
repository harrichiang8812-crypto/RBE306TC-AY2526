import torch
from safetensors.torch import load_file
import sys

# 你的模型路径
unet_path = "/data-shared/NAS/NewModels/Flux2/flux2_dev_fp8mixed.safetensors"

print(f"正在读取模型文件: {unet_path} ...")
try:
    state_dict = load_file(unet_path)
except Exception as e:
    print(f"读取失败: {e}")
    sys.exit()

print(f"\n✅ 读取成功！模型包含 {len(state_dict)} 个参数张量。")

# 1. 打印前 20 个 Key，看看长什么样
print("\n--- [1] 头部参数 (Top 20) ---")
for i, key in enumerate(list(state_dict.keys())[:20]):
    print(f"{i}: {key}  |  Shape: {state_dict[key].shape}")

# 2. 专门寻找 'vector' 相关的参数
print("\n--- [2] 寻找 'vector' 相关参数 ---")
vector_keys = [k for k in state_dict.keys() if "vector" in k]
if vector_keys:
    for k in vector_keys[:10]: # 只打印前10个
        print(f"Found: {k}")
else:
    print("❌ 警告：没找到任何包含 'vector' 的参数！")

# 3. 专门寻找 'time' 相关的参数
print("\n--- [3] 寻找 'time' 相关参数 ---")
time_keys = [k for k in state_dict.keys() if "time" in k]
for k in time_keys[:5]:
    print(f"Found: {k}")

# 4. 专门寻找 'img' 相关的参数
print("\n--- [4] 寻找 'img' 相关参数 ---")
img_keys = [k for k in state_dict.keys() if "img" in k]
for k in img_keys[:5]:
    print(f"Found: {k}")

print("\n诊断结束。请把上面的输出截图发给我！")
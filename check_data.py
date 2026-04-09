import numpy as np
import torch
import glob
import os

# 1. 检查文件是否存在
data_dir = "./sted_dataset_v2"
images = sorted(glob.glob(os.path.join(data_dir, "images", "*.npy")))
labels = sorted(glob.glob(os.path.join(data_dir, "labels", "*.npy")))

print(f"找到图片: {len(images)} 张")
print(f"找到标签: {len(labels)} 张")

if len(images) == 0:
    print("错误：没有找到数据，请先运行生成代码！")
    exit()

# 2. 试读取一张，看看形状和数值
img = np.load(images[0])
lbl = np.load(labels[0])

print(f"\n图片形状: {img.shape}, 数据类型: {img.dtype}")
print(f"图片最大值(光子数): {img.max()}")
print(f"标签形状: {lbl.shape}, 数据类型: {lbl.dtype}")

# 3. 模拟转为 Tensor
img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0) # 增加通道维度
print(f"Tensor 形状: {img_tensor.shape}") # 应该是 (1, 64, 64)

print("\n准备工作完成，可以开始写 train.py 了！")
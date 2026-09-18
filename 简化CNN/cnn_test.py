import torch

from data_io import load_image
from model import SimpleCNN


# 读取一张图片
image = load_image(
    "raw_dataset.npz",
    index=0,
)

print("NumPy图片：")
print(image.shape)


# NumPy → Tensor
x = torch.from_numpy(image).float()

# [H, W]
# ↓
# [B, C, H, W]
x = x.unsqueeze(0).unsqueeze(0)

print("CNN输入：")
print(x.shape)


# 创建模型
model = SimpleCNN()

# 前向传播
output = model(x)

print("模型输出：")
print(output.shape)

print(output)
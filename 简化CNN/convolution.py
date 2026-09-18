import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data_io import load_image


# ==================================================
# 0. 创建输出文件夹
# ==================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(script_dir, "卷积测试层")

os.makedirs(output_dir, exist_ok=True)


# ==================================================
# 1. 读取图像
# ==================================================

x = load_image(
    "raw_dataset.npz",
    index=0,
)


# ==================================================
# 2. 数据类型转换与归一化
# ==================================================

x = torch.from_numpy(x).float()
x = x / 255.0


# ==================================================
# 3. [H, W] → [B, C, H, W]
# ==================================================

x = x.unsqueeze(0).unsqueeze(0)


# ==================================================
# 4. 创建卷积层
# ==================================================

conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=3,
    padding=1
)


# ==================================================
# 5. 进行卷积
# ==================================================

features = conv(x)


# ==================================================
# 6. 查看数据形状
# ==================================================

print("输入：", x.shape)
print("卷积后：", features.shape)


# ==================================================
# 7. 显示并保存第一个卷积特征图
# ==================================================

plt.imshow(
    features[0, 0].detach().numpy(),
    cmap="gray"
)

plt.title("Feature Map 1")
plt.axis("off")

plt.savefig(
    os.path.join(output_dir, "卷积特征图.png"),
    bbox_inches="tight"
)

plt.show()
plt.close()


# ==================================================
# 8. 创建最大池化层
# ==================================================

pool = nn.MaxPool2d(
    kernel_size=2,
    stride=2
)


# ==================================================
# 9. 进行池化
# ==================================================

pooled_features = pool(features)

print("\n池化之后：")
print("shape:", pooled_features.shape)


# ==================================================
# 10. 取出第一张图片的 8 个特征图
# ==================================================

pooled = pooled_features[0].detach().cpu()


# ==================================================
# 11. 显示并保存池化后的特征图
# ==================================================

fig, axes = plt.subplots(
    2,
    4,
    figsize=(10, 5)
)

for i in range(pooled.shape[0]):
    row = i // 4
    col = i % 4

    axes[row, col].imshow(
        pooled[i],
        cmap="gray"
    )

    axes[row, col].set_title(
        f"Channel {i + 1}"
    )

    axes[row, col].axis("off")


plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "池化特征图.png"),
    bbox_inches="tight"
)

plt.show()
plt.close()


print("\n图片已保存到：")
print(output_dir)


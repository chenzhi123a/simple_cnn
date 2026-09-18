"""
CNN 测试示例。

当前功能：
    1. 从 test.npz 中读取测试数据
    2. 随机抽取一张测试图片
    3. 加载训练过程中保存的最佳模型
    4. 使用 CNN 对图片进行分类
    5. 将预测结果与真实标签进行比较
    6. 可视化测试图片、真实类别、预测类别和置信度

测试流程：

    test.npz
        ↓
    随机抽取一张图片
        ↓
    NumPy → Tensor
        ↓
    增加 Channel 维度
        ↓
    加载最佳模型
        ↓
    前向传播
        ↓
    Logits
        ↓
    Softmax
        ↓
    预测类别
        ↓
    与真实标签比较
        ↓
    可视化结果
"""


import random

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

from data_io import load_dataset
from model import SimpleCNN


# ============================================================
# 1. 数据路径
# ============================================================

DATA_PATH_test = (
    r"F:\chenzhi\MRI研一学习计划"
    r"\MRI研一学习计划\03_深度学习基础"
    r"\资料\简化CNN\dataset\test.npz"
)

# 最佳模型路径
MODEL_PATH = "best_model.pth"


# ============================================================
# 2. 类别名称
# ============================================================

class_names = {
    0: "圆形",
    1: "方形",
}


# ============================================================
# 3. 读取测试数据
# ============================================================

images, labels = load_dataset(DATA_PATH_test)

print("测试数据读取完成")
print("images.shape:", images.shape)
print("labels.shape:", labels.shape)


# ============================================================
# 4. 随机选择一张测试图片
# ============================================================

index = random.randrange(len(images))

image = images[index]
true_label = int(labels[index])

print("\n随机抽取测试图片")
print("图片索引:", index)
print("图片 shape:", image.shape)
print("真实标签:", true_label)
print("真实类别:", class_names[true_label])


# ============================================================
# 5. NumPy → Tensor
# ============================================================

# NumPy：
#
# [64, 64]
#
# ↓ torch.from_numpy()
#
# [64, 64]
#
# ↓ unsqueeze(0)
#
# [1, 64, 64]
#
# ↓ unsqueeze(0)
#
# [1, 1, 64, 64]
#
# 最终符合 CNN 的输入格式：
#
# [Batch, Channel, Height, Width]

image_tensor = torch.from_numpy(image).float()

image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)

print("\nCNN 输入 shape:", image_tensor.shape)


# ============================================================
# 6. 创建模型
# ============================================================

model = SimpleCNN()


# ============================================================
# 7. 加载最佳模型参数
# ============================================================

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu",
    )
)

print("\n最佳模型加载完成")


# ============================================================
# 8. 切换到测试模式
# ============================================================

model.eval()


# ============================================================
# 9. 使用模型进行预测
# ============================================================

with torch.no_grad():

    # 前向传播
    output = model(image_tensor)


print("\n模型输出 Logits:")
print(output)


# ============================================================
# 10. Logits → 概率
# ============================================================

probabilities = torch.softmax(
    output,
    dim=1,
)

print("\n分类概率:")
print(probabilities)


# ============================================================
# 11. 获取预测类别
# ============================================================

predicted_label = torch.argmax(
    probabilities,
    dim=1,
).item()

predicted_probability = probabilities[
    0,
    predicted_label,
].item()


print("\n预测结果")
print("预测标签:", predicted_label)
print("预测类别:", class_names[predicted_label])
print(
    f"预测概率: {predicted_probability:.2%}"
)


# ============================================================
# 12. 判断预测是否正确
# ============================================================

is_correct = predicted_label == true_label


if is_correct:

    print("\n预测结果：正确 ✓")

else:

    print("\n预测结果：错误 ✗")


# ============================================================
# 13. 可视化
# ============================================================

plt.figure(figsize=(7, 7))

plt.imshow(
    image,
    cmap="gray",
)

plt.axis("off")


# 根据预测结果设置标题
if is_correct:

    result_text = "预测正确 ✓"

else:

    result_text = "预测错误 ✗"


plt.title(
    f"{result_text}\n"
    f"真实类别：{class_names[true_label]}\n"
    f"预测类别：{class_names[predicted_label]}\n"
    f"置信度：{predicted_probability:.2%}",
    fontsize=14,
)


plt.tight_layout()

plt.show()


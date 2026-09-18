"""
生成圆形和方形图像数据集。

任务：
    0 -> 圆形
    1 -> 方形

本脚本只负责：
    1. 生成图像
    2. 生成对应标签
    3. 保存完整的原始数据

不负责：
    - 训练集/验证集/测试集划分
    - CNN训练
    - 模型验证
    - 模型测试

生成的数据保存在 data.py 同级目录：
    raw_dataset.npz
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# =========================
# 基本参数
# =========================

IMAGE_SIZE = 64

# 每种图形生成的数量
SAMPLES_PER_CLASS = 1000

# 随机种子
SEED = 42


# =========================
# 生成圆形
# =========================

def generate_circle(rng):
    """生成一张圆形图片。"""

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=0,
    )

    draw = ImageDraw.Draw(image)

    # 随机半径
    radius = rng.integers(8, 20)

    # 随机圆心
    center_x = rng.integers(
        radius,
        IMAGE_SIZE - radius,
    )

    center_y = rng.integers(
        radius,
        IMAGE_SIZE - radius,
    )

    # 随机灰度
    intensity = int(rng.integers(180, 256))

    draw.ellipse(
        (
            center_x - radius,
            center_y - radius,
            center_x + radius,
            center_y + radius,
        ),
        fill=intensity,
    )

    return np.array(image, dtype=np.uint8)


# =========================
# 生成方形
# =========================

def generate_square(rng):
    """生成一张方形图片。"""

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=0,
    )

    draw = ImageDraw.Draw(image)

    # 随机边长
    side = rng.integers(16, 40)

    # 随机左上角位置
    x1 = rng.integers(
        0,
        IMAGE_SIZE - side,
    )

    y1 = rng.integers(
        0,
        IMAGE_SIZE - side,
    )

    x2 = x1 + side
    y2 = y1 + side

    # 随机灰度
    intensity = int(rng.integers(180, 256))

    draw.rectangle(
        (x1, y1, x2, y2),
        fill=intensity,
    )

    return np.array(image, dtype=np.uint8)


# =========================
# 生成原始数据集
# =========================

def generate_dataset():
    """生成所有图像以及对应标签。"""

    rng = np.random.default_rng(SEED)

    images = []
    labels = []

    # -------------------------
    # 生成圆形
    # -------------------------

    for _ in range(SAMPLES_PER_CLASS):

        image = generate_circle(rng)

        images.append(image)
        labels.append(0)

    # -------------------------
    # 生成方形
    # -------------------------

    for _ in range(SAMPLES_PER_CLASS):

        image = generate_square(rng)

        images.append(image)
        labels.append(1)

    # 转换为 NumPy 数组
    images = np.array(images, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int64)

    return images, labels


# =========================
# 保存数据
# =========================

def save_dataset(images, labels):

    # 获取 data.py 所在目录
    current_dir = Path(__file__).resolve().parent

    # 保存到同级目录
    save_path = current_dir / "raw_dataset.npz"

    np.savez_compressed(
        save_path,
        images=images,
        labels=labels,
    )

    return save_path


# =========================
# 主程序
# =========================

def main():

    print("开始生成原始数据集...")

    images, labels = generate_dataset()

    print()
    print("数据生成完成")
    print("-------------------------")
    print(f"图像数量：{len(images)}")
    print(f"图像 shape：{images.shape}")
    print(f"标签 shape：{labels.shape}")
    print(f"数据类型：{images.dtype}")
    print(f"标签类型：{labels.dtype}")
    print("-------------------------")

    # 保存
    save_path = save_dataset(
        images,
        labels,
    )

    print()
    print(f"数据已保存到：")
    print(save_path)


if __name__ == "__main__":
    main()


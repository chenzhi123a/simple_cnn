"""
数据集划分工具。

功能：
    读取原始数据集，并按照指定比例划分为：

        训练集
        验证集
        测试集

输入数据格式：
    .npz 文件

其中包含：
    images
    labels

默认输入：
    data.py 同级目录下的 raw_dataset.npz

默认输出：
    data.py 同级目录下的 dataset/

使用示例：

    python data_process.py

或者：

    python data_process.py --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15

也可以指定输入和输出：

    python data_process.py \
        --input my_data.npz \
        --output my_dataset
"""

from pathlib import Path
import argparse

import numpy as np


# ============================================================
# 1. 读取数据
# ============================================================

def load_dataset(path):
    """
    读取 .npz 数据集。

    参数
    ----
    path : Path
        数据集文件路径。

    返回
    ----
    images : np.ndarray
        图像数据。

    labels : np.ndarray
        标签数据。
    """

    data = np.load(path)

    images = data["images"]
    labels = data["labels"]

    # 基本检查
    if len(images) != len(labels):
        raise ValueError(
            "images 和 labels 的样本数量不一致。"
        )

    return images, labels


# ============================================================
# 2. 划分数据集
# ============================================================

def split_dataset(
    images,
    labels,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
):
    """
    将数据随机划分为训练集、验证集和测试集。

    参数
    ----
    images : np.ndarray
        所有图像。

    labels : np.ndarray
        所有标签。

    train_ratio : float
        训练集比例。

    val_ratio : float
        验证集比例。

    test_ratio : float
        测试集比例。

    seed : int
        随机种子，保证每次划分结果一致。

    返回
    ----
    train_images, train_labels
    val_images, val_labels
    test_images, test_labels
    """

    # --------------------------------------------------------
    # 检查比例
    # --------------------------------------------------------

    ratio_sum = train_ratio + val_ratio + test_ratio

    if not np.isclose(ratio_sum, 1.0):
        raise ValueError(
            "train_ratio + val_ratio + test_ratio 必须等于 1。"
        )

    if min(train_ratio, val_ratio, test_ratio) < 0:
        raise ValueError(
            "数据集比例不能小于 0。"
        )

    # --------------------------------------------------------
    # 检查数据
    # --------------------------------------------------------

    if len(images) != len(labels):
        raise ValueError(
            "images 和 labels 的样本数量不一致。"
        )

    # --------------------------------------------------------
    # 创建随机数生成器
    # --------------------------------------------------------

    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # 生成随机索引
    # --------------------------------------------------------

    indices = rng.permutation(len(images))

    # --------------------------------------------------------
    # 根据比例计算分界位置
    # --------------------------------------------------------

    total = len(images)

    train_end = int(total * train_ratio)

    val_end = train_end + int(total * val_ratio)

    # --------------------------------------------------------
    # 根据索引划分
    # --------------------------------------------------------

    train_indices = indices[:train_end]

    val_indices = indices[train_end:val_end]

    test_indices = indices[val_end:]

    # --------------------------------------------------------
    # 获取数据
    # --------------------------------------------------------

    train_images = images[train_indices]
    train_labels = labels[train_indices]

    val_images = images[val_indices]
    val_labels = labels[val_indices]

    test_images = images[test_indices]
    test_labels = labels[test_indices]

    return (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
    )


# ============================================================
# 3. 保存数据
# ============================================================

def save_split_dataset(
    images,
    labels,
    output_dir,
    name,
):
    """
    保存一个数据集子集。

    参数
    ----
    images : np.ndarray
        图像数据。

    labels : np.ndarray
        标签数据。

    output_dir : Path
        输出目录。

    name : str
        数据集名称，例如 train / val / test。
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = output_dir / f"{name}.npz"

    np.savez_compressed(
        save_path,
        images=images,
        labels=labels,
    )

    return save_path


# ============================================================
# 4. 打印数据集信息
# ============================================================

def print_dataset_info(
    name,
    images,
    labels,
):
    """打印数据集的基本信息。"""

    print(
        f"{name:>5}："
        f"images={images.shape}, "
        f"labels={labels.shape}"
    )


# ============================================================
# 5. 参数解析
# ============================================================

def parse_args():
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(
        description="将原始数据集划分为训练集、验证集和测试集。"
    )

    # --------------------------------------------------------
    # 输入文件
    # --------------------------------------------------------

    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="原始 .npz 数据集路径。",
    )

    # --------------------------------------------------------
    # 输出目录
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="划分后数据集的输出目录。",
    )

    # --------------------------------------------------------
    # 数据集比例
    # --------------------------------------------------------

    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="训练集比例，默认 0.7。",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="验证集比例，默认 0.15。",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="测试集比例，默认 0.15。",
    )

    # --------------------------------------------------------
    # 随机种子
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42。",
    )

    return parser.parse_args()


# ============================================================
# 6. 主程序
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # 获取脚本所在目录
    # --------------------------------------------------------

    current_dir = Path(__file__).resolve().parent

    # --------------------------------------------------------
    # 设置默认输入路径
    # --------------------------------------------------------

    if args.input is None:
        input_path = current_dir / "raw_dataset.npz"
    else:
        input_path = Path(args.input)

    # --------------------------------------------------------
    # 设置默认输出路径
    # --------------------------------------------------------

    if args.output is None:
        output_dir = current_dir / "dataset"
    else:
        output_dir = Path(args.output)

    # --------------------------------------------------------
    # 检查输入文件
    # --------------------------------------------------------

    if not input_path.exists():
        raise FileNotFoundError(
            f"找不到输入数据集：{input_path}"
        )

    print("=" * 50)
    print("数据集处理")
    print("=" * 50)

    print(f"输入数据：{input_path}")
    print(f"输出目录：{output_dir}")
    print()

    # --------------------------------------------------------
    # 读取数据
    # --------------------------------------------------------

    images, labels = load_dataset(input_path)

    print("原始数据：")
    print_dataset_info(
        "全部",
        images,
        labels,
    )

    print()

    # --------------------------------------------------------
    # 划分数据集
    # --------------------------------------------------------

    (
        train_images,
        train_labels,
        val_images,
        val_labels,
        test_images,
        test_labels,
    ) = split_dataset(
        images,
        labels,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # --------------------------------------------------------
    # 打印划分结果
    # --------------------------------------------------------

    print("划分结果：")

    print_dataset_info(
        "训练集",
        train_images,
        train_labels,
    )

    print_dataset_info(
        "验证集",
        val_images,
        val_labels,
    )

    print_dataset_info(
        "测试集",
        test_images,
        test_labels,
    )

    print()

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    train_path = save_split_dataset(
        train_images,
        train_labels,
        output_dir,
        "train",
    )

    val_path = save_split_dataset(
        val_images,
        val_labels,
        output_dir,
        "val",
    )

    test_path = save_split_dataset(
        test_images,
        test_labels,
        output_dir,
        "test",
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print("数据保存完成：")

    print(f"训练集：{train_path}")
    print(f"验证集：{val_path}")
    print(f"测试集：{test_path}")

    print("=" * 50)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":
    main()

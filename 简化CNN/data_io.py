"""
数据读取工具。

当前功能：
    1. 从 .npz 文件中读取指定索引的一张图片
    2. 从 .npz 文件中批量读取图片
    3. 从 .npz 文件中读取图片和标签

输入的 .npz 文件需要包含：
    images
    labels
"""

from pathlib import Path

import numpy as np


def load_image(
    file_path: str | Path,
    index: int = 0,
) -> np.ndarray:
    """
    从 .npz 文件中读取一张图片。

    参数
    ----------
    file_path : str | Path
        .npz 文件路径。

    index : int
        图片索引，默认读取第 0 张。

    返回
    ----------
    np.ndarray
        一张图片。
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{file_path}"
        )

    data = np.load(file_path)

    images = data["images"]

    if index < 0 or index >= len(images):
        raise IndexError(
            f"图片索引 {index} 超出范围，"
            f"当前共有 {len(images)} 张图片。"
        )

    return images[index]


def load_images(
    file_path: str | Path,
    indices: list[int] | None = None,
) -> np.ndarray:
    """
    从 .npz 文件中批量读取图片。

    参数
    ----------
    file_path : str | Path
        .npz 文件路径。

    indices : list[int] | None
        要读取的图片索引。

        例如：
            [0, 1, 2, 3]

        如果为 None，则读取全部图片。

    返回
    ----------
    np.ndarray
        多张图片组成的数组。

        例如：
            (100, 64, 64)

        表示：
            100 张图片
            每张图片大小为 64 × 64。
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{file_path}"
        )

    data = np.load(file_path)

    images = data["images"]

    # 如果没有指定索引，就返回全部图片
    if indices is None:
        return images

    # 检查每一个索引是否合法
    for index in indices:
        if index < 0 or index >= len(images):
            raise IndexError(
                f"图片索引 {index} 超出范围，"
                f"当前共有 {len(images)} 张图片。"
            )

    return images[indices]


def load_dataset(
    file_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    从 .npz 文件中读取完整数据集。

    返回：
        images：所有图片
        labels：所有标签

    例如：

        images.shape
            (1000, 64, 64)

        labels.shape
            (1000,)
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{file_path}"
        )

    data = np.load(file_path)

    images = data["images"]
    labels = data["labels"]

    # 检查图片数量和标签数量是否一致
    if len(images) != len(labels):
        raise ValueError(
            f"图片数量与标签数量不一致："
            f"images={len(images)}, "
            f"labels={len(labels)}"
        )

    return images, labels


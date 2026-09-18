"""
简易 CNN 模型。

网络结构：
    输入图像
        ↓
    卷积层 1
        ↓
      ReLU
        ↓
    卷积层 2
        ↓
      ReLU
        ↓
      最大池化
        ↓
      Flatten
        ↓
      全连接层
        ↓
      分类结果

当前任务：
    圆形 vs 方形（二分类）
"""

import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
    """两层卷积 + ReLU + 池化 + 全连接的简易 CNN。"""

    def __init__(self):
        super().__init__()

        # 第一层卷积：
        # 输入：1个通道（灰度图）
        # 输出：8张特征图
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=8,
            kernel_size=3,
            padding=1,
        )

        # 第一层 ReLU
        self.relu1 = nn.ReLU()

        # 第二层卷积：
        # 输入：上一层的8张特征图
        # 输出：16张特征图
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1,
        )

        # 第二层 ReLU
        self.relu2 = nn.ReLU()

        # 最大池化：
        # 2×2区域压缩成1个值
        self.pool = nn.MaxPool2d(
            kernel_size=2,
        )

        # 全连接层：
        #
        # 假设输入图片大小为64×64。
        #
        # 两次卷积 padding=1、kernel_size=3
        # 不改变空间尺寸：
        #
        # 64×64
        #   ↓ Conv1
        # 64×64
        #   ↓ Conv2
        # 64×64
        #   ↓ Pool
        # 32×32
        #
        # 此时有16张特征图：
        #
        # 16 × 32 × 32 = 16384
        #
        # 最终输出2个类别：
        # 0：圆形
        # 1：方形
        self.fc = nn.Linear(
            16 * 32 * 32,
            2,
        )

    def forward(self, x):
        """定义数据经过网络的顺序。"""

        # 第一次特征提取
        x = self.conv1(x)

        # 引入非线性
        x = self.relu1(x)

        # 第二次特征提取
        x = self.conv2(x)

        # 引入非线性
        x = self.relu2(x)

        # 压缩空间尺寸
        x = self.pool(x)

        # 展平
        x = x.flatten(1)

        # 全连接分类
        x = self.fc(x)

        return x



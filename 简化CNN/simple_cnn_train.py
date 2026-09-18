"""
CNN 最简单训练示例。

当前功能：
    1. 从 train.npz 中读取训练数据
    2. 从 val.npz 中读取验证数据
    3. 使用 DataLoader 将数据分成 batch
    4. 使用训练集更新模型参数
    5. 每个 Epoch 结束后使用验证集评估模型
    6. 保存 Val Loss 最低时的模型参数

训练流程：

    Train
        ↓
    前向传播
        ↓
    Loss
        ↓
    反向传播
        ↓
    参数更新
        ↓
    一个 Epoch 结束
        ↓
    Validation
        ↓
    计算 Val Loss
        ↓
    是否优于历史最佳？
        ↓
    是 → 保存模型参数
"""


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from data_io import load_dataset
from model import SimpleCNN


# ============================================================
# 1. 数据路径
# ============================================================

DATA_PATH_train = (
    r"F:\chenzhi\MRI研一学习计划"
    r"\MRI研一学习计划\03_深度学习基础"
    r"\资料\简化CNN\dataset\train.npz"
)

DATA_PATH_val = (
    r"F:\chenzhi\MRI研一学习计划"
    r"\MRI研一学习计划\03_深度学习基础"
    r"\资料\简化CNN\dataset\val.npz"
)


# ============================================================
# 2. 训练参数
# ============================================================

batch_size = 32
num_epochs = 10
learning_rate = 0.001


# ============================================================
# 3. 模型保存路径
# ============================================================

BEST_MODEL_PATH = "best_model.pth"


# ============================================================
# 4. 读取数据
# ============================================================

images, labels = load_dataset(DATA_PATH_train)

val_images, val_labels = load_dataset(DATA_PATH_val)


print("读取数据完成")

print("\n训练集：")
print("images.shape:", images.shape)
print("labels.shape:", labels.shape)

print("\n验证集：")
print("val_images.shape:", val_images.shape)
print("val_labels.shape:", val_labels.shape)


# ============================================================
# 5. NumPy → Tensor
# ============================================================

images = torch.from_numpy(images).float()
labels = torch.from_numpy(labels).long()

val_images = torch.from_numpy(val_images).float()
val_labels = torch.from_numpy(val_labels).long()


# ============================================================
# 6. 增加 Channel 维度
# ============================================================

# 原始数据：
#
# [N, 64, 64]
#
# 增加 Channel 后：
#
# [N, 1, 64, 64]

images = images.unsqueeze(1)
val_images = val_images.unsqueeze(1)


print("\nTensor 数据：")

print("训练集 images.shape:", images.shape)
print("训练集 labels.shape:", labels.shape)

print("验证集 val_images.shape:", val_images.shape)
print("验证集 val_labels.shape:", val_labels.shape)


# ============================================================
# 7. 创建 Dataset
# ============================================================

train_dataset = TensorDataset(
    images,
    labels,
)

val_dataset = TensorDataset(
    val_images,
    val_labels,
)


# ============================================================
# 8. 创建 DataLoader
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
)


print("\nDataLoader 创建完成")
print("batch_size:", batch_size)


# ============================================================
# 9. 创建模型
# ============================================================

model = SimpleCNN()


# ============================================================
# 10. 创建 Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# 11. 创建 Optimizer
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate,
)


# ============================================================
# 12. 初始化最佳验证 Loss
# ============================================================

# 一开始还没有任何验证结果，
# 所以把最佳 Loss 设置为正无穷。

best_val_loss = float("inf")


# ============================================================
# 13. 多个 Epoch 的训练
# ============================================================

for epoch in range(num_epochs):

    # ========================================================
    # Train
    # ========================================================

    model.train()

    total_train_loss = 0.0

    for images_batch, labels_batch in train_loader:

        # ----------------------------------------------------
        # ① 清除上一轮梯度
        # ----------------------------------------------------

        optimizer.zero_grad()

        # ----------------------------------------------------
        # ② 前向传播
        # ----------------------------------------------------

        output = model(images_batch)

        # ----------------------------------------------------
        # ③ 计算 Loss
        # ----------------------------------------------------

        loss = criterion(
            output,
            labels_batch,
        )

        # ----------------------------------------------------
        # ④ 反向传播
        # ----------------------------------------------------

        loss.backward()

        # ----------------------------------------------------
        # ⑤ 更新模型参数
        # ----------------------------------------------------

        optimizer.step()

        # ----------------------------------------------------
        # ⑥ 累加 Loss
        # ----------------------------------------------------

        total_train_loss += loss.item()


    # ========================================================
    # 计算当前 Epoch 的平均 Train Loss
    # ========================================================

    train_loss = total_train_loss / len(train_loader)


    # ========================================================
    # Validation
    # ========================================================

    # 切换到验证模式
    model.eval()

    total_val_loss = 0.0

    # 验证阶段不需要计算梯度
    with torch.no_grad():

        for images_batch, labels_batch in val_loader:

            # ------------------------------------------------
            # ① 前向传播
            # ------------------------------------------------

            output = model(images_batch)

            # ------------------------------------------------
            # ② 计算 Loss
            # ------------------------------------------------

            loss = criterion(
                output,
                labels_batch,
            )

            # ------------------------------------------------
            # ③ 累加 Loss
            # ------------------------------------------------

            total_val_loss += loss.item()


    # ========================================================
    # 计算当前 Epoch 的平均 Val Loss
    # ========================================================

    val_loss = total_val_loss / len(val_loader)


    # ========================================================
    # 判断是否为当前最佳模型
    # ========================================================

    if val_loss < best_val_loss:

        # 更新历史最佳 Val Loss
        best_val_loss = val_loss

        # 保存当前模型参数
        torch.save(
            model.state_dict(),
            BEST_MODEL_PATH,
        )

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss = {train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f} | "
            f"保存最佳模型"
        )

    else:

        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss = {train_loss:.4f} | "
            f"Val Loss = {val_loss:.4f}"
        )


# ============================================================
# 14. 训练结束
# ============================================================

print("\n训练完成！")

print(
    f"最佳 Val Loss = {best_val_loss:.4f}"
)

print(
    f"最佳模型已保存到：{BEST_MODEL_PATH}"
)

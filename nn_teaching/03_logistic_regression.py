# -*- coding: utf-8 -*-
"""
脚本：使用 PyTorch 实现二维空间中的逻辑回归二分类。

使用单个线性层 + Sigmoid，配合 BCELoss；补充特征 Dropout、训练/验证划分、
余弦退火与早停。绘制决策边界与样本散点图。
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification

from training_utils import EarlyStopping

# 全局中文字体与负号显示（兼容 macOS / Windows）
matplotlib.rcParams["font.sans-serif"] = [
    "PingFang SC",
    "SimHei",
    "Arial Unicode MS",
    "Microsoft YaHei",
    "Heiti TC",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def configure_chinese_matplotlib() -> None:
    """
    配置 matplotlib 中文显示与负号正常渲染。

    Returns:
        None: 无返回值。
    """
    matplotlib.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "SimHei",
        "Arial Unicode MS",
        "Microsoft YaHei",
        "Heiti TC",
        "DejaVu Sans",
    ]
    matplotlib.rcParams["axes.unicode_minus"] = False


def make_2d_binary_data(
    n_samples: int = 400,
    random_state: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成二维二分类模拟数据（线性可分性较好，便于观察直线决策边界）。

    Args:
        n_samples: 样本数。
        random_state: 随机种子。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - X: 形状 (N, 2) 的特征。
            - y: 形状 (N, 1) 的标签，取值为 0 或 1。
    """
    x_np, y_np = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        flip_y=0.02,
        class_sep=1.6,
        random_state=random_state,
    )
    x = torch.tensor(x_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32).view(-1, 1)
    return x, y


class LogisticRegressionModel(nn.Module):
    """
    逻辑回归模型：线性变换后接 Sigmoid，将实数输出映射到 (0, 1) 概率区间。

    数学形式：p = sigmoid(W x + b)。训练时使用 BCELoss(p, y)。

    说明：若改用 nn.BCEWithLogitsLoss，则通常不在末尾加 Sigmoid，损失函数内部会合并 sigmoid，
    数值更稳定；本为直观展示“线性层 + 激活”，采用显式 Sigmoid。
    """

    def __init__(self, in_features: int = 2) -> None:
        """
        初始化单层线性变换与 Sigmoid 激活。

        Args:
            in_features: 输入特征维度，二维分类任务中为 2。

        Returns:
            None: 构造函数无返回值。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(in_features=in_features, out_features=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：计算每个样本属于正类的预测概率。

        Args:
            x: 输入特征，形状 (N, 2)。

        Returns:
            torch.Tensor: 形状 (N, 1) 的概率值。
        """
        x = self.dropout(x)
        logits = self.linear(x)
        return self.sigmoid(logits)


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 2000,
    lr: float = 0.5,
) -> tuple[list[float], list[float]]:
    """
    使用二元交叉熵损失与 SGD 训练；CosineAnnealingLR 与验证集早停。

    Args:
        model: LogisticRegressionModel 实例。
        x_train, y_train: 训练集。
        x_val, y_val: 验证集。
        max_epochs: 最大轮数。
        lr: 初始学习率。

    Returns:
        tuple: (训练 BCE 历史, 验证 BCE 历史)。
    """
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4)
    early = EarlyStopping(patience=120, min_delta=1e-4, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        prob = model(x_train)
        loss = criterion(prob, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tr = criterion(model(x_train), y_train).item()
            va = criterion(model(x_val), y_val).item()
        train_losses.append(tr)
        val_losses.append(va)

        if early.step(va):
            break

    return train_losses, val_losses


def plot_decision_boundary(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    grid_steps: int = 200,
) -> None:
    """
    在二维平面上绘制分类散点与决策边界（概率等于 0.5 的等高线）。

    Args:
        model: 训练好的模型。
        x: 特征 (N, 2)。
        y: 标签 (N, 1)。
        grid_steps: 网格分辨率，越大边界越平滑但计算稍慢。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy().ravel()

    x_min, x_max = float(x_np[:, 0].min() - 0.8), float(x_np[:, 0].max() + 0.8)
    y_min, y_max = float(x_np[:, 1].min() - 0.8), float(x_np[:, 1].max() + 0.8)
    gx, gy = np.meshgrid(
        np.linspace(x_min, x_max, grid_steps),
        np.linspace(y_min, y_max, grid_steps),
    )
    grid = torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)
    with torch.no_grad():
        prob_grid = model(grid).numpy().reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(gx, gy, prob_grid, levels=30, cmap="RdBu_r", alpha=0.75)
    ax.contour(gx, gy, prob_grid, levels=[0.5], colors="k", linewidths=2)
    mask0 = y_np < 0.5
    mask1 = y_np >= 0.5
    ax.scatter(
        x_np[mask0, 0],
        x_np[mask0, 1],
        c="#4C72B0",
        edgecolors="k",
        s=35,
        label="负类 0",
    )
    ax.scatter(
        x_np[mask1, 0],
        x_np[mask1, 1],
        c="#C44E52",
        edgecolors="k",
        s=35,
        label="正类 1",
    )
    ax.set_title("逻辑回归：二维二分类与决策边界（黑线为 p=0.5）")
    ax.set_xlabel("特征维度 1")
    ax.set_ylabel("特征维度 2")
    fig.colorbar(contour, ax=ax, fraction=0.046, pad=0.04, label="预测为正类概率")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_loss_curve(train_losses: list[float], val_losses: list[float]) -> None:
    """
    绘制训练与验证 BCE 随 epoch 变化的曲线。

    Args:
        train_losses: 训练集损失序列。
        val_losses: 验证集损失序列。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    plt.figure(figsize=(7, 4))
    ep = range(1, len(train_losses) + 1)
    plt.plot(ep, train_losses, color="#4C72B0", label="训练集")
    plt.plot(ep, val_losses, color="#C44E52", label="验证集")
    plt.title("逻辑回归：BCE 损失（余弦退火 + 早停）")
    plt.xlabel("Epoch")
    plt.ylabel("BCE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    主函数：生成数据、训练、打印损失并展示决策边界。

    Returns:
        None: 无返回值。
    """
    torch.manual_seed(0)
    configure_chinese_matplotlib()

    x, y = make_2d_binary_data()
    n = x.size(0)
    n_train = int(0.85 * n)
    idx = torch.randperm(n)
    tr_idx, va_idx = idx[:n_train], idx[n_train:]
    x_train, y_train = x[tr_idx], y[tr_idx]
    x_val, y_val = x[va_idx], y[va_idx]

    model = LogisticRegressionModel(in_features=2)
    train_losses, val_losses = train_model(model, x_train, y_train, x_val, y_val, max_epochs=2000, lr=0.5)
    print(f"训练结束，最终验证 BCE: {val_losses[-1]:.6f}，总轮数: {len(val_losses)}")

    plot_loss_curve(train_losses, val_losses)
    model.eval()
    plot_decision_boundary(model, x, y)


if __name__ == "__main__":
    main()

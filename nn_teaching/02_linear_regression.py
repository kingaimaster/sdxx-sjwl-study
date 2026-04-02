# -*- coding: utf-8 -*-
"""
脚本：使用 PyTorch 实现一元线性回归。

演示 torch.nn.Linear、MSELoss、SGD 优化器的完整训练循环；
补充 Dropout（输入端随机失活，教学演示用）、训练/验证划分、余弦退火学习率与早停。
使用模拟数据并绘制拟合直线与损失曲线。
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
    配置 matplotlib 的中文字体与负号显示，避免中文乱码与负号显示为方块。

    Returns:
        None: 仅修改全局绘图参数，无返回值。
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


def make_synthetic_linear_data(
    n: int = 300,
    w_true: float = 2.5,
    b_true: float = 1.0,
    noise_std: float = 0.8,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成一元线性关系 y = w * x + b + noise 的模拟数据。

    Args:
        n: 样本数量。
        w_true: 真实斜率。
        b_true: 真实截距（偏置）。
        noise_std: 高斯噪声标准差。
        seed: 随机种子。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - x: 形状为 (n, 1) 的特征张量。
            - y: 形状为 (n, 1) 的标签张量。
    """
    torch.manual_seed(seed)
    x = torch.linspace(-3, 3, n).unsqueeze(1)
    noise = torch.randn(n, 1) * noise_std
    y = w_true * x + b_true + noise
    return x, y


class LinearRegressionNet(nn.Module):
    """
    一元线性回归：输入一维特征经可选 Dropout 后接单个 Linear。

    说明：经典线性回归通常不使用 Dropout；此处仅在训练阶段演示正则化接口，
    eval 时 Dropout 自动关闭，预测与常规线性回归一致。
    """

    def __init__(self, dropout_p: float = 0.05) -> None:
        """
        Args:
            dropout_p: 特征维上的 Dropout 概率；设为 0 即关闭。
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_p)
        self.linear = nn.Linear(1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, 1)。

        Returns:
            torch.Tensor: (N, 1) 预测值。
        """
        return self.linear(self.dropout(x))


def train_with_val_and_schedule(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 600,
    lr: float = 0.08,
) -> tuple[nn.Module, list[float], list[float]]:
    """
    在训练子集上优化，每轮计算验证 MSE；使用 CosineAnnealingLR 与早停。

    Args:
        x_train, y_train: 训练特征与标签。
        x_val, y_val: 验证特征与标签。
        max_epochs: 最大轮数。
        lr: SGD 初始学习率。

    Returns:
        tuple: (模型, 训练 MSE 历史, 验证 MSE 历史)。
    """
    model = LinearRegressionNet(dropout_p=0.05)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    early = EarlyStopping(patience=80, min_delta=1e-5, mode="min")

    train_hist: list[float] = []
    val_hist: list[float] = []

    for _epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        pred = model(x_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            train_mse = criterion(model(x_train), y_train).item()
            val_mse = criterion(model(x_val), y_val).item()
        train_hist.append(train_mse)
        val_hist.append(val_mse)

        if early.step(val_mse):
            break

    return model, train_hist, val_hist


def plot_fit_and_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    model: nn.Module,
    train_losses: list[float],
    val_losses: list[float],
) -> None:
    """
    绘制原始散点、拟合直线以及训练/验证 MSE 曲线。

    Args:
        x: 全体特征 (N, 1)。
        y: 全体标签 (N, 1)。
        model: 训练后的模型（eval）。
        train_losses: 每轮训练集 MSE。
        val_losses: 每轮验证集 MSE。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    model.eval()
    x_np = x.detach().cpu().numpy().ravel()
    y_np = y.detach().cpu().numpy().ravel()
    xx = torch.linspace(float(x_np.min()), float(x_np.max()), steps=100).unsqueeze(1)
    with torch.no_grad():
        yy = model(xx).detach().cpu().numpy().ravel()

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(x_np, y_np, s=18, alpha=0.65, label="样本点")
    axes[0].plot(xx.numpy().ravel(), yy, color="#C44E52", linewidth=2, label="拟合直线")
    axes[0].set_title("一元线性回归：样本与拟合结果（eval 时 Dropout 关闭）")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].legend()
    axes[0].grid(True, alpha=0.25)

    ep = range(1, len(train_losses) + 1)
    axes[1].plot(ep, train_losses, color="#4C72B0", label="训练集 MSE")
    axes[1].plot(ep, val_losses, color="#C44E52", label="验证集 MSE")
    axes[1].set_title("MSE 曲线（余弦退火 + 早停）")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    plt.show()


def main() -> None:
    """
    脚本入口：划分训练/验证、训练模型、打印参数并绘图。

    Returns:
        None: 无返回值。
    """
    torch.manual_seed(0)
    configure_chinese_matplotlib()

    x, y = make_synthetic_linear_data(n=300)
    n_train = int(0.85 * len(x))
    x_train, x_val = x[:n_train], x[n_train:]
    y_train, y_val = y[:n_train], y[n_train:]

    model, train_hist, val_hist = train_with_val_and_schedule(x_train, y_train, x_val, y_val)

    w_learned = float(model.linear.weight.detach().cpu().item())
    b_learned = float(model.linear.bias.detach().cpu().item())
    print("=== 线性层参数（可与真实 w=2.5, b=1.0 对比） ===")
    print(f"斜率 weight: {w_learned:.4f}")
    print(f"截距 bias:   {b_learned:.4f}")
    print(f"最终验证 MSE: {val_hist[-1]:.6f}，总训练轮数: {len(train_hist)}")

    plot_fit_and_loss(x, y, model, train_hist, val_hist)


if __name__ == "__main__":
    main()

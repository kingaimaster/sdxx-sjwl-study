# -*- coding: utf-8 -*-
"""
脚本：多层感知机（MLP）解决多分类问题。

使用至少一个隐藏层、ReLU、Dropout 与 CrossEntropyLoss；
补充训练/验证划分、余弦退火与早停；注释中说明张量维度变换。

"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
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
    配置 matplotlib 中文与负号显示。

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


def make_multiclass_data(
    n_samples: int = 1200,
    n_features: int = 20,
    n_classes: int = 4,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    使用随机数构造多分类模拟数据集（特征与标签均为合成）。

    Args:
        n_samples: 样本数量。
        n_features: 每个样本的特征维度。
        n_classes: 类别数。
        seed: 随机种子。

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - X: 形状 (N, n_features) 的 float32 张量。
            - y: 形状 (N,) 的 int64 类别索引，取值 0..C-1。
    """
    g = torch.Generator().manual_seed(seed)
    # 为每类生成不同中心，使数据非平凡可分
    centers = torch.randn(n_classes, n_features, generator=g) * 2.0
    y = torch.randint(low=0, high=n_classes, size=(n_samples,), generator=g)
    x = centers[y] + torch.randn(n_samples, n_features, generator=g) * 0.85
    return x, y


class MultiLayerPerceptron(nn.Module):
    """
    全连接神经网络（MLP）用于多分类。

    网络结构（以默认参数为例）：
    - 输入维度 in_dim；
    - 隐藏层：Linear(in_dim -> hidden) + ReLU；
    - 输出层：Linear(hidden -> num_classes)，**不带 Softmax**。

    重要说明（维度变换，设批量大小为 N）：
    1) 输入 x 的形状为 (N, in_dim)。第一层线性层将每个样本从 in_dim 维映射到 hidden 维，
       得到 (N, hidden)。ReLU 逐元素作用，不改变形状。
    2) 第二层线性层将 (N, hidden) 映射为 (N, num_classes)，得到每个样本对各类的“打分”，
       称为 logits。
    3) nn.CrossEntropyLoss 期望：
       - 模型输出：形状 (N, C) 的 logits（未归一化的类分数）；
       - 目标标签：形状 (N,) 的整型类别索引，取值 0 到 C-1；
       内部会对 logits 做 log-softmax 并与 NLL 结合，等价于多类逻辑回归的负对数似然。

    因此最后一层不要使用 Softmax；若手动使用 Softmax，则需改用 NLLLoss 等组合，
    初学者直接使用 CrossEntropyLoss + logits 是最常见写法。
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        """
        构建两层线性变换与中间 ReLU。

        Args:
            in_dim: 输入特征维度。
            hidden: 隐藏层神经元个数。
            num_classes: 输出类别数。

        Returns:
            None: 构造函数无返回值。
        """
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(hidden, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：fc1 -> ReLU -> Dropout -> fc2，返回 logits。

        Args:
            x: 形状 (N, in_dim) 的输入张量。

        Returns:
            torch.Tensor: 形状 (N, num_classes) 的 logits。
        """
        # h: (N, in_dim) -> (N, hidden)
        h = self.fc1(x)
        h = self.dropout(self.relu(h))
        # logits: (N, hidden) -> (N, num_classes)
        logits = self.fc2(h)
        return logits


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    根据 logits 计算分类准确率。

    Args:
        logits: 形状 (N, C)。
        y: 形状 (N,) 的真实类别。

    Returns:
        float: 准确率，范围 [0, 1]。
    """
    pred = torch.argmax(logits, dim=1)
    return float((pred == y).float().mean().item())


def train_mlp(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 500,
    lr: float = 0.05,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    全批量训练 MLP，记录训练/验证损失与准确率；CosineAnnealingLR 与早停。

    Args:
        model: MLP 模型。
        x_train, y_train: 训练集。
        x_val, y_val: 验证集。
        max_epochs: 最大轮数。
        lr: 初始学习率。

    Returns:
        tuple: (train_loss, val_loss, train_acc, val_acc) 各为逐 epoch 列表。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    early = EarlyStopping(patience=60, min_delta=1e-4, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tl = criterion(model(x_train), y_train).item()
            vl = criterion(model(x_val), y_val).item()
            ta = accuracy_from_logits(model(x_train), y_train)
            va = accuracy_from_logits(model(x_val), y_val)
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)

        if early.step(vl):
            break

    return train_losses, val_losses, train_accs, val_accs


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    """
    绘制训练集与验证集上的 Loss 与 Accuracy 曲线。

    Args:
        train_losses, val_losses: 训练/验证交叉熵。
        train_accs, val_accs: 训练/验证准确率。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_losses, color="#4C72B0", label="训练集")
    axes[0].plot(epochs, val_losses, color="#C44E52", label="验证集")
    axes[0].set_title("CrossEntropy 损失（余弦退火 + 早停）")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, color="#4C72B0", label="训练集")
    axes[1].plot(epochs, val_accs, color="#C44E52", label="验证集")
    axes[1].set_title("准确率")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def main() -> None:
    """
    主流程：构造数据、训练、打印最终指标并绘图。

    Returns:
        None: 无返回值。
    """
    torch.manual_seed(0)
    configure_chinese_matplotlib()

    in_dim = 20
    hidden = 64
    num_classes = 4
    x, y = make_multiclass_data(n_samples=1200, n_features=in_dim, n_classes=num_classes)
    n = x.size(0)
    n_train = int(0.85 * n)
    perm = torch.randperm(n)
    tr, va = perm[:n_train], perm[n_train:]
    x_train, y_train = x[tr], y[tr]
    x_val, y_val = x[va], y[va]

    model = MultiLayerPerceptron(in_dim=in_dim, hidden=hidden, num_classes=num_classes)
    tl, vl, ta, va = train_mlp(model, x_train, y_train, x_val, y_val, max_epochs=500, lr=0.05)

    print("=== 多分类 MLP 训练完成 ===")
    print(f"最终验证 CrossEntropy: {vl[-1]:.4f}，验证准确率: {va[-1]:.4f}，总轮数: {len(vl)}")

    plot_training_curves(tl, vl, ta, va)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
脚本：卷积神经网络（CNN）前向与反向传播演示。

使用随机图像张量 [N, 3, 28, 28] 完成前向与反向传播演示，并追加短训练：
全连接前 Dropout、训练/验证划分、余弦退火与早停（仍无真实数据集）。
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


class SimpleCNN(nn.Module):
    """
    基础 CNN：Conv -> BN -> ReLU -> Pool 堆叠，最后接全连接分类头。

    以下假定输入张量形状为 (N, 3, 28, 28)，其中 N 为批量大小（本示例取 N=32）。
    记 Conv2d 的卷积核大小为 k，步幅为 s，填充为 p，输入高宽为 H_in、W_in，
    则单个方向（高度）上的输出尺寸为：
        H_out = floor((H_in + 2p - k) / s + 1)
    宽度方向同理。MaxPool2d 默认 kernel_size=stride=2 时：
        H_out = floor((H_in - k) / s + 1)（无 padding 时）。

    各层维度变化（与本类 __init__ 中参数一致时）：

    1) conv1: Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
       - 输入：(N, 3, 28, 28)
       - 输出通道变为 16；高宽：H_out = (28 + 2*1 - 3) / 1 + 1 = 28
       - 形状：(N, 16, 28, 28)

    2) bn1: BatchNorm2d(16)
       - 不改变形状，仍为 (N, 16, 28, 28)

    3) relu: 逐元素 ReLU，不改变形状

    4) pool1: MaxPool2d(kernel_size=2, stride=2)
       - H_out = (28 - 2) / 2 + 1 = 14（PyTorch 默认向下取整的整数运算）
       - 形状：(N, 16, 14, 14)

    5) conv2: Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
       - H_out = (14 + 2*1 - 3) / 1 + 1 = 14
       - 形状：(N, 32, 14, 14)

    6) bn2 + relu：形状仍为 (N, 32, 14, 14)

    7) pool2: MaxPool2d(2, stride=2)
       - 形状：(N, 32, 7, 7)

    8) conv3: Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
       - H_out = (7 + 2*1 - 3) / 1 + 1 = 7
       - 形状：(N, 64, 7, 7)

    9) bn3 + relu：形状仍为 (N, 64, 7, 7)

    10) pool3: MaxPool2d(2, stride=2)
        - H_out = (7 - 2) / 2 + 1 = 3
        - 形状：(N, 64, 3, 3)

    11) flatten: 将 (N, 64, 3, 3) 展平为 (N, 64*3*3) = (N, 576)

    12) flatten -> Dropout -> fc: Linear(576, num_classes)
        - 输出 logits：(N, num_classes)
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.3) -> None:
        """
        构建卷积块与全连接分类头。

        Args:
            num_classes: 输出类别数（例如 10 类）。
            dropout_p: 全连接层前的 Dropout 概率。

        Returns:
            None: 构造函数无返回值。
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(64 * 3 * 3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：依次经过卷积、批归一化、激活与池化，最后全连接输出 logits。

        Args:
            x: 输入图像张量，形状 (N, 3, 28, 28)。

        Returns:
            torch.Tensor: 分类 logits，形状 (N, num_classes)。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x)

        x = self.dropout(self.flatten(x))
        x = self.fc(x)
        return x


def plot_loss_bar(loss_value: float) -> None:
    """
    将一次反向传播得到的标量损失以条形图形式展示（可视化）。

    Args:
        loss_value: 损失数值。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["本次迭代的交叉熵损失"], [loss_value], color="#4C72B0")
    ax.set_ylabel("Loss")
    ax.set_title("CNN 单次训练步：损失值示意")
    plt.tight_layout()
    plt.show()


def run_demo() -> None:
    """
    构造随机批次数据，执行前向、交叉熵损失、反向传播一步，并打印张量形状。

    Returns:
        None: 无返回值。
    """
    torch.manual_seed(0)
    configure_chinese_matplotlib()

    batch_size = 32
    num_classes = 10
    images = torch.randn(batch_size, 3, 28, 28)
    targets = torch.randint(low=0, high=num_classes, size=(batch_size,))

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    print("=== 输入与标签形状 ===")
    print(f"images:  {tuple(images.shape)}  # (N, C, H, W)")
    print(f"targets: {tuple(targets.shape)}  # (N,) 类别索引")

    optimizer.zero_grad()
    logits = model(images)
    loss = criterion(logits, targets)
    loss.backward()
    optimizer.step()

    print("\n=== 前向输出与损失 ===")
    print(f"logits: {tuple(logits.shape)}  # (N, num_classes)")
    print(f"loss:   {float(loss.detach().cpu().item()):.6f}")

    print("\n=== 若干参数的梯度是否存在（示意） ===")
    print(f"conv1.weight.grad is not None: {model.conv1.weight.grad is not None}")
    print(f"fc.weight.grad is not None:     {model.fc.weight.grad is not None}")

    plot_loss_bar(float(loss.detach().cpu().item()))

    run_short_training_demo()


def run_short_training_demo() -> None:
    """
    在随机生成的训练/验证集上短训练若干轮，演示 Dropout、余弦退火与早停。

    Returns:
        None: 无返回值。
    """
    configure_chinese_matplotlib()
    torch.manual_seed(1)
    n = 512
    num_classes = 10
    images = torch.randn(n, 3, 28, 28)
    targets = torch.randint(0, num_classes, (n,))
    n_train = int(0.85 * n)
    perm = torch.randperm(n)
    tr, va = perm[:n_train], perm[n_train:]
    img_tr, tar_tr = images[tr], targets[tr]
    img_va, tar_va = images[va], targets[va]

    model = SimpleCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    max_epochs = 80
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    early = EarlyStopping(patience=15, min_delta=1e-3, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []

    print("\n=== 短训练（随机数据，验证集仅用于早停与曲线） ===")
    for ep in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(img_tr)
        loss = criterion(logits, tar_tr)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tl = criterion(model(img_tr), tar_tr).item()
            vl = criterion(model(img_va), tar_va).item()
        train_losses.append(tl)
        val_losses.append(vl)
        if early.step(vl):
            print(f"早停于第 {ep} 轮")
            break

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="训练集")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="验证集")
    ax.set_title("CNN 随机数据短训练：交叉熵（余弦退火 + 早停）")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    脚本入口。

    Returns:
        None: 无返回值。
    """
    run_demo()


if __name__ == "__main__":
    main()

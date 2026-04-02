# -*- coding: utf-8 -*-
"""
脚本：使用全连接神经网络（MLP）对 MNIST 手写数字数据集进行完整流程实践。

流程包括：数据加载、模型构建（含 Kaiming / Xavier、Dropout）、
余弦退火学习率、验证集早停、训练、保存检查点、重新加载后进行推理与测试集评估，
以及中文可视化（损失曲线、准确率曲线、混淆矩阵等）。

依赖：torch、torchvision、matplotlib、seaborn、scikit-learn。
首次运行会自动下载 MNIST 数据到本地 data 目录。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

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


def init_module_weights(module: nn.Module) -> None:
    """
    对全连接网络中各层采用与激活函数匹配的初始化策略。

    说明：
    - 隐藏层 Linear 后接 ReLU：使用 Kaiming He 正态初始化（fan_in，relu），
      使 ReLU 层输入方差在合理范围，减轻梯度消失或爆炸。
    - 最后一层输出 logits（其后无 ReLU，接 CrossEntropyLoss）：使用 Xavier 均匀初始化，
      适用于线性输出层的一种常见做法。

    Args:
        module: 任意子模块；仅对 nn.Linear 生效。

    Returns:
        None: 原地修改参数。
    """
    if isinstance(module, nn.Linear):
        if module.out_features != 10:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class MnistMLP(nn.Module):
    """
    MNIST 全连接分类器：将 28x28 图像展平为 784 维向量，经两层隐藏 ReLU 与 Dropout 映射到 10 类 logits。

    展平后维度：(N, 1, 28, 28) -> (N, 784)。训练时 Dropout 随机失活部分神经元，eval 时自动关闭。
    """

    def __init__(
        self,
        hidden1: int = 256,
        hidden2: int = 128,
        num_classes: int = 10,
        dropout_p: float = 0.25,
    ) -> None:
        """
        构建 MLP 并在构建完成后对整个模块树应用 init_module_weights。

        Args:
            hidden1: 第一隐藏层神经元个数。
            hidden2: 第二隐藏层神经元个数。
            num_classes: 类别数，MNIST 为 10。
            dropout_p: 两个隐藏层后 Dropout 的概率（训练阶段生效）。

        Returns:
            None: 构造函数无返回值。
        """
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.apply(init_module_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播，返回各类别 logits。

        Args:
            x: 输入图像张量，形状 (N, 1, 28, 28)。

        Returns:
            torch.Tensor: 形状 (N, num_classes) 的 logits。
        """
        x = self.flatten(x)
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    在训练集上执行一个 epoch 的训练。

    Args:
        model: 神经网络模型。
        loader: 训练数据 DataLoader。
        criterion: 损失函数。
        optimizer: 优化器。
        device: cpu 或 cuda。

    Returns:
        tuple[float, float]: (平均损失, 准确率)。
    """
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    n = len(all_labels)
    return total_loss / n, accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    在指定数据加载器上评估模型，返回损失、准确率、真实标签与预测标签。

    Args:
        model: 模型。
        loader: 数据加载器。
        criterion: 损失函数。
        device: 设备。

    Returns:
        tuple: (平均损失, 准确率, y_true 数组, y_pred 数组)。
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    n = len(all_labels)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return total_loss / n, accuracy_score(all_labels, all_preds), y_true, y_pred


def save_checkpoint(
    model: nn.Module,
    path: Path,
    epoch: int,
    best_acc: float,
    extra: dict | None = None,
) -> None:
    """
    保存模型状态与元数据，便于后续加载推理。

    Args:
        model: 要保存的模型。
        path: 保存路径（.pt 文件）。
        epoch: 训练轮次。
        best_acc: 当前最佳验证准确率。
        extra: 可选附加字典。

    Returns:
        None: 无返回值。
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_acc,
        "model_class": "MnistMLP",
        "num_classes": 10,
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model_for_inference(path: Path, device: torch.device) -> MnistMLP:
    """
    从检查点加载权重并置于 eval 模式。

    Args:
        path: 检查点文件路径。
        device: 目标设备。

    Returns:
        MnistMLP: 已加载权重的模型。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MnistMLP()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def plot_training_curves(train_losses: list[float], val_losses: list[float], train_accs: list[float], val_accs: list[float]) -> None:
    """
    绘制训练与验证的损失、准确率曲线（中文标题与坐标轴）。

    Args:
        train_losses: 每轮训练平均损失。
        val_losses: 每轮验证平均损失。
        train_accs: 每轮训练准确率。
        val_accs: 每轮验证准确率。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_losses, label="训练集", color="#4C72B0")
    axes[0].plot(epochs, val_losses, label="验证集", color="#C44E52")
    axes[0].set_xlabel("训练轮次 Epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("MNIST 全连接网络：损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="训练集", color="#4C72B0")
    axes[1].plot(epochs, val_accs, label="验证集", color="#C44E52")
    axes[1].set_xlabel("训练轮次 Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("MNIST 全连接网络：准确率曲线")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix_heatmap(cm: np.ndarray, class_labels: list[str], title: str) -> None:
    """
    使用 seaborn 绘制混淆矩阵热力图（中文标题）。

    Args:
        cm: 混淆矩阵，形状 (C, C)。
        class_labels: 各类别显示名称。
        title: 图标题。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    主流程：准备数据、训练、保存、加载、测试集评估与可视化。

    Returns:
        None: 无返回值。
    """
    configure_chinese_matplotlib()
    torch.manual_seed(42)
    np.random.seed(42)

    data_root = Path(__file__).resolve().parent / "data"
    ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
    ckpt_path = ckpt_dir / "mnist_fcnn_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_full = datasets.MNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(data_root), train=False, download=True, transform=transform)

    train_size = int(0.9 * len(train_full))
    val_size = len(train_full) - train_size
    train_set, val_set = torch.utils.data.random_split(
        train_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    batch_size = 128
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model = MnistMLP().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 30
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=6, min_delta=1e-4, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    best_val = 0.0
    actual_epochs = 0

    print("开始训练 MNIST 全连接网络（余弦退火学习率 + 验证集早停）...")
    for epoch in range(1, num_epochs + 1):
        tl, ta = train_one_epoch(model, train_loader, criterion, optimizer, device)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        actual_epochs = epoch
        if va > best_val:
            best_val = va
            save_checkpoint(model, ckpt_path, epoch, best_val)
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{num_epochs}  lr={lr_now:.2e}  "
            f"训练损失={tl:.4f}  验证损失={vl:.4f}  训练准确率={ta:.4f}  验证准确率={va:.4f}"
        )
        if early_stopping.step(vl):
            print(f"早停触发：验证损失连续 {early_stopping.patience} 轮未明显下降，在第 {epoch} 轮结束训练。")
            break

    print(f"\n最佳验证准确率: {best_val:.4f}，检查点已保存至: {ckpt_path}（实际训练 {actual_epochs} 轮）")

    print("\n加载检查点并在测试集上评估（推理阶段）...")
    model_loaded = load_model_for_inference(ckpt_path, device)
    test_loss, test_acc, y_true, y_pred = evaluate(model_loaded, test_loader, criterion, device)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    digit_names = [f"数字{i}" for i in range(10)]
    print("\n分类报告（精确率、召回率、F1）:\n")
    print(classification_report(y_true, y_pred, target_names=digit_names, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix_heatmap(cm, digit_names, "MNIST 测试集混淆矩阵（全连接网络）")

    print("\n要点：")
    print("- 隐藏层使用 Kaiming 初始化配合 ReLU；输出层使用 Xavier 初始化；Dropout 缓解过拟合。")
    print("- CosineAnnealingLR 按 epoch 衰减学习率；EarlyStopping 依据验证损失提前结束。")
    print("- 保存 state_dict 便于部署；加载时需先构造相同结构的模型再 load_state_dict。")


if __name__ == "__main__":
    main()

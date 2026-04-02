# -*- coding: utf-8 -*-
"""
脚本：使用卷积神经网络（CNN）对 Fashion-MNIST 服装图像数据集进行完整流程实践。

包含：数据加载、含 BatchNorm 与 Dropout 的 CNN、卷积与全连接层的合适权重初始化（Kaiming / Xavier）、
余弦退火学习率、验证集早停、训练、保存检查点、导出 ONNX、加载推理、测试集评估，
以及中文可视化（损失、准确率、混淆矩阵、样例预测网格）。

首次运行会下载 Fashion-MNIST 至 data 目录。
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

from training_utils import EarlyStopping, export_model_onnx

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
    配置 matplotlib 中文显示与负号渲染。

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


def init_cnn_weights(module: nn.Module) -> None:
    """
    对 CNN 子模块进行权重初始化。

    说明：
    - Conv2d：使用 Kaiming He 正态初始化（mode=fan_out, nonlinearity=relu），
      与 ReLU 激活配套，有利于卷积层训练稳定。
    - BatchNorm2d：权重置 1、偏置置 0（与 PyTorch 默认一致，显式写出便于）。
    - Linear：若输出维度不为类别数，则视为隐藏全连接层，使用 Kaiming；
      最后一层输出 10 类 logits，使用 Xavier 均匀初始化。

    Args:
        module: 网络子模块。

    Returns:
        None: 原地修改。
    """
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        if module.out_features != 10:
            nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
        else:
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class FashionCNN(nn.Module):
    """
    面向 28x28 灰度图的典型小型 CNN。

    结构：Conv-BN-ReLU-Pool 重复两次后展平，全连接前使用 Dropout，再经两层全连接输出 10 类。
    特征图尺寸：28 -> 卷积保持 -> 池化 14 -> 卷积保持 -> 池化 7。
    """

    def __init__(self, num_classes: int = 10, dropout_p: float = 0.3) -> None:
        """
        构建各层并在末尾 apply(init_cnn_weights)。

        Args:
            num_classes: 类别数，Fashion-MNIST 为 10。
            dropout_p: 展平后第一个全连接层前的 Dropout 概率。

        Returns:
            None: 无返回值。
        """
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, num_classes)
        self.apply(init_cnn_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播得到 logits。

        Args:
            x: (N, 1, 28, 28)。

        Returns:
            torch.Tensor: (N, num_classes)。
        """
        x = self.features(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """
    单轮训练，返回平均损失与准确率。

    Args:
        model: 模型。
        loader: 训练 DataLoader。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 设备。

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
    评估模型并返回损失、准确率与标签数组。

    Args:
        model: 模型。
        loader: DataLoader。
        criterion: 损失。
        device: 设备。

    Returns:
        tuple: (平均损失, 准确率, y_true, y_pred)。
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


def save_checkpoint(model: nn.Module, path: Path, epoch: int, best_acc: float) -> None:
    """
    保存训练检查点。

    Args:
        model: 模型。
        path: 保存路径。
        epoch: 轮次。
        best_acc: 最佳验证准确率。

    Returns:
        None: 无返回值。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "best_val_acc": best_acc,
            "num_classes": 10,
        },
        path,
    )


def load_model_for_inference(path: Path, device: torch.device) -> FashionCNN:
    """
    加载权重构建推理用模型。

    Args:
        path: .pt 路径。
        device: 设备。

    Returns:
        FashionCNN: eval 模式模型。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = FashionCNN()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    """
    绘制中文标题的训练/验证曲线。

    Args:
        train_losses: 训练损失序列。
        val_losses: 验证损失序列。
        train_accs: 训练准确率序列。
        val_accs: 验证准确率序列。

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
    axes[0].set_title("Fashion-MNIST CNN：损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="训练集", color="#4C72B0")
    axes[1].plot(epochs, val_accs, label="验证集", color="#C44E52")
    axes[1].set_xlabel("训练轮次 Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("Fashion-MNIST CNN：准确率曲线")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix(cm: np.ndarray, names_cn: list[str], title: str) -> None:
    """
    混淆矩阵热力图（中文类别名）。

    Args:
        cm: 混淆矩阵。
        names_cn: 中文类别名称列表。
        title: 图标题。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=names_cn, yticklabels=names_cn)
    plt.xlabel("预测类别")
    plt.ylabel("真实类别")
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_sample_predictions(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    class_names: list[str],
    device: torch.device,
    n_show: int = 16,
) -> None:
    """
    随机抽取若干测试样本展示图像与预测结果（中文标题与图例）。

    Args:
        model: 已加载的模型。
        dataset: 测试集 Dataset。
        class_names: 类别中文名。
        device: 设备。
        n_show: 展示图片数量（建议为完全平方数）。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    model.eval()
    indices = torch.randperm(len(dataset))[:n_show]
    cols = int(np.sqrt(n_show))
    rows = n_show // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.flatten()
    for ax, idx in zip(axes, indices):
        img, label = dataset[int(idx)]
        with torch.no_grad():
            logits = model(img.unsqueeze(0).to(device))
            pred = int(logits.argmax(dim=1).item())
        img_np = img.squeeze().numpy()
        ax.imshow(img_np, cmap="gray")
        color = "green" if pred == label else "red"
        ax.set_title(f"真实:{class_names[label]}\n预测:{class_names[pred]}", fontsize=8, color=color)
        ax.axis("off")
    fig.suptitle("Fashion-MNIST 测试样本：真实标签与模型预测对比（绿=正确，红=错误）")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    主入口：数据、训练、保存、加载评估、中文可视化。

    Returns:
        None: 无返回值。
    """
    configure_chinese_matplotlib()
    torch.manual_seed(42)
    np.random.seed(42)

    data_root = Path(__file__).resolve().parent / "data"
    ckpt_path = Path(__file__).resolve().parent / "checkpoints" / "fashionmnist_cnn_best.pt"

    class_names_cn = [
        "T恤",
        "裤子",
        "套衫",
        "连衣裙",
        "外套",
        "凉鞋",
        "衬衫",
        "运动鞋",
        "包",
        "短靴",
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,)),
        ]
    )

    train_full = datasets.FashionMNIST(root=str(data_root), train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=str(data_root), train=False, download=True, transform=transform)

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

    model = FashionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 25
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=6, min_delta=1e-4, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    best_val = 0.0
    actual_epochs = 0

    print("开始训练 Fashion-MNIST CNN（余弦退火 + 早停）...")
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

    print(f"\n最佳验证准确率: {best_val:.4f}，模型已保存: {ckpt_path}（实际训练 {actual_epochs} 轮）")

    print("\n加载检查点并在测试集上评估...")
    model_loaded = load_model_for_inference(ckpt_path, device)
    test_loss, test_acc, y_true, y_pred = evaluate(model_loaded, test_loader, criterion, device)
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")
    print("\n分类报告:\n")
    print(classification_report(y_true, y_pred, target_names=class_names_cn, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix(cm, class_names_cn, "Fashion-MNIST 测试集混淆矩阵（CNN）")
    plot_sample_predictions(model_loaded, test_set, class_names_cn, device)

    onnx_path = Path(__file__).resolve().parent / "checkpoints" / "fashionmnist_cnn.onnx"
    print(f"\n正在导出 ONNX 至: {onnx_path}")
    try:
        export_model_onnx(
            model_loaded,
            onnx_path,
            torch.randn(1, 1, 28, 28, device=device),
            input_names=("input",),
            output_names=("logits",),
            dynamic_batch=True,
        )
        print("ONNX 导出成功，可用 onnxruntime 等引擎加载推理。")
    except Exception as exc:
        print(f"ONNX 导出失败（可检查 torch/onnx 版本）: {exc}")

    print("\n权重初始化小结：卷积与隐藏全连接层使用 Kaiming（配合 ReLU）；输出层使用 Xavier。")


if __name__ == "__main__":
    main()

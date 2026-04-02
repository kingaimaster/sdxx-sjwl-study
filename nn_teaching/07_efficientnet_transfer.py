# -*- coding: utf-8 -*-
"""
脚本：加载 torchvision 预训练 EfficientNet-B0，冻结骨干网络并替换分类头，
仅对分类头参数进行优化；补充训练/验证划分、余弦退火与早停（分类头内已有 Dropout）。

使用随机张量模拟 224x224 图像，无需下载额外数据集即可运行。
首次运行若选择加载 ImageNet 预训练权重，需联网下载权重文件。
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

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


def build_efficientnet_for_transfer(num_classes: int = 10, pretrained: bool = True) -> nn.Module:
    """
    构建 EfficientNet-B0 并替换最后的分类层以适应自定义类别数。

    EfficientNet 的 features 部分作为特征提取器；classifier 通常为
    Sequential(Dropout, Linear)。将最后一层 Linear 的 out_features 改为 num_classes。

    冻结策略：将 model.features 中所有参数的 requires_grad 设为 False，
    仅训练 model.classifier 中的参数（本示例中即为新的全连接层等）。

    Args:
        num_classes: 目标任务的类别数，例如 10。
        pretrained: 是否加载 ImageNet 预训练权重；若 False 则随机初始化主干。

    Returns:
        nn.Module: 已替换分类头并冻结骨干的 EfficientNet 模型。
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    dropout_p = model.classifier[0].p
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def train_classifier_only(
    model: nn.Module,
    images_train: torch.Tensor,
    labels_train: torch.Tensor,
    images_val: torch.Tensor,
    labels_val: torch.Tensor,
    max_epochs: int = 30,
    lr: float = 0.05,
) -> tuple[list[float], list[float]]:
    """
    仅针对分类头参数优化；记录训练/验证损失，使用 CosineAnnealingLR 与验证集早停。

    Args:
        model: EfficientNet 模型。
        images_train, labels_train: 训练批张量。
        images_val, labels_val: 验证批张量。
        max_epochs: 最大轮数。
        lr: 初始学习率。

    Returns:
        tuple: (训练损失序列, 验证损失序列)。
    """
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    early = EarlyStopping(patience=8, min_delta=1e-3, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []

    for _ in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(images_train)
        loss = criterion(logits, labels_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            tl = criterion(model(images_train), labels_train).item()
            vl = criterion(model(images_val), labels_val).item()
        train_losses.append(tl)
        val_losses.append(vl)
        if early.step(vl):
            break

    return train_losses, val_losses


def plot_training_loss(train_losses: list[float], val_losses: list[float]) -> None:
    """
    绘制训练与验证损失曲线。

    Args:
        train_losses: 训练集损失。
        val_losses: 验证集损失。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    plt.figure(figsize=(7, 4))
    ep = range(1, len(train_losses) + 1)
    plt.plot(ep, train_losses, marker="o", color="#4C72B0", label="训练集")
    plt.plot(ep, val_losses, marker="s", color="#C44E52", label="验证集")
    plt.title("迁移学习：仅优化分类头（余弦退火 + 早停，随机数据演示）")
    plt.xlabel("Epoch")
    plt.ylabel("CrossEntropy Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def verify_backbone_frozen(model: nn.Module) -> None:
    """
    打印骨干与分类头的参数是否参与梯度计算（requires_grad 状态）。

    Args:
        model: EfficientNet 模型。

    Returns:
        None: 仅打印信息。
    """
    feat_grad = [p.requires_grad for p in model.features.parameters()]
    cls_grad = [p.requires_grad for p in model.classifier.parameters()]
    print("=== 参数冻结检查（骨干应全为 False，分类头应为 True） ===")
    print(f"features 中 requires_grad 为 True 的数量: {sum(feat_grad)} / {len(feat_grad)}")
    print(f"classifier 中 requires_grad 为 True 的数量: {sum(cls_grad)} / {len(cls_grad)}")


def main() -> None:
    """
    主流程：构建模型、构造随机批次、训练分类头、绘图。

    Returns:
        None: 无返回值。
    """
    torch.manual_seed(0)
    configure_chinese_matplotlib()

    num_classes = 10
    n_samples = 48

    # 若环境无法下载预训练权重，可将 pretrained 改为 False，仅演示结构与训练流程
    use_pretrained = True
    try:
        model = build_efficientnet_for_transfer(num_classes=num_classes, pretrained=use_pretrained)
    except Exception as exc:
        print(f"加载预训练权重失败（可能无网络或版本不匹配），将使用随机初始化主干: {exc}")
        model = build_efficientnet_for_transfer(num_classes=num_classes, pretrained=False)

    verify_backbone_frozen(model)

    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(low=0, high=num_classes, size=(n_samples,))
    n_tr = int(0.85 * n_samples)
    perm = torch.randperm(n_samples)
    tr, va = perm[:n_tr], perm[n_tr:]
    img_tr, lab_tr = images[tr], labels[tr]
    img_va, lab_va = images[va], labels[va]

    train_losses, val_losses = train_classifier_only(
        model, img_tr, lab_tr, img_va, lab_va, max_epochs=30, lr=0.05
    )
    print("\n=== 训练/验证损失（随机标签，仅演示计算图与调度；总轮数 %d） ===" % len(train_losses))
    for i, (a, b) in enumerate(zip(train_losses, val_losses), start=1):
        print(f"Epoch {i}: train={a:.6f}  val={b:.6f}")

    plot_training_loss(train_losses, val_losses)


if __name__ == "__main__":
    main()

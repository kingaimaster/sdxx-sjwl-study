# -*- coding: utf-8 -*-
"""
脚本：使用 torchvision 预训练 EfficientNet-B0 在 CIFAR-100 数据集上进行迁移学习完整流程。

包含：将 32x32 图像缩放至 224x224、替换分类头为 100 类、对**新分类头**采用 Xavier 初始化、
骨干与头部差异化学习率微调、余弦退火学习率调度、验证集早停、训练、保存检查点、加载推理、测试集评估，以及中文可视化。
EfficientNet 分类头中已含 Dropout（与 torchvision 结构一致）。

说明：CIFAR-100 与 ImageNet 分布不同，需足够轮次与数据增强才可能达到较高精度；
本脚本以为主，轮次适中，可自行调大 `num_epochs` 与 batch。

首次运行需下载 CIFAR-100 与（可选）预训练权重，请保持网络畅通。

终端进度：训练与验证/测试循环使用 tqdm 显示进度条（`pip install tqdm`）。
若未安装 tqdm，将退化为每隔若干 batch 打印文本进度。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False

    def tqdm(iterable, **_kwargs):  # type: ignore[misc]
        """未安装 tqdm 时的占位，直接返回可迭代对象。"""
        return iterable
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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


def build_efficientnet_cifar100(num_classes: int = 100, pretrained: bool = True) -> nn.Module:
    """
    构建 EfficientNet-B0，加载 ImageNet 预训练权重，并将分类头替换为 num_classes 维输出。

    权重初始化策略（要点）：
    - **骨干网络 features**：沿用预训练权重，不在此函数中改写。
    - **新替换的 Linear 分类层**：采用 **Xavier 均匀初始化**（Glorot init），
      因该层直接输出 logits，其后无 ReLU，Xavier 常用于全连接输出层的一种合理选择；
      偏置置零。

    Args:
        num_classes: CIFAR-100 为 100。
        pretrained: 是否加载 ImageNet 预训练；若 False 则整网随机初始化（仅用于无网络环境）。

    Returns:
        nn.Module: EfficientNet 模型。
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    # progress=True 时，下载预训练权重文件会在终端显示进度条（依赖 tqdm）
    model = efficientnet_b0(weights=weights, progress=True)
    in_features = model.classifier[1].in_features
    dropout_p = model.classifier[0].p
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_p, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    nn.init.xavier_uniform_(model.classifier[1].weight)
    nn.init.zeros_(model.classifier[1].bias)
    return model


def get_optimizer(model: nn.Module, lr_backbone: float, lr_head: float) -> optim.Optimizer:
    """
    构造参数组：骨干网络与分类头使用不同学习率，便于稳定微调预训练特征。

    Args:
        model: EfficientNet 模型。
        lr_backbone: 特征提取部分学习率（宜较小）。
        lr_head: 分类头学习率（可略大）。

    Returns:
        optim.Optimizer: Adam 优化器。
    """
    backbone_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("classifier"):
            head_params.append(p)
        else:
            backbone_params.append(p)
    return optim.Adam(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ],
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch_tag: str = "",
    log_interval: int = 20,
) -> tuple[float, float]:
    """
    训练一个 epoch。

    优先使用 tqdm 包裹 DataLoader，在终端显示实时进度条与当前 batch 损失。
    若未安装 tqdm，则每隔 log_interval 个 batch 打印一行文本进度。

    Args:
        model: 模型。
        loader: 训练 DataLoader。
        criterion: 损失函数。
        optimizer: 优化器。
        device: 设备。
        epoch_tag: 显示在进度条描述中，例如 \"Epoch 1/8 训练\"。
        log_interval: 无 tqdm 时，每隔多少个 batch 打印一次；有 tqdm 时忽略。

    Returns:
        tuple[float, float]: (平均损失, 准确率)。
    """
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)

    if _TQDM_AVAILABLE:
        pbar = tqdm(
            loader,
            desc=epoch_tag or "训练",
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.2,
        )
    else:
        pbar = loader

    for batch_idx, (images, labels) in enumerate(pbar):
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

        if _TQDM_AVAILABLE:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        elif log_interval and n_batches > 0:
            done = batch_idx + 1
            if done == 1 or done % log_interval == 0 or done == n_batches:
                pct = 100.0 * done / n_batches
                print(
                    f"  [{epoch_tag}] 训练中: batch {done}/{n_batches} ({pct:.1f}%), "
                    f"当前 batch 损失={loss.item():.4f}",
                    flush=True,
                )

    n = len(all_labels)
    return total_loss / n, accuracy_score(all_labels, all_preds)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "验证",
    log_interval: int = 20,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    评估模型。

    使用 tqdm 显示评估进度条；无 tqdm 时按 log_interval 打印文本。

    Args:
        model: 模型。
        loader: DataLoader。
        criterion: 损失。
        device: 设备。
        desc: 进度条描述，例如 \"Epoch 1/8 验证集\"。
        log_interval: 无 tqdm 时的打印间隔。

    Returns:
        tuple: (平均损失, 准确率, y_true, y_pred)。
    """
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    n_batches = len(loader)

    if _TQDM_AVAILABLE:
        pbar = tqdm(
            loader,
            desc=desc,
            unit="batch",
            dynamic_ncols=True,
            mininterval=0.2,
        )
    else:
        pbar = loader

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        if _TQDM_AVAILABLE:
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        elif log_interval and n_batches > 0:
            done = batch_idx + 1
            if done == 1 or done % log_interval == 0 or done == n_batches:
                pct = 100.0 * done / n_batches
                print(
                    f"  [{desc}] 评估中: batch {done}/{n_batches} ({pct:.1f}%)",
                    flush=True,
                )

    n = len(all_labels)
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    return total_loss / n, accuracy_score(all_labels, all_preds), y_true, y_pred


def save_checkpoint(model: nn.Module, path: Path, epoch: int, best_acc: float, extra: dict | None = None) -> None:
    """
    保存检查点。

    Args:
        model: 模型。
        path: 路径。
        epoch: 轮次。
        best_acc: 最佳验证准确率。
        extra: 附加字段。

    Returns:
        None: 无返回值。
    """
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "best_val_acc": best_acc,
        "num_classes": 100,
        "arch": "efficientnet_b0",
    }
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_model_for_inference(path: Path, device: torch.device, num_classes: int = 100) -> nn.Module:
    """
    从检查点恢复模型（需先构建同结构网络再加载权重）。

    Args:
        path: .pt 文件。
        device: 设备。
        num_classes: 类别数。

    Returns:
        nn.Module: eval 模式 EfficientNet。
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_efficientnet_cifar100(num_classes=num_classes, pretrained=False)
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
    绘制训练过程损失与准确率（中文）。

    Args:
        train_losses: 训练损失。
        val_losses: 验证损失。
        train_accs: 训练准确率。
        val_accs: 验证准确率。

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
    axes[0].set_title("CIFAR-100 + EfficientNet-B0：损失曲线")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accs, label="训练集", color="#4C72B0")
    axes[1].plot(epochs, val_accs, label="验证集", color="#C44E52")
    axes[1].set_xlabel("训练轮次 Epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("CIFAR-100 + EfficientNet-B0：准确率曲线")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    plt.show()


def plot_confusion_matrix_large(cm: np.ndarray, title: str, tick_step: int = 10) -> None:
    """
    绘制类别数较多时的混淆矩阵热力图（不对每个格子标注数字，避免拥挤）。

    Args:
        cm: 混淆矩阵，形状 (100, 100)。
        title: 中文标题。
        tick_step: 坐标轴刻度间隔（每隔多少类显示一个刻度编号）。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        cbar_kws={"label": "样本数量"},
        ax=ax,
    )
    ticks = list(range(0, cm.shape[0], tick_step))
    ax.set_xticks([t + 0.5 for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_yticks([t + 0.5 for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)
    ax.set_xlabel("预测类别编号（0 至 99）")
    ax.set_ylabel("真实类别编号（0 至 99）")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    主流程：CIFAR-100 数据、EfficientNet 微调、保存、测试评估与中文可视化。

    Returns:
        None: 无返回值。
    """
    configure_chinese_matplotlib()
    torch.manual_seed(42)
    np.random.seed(42)

    data_root = Path(__file__).resolve().parent / "data"
    ckpt_path = Path(__file__).resolve().parent / "checkpoints" / "cifar100_efficientnet_b0_best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}", flush=True)
    if _TQDM_AVAILABLE:
        print("已检测到 tqdm：训练、验证与测试将显示实时进度条。", flush=True)
    else:
        print(
            "未检测到 tqdm，无法显示进度条，仅每隔若干 batch 打印文本。"
            "建议安装: pip install tqdm",
            flush=True,
        )
    if device.type == "cpu":
        print("提示：CIFAR-100 + EfficientNet 在 CPU 上训练较慢，如有 GPU 建议改用 CUDA。", flush=True)

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    try:
        print(
            "正在加载 CIFAR-100（首次运行会从网络下载数据压缩包，可能需要数分钟）...",
            flush=True,
        )
        # 训练集与验证集应对同一样本索引划分，但验证集不使用随机增强，仅使用 eval_transform
        train_aug = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=train_transform)
        train_eval = datasets.CIFAR100(root=str(data_root), train=True, download=True, transform=eval_transform)
        test_set = datasets.CIFAR100(root=str(data_root), train=False, download=True, transform=eval_transform)
        print("CIFAR-100 数据已加载完成。", flush=True)
    except Exception as exc:
        print(f"数据集下载或加载失败: {exc}")
        raise

    n_train = len(train_aug)
    train_size = int(0.9 * n_train)
    val_size = n_train - train_size
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_train, generator=g).tolist()
    train_idx = perm[:train_size]
    val_idx = perm[train_size:]
    train_set = torch.utils.data.Subset(train_aug, train_idx)
    val_set = torch.utils.data.Subset(train_eval, val_idx)

    batch_size = 32
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print(
        f"数据划分: 训练样本 {len(train_set)}, 验证样本 {len(val_set)}, 测试样本 {len(test_set)}；"
        f"每 epoch 约 {len(train_loader)} 个训练 batch。",
        flush=True,
    )
    print(
        "若使用 CPU，单步前向较慢，首个 epoch 可能需较长时间；请观察 tqdm 进度条或文本 batch 日志。",
        flush=True,
    )

    use_pretrained = True
    print(
        "\n正在构建 EfficientNet-B0 并加载 ImageNet 预训练权重（首次运行需从网络下载，可能占用数分钟，请耐心等待）...",
        flush=True,
    )
    try:
        model = build_efficientnet_cifar100(num_classes=100, pretrained=use_pretrained).to(device)
        print("预训练权重已就绪，分类头已替换为 100 类并完成 Xavier 初始化。", flush=True)
    except Exception as exc:
        print(f"加载预训练 EfficientNet 失败，将从头训练骨干（精度可能明显下降）: {exc}", flush=True)
        model = build_efficientnet_cifar100(num_classes=100, pretrained=False).to(device)
        print("已使用随机初始化骨干。", flush=True)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"当前可训练参数约 {n_params / 1e6:.2f} M 个。", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, lr_backbone=3e-4, lr_head=3e-3)

    num_epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    early_stopping = EarlyStopping(patience=4, min_delta=5e-3, mode="min")

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []
    best_val = 0.0
    actual_epochs = 0

    print("\n开始微调 EfficientNet-B0（骨干与分类头不同学习率 + 余弦退火 + 早停）...", flush=True)
    for epoch in range(1, num_epochs + 1):
        tag = f"Epoch {epoch}/{num_epochs}"
        tl, ta = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch_tag=tag,
            log_interval=40,
        )
        print(f"  [{tag}] 训练阶段结束，开始在验证集上评估...", flush=True)
        vl, va, _, _ = evaluate(model, val_loader, criterion, device, desc=f"{tag} 验证集", log_interval=40)
        scheduler.step()
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        actual_epochs = epoch
        if va > best_val:
            best_val = va
            save_checkpoint(model, ckpt_path, epoch, best_val)
        lr0 = optimizer.param_groups[0]["lr"]
        lr1 = optimizer.param_groups[1]["lr"]
        print(
            f"Epoch {epoch:02d}/{num_epochs}  lr骨干={lr0:.2e} lr头={lr1:.2e}  "
            f"训练损失={tl:.4f}  验证损失={vl:.4f}  训练准确率={ta:.4f}  验证准确率={va:.4f}",
            flush=True,
        )
        if early_stopping.step(vl):
            print(f"早停触发：验证损失连续 {early_stopping.patience} 轮未明显下降，在第 {epoch} 轮结束。", flush=True)
            break

    print(f"\n最佳验证准确率: {best_val:.4f}，检查点: {ckpt_path}（实际训练 {actual_epochs} 轮）", flush=True)

    print("\n加载最佳权重并在测试集上评估...", flush=True)
    model_loaded = load_model_for_inference(ckpt_path, device, num_classes=100)
    test_loss, test_acc, y_true, y_pred = evaluate(
        model_loaded,
        test_loader,
        criterion,
        device,
        desc="测试集",
        log_interval=40,
    )
    print(f"测试集损失: {test_loss:.4f}")
    print(f"测试集准确率: {test_acc:.4f}")

    cifar100_meta = datasets.CIFAR100(root=str(data_root), train=True, download=False)
    fine_names = cifar100_meta.classes

    print("\n sklearn 分类报告（因 100 类，仅展示汇总行与前几行示意）:")
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(100)),
        target_names=fine_names,
        digits=3,
        zero_division=0,
    )
    lines = report.splitlines()
    print("\n".join(lines[:25]))
    print("...（其余类别行略，完整报告可保存到文件自行查看）...")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(100)))
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    plot_confusion_matrix_large(cm, "CIFAR-100 测试集混淆矩阵（EfficientNet-B0，行=真实，列=预测）")

    print("\n说明：骨干为 ImageNet 预训练权重；新分类层 Xavier 初始化；分类头内含 Dropout；学习率为 CosineAnnealingLR。")


if __name__ == "__main__":
    main()

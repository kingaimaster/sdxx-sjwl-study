# -*- coding: utf-8 -*-
"""
脚本：对比不同神经网络权重初始化对前向激活分布与短期训练的影响。

实现三种初始化：全零、随机正态分布、Kaiming He（配合 ReLU）。
网络隐藏层含 Dropout；短训练使用训练/验证划分、余弦退火与早停；
并保留一次前向抓取激活直方图与损失曲线对比。
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


def init_all_zeros(module: nn.Module) -> None:
    """
    将模块中所有可训练参数（权重与偏置）初始化为 0。

    提示：全零初始化会导致隐藏层神经元完全对称，反向传播时梯度相同，
    多个神经元等价于一个神经元，深度网络难以学习丰富特征。

    Args:
        module: 任意 nn.Module（通常传入 Sequential 或自定义网络）。

    Returns:
        None: 原地修改参数，无返回值。
    """
    if isinstance(module, nn.Linear):
        nn.init.constant_(module.weight, 0.0)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_normal_small(module: nn.Module, std: float = 0.05) -> None:
    """
    使用较小的随机正态分布初始化线性层权重；偏置置 0。

    Args:
        module: 网络模块。
        std: 正态分布标准差。

    Returns:
        None: 原地修改参数。
    """
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


def init_kaiming_relu(module: nn.Module) -> None:
    """
    对 Linear 层使用 Kaiming 均匀初始化（适用于 ReLU）。

    该初始化根据 fan_in 调整方差，有助于缓解深层网络中的梯度消失/爆炸问题，
    使各层激活值在合理范围内波动。

    Args:
        module: 网络模块。

    Returns:
        None: 原地修改参数。
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=0.0, mode="fan_in", nonlinearity="relu")
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class DeepMLP(nn.Module):
    """
    用于演示初始化的较深全连接网络：输入 -> 三层隐藏 ReLU -> 输出 logits。

    说明：为突出初始化对“隐藏层激活分布”的影响，隐藏层宽度设为 256。
    """

    def __init__(self, in_dim: int, hidden: int, num_classes: int) -> None:
        """
        构建四层 Linear（三层隐藏）结构。

        Args:
            in_dim: 输入维度。
            hidden: 各隐藏层宽度（本示例取相同宽度）。
            num_classes: 输出类别数。

        Returns:
            None: 构造函数无返回值。
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算 logits。

        Args:
            x: (N, in_dim)。

        Returns:
            torch.Tensor: (N, num_classes) logits。
        """
        return self.net(x)


def register_hidden_activation_capture(model: DeepMLP) -> tuple[list[torch.Tensor], list]:
    """
    在第三个隐藏层 ReLU 之后注册前向钩子，用于抓取该层输出激活值。

    Args:
        model: DeepMLP 实例。

    Returns:
        tuple[list[torch.Tensor], list]:
            - 用于存放激活张量的列表（钩子内 append）；
            - 钩子句柄列表（便于后续移除，本脚本可省略移除步骤）。
    """
    activations: list[torch.Tensor] = []
    handles = []

    def hook_fn(_module: nn.Module, _inp: tuple, out: torch.Tensor) -> None:
        activations.append(out.detach().reshape(-1).cpu())

    # 下标：Linear0, ReLU1, Dropout2, ... 第三个 ReLU 为索引 7
    target = model.net[7]
    handles.append(target.register_forward_hook(hook_fn))
    return activations, handles


def capture_activations_once(model: DeepMLP, x: torch.Tensor) -> torch.Tensor:
    """
    运行一次前向传播并返回抓取到的隐藏层激活向量（展平后拼接）。

    Args:
        model: 已注册钩子的模型（需与 register_hidden_activation_capture 配合使用）。
        x: 输入批量。

    Returns:
        torch.Tensor: 一维张量，包含该次前向中目标层所有神经元的激活值。
    """
    activations, _handles = register_hidden_activation_capture(model)
    model.eval()
    with torch.no_grad():
        _ = model(x)
    if not activations:
        return torch.tensor([])
    return activations[0]


def build_initialized_model(
    init_name: str,
    in_dim: int,
    hidden: int,
    num_classes: int,
) -> DeepMLP:
    """
    根据名称创建 DeepMLP 并应用对应初始化函数。

    Args:
        init_name: 'zeros' | 'normal' | 'kaiming'。
        in_dim: 输入维度。
        hidden: 隐藏层宽度。
        num_classes: 类别数。

    Returns:
        DeepMLP: 新模型实例。
    """
    m = DeepMLP(in_dim, hidden, num_classes)
    if init_name == "zeros":
        m.apply(init_all_zeros)
    elif init_name == "normal":
        m.apply(init_normal_small)
    elif init_name == "kaiming":
        m.apply(init_kaiming_relu)
    else:
        raise ValueError(f"未知初始化类型: {init_name}")
    return m


def short_train(
    model: DeepMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    max_epochs: int = 120,
    lr: float = 0.05,
) -> tuple[list[float], list[float]]:
    """
    短训练：记录训练/验证交叉熵；CosineAnnealingLR 与验证集早停。

    Args:
        model: 模型。
        x_train, y_train: 训练集。
        x_val, y_val: 验证集。
        max_epochs: 最大轮数。
        lr: 初始学习率。

    Returns:
        tuple: (训练损失序列, 验证损失序列)。
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-5)
    early = EarlyStopping(patience=25, min_delta=1e-4, mode="min")
    train_losses: list[float] = []
    val_losses: list[float] = []

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
        train_losses.append(tl)
        val_losses.append(vl)
        if early.step(vl):
            break

    return train_losses, val_losses


def plot_activation_histograms(
    tensors: dict[str, torch.Tensor],
    title: str,
) -> None:
    """
    绘制多种初始化下隐藏层激活的直方图对比。

    Args:
        tensors: 键为方法名称，值为激活值一维张量。
        title: 图标题。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    n = len(tensors)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]
    colors = ["#4C72B0", "#55A868", "#C44E52"]
    for ax, (name, vec), c in zip(axes, tensors.items(), colors):
        if vec.numel() == 0:
            continue
        ax.hist(vec.numpy(), bins=60, color=c, alpha=0.85, edgecolor="white")
        ax.set_title(f"{name}\n激活值分布")
        ax.set_xlabel("激活值")
        ax.set_ylabel("频数")
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_loss_curves(train_curves: dict[str, list[float]], val_curves: dict[str, list[float]]) -> None:
    """
    绘制不同初始化下训练与验证损失对比。

    Args:
        train_curves: 方法名到训练损失序列。
        val_curves: 方法名到验证损失序列。

    Returns:
        None: 仅绘图。
    """
    configure_chinese_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name, seq in train_curves.items():
        axes[0].plot(range(1, len(seq) + 1), seq, label=name, linewidth=2)
    axes[0].set_title("训练集损失（Dropout + 余弦退火 + 早停）")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CrossEntropy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for name, seq in val_curves.items():
        axes[1].plot(range(1, len(seq) + 1), seq, label=name, linewidth=2)
    axes[1].set_title("验证集损失")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("CrossEntropy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()


def make_dataset(
    n_samples: int = 2000,
    in_dim: int = 64,
    n_classes: int = 5,
    seed: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    生成多类分类合成数据。

    Args:
        n_samples: 样本数。
        in_dim: 特征维度。
        n_classes: 类别数。
        seed: 随机种子。

    Returns:
        tuple: (x, y)，y 为 long 类型类别索引。
    """
    g = torch.Generator().manual_seed(seed)
    centers = torch.randn(n_classes, in_dim, generator=g) * 1.5
    y = torch.randint(0, n_classes, (n_samples,), generator=g)
    x = centers[y] + torch.randn(n_samples, in_dim, generator=g) * 0.5
    return x, y


def main() -> None:
    """
    主流程：对三种初始化分别抓取激活并短训练，最后绘图。

    Returns:
        None: 无返回值。
    """
    torch.manual_seed(0)
    configure_chinese_matplotlib()

    in_dim = 64
    hidden = 256
    num_classes = 5
    x, y = make_dataset(in_dim=in_dim, n_classes=num_classes)
    n = x.size(0)
    n_train = int(0.85 * n)
    perm = torch.randperm(n)
    tr, va = perm[:n_train], perm[n_train:]
    x_train, y_train = x[tr], y[tr]
    x_val, y_val = x[va], y[va]

    init_kinds = {
        "全零初始化": "zeros",
        "随机正态（小方差）": "normal",
        "Kaiming He（ReLU）": "kaiming",
    }

    act_map: dict[str, torch.Tensor] = {}
    train_curves: dict[str, list[float]] = {}
    val_curves: dict[str, list[float]] = {}

    for label, key in init_kinds.items():
        model = build_initialized_model(key, in_dim, hidden, num_classes)
        vec = capture_activations_once(model, x[:512])
        act_map[label] = vec
        m_train = build_initialized_model(key, in_dim, hidden, num_classes)
        tl, vl = short_train(m_train, x_train, y_train, x_val, y_val, max_epochs=120, lr=0.05)
        train_curves[label] = tl
        val_curves[label] = vl

    plot_activation_histograms(
        act_map,
        title="第三个隐藏层 ReLU 之后的激活值分布（单次前向，批量样本拼接）",
    )
    plot_loss_curves(train_curves, val_curves)

    print("=== 小结 ===")
    print("1）全零初始化通常导致对称性与梯度相同问题，隐藏层难以学习多样特征。")
    print("2）过小或过大的随机初始化可能使激活饱和或方差不稳定。")
    print("3）Kaiming/Xavier 等初始化根据层宽度与非线性类型缩放方差，有助于深层网络稳定训练。")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""
教学用公共工具：早停（Early Stopping）与 ONNX 导出辅助函数。

供同目录下多个脚本 `import` 使用；若仅需独立运行单文件，可将类复制到该文件中。
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn


class EarlyStopping:
    """
    早停：当验证集指标在若干轮内未提升时终止训练，减轻过拟合与无效计算。

    Args 说明在 __init__ 中给出；通过 step 方法每轮传入验证集指标并判断是否应停止。
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min") -> None:
        """
        Args:
            patience: 验证指标连续未改善的最大轮数，达到则触发早停。
            min_delta: 视为「有改善」的最小变化量。
            mode: \"min\" 表示指标越小越好（如损失）；\"max\" 表示越大越好（如准确率）。
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.counter = 0

    def step(self, metric: float) -> bool:
        """
        根据本轮验证指标更新内部状态。

        Args:
            metric: 当前 epoch 的验证集指标（如验证损失或验证准确率）。

        Returns:
            bool: 若应提前结束训练则为 True，否则为 False。
        """
        if self.best is None:
            self.best = metric
            return False
        if self.mode == "min":
            improved = metric < self.best - self.min_delta
        else:
            improved = metric > self.best + self.min_delta
        if improved:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

    def reset(self) -> None:
        """重置状态，便于在同一脚本内多次复用同一实例。"""
        self.best = None
        self.counter = 0


def export_model_onnx(
    model: nn.Module,
    export_path: Path,
    dummy_input: torch.Tensor,
    input_names: tuple[str, ...] = ("input",),
    output_names: tuple[str, ...] = ("output",),
    dynamic_batch: bool = True,
    opset_version: int = 17,
) -> None:
    """
    将 PyTorch 模型导出为 ONNX，便于在 ONNX Runtime、TensorRT 等环境部署。

    Args:
        model: 模型；导出前会置于 eval。
        export_path: 保存路径，一般以 .onnx 为后缀。
        dummy_input: 与真实推理时形状一致的示例张量，设备与模型一致。
        input_names: ONNX 输入名。
        output_names: ONNX 输出名。
        dynamic_batch: 是否声明 batch 维为动态。
        opset_version: ONNX 算子集版本。

    Returns:
        None: 文件写入磁盘。
    """
    model.eval()
    device = next(model.parameters()).device
    dummy = dummy_input.to(device)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {input_names[0]: {0: "batch_size"}}
        if output_names:
            dynamic_axes[output_names[0]] = {0: "batch_size"}
    # dynamo=False 使用 TorchScript 路径导出，避免默认 Dynamo 路径依赖 onnxscript（教学环境常未安装）
    export_kw: dict = {
        "input_names": list(input_names),
        "output_names": list(output_names),
        "dynamic_axes": dynamic_axes,
        "opset_version": opset_version,
        "dynamo": False,
    }
    try:
        torch.onnx.export(model, dummy, str(export_path), **export_kw)
    except TypeError:
        export_kw.pop("dynamo", None)
        torch.onnx.export(model, dummy, str(export_path), **export_kw)

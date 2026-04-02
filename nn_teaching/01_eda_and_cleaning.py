# -*- coding: utf-8 -*-
"""
脚本：探索性数据分析（EDA）与数据清洗入门。

本脚本使用 pandas 与 scikit-learn 构造带有缺失值与异常值的模拟数据，
演示均值填充、标准化以及清洗前后分布的可视化对比。
独立运行即可在控制台看到统计摘要并弹出图表窗口。
"""

from __future__ import annotations

import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

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

warnings.filterwarnings("ignore", category=UserWarning)


def configure_chinese_matplotlib() -> None:
    """
    显式配置 matplotlib 的中文字体与坐标轴负号显示。

    在部分系统上字体列表中靠前的字体若不存在，matplotlib 会依次尝试后续字体，
    从而尽量保证中文标题与图例正常显示。

    Returns:
        None: 本函数无返回值，仅修改全局 rcParams。
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


def build_dirty_regression_dataset(
    n_samples: int = 500,
    n_features: int = 3,
    random_state: int = 42,
    missing_ratio: float = 0.08,
    outlier_ratio: float = 0.03,
) -> pd.DataFrame:
    """
    构造带有缺失值与异常值的模拟回归数据集。

    先使用 sklearn 生成干净的回归数据，再通过随机掩码注入缺失值，
    并对少量样本的目标变量乘以较大系数以模拟异常值。

    Args:
        n_samples: 样本数量。
        n_features: 特征数量（本演示主要关注第一个特征的分布对比）。
        random_state: 随机种子，保证结果可复现。
        missing_ratio: 在第一个特征上随机置为缺失的比例（近似）。
        outlier_ratio: 目标列异常值所占比例（近似）。

    Returns:
        pd.DataFrame: 包含特征列 feature_0.. 与目标列 target 的数据框。
    """
    rng = np.random.default_rng(random_state)
    x_arr, y_arr, _coef = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=12.0,
        random_state=random_state,
        coef=True,
    )
    df = pd.DataFrame(x_arr, columns=[f"feature_{i}" for i in range(n_features)])
    df["target"] = y_arr

    # 在第一个特征上随机注入缺失值（NaN）
    n_missing = int(n_samples * missing_ratio)
    miss_idx = rng.choice(n_samples, size=n_missing, replace=False)
    df.loc[miss_idx, "feature_0"] = np.nan

    # 对目标列注入少量极端异常值
    n_out = max(1, int(n_samples * outlier_ratio))
    out_idx = rng.choice(n_samples, size=n_out, replace=False)
    df.loc[out_idx, "target"] = df.loc[out_idx, "target"] * rng.uniform(8.0, 15.0, size=n_out)

    return df


def plot_distribution_comparison(
    series_before: pd.Series,
    series_after: np.ndarray,
    title: str,
    xlabel: str,
) -> None:
    """
    绘制清洗前（原始）与清洗后（处理完毕）的分布对比直方图。

    Args:
        series_before: 清洗前的 pandas 序列（允许含 NaN，绘图时会自动忽略）。
        series_after: 清洗并（若适用）标准化后的 numpy 数组。
        title: 图总标题。
        xlabel: 横轴标签文字。

    Returns:
        None: 本函数仅负责展示图像，无返回值。
    """
    configure_chinese_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(series_before.dropna(), bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_title("清洗前：特征 feature_0 分布")
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("频数")

    axes[1].hist(series_after, bins=30, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_title("清洗后：特征 feature_0（均值填充 + 标准化）")
    axes[1].set_xlabel(xlabel + "（标准化后）")
    axes[1].set_ylabel("频数")

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_boxplot_comparison(
    target_before: pd.Series,
    target_after_clip: pd.Series,
) -> None:
    """
    使用箱线图对比目标变量在异常值处理前后的分布变化。

    说明：本函数对“清洗后”的目标采用 winsorize 思路的简易截断
    （按分位数裁剪极端值），以便肉眼观察箱线图差异；实际项目中应结合业务含义
    选择删除、截断或稳健回归等策略。

    Args:
        target_before: 含异常值的目标列。
        target_after_clip: 经分位数裁剪后的目标列。

    Returns:
        None: 本函数仅负责展示图像，无返回值。
    """
    configure_chinese_matplotlib()
    fig, ax = plt.subplots(figsize=(7, 5))
    data = [target_before.values, target_after_clip.values]
    ax.boxplot(data, labels=["清洗前（含异常值）", "清洗后（分位数裁剪示意）"])
    ax.set_title("目标变量 target 的箱线图对比")
    ax.set_ylabel("目标值")
    plt.tight_layout()
    plt.show()


def main() -> None:
    """
    主流程：生成脏数据、打印摘要、执行缺失值填充与标准化，并绘图对比。

    Returns:
        None: 脚本入口无返回值。
    """
    configure_chinese_matplotlib()
    df_dirty = build_dirty_regression_dataset()
    print("=== 原始数据前 5 行（含缺失） ===")
    print(df_dirty.head())
    print("\n=== 缺失值统计 ===")
    print(df_dirty.isna().sum())

    # 仅对特征列做缺失值填充，避免对目标列误填
    feature_cols = [c for c in df_dirty.columns if c.startswith("feature_")]
    imputer = SimpleImputer(strategy="mean")
    x_imputed = imputer.fit_transform(df_dirty[feature_cols])
    df_clean = pd.DataFrame(x_imputed, columns=feature_cols, index=df_dirty.index)
    df_clean["target"] = df_dirty["target"].values

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df_clean[feature_cols])
    df_scaled = pd.DataFrame(x_scaled, columns=[f"{c}_scaled" for c in feature_cols], index=df_clean.index)
    df_scaled["target"] = df_clean["target"].values

    print("\n=== 均值填充 + 标准化后的特征描述性统计 ===")
    print(df_scaled.describe())

    # 第一个特征：清洗前 vs 清洗后（标准化后）直方图
    feat0_before = df_dirty["feature_0"]
    feat0_after = df_scaled["feature_0_scaled"].values
    plot_distribution_comparison(
        feat0_before,
        feat0_after,
        title="特征分布：数据清洗与标准化前后对比",
        xlabel="特征值",
    )

    # 目标列：简易异常值处理（按 1% 与 99% 分位裁剪）用于箱线图展示
    q_low, q_high = df_clean["target"].quantile([0.01, 0.99])
    target_clipped = df_clean["target"].clip(lower=q_low, upper=q_high)
    plot_boxplot_comparison(df_clean["target"], target_clipped)

    print("\n说明：本脚本为探索性数据分析，不涉及神经网络训练，故不演示 Dropout、早停与学习率调度。")

    print("\n要点小结：")
    print("1）缺失值会改变均值、方差等统计量，直接建模可能引入偏差；常用策略包括删除、均值/中位数填充、或更复杂的插值。")
    print("2）StandardScaler 将每个特征转为零均值单位方差，有助于梯度下降类算法稳定收敛。")
    print("3）异常值会拉长分布尾部，箱线图对离群点非常敏感；需结合领域知识决定处理策略。")


if __name__ == "__main__":
    main()

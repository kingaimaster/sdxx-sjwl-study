# PyTorch 神经网络核心概念教学代码库

## 项目背景

本目录包含一套面向初学者的 Python 脚本，覆盖数据预处理、线性/逻辑回归、多层感知机、权重初始化、卷积网络、迁移学习，以及 **MNIST / Fashion-MNIST / CIFAR-100** 的完整建模流程（训练、保存、推理、评估）。除 **01** 为纯数据分析外，神经网络相关脚本普遍包含 **Dropout**、验证集 **早停（Early Stopping）** 与 **余弦退火（CosineAnnealingLR）** 学习率调度；**08～10** 另对分类任务使用余弦调度（详见各文件）。每个脚本含详尽 **中文注释** 与 **Docstring**。图表使用 **matplotlib**（部分配合 **seaborn**），并配置中文字体与负号。

## 目录与生成文件

| 路径 | 说明 |
|------|------|
| `training_utils.py` | 公共模块：`EarlyStopping` 早停、`export_model_onnx` ONNX 导出（供多脚本 `import`） |
| `data/` | 运行 08、09、10 时自动下载的数据集（MNIST、Fashion-MNIST、CIFAR-100 等） |
| `checkpoints/` | 运行 08、09、10 时保存的模型权重（`.pt`）；**09** 另导出 `fashionmnist_cnn.onnx` |

若目录不存在，脚本会在首次保存时创建。

**运行目录**：请在 **`nn_teaching` 目录下**执行 `python xxx.py`，以便正确导入 `training_utils`（除 **01** 仅数据分析外，其余脚本均依赖该模块或与其一致的工程约定）。

## 环境要求

- Python 3.10 及以上（推荐 3.11）
- **PyTorch 2.11.0** 及与之匹配的 **torchvision**（版本以 [PyTorch 官网](https://pytorch.org/get-started/locally/) 为准；若暂无 2.11.0  wheel，可使用官网当前推荐的 2.x 稳定版）
- 常用库：NumPy、pandas、scikit-learn、matplotlib、seaborn、**tqdm**（脚本 10 强烈建议安装以显示训练进度条）

### 使用 conda（示例）

```bash
conda activate pytorch
cd /path/to/nn_teaching
```

### 安装 PyTorch 与依赖

按本机是否需要 GPU，在官网选择对应命令安装 **torch** 与 **torchvision**。CPU 示例如下（索引地址与版本请以官网为准）：

```bash
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cpu
```

其余依赖可一键安装（版本见文件内说明）：

```bash
pip install -r requirements.txt
```

若需手动逐个安装，亦可执行：

```bash
pip install numpy pandas scikit-learn matplotlib seaborn tqdm
```

## 脚本总览

| 文件 | 内容概要 |
|------|----------|
| `01_eda_and_cleaning.py` | 模拟数据的缺失值与异常值；均值填充、标准化；清洗前后分布对比图 |
| `02_linear_regression.py` | 线性回归 + Dropout 演示；训练/验证；余弦退火 + 早停 |
| `03_logistic_regression.py` | 逻辑回归 + 特征 Dropout；BCE；余弦退火 + 早停；决策边界 |
| `04_fully_connected_nn.py` | MLP + Dropout；多分类；余弦退火 + 早停 |
| `05_weight_initialization.py` | 深度 MLP + Dropout；初始化对比；余弦退火 + 早停 |
| `06_cnn_network.py` | CNN + 全连接前 Dropout；单步反向后短训练；余弦退火 + 早停 |
| `07_efficientnet_transfer.py` | EfficientNet 分类头（含 Dropout）；随机数据；余弦退火 + 早停 |
| `08_mnist_fcnn.py` | **MNIST** + MLP + Dropout；余弦退火 + 早停；保存/加载；混淆矩阵 |
| `09_fashionmnist_cnn.py` | **Fashion-MNIST** + CNN；余弦退火 + 早停；**导出 ONNX**；中文可视化 |
| `10_cifar100_efficientnet.py` | **CIFAR-100** + EfficientNet；骨干/头不同 lr；余弦退火 + 早停；**tqdm** |

## 运行方式

在 `nn_teaching` 目录下执行：

```bash
python 01_eda_and_cleaning.py
```

将文件名依次替换为 `02_...` 至 `10_...` 即可。脚本 **01～07** 主要使用随机或 sklearn 模拟数据；**08～10** 需联网 **首次下载** 数据集或（10 中）ImageNet 预训练权重，耗时因网络而异。

## 各脚本补充说明

### 01～07（基础概念）

- **01**：`pandas`、`sklearn` 生成回归数据；无神经网络，**不**使用 Dropout/早停/调度（脚本内另有说明）。
- **02～07**：均含 `training_utils.EarlyStopping` 与 `CosineAnnealingLR`（超参数见各文件）。
- **06**：先单步前向/反向，再随机数据短训练；输入形状示例 `[32, 3, 28, 28]`。
- **07**：分类头内含 torchvision 默认 `Dropout`；随机 224 图像划分训练/验证。

### 08～10（真实数据集与工程流程）

- **08**：`checkpoints/mnist_fcnn_best.pt`；分类报告与混淆矩阵。
- **09**：样例预测网格；训练结束后导出 `checkpoints/fashionmnist_cnn.onnx`（需 PyTorch ONNX 支持；失败时脚本打印原因）。
- **10**：输入缩放到 224；骨干与分类头 **不同学习率** + 余弦退火；**tqdm**；CPU 上很慢，建议 GPU。

## 中文图表与字体

若标题或标签显示为方框，请在本机安装 **黑体 / 微软雅黑 / PingFang SC** 等字体，或编辑各脚本中的 `matplotlib.rcParams["font.sans-serif"]` 列表，将本机已有中文字体置于靠前位置。

## 学习建议

建议按 **01→10** 顺序阅读；**01～07** 夯实张量、层与训练循环，**08～10** 再过渡到真实数据与保存部署。调试数据集脚本时，可先减小 epoch 或 batch（在对应 `.py` 内修改常量）以缩短单次运行时间。

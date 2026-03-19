核心是在做一个计算机视觉/深度学习实验，目标是用乳腺组织病理图像去判断某个图像 patch 里是否出现了浸润性导管癌。

我们使用的数据集是 Kaggle 上公开的 Breast Histopathology Images（IDC）数据集：
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images

这个数据集本质上是从乳腺癌病理切片中裁剪得到的小图像块。每个样本是一张 50×50 的 RGB 图像，对应一个二分类标签，用来表示该区域中是否包含浸润性导管癌：

- 0 表示非癌组织
- 1 表示癌组织

整个数据集大约包含 27 万张图像，其中正负样本分布不均衡，大约 28% 为阳性样本。

数据在组织形式上是按 patient 进行分组的，每个患者目录下包含若干 patch，并按照标签分别存放在 0 和 1 子目录中。这一点在工程上非常关键，因为同一个患者的 patch 在分布上具有强相关性。如果直接在 patch 级别进行随机划分，很容易导致训练集和测试集之间的信息泄漏，从而高估模型性能。因此在实际使用中，我们采用基于 patient 的划分方式来构建训练集和验证集。

而我们的任务本质上是在做这样一个映射去拟合伯努利分布：

输入
$$ x \in \mathbb{R}^{h \times w \times 3} $$

其中，$x$ 是一张高为 $h$、宽为 $w$、3 个颜色通道的 RGB 组织图像 patch。

输出
$$ y \in {0,1} $$

其中，$y=1$ 表示该 patch 含有 IDC，$y=0$ 表示不含 IDC。

模型要学习的是一个函数
$$ f_\theta : \mathbb{R}^{h \times w \times 3} \rightarrow [0,1] $$

其中，$\theta$ 是模型参数，$f_\theta(x)$ 输出这个 patch 属于 IDC 的概率。

另外，由于标签是基于局部区域生成的，在肿瘤边界附近可能存在一定程度的标注噪声。这在训练过程中可能会对模型带来影响，需要通过数据增强、正则化或更鲁棒的训练策略来缓解。


```
histo-patch-cls/
├── README.md
├── pyproject.toml
├── configs/
│   ├── default.json
│   └── ...
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── scripts/
│   └── ...
├── src/
│   ├── histoclass_cli/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── main.py
│   └── histoclass/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── engine/
│       │       ├── __init__.py
│       │       ├── trainer.py
│       │       └── evaluator.py
│       └── utils/
│           ├── __init__.py
│           ├── config.py
│           ├── logger.py
│           ├── metrics.py
│           └── seed.py
├── outputs/
└── tests/
```

要求：  
第一，目录结构清晰，以后不会越写越乱。
第二，配置驱动，实验可复现。
第三，数据流单向且干净。
第四，训练、验证、推理、配置解析彼此解耦

我们使用 PyTorch，Python 3.11。
使用 dataclass 将参数和配置项提出，之后交给 utils/config.py 统一解析。类型注解清晰。

考虑横向拓展处使用状态机或者工厂模式构造模块，避免到处塞 if-else，但对象生命周期必须由模块自身或 engine 管理。

统一使用 utils.logger 中提供的 get_logger 获取 logger 进行注释。

histoclass_cli.pipeline 负责将 histoclass 库中的组件编排成应用层流水线的逻辑，main 单独提供命令行参数的解析与环境的初始化。

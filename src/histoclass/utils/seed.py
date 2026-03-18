"""Seed utilities for reproducible experiments."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from .logger import get_logger


LOGGER = get_logger(__name__)
UINT32_MAX: int = 2**32


@dataclass(frozen=True, slots=True)
class SeedState:
    """@brief 随机种子配置结果；Materialized seed configuration result.

    @param seed 主随机种子；Primary random seed.
    @param deterministic 是否启用确定性算法；Whether deterministic algorithms are enabled.
    @param benchmark 是否启用 cuDNN benchmark；Whether cuDNN benchmark is enabled.
    @param cuda_available 当前 CUDA 可用性；Current CUDA availability.
    """

    seed: int
    deterministic: bool
    benchmark: bool
    cuda_available: bool


def seed_everything(
    seed: int,
    *,
    deterministic: bool = True,
    benchmark: bool = False,
) -> SeedState:
    """@brief 统一设置实验随机种子；Set random seeds across runtime components.

    @param seed 期望使用的随机种子；Desired random seed.
    @param deterministic 是否启用确定性算法；Enable deterministic algorithms if True.
    @param benchmark 是否启用 cuDNN benchmark；Enable cuDNN benchmark if True.
    @return 实际生效的种子状态；Final applied seed state.
    @note deterministic=True 与 benchmark=True 通常互斥，后者可能引入非确定性。
    """
    normalized_seed = _normalize_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(normalized_seed)

    random.seed(normalized_seed)
    np.random.seed(normalized_seed)
    torch.manual_seed(normalized_seed)

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed(normalized_seed)
        torch.cuda.manual_seed_all(normalized_seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if deterministic and benchmark:
        LOGGER.warning(
            "Both deterministic=True and benchmark=True were requested; "
            "benchmark may reduce reproducibility."
        )

    LOGGER.info(
        "Seed configured: seed=%d, deterministic=%s, benchmark=%s, cuda_available=%s",
        normalized_seed,
        deterministic,
        benchmark,
        cuda_available,
    )

    return SeedState(
        seed=normalized_seed,
        deterministic=deterministic,
        benchmark=benchmark,
        cuda_available=cuda_available,
    )


def seed_worker(worker_id: int) -> None:
    """@brief 为 DataLoader worker 设置随机种子；Seed a DataLoader worker process.

    @param worker_id worker 编号；DataLoader worker identifier.
    @return 无返回值；No return value.
    @note 使用 PyTorch 传入的初始种子派生 NumPy 与 Python random 种子。
    """
    del worker_id
    worker_seed = torch.initial_seed() % UINT32_MAX
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def build_torch_generator(seed: int) -> torch.Generator:
    """@brief 构建带固定种子的 Torch 生成器；Build a seeded torch.Generator.

    @param seed 用于生成器的随机种子；Random seed for generator.
    @return 可注入 DataLoader 的 torch.Generator；A generator ready for DataLoader.
    """
    generator = torch.Generator()
    generator.manual_seed(_normalize_seed(seed))
    return generator


def _normalize_seed(seed: int) -> int:
    """@brief 规整种子到无符号 32 位范围；Normalize seed into uint32 range.

    @param seed 任意整数种子；Any integer seed.
    @return 规整后的非负整数种子；Normalized non-negative integer seed.
    """
    return int(seed) % UINT32_MAX


__all__ = ["SeedState", "build_torch_generator", "seed_everything", "seed_worker"]

"""Model definitions for IDC patch binary classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torchvision.models import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    resnet18,
    resnet34,
    resnet50,
)

from .utils import get_logger


LOGGER = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """@brief 模型配置；Model configuration.

    @param backbone_name ResNet 主干名称；ResNet backbone name.
    @param hidden_dim 分类头隐藏层维度；Hidden dimension of classifier head.
    @param dropout Dropout 概率；Dropout probability.
    @param pretrained 是否加载 ImageNet 预训练；Whether to load ImageNet pretrained weights.
    @param freeze_backbone 是否冻结 backbone 参数；Whether to freeze backbone parameters.
    """

    backbone_name: str = "resnet18"
    hidden_dim: int = 256
    dropout: float = 0.2
    pretrained: bool = True
    freeze_backbone: bool = False


class IDCResNetClassifier(nn.Module):
    """@brief IDC 二分类模型；IDC binary classifier with ResNet backbone.

    @note 网络结构：ResNet -> GAP -> Linear(hidden_dim) -> SiLU -> Dropout -> Linear(1).
    """

    def __init__(
        self,
        config: ModelConfig = ModelConfig(),
    ) -> None:
        """@brief 初始化 IDC 分类器；Initialize IDC classifier.

        @param config 模型配置；Model configuration.
        """
        super().__init__()
        _validate_model_config(config)
        self.config = config

        backbone = build_resnet(config.backbone_name, config.pretrained)
        feature_dim = _replace_backbone_fc_with_identity(backbone)

        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Dropout(p=config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

        if config.freeze_backbone:
            _freeze_module(self.backbone)

        LOGGER.info(
            "Model initialized: backbone=%s, feature_dim=%d, hidden_dim=%d, dropout=%.3f, freeze_backbone=%s",
            config.backbone_name,
            feature_dim,
            config.hidden_dim,
            config.dropout,
            config.freeze_backbone,
        )

    def forward(self, images: Tensor) -> Tensor:
        """@brief 前向计算 logits；Forward pass returning logits.

        @param images 输入图像张量，形状 `[B, C, H, W]`；Input images tensor `[B, C, H, W]`.
        @return 二分类 logits，形状 `[B, 1]`；Binary logits tensor `[B, 1]`.
        """
        features = self.backbone(images)
        return self.classifier(features)

    def predict_proba(self, images: Tensor) -> Tensor:
        """@brief 预测正类概率；Predict positive-class probabilities.

        @param images 输入图像张量；Input images tensor.
        @return 正类概率，形状 `[B, 1]`；Positive probabilities `[B, 1]`.
        """
        return torch.sigmoid(self.forward(images))

    def num_trainable_parameters(self) -> int:
        """@brief 统计可训练参数量；Count trainable parameters.

        @return 可训练参数总数；Total number of trainable parameters.
        """
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


def build_model(
    config: ModelConfig,
) -> IDCResNetClassifier:
    """@brief 按配置构建模型；Build model from config.

    @param config 模型配置；Model configuration.
    @return 构建完成的模型；Constructed model instance.
    """
    return IDCResNetClassifier(config=config)


def build_resnet(backbone_name: str, pretrained: bool) -> nn.Module:
    """@brief 按名称构建 ResNet 主干；Build ResNet backbone by name.

    @param backbone_name ResNet 名称（如 resnet18）；ResNet name (e.g., resnet18).
    @param pretrained 是否加载预训练权重；Whether to load pretrained weights.
    @return ResNet 主干模型；Instantiated ResNet backbone.
    """
    normalized_name = backbone_name.lower().strip()
    if normalized_name == "resnet18":
        LOGGER.info("Building backbone '%s' (pretrained=%s)", normalized_name, pretrained)
        return _build_resnet18(pretrained)
    if normalized_name == "resnet34":
        LOGGER.info("Building backbone '%s' (pretrained=%s)", normalized_name, pretrained)
        return _build_resnet34(pretrained)
    if normalized_name == "resnet50":
        LOGGER.info("Building backbone '%s' (pretrained=%s)", normalized_name, pretrained)
        return _build_resnet50(pretrained)

    available = "resnet18, resnet34, resnet50"
    LOGGER.error("Unknown backbone '%s'. Available: [%s]", backbone_name, available)
    raise KeyError(f"Unknown backbone '{backbone_name}'. Available: [{available}]")


def _build_resnet18(pretrained: bool) -> nn.Module:
    """@brief 构建 ResNet-18；Build ResNet-18 backbone."""
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    return resnet18(weights=weights)


def _build_resnet34(pretrained: bool) -> nn.Module:
    """@brief 构建 ResNet-34；Build ResNet-34 backbone."""
    weights = ResNet34_Weights.DEFAULT if pretrained else None
    return resnet34(weights=weights)


def _build_resnet50(pretrained: bool) -> nn.Module:
    """@brief 构建 ResNet-50；Build ResNet-50 backbone."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    return resnet50(weights=weights)


def _replace_backbone_fc_with_identity(backbone: nn.Module) -> int:
    """@brief 替换 fc 为恒等映射并返回特征维度；Swap fc with identity and return feature dim.

    @param backbone ResNet 主干模型；ResNet backbone module.
    @return 全局池化后特征维度；Feature dimension after global pooling.
    """
    if not hasattr(backbone, "fc"):
        raise TypeError("backbone must define attribute 'fc'.")

    fc_layer = backbone.fc
    if not isinstance(fc_layer, nn.Linear):
        raise TypeError(
            f"Expected backbone.fc to be nn.Linear, got {type(fc_layer).__name__}."
        )

    feature_dim = int(fc_layer.in_features)
    backbone.fc = nn.Identity()
    return feature_dim


def _freeze_module(module: nn.Module) -> None:
    """@brief 冻结模块参数；Freeze parameters in a module.

    @param module 待冻结模块；Module to freeze.
    """
    for parameter in module.parameters():
        parameter.requires_grad = False


def _validate_model_config(config: ModelConfig) -> None:
    """@brief 校验模型配置；Validate model configuration.

    @param config 待校验配置；Configuration to validate.
    """
    if config.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got {config.hidden_dim}.")

    if not (0.0 <= config.dropout < 1.0):
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}.")


__all__ = [
    "IDCResNetClassifier",
    "ModelConfig",
    "build_resnet",
    "build_model",
]

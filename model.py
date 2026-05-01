"""
PyTorch model classes for Host NAIP SDM binary classification.

The code supports seven model groups from three predictor families:

1. NAIP imagery
2. Topographic image chips
3. Tabular environmental covariates

Canonical model_type keys:
    image_only
    tabular_only
    topo_only
    image_tabular
    topo_tabular
    image_topo
    image_topo_tabular

Backward-compatible class names are retained, including HostClimateOnlyModel and
HostImageryClimateModel. In new writing, treat "climate" as "tabular env" when
the vector also includes GHM, lat/lon, and topographic scalar summaries.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights


MODEL_CLASS_ALIASES = {
    "HostImageryOnlyModel": "image_only",
    "HostClimateOnlyModel": "tabular_only",
    "HostTabularOnlyModel": "tabular_only",
    "HostTopoOnlyModel": "topo_only",
    "HostImageryClimateModel": "image_tabular",
    "HostImageTabularModel": "image_tabular",
    "HostTopoClimateModel": "topo_tabular",
    "HostTopoTabularModel": "topo_tabular",
    "HostImageTopoModel": "image_topo",
    "HostImageryTopoModel": "image_topo",
    "HostImageClimateTopoModel": "image_topo_tabular",
    "HostImageryClimateTopoModel": "image_topo_tabular",
    "HostImageTopoClimateModel": "image_topo_tabular",
    "HostNAIPTopoClimateModel": "image_topo_tabular",
}

MODEL_TYPE_ALIASES = {
    "image_only": "image_only",
    "naip_only": "image_only",
    "imagery_only": "image_only",
    "tabular_only": "tabular_only",
    "env_only": "tabular_only",
    "climate_only": "tabular_only",
    "topo_only": "topo_only",
    "topography_only": "topo_only",
    "image_tabular": "image_tabular",
    "image_climate": "image_tabular",
    "naip_climate": "image_tabular",
    "topo_tabular": "topo_tabular",
    "topo_climate": "topo_tabular",
    "image_topo": "image_topo",
    "naip_topo": "image_topo",
    "image_topo_tabular": "image_topo_tabular",
    "image_tabular_topo": "image_topo_tabular",
    "image_climate_topo": "image_topo_tabular",
    "image_topo_climate": "image_topo_tabular",
    "naip_topo_climate": "image_topo_tabular",
    "naip_climate_topo": "image_topo_tabular",
    "full": "image_topo_tabular",
}


def normalize_model_type(model_type: str) -> str:
    key = str(model_type).strip()
    if key in MODEL_CLASS_ALIASES:
        return MODEL_CLASS_ALIASES[key]
    key = key.lower()
    if key not in MODEL_TYPE_ALIASES:
        valid = ", ".join(sorted(MODEL_TYPE_ALIASES))
        raise ValueError(f"Unknown model_type '{model_type}'. Valid aliases include: {valid}")
    return MODEL_TYPE_ALIASES[key]


def get_resnet_model(pretrained: bool = True, in_channels: int = 4) -> nn.Module:
    """
    Create a ResNet18 encoder that accepts `in_channels` channels.

    For NAIP, pretrained ImageNet weights are useful for RGB-like bands. The
    extra NIR channel is initialized as the mean of the RGB filters.
    """
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    original_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=original_conv.bias is not None,
    )

    if pretrained:
        with torch.no_grad():
            copy_channels = min(3, in_channels)
            model.conv1.weight[:, :copy_channels, :, :] = original_conv.weight[:, :copy_channels, :, :]
            if in_channels > 3:
                mean_rgb = original_conv.weight.mean(dim=1)
                for c in range(3, in_channels):
                    model.conv1.weight[:, c, :, :] = mean_rgb
    else:
        nn.init.kaiming_normal_(model.conv1.weight, mode="fan_out", nonlinearity="relu")

    return model


class SmallTopoCNN(nn.Module):
    """
    Compact CNN for 4-band topographic chips.

    This branch intentionally does not use ImageNet pretraining because
    topographic chips are continuous geophysical layers, not natural RGB images.
    GroupNorm is used instead of BatchNorm for testing - but use BatchNorm in practice?
    """

    def __init__(self, in_channels: int = 4, feature_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.out_features = feature_dim

    def forward(self, x):
        return self.proj(self.encoder(x))


class TabularMLP(nn.Module):
    """Small MLP for tabular environmental predictors."""

    def __init__(self, num_features: int, feature_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be > 0 for TabularMLP")
        self.net = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(inplace=True),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),
        )
        self.out_features = feature_dim

    def forward(self, x):
        return self.net(x)


class FusionHead(nn.Module):
    """Binary classifier head for one or more fused feature vectors."""

    def __init__(self, in_features: int, hidden_dim: int = 256, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class HostModelBase(nn.Module):
    """Base class with training/validation utilities used by train_utils.py."""

    model_type_key = "base"

    def forward_from_batch(self, batch):
        raise NotImplementedError

    def training_step(self, batch):
        labels = batch["label"].to(next(self.parameters()).device)
        out = self.forward_from_batch(batch)
        return F.binary_cross_entropy_with_logits(out, labels)

    def validation_step(self, batch):
        labels = batch["label"].to(next(self.parameters()).device)
        out = self.forward_from_batch(batch)
        loss = F.binary_cross_entropy_with_logits(out, labels)
        preds = (torch.sigmoid(out) > 0.5).float()
        acc = (preds == labels).float().mean()
        return {"val_loss": loss.detach(), "val_acc": acc.detach()}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        return {"val_loss": loss.item(), "val_acc": acc.item()}

    def epoch_end(self, epoch, result):
        print(
            f"Epoch [{epoch + 1}], "
            f"train_loss: {result['train_loss']:.4f}, "
            f"val_loss: {result['val_loss']:.4f}, "
            f"val_acc: {result['val_acc']:.4f}"
        )


class HostImageryOnlyModel(HostModelBase):
    """NAIP imagery only."""

    model_type_key = "image_only"

    def __init__(self, naip_channels=4, hidden_dim=256, dropout=0.25, pretrained_image=True):
        super().__init__()
        self.image_encoder = get_resnet_model(pretrained=pretrained_image, in_channels=naip_channels)
        self.image_encoder.fc = nn.Identity()
        self.classifier = FusionHead(512, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, img):
        return self.classifier(self.image_encoder(img))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(batch["image"].to(device))


class HostClimateOnlyModel(HostModelBase):
    """
    Tabular environmental predictors only.

    The historical class name is retained for compatibility. The tabular vector
    may include WorldClim, GHM, lat/lon, and topographic scalar summaries.
    """

    model_type_key = "tabular_only"

    def __init__(self, num_env_features, hidden_dim=256, dropout=0.25, env_feature_dim=128):
        super().__init__()
        self.env_encoder = TabularMLP(num_env_features, feature_dim=env_feature_dim, dropout=dropout)
        self.classifier = FusionHead(env_feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, env):
        return self.classifier(self.env_encoder(env))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(batch["env"].to(device))


class HostTopoOnlyModel(HostModelBase):
    """Topographic image chips only."""

    model_type_key = "topo_only"

    def __init__(self, topo_channels=4, hidden_dim=256, dropout=0.25, topo_feature_dim=128):
        super().__init__()
        self.topo_encoder = SmallTopoCNN(topo_channels, feature_dim=topo_feature_dim, dropout=dropout)
        self.classifier = FusionHead(topo_feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, topo):
        return self.classifier(self.topo_encoder(topo))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(batch["topo"].to(device))


class HostImageryClimateModel(HostModelBase):
    """NAIP imagery + tabular environmental predictors."""

    model_type_key = "image_tabular"

    def __init__(
        self,
        num_env_features,
        naip_channels=4,
        hidden_dim=256,
        dropout=0.25,
        env_feature_dim=128,
        pretrained_image=True,
    ):
        super().__init__()
        self.image_encoder = get_resnet_model(pretrained=pretrained_image, in_channels=naip_channels)
        self.image_encoder.fc = nn.Identity()
        self.env_encoder = TabularMLP(num_env_features, feature_dim=env_feature_dim, dropout=dropout)
        self.classifier = FusionHead(512 + env_feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, img, env):
        img_feat = self.image_encoder(img)
        env_feat = self.env_encoder(env)
        return self.classifier(torch.cat([img_feat, env_feat], dim=1))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(batch["image"].to(device), batch["env"].to(device))


class HostTopoClimateModel(HostModelBase):
    """Topographic image chips + tabular environmental predictors."""

    model_type_key = "topo_tabular"

    def __init__(
        self,
        num_env_features,
        topo_channels=4,
        hidden_dim=256,
        dropout=0.25,
        topo_feature_dim=128,
        env_feature_dim=128,
    ):
        super().__init__()
        self.topo_encoder = SmallTopoCNN(topo_channels, feature_dim=topo_feature_dim, dropout=dropout)
        self.env_encoder = TabularMLP(num_env_features, feature_dim=env_feature_dim, dropout=dropout)
        self.classifier = FusionHead(topo_feature_dim + env_feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, topo, env):
        topo_feat = self.topo_encoder(topo)
        env_feat = self.env_encoder(env)
        return self.classifier(torch.cat([topo_feat, env_feat], dim=1))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(batch["topo"].to(device), batch["env"].to(device))


class HostImageTopoModel(HostModelBase):
    """NAIP imagery + topographic image chips."""

    model_type_key = "image_topo"

    def __init__(
        self,
        naip_channels=4,
        topo_channels=4,
        hidden_dim=256,
        dropout=0.25,
        topo_feature_dim=128,
        pretrained_image=True,
    ):
        super().__init__()
        self.image_encoder = get_resnet_model(pretrained=pretrained_image, in_channels=naip_channels)
        self.image_encoder.fc = nn.Identity()
        self.topo_encoder = SmallTopoCNN(topo_channels, feature_dim=topo_feature_dim, dropout=dropout)
        self.classifier = FusionHead(512 + topo_feature_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, img, topo):
        img_feat = self.image_encoder(img)
        topo_feat = self.topo_encoder(topo)
        return self.classifier(torch.cat([img_feat, topo_feat], dim=1))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(batch["image"].to(device), batch["topo"].to(device))


class HostImageClimateTopoModel(HostModelBase):
    """NAIP imagery + topographic image chips + tabular predictors."""

    model_type_key = "image_topo_tabular"

    def __init__(
        self,
        num_env_features,
        naip_channels=4,
        topo_channels=4,
        hidden_dim=256,
        dropout=0.25,
        topo_feature_dim=128,
        env_feature_dim=128,
        pretrained_image=True,
    ):
        super().__init__()
        self.image_encoder = get_resnet_model(pretrained=pretrained_image, in_channels=naip_channels)
        self.image_encoder.fc = nn.Identity()
        self.topo_encoder = SmallTopoCNN(topo_channels, feature_dim=topo_feature_dim, dropout=dropout)
        self.env_encoder = TabularMLP(num_env_features, feature_dim=env_feature_dim, dropout=dropout)
        self.classifier = FusionHead(
            512 + topo_feature_dim + env_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, img, topo, env):
        img_feat = self.image_encoder(img)
        topo_feat = self.topo_encoder(topo)
        env_feat = self.env_encoder(env)
        return self.classifier(torch.cat([img_feat, topo_feat, env_feat], dim=1))

    def forward_from_batch(self, batch):
        device = next(self.parameters()).device
        return self(
            batch["image"].to(device),
            batch["topo"].to(device),
            batch["env"].to(device),
        )


def build_model(
    model_type: str,
    num_env_features: int = 0,
    hidden_dim: int = 256,
    dropout: float = 0.25,
    naip_channels: int = 4,
    topo_channels: int = 4,
    topo_feature_dim: int = 128,
    env_feature_dim: int = 128,
    pretrained_image: bool = True,
) -> HostModelBase:
    """Factory for the seven supported model groups."""
    key = normalize_model_type(model_type)

    if key == "image_only":
        return HostImageryOnlyModel(
            naip_channels=naip_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained_image=pretrained_image,
        )
    if key == "tabular_only":
        return HostClimateOnlyModel(
            num_env_features=num_env_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            env_feature_dim=env_feature_dim,
        )
    if key == "topo_only":
        return HostTopoOnlyModel(
            topo_channels=topo_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            topo_feature_dim=topo_feature_dim,
        )
    if key == "image_tabular":
        return HostImageryClimateModel(
            num_env_features=num_env_features,
            naip_channels=naip_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            env_feature_dim=env_feature_dim,
            pretrained_image=pretrained_image,
        )
    if key == "topo_tabular":
        return HostTopoClimateModel(
            num_env_features=num_env_features,
            topo_channels=topo_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            topo_feature_dim=topo_feature_dim,
            env_feature_dim=env_feature_dim,
        )
    if key == "image_topo":
        return HostImageTopoModel(
            naip_channels=naip_channels,
            topo_channels=topo_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            topo_feature_dim=topo_feature_dim,
            pretrained_image=pretrained_image,
        )
    if key == "image_topo_tabular":
        return HostImageClimateTopoModel(
            num_env_features=num_env_features,
            naip_channels=naip_channels,
            topo_channels=topo_channels,
            hidden_dim=hidden_dim,
            dropout=dropout,
            topo_feature_dim=topo_feature_dim,
            env_feature_dim=env_feature_dim,
            pretrained_image=pretrained_image,
        )

    raise ValueError(f"Unsupported model_type after normalization: {key}")


# Backward-compatible aliases used by earlier scripts/checkpoints.
HostTabularOnlyModel = HostClimateOnlyModel
HostImageTabularModel = HostImageryClimateModel
HostTopoTabularModel = HostTopoClimateModel
HostImageryTopoModel = HostImageTopoModel
HostImageryClimateTopoModel = HostImageClimateTopoModel
HostImageTopoClimateModel = HostImageClimateTopoModel
HostNAIPTopoClimateModel = HostImageClimateTopoModel

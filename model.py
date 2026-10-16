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

    def __init__(self):
        super().__init__()
        self.register_buffer(
            "_training_pos_weight",
            torch.tensor(float("nan"), dtype=torch.float32),
            persistent=False,
        )
        self._training_label_smoothing = 0.0
        self._radiometric_consistency_weight = 0.0
        self._image_modality_dropout_probability = 0.0
        # October 2026 options (defaults reproduce the August behaviour exactly).
        self._image_modality_mode = "blank"
        self._image_feature_dim_amplitude = 0.0
        self._same_ground_consistency_weight = 0.0
        self._last_training_loss_components = {}

    def configure_training_loss(
        self,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
        radiometric_consistency_weight: float = 0.0,
        image_modality_dropout_probability: float = 0.0,
        image_modality_mode: str = "blank",
        image_feature_dim_amplitude: float = 0.0,
        same_ground_consistency_weight: float = 0.0,
    ) -> None:
        """Configure training-only weighting/smoothing.

        Validation always uses ordinary BCE against unsmoothed labels so model
        selection remains comparable across August loss arms.
        """
        if pos_weight is not None and (not torch.isfinite(torch.tensor(pos_weight)) or pos_weight <= 0):
            raise ValueError("pos_weight must be finite and positive")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if (
            not torch.isfinite(torch.tensor(radiometric_consistency_weight))
            or radiometric_consistency_weight < 0
        ):
            raise ValueError(
                "radiometric_consistency_weight must be finite and non-negative"
            )
        if (
            not torch.isfinite(torch.tensor(image_modality_dropout_probability))
            or not 0 <= image_modality_dropout_probability < 1
        ):
            raise ValueError(
                "image_modality_dropout_probability must be finite and in [0, 1)"
            )
        value = float("nan") if pos_weight is None else float(pos_weight)
        self._training_pos_weight.fill_(value)
        self._training_label_smoothing = float(label_smoothing)
        self._radiometric_consistency_weight = float(radiometric_consistency_weight)
        self._image_modality_dropout_probability = float(
            image_modality_dropout_probability
        )
        if image_modality_mode not in {"blank", "feature", "dim", "none"}:
            raise ValueError("image_modality_mode must be blank, feature, dim or none")
        if not 0 <= image_feature_dim_amplitude < 1:
            raise ValueError("image_feature_dim_amplitude must be in [0, 1)")
        if not same_ground_consistency_weight >= 0:
            raise ValueError("same_ground_consistency_weight must be non-negative")
        self._image_modality_mode = str(image_modality_mode)
        self._image_feature_dim_amplitude = float(image_feature_dim_amplitude)
        self._same_ground_consistency_weight = float(same_ground_consistency_weight)

    def apply_image_modality_dropout(self, features: torch.Tensor) -> torch.Tensor:
        """Training-only regularisation of the image embedding; identity at inference.

        Modes (October 2026):
          blank   August behaviour: zero the whole embedding per sample with the configured probability,
                  scaling kept embeddings by 1 / (1 - p).
          feature ordinary dropout on individual embedding units with the configured probability.
          dim     multiply each sample's embedding by u ~ Uniform(1 - a, 1 + a) (mean 1), a = amplitude.
          none    no image-embedding regularisation.
        The feature and dim modes keep the expected embedding equal to the inference embedding, so the
        fusion head never sees an inference-time magnitude it was not trained on.
        """
        if not self.training:
            return features
        mode = self._image_modality_mode
        if mode == "none":
            return features
        if mode == "feature":
            return F.dropout(features, p=self._image_modality_dropout_probability, training=True)
        if mode == "dim":
            a = self._image_feature_dim_amplitude
            if a <= 0:
                return features
            u = 1.0 + a * (2.0 * torch.rand((features.shape[0], 1), device=features.device) - 1.0)
            return features * u.to(features.dtype)
        probability = self._image_modality_dropout_probability
        if probability <= 0:
            return features
        keep = (
            torch.rand((features.shape[0], 1), device=features.device)
            >= probability
        ).to(features.dtype)
        return features * keep / (1.0 - probability)

    def forward_from_batch(self, batch):
        raise NotImplementedError

    def training_step(self, batch):
        labels = batch["label"].to(next(self.parameters()).device)
        out = self.forward_from_batch(batch)
        smoothing = self._training_label_smoothing
        targets = labels * (1.0 - smoothing) + 0.5 * smoothing
        pos_weight = (
            self._training_pos_weight
            if torch.isfinite(self._training_pos_weight)
            else None
        )
        supervised = F.binary_cross_entropy_with_logits(
            out, targets, pos_weight=pos_weight
        )
        consistency = torch.zeros((), dtype=supervised.dtype, device=supervised.device)
        if self._radiometric_consistency_weight > 0:
            if "image" not in batch or "image_consistency" not in batch:
                raise ValueError(
                    "Radiometric consistency requires image and image_consistency views"
                )
            paired_batch = dict(batch)
            paired_batch["image"] = batch["image_consistency"]
            paired_out = self.forward_from_batch(paired_batch)
            consistency = F.mse_loss(torch.sigmoid(out), torch.sigmoid(paired_out))
        total = supervised + self._radiometric_consistency_weight * consistency
        same_ground = torch.zeros((), dtype=supervised.dtype, device=supervised.device)
        if self._same_ground_consistency_weight > 0 and "sg_image_a" in batch:
            same_ground = self.same_ground_consistency(batch)
            total = total + self._same_ground_consistency_weight * same_ground
        self._last_training_loss_components = {
            "supervised": supervised.detach(),
            "radiometric_consistency": consistency.detach(),
            "radiometric_consistency_weight": self._radiometric_consistency_weight,
            "same_ground_consistency": same_ground.detach(),
            "same_ground_consistency_weight": self._same_ground_consistency_weight,
        }
        return total

    def same_ground_consistency(self, batch) -> torch.Tensor:
        """Mean squared logit difference between two acquisitions of the same ground (October 2026).

        The pair forward pass runs with the model in eval mode (BatchNorm running statistics, no dropout,
        no image-embedding regularisation), i.e. exactly as at inference, with gradients enabled. Both
        images share the same environmental vector. The A and B images go through one concatenated batch.
        """
        device = next(self.parameters()).device
        img = torch.cat([batch["sg_image_a"], batch["sg_image_b"]], dim=0).to(device)
        env = batch["sg_env"].to(device)
        was_training = self.training
        self.eval()
        try:
            out = self.forward_from_batch({"image": img, "env": torch.cat([env, env], dim=0)})
        finally:
            self.train(was_training)
        out_a, out_b = out.float().chunk(2, dim=0)
        return F.mse_loss(out_a, out_b)

    def training_loss_components(self) -> dict:
        return dict(self._last_training_loss_components)

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
        img_feat = self.apply_image_modality_dropout(img_feat)
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
        img_feat = self.apply_image_modality_dropout(img_feat)
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
        img_feat = self.apply_image_modality_dropout(img_feat)
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

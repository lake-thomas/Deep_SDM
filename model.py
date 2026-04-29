# Pytorch model classes for host tree classification using NAIP imagery and environmental variables
# Thomas Lake, January 2026

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights
import torch.nn.functional as F

class HostImageClimateModelBase(torch.nn.Module):
    """
    Base class for Imagery Climate Model
    """
    def training_step(self, batch):
        """
        Perform a training step on the model
        """
        images = batch["image"]
        envs   = batch["env"]
        labels = batch["label"]

        device = next(self.parameters()).device
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)

        if isinstance(self, HostImageryClimateModel):
            out = self(images, envs)
        elif isinstance(self, HostImageryClimateTopoModel):
            out = self(images, batch["topo"].to(device), envs)
        elif isinstance(self, HostImageryOnlyModel):
            out = self(images)
        elif isinstance(self, HostClimateOnlyModel):
            out = self(envs)
        elif isinstance(self, HostTopoOnlyModel):
            out = self(batch["topo"].to(device))
            out = self(batch["topo"].to(device))
        else:
            raise NotImplementedError("Unknown model type for training_step")

        loss = F.binary_cross_entropy_with_logits(out, labels) # Binary cross entropy loss 
        return loss
    
    def validation_step(self, batch):
        """
        Perform a validation step on the model
        """
        images = batch["image"]
        envs   = batch["env"]
        labels = batch["label"]

        device = next(self.parameters()).device
        images = images.to(device)
        envs = envs.to(device)
        labels = labels.to(device)

        if isinstance(self, HostImageryClimateModel):
            out = self(images, envs)
        elif isinstance(self, HostImageryClimateTopoModel):
            out = self(images, batch["topo"].to(device), envs)
        elif isinstance(self, HostImageryOnlyModel):
            out = self(images)
        elif isinstance(self, HostClimateOnlyModel):
            out = self(envs)
        elif isinstance(self, HostTopoOnlyModel):
            out = self(batch["topo"].to(device))
        else:
            raise NotImplementedError("Unknown model type for validation_step")

        loss = F.binary_cross_entropy_with_logits(out, labels)
        preds = (torch.sigmoid(out) > 0.5).float() # Binary sigmoid for predictions with threshold 0.5
        acc = (preds == labels).float().mean()
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
    
    def validation_epoch_end(self, outputs):
        """
        Aggregate validation results at the end of an epoch
        """
        loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        acc = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'val_loss': loss.item(), 'val_acc': acc.item()}
    
    def epoch_end(self, epoch, result):
        """
        Print the results at the end of an epoch
        """
        print(f"Epoch [{epoch+1}], train_loss: {result['train_loss']:.4f}, val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")


def get_resnet_model(pretrained=True, in_channels=4):
    """
    Create a ResNet model that accepts 4-channel input (NAIP RGB + NIR)
    """
    # Load the ResNet18 model with pretrained weights
    # Note: ResNet18 is commonly used in deep-sdms, using deeper models might not be necessary
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    # Modify the first convolution layer to accept 4 channels instead of 3 (RBG + NIR for NAIP)
    original_conv = model.conv1
    model.conv1 = nn.Conv2d(in_channels=in_channels,
                            out_channels=original_conv.out_channels,
                            kernel_size=original_conv.kernel_size,
                            stride=original_conv.stride,
                            padding=original_conv.padding,
                            bias=original_conv.bias is not None)
    
    # Initalize the new conv layer weights for the 4th channel as the mean of the first 3 channels
    with torch.no_grad():
        model.conv1.weight[:, :3, :, :] = original_conv.weight
        if in_channels >= 4:
            model.conv1.weight[:, 3, :, :] = original_conv.weight.mean(dim=1)
        if in_channels > 4:
            for c in range(4, in_channels):
                model.conv1.weight[:, c, :, :] = original_conv.weight.mean(dim=1)

    return model


class HostImageryClimateModel(HostImageClimateModelBase):
    """
    Inherits from HostImageClimateModelBase and combines NAIP imagery with environmental variables
    to predict the presence of a species.
    Args:
        num_env_features (int): Number of environmental features.
        hidden_dim (int): Dimension of the hidden layer in the classifier.
    """
    def __init__(self, num_env_features, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.resnet = get_resnet_model(pretrained=True)
        self.resnet.fc = nn.Identity() # Remove the final fully connected layer

        # MLP 
        self.climate_mlp = nn.Sequential(
            nn.Linear(num_env_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 2000, hidden_dim),  # 512 from Resnet18 or 2048 from ResNet50 + 2000 from climate features
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification logits
            # nn.Sigmoid()  # Output probability between 0 and 1
        )

        # A Shallower MLP alternative has lower performance
        # self.climate_mlp = nn.Sequential(
        #     nn.Linear(num_env_features, 128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128),
        #     nn.Linear(128, 64),
        #     nn.ReLU()
        # )

        # self.classifier = nn.Sequential(
        #     nn.Linear(512 + 64, hidden_dim),  # 512 from Resnet18 or 2048 from ResNet50 + 64 from climate features
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_dim, 1),  # Binary classification
        # )

    def forward(self, img, env):
        img_feat = self.resnet(img)  # Shape: (batch_size, 512) for ResNet18
        env_feat = self.climate_mlp(env)  # Shape: (batch_size, 64)
        fused = torch.cat((img_feat, env_feat), dim=1)
        out = self.classifier(fused) # Shape: (batch_size, 1)
        return out.squeeze(1) # Return shape (batch_size,)


class HostImageryOnlyModel(HostImageClimateModelBase):
    """
    Inherits from HostImageClimateModelBase and uses only NAIP imagery to predict the presence of a species.
    Args:
        hidden_dim (int): Dimension of the hidden layer in the classifier.
    """
    def __init__(self, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.resnet = get_resnet_model(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        self.classifier = nn.Sequential(
            nn.Linear(512, hidden_dim),  # 512 from Resnet18 or 2048 from ResNet50
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification
            # nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, img):
        img_feat = self.resnet(img)  # Shape: (batch_size, 512) for ResNet18
        out = self.classifier(img_feat) # Shape: (batch_size, 1)
        return out.squeeze(1) # Return shape (batch_size,)


class HostClimateOnlyModel(HostImageClimateModelBase):
    """
    Inherits from HostImageClimateModelBase and uses only environmental variables to predict the presence of a species.
    Args:
        num_env_features (int): Number of environmental features.
        hidden_dim (int): Dimension of the hidden layer in the classifier.
    """
    def __init__(self, num_env_features, hidden_dim=256, dropout=0.25):
        super().__init__()
        
        # MLP 
        self.climate_mlp = nn.Sequential(
            nn.Linear(num_env_features, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000),
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.BatchNorm1d(2000)
        )

        self.classifier = nn.Sequential(
            nn.Linear(2000, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # Binary classification
            # nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, env):
        env_feat = self.climate_mlp(env)  # Shape: (batch_size, 64)
        out = self.classifier(env_feat) # Shape: (batch_size, 1)
        return out.squeeze(1) # Return shape (batch_size,)



class HostImageryClimateTopoModel(HostImageClimateModelBase):
    def __init__(self, num_env_features, naip_channels=4, topo_channels=4, topo_features=128, env_features=128, hidden_features=256, dropout=0.25):
        super().__init__()
        self.image_resnet = get_resnet_model(pretrained=True, in_channels=naip_channels)
        self.image_resnet.fc = nn.Identity()
        self.topo_resnet = get_resnet_model(pretrained=True, in_channels=topo_channels)
        self.topo_resnet.fc = nn.Identity()
        self.topo_proj = nn.Sequential(nn.Linear(512, topo_features), nn.ReLU(), nn.BatchNorm1d(topo_features))
        self.climate_mlp = nn.Sequential(nn.Linear(num_env_features, env_features), nn.ReLU(), nn.BatchNorm1d(env_features))
        self.classifier = nn.Sequential(nn.Linear(512 + topo_features + env_features, hidden_features), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_features, 1))

    def forward(self, img, topo, env):
        x1 = self.image_resnet(img)
        x2 = self.topo_proj(self.topo_resnet(topo))
        x3 = self.climate_mlp(env)
        out = self.classifier(torch.cat([x1, x2, x3], dim=1))
        return out.squeeze(1)


# Backward-compatible alias
HostNAIPTopoClimateModel = HostImageryClimateTopoModel


class HostTopoOnlyModel(HostImageClimateModelBase):
    """Topography-only ResNet18 binary classifier."""
    def __init__(self, topo_channels=4, hidden_dim=256, dropout=0.25):
        super().__init__()
        self.topo_resnet = get_resnet_model(pretrained=True, in_channels=topo_channels)
        self.topo_resnet.fc = nn.Identity()
        self.classifier = nn.Sequential(nn.Linear(512, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(self, topo):
        feat = self.topo_resnet(topo)
        return self.classifier(feat).squeeze(1)

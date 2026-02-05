import torch
import torch.nn as nn
import torchvision.models as models
from src.config import Config

class ResNetBackbone(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNetBackbone, self).__init__()
        
        # 1. Dynamic Backbone Selection
        backbone_name = Config.BACKBONE.lower()
        weights = 'DEFAULT' if pretrained else None
        
        if backbone_name == "resnet18":
            self.resnet = models.resnet18(weights=weights)
            expansion = 1
        elif backbone_name == "resnet34":
            self.resnet = models.resnet34(weights=weights)
            expansion = 1
        elif backbone_name == "resnet50":
            self.resnet = models.resnet50(weights=weights)
            expansion = 4 # ResNet50 block output is 4x input
        elif backbone_name == "wide_resnet50_2":
            self.resnet = models.wide_resnet50_2(weights=weights)
            expansion = 4
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 2. Adaptive First Layer (CIFAR vs ImageNet)
        if Config.IMAGE_SIZE <= 64:
            # === CIFAR (32x32) Modification ===
            # Use 3x3 kernel, stride=1, padding=1, remove maxpool
            # Note: Conv1 input channels always 3->64 for these ResNets
            self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.resnet.maxpool = nn.Identity()
        else:
            # === ImageNet (224x224) Standard ===
            # Keep original 7x7 conv, stride=2, maxpool
            pass

        # 3. Compute Feature Dimension
        # ResNet18/34: 512, ResNet50/Wide: 2048
        self.num_features = 512 * expansion
        
        # 4. Feature Normalization & FC
        self.bn = nn.BatchNorm1d(self.num_features)
        self.resnet.fc = nn.Linear(self.num_features, num_classes)
        
    def forward(self, x):
        # Standard ResNet forward until layer4
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        # GAP
        out = self.resnet.avgpool(x)
        features = torch.flatten(out, 1)
        
        # BN (Feature Norm)
        features = self.bn(features)
        
        logits = self.resnet.fc(features)
        
        # Return features before BN or after?
        # Usually for metric learning we want the normalized features that were used for classification.
        # But for OT (SPP) we need spatial features.
        # Wait, SPP uses 'x' (before GAP).
        # The 'features' returned here are usually used for visualization or simple metric.
        # The OT module uses the 'x' (spatial features).
        # However, Center Loss needs the flattened features 'features'.
        
        return x, features, logits

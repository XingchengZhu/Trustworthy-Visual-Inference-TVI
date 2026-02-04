import torch
import torch.nn as nn
from torchvision.models import resnet18

class ResNet18Backbone(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(ResNet18Backbone, self).__init__()
        # Use simple weights param for compatibility if pretrained=True, though we likely train from scratch or load state_dict
        # For torch 2.0+, weights='DEFAULT' is preferred but 'pretrained=True' is deprecated. 
        # We will use weights=None for scratch or manually load if needed.
        # Given "pretrained" param here is legacy-like, let's just stick to default initialization (scratch) 
        # unless user wants pretrained. For CIFAR-10, ImageNet weights are okay but size mismatch needs handling if not careful.
        # But here we will train from scratch on CIFAR-10 usuallly, or fine-tune.
        # However, standard resnet18 expects 224x224. CIFAR is 32x32. We need to modify the first conv.
        
        self.resnet = resnet18(weights='DEFAULT' if pretrained else None)
        
        # Modify the first conv layer for CIFAR-10 (3 channels, 32x32)
        # Original: nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # CIFAR version usually uses 3x3 kernel, stride 1, padding 1 to keep spatial dim
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity() # Remove maxpool to preserve spatial dim (32 -> 16 -> 8 -> 4)
        
        self.num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.num_features, num_classes)
        
    def forward(self, x):
        # We need to extract features before the final FC and GAP
        # Standard implementation of ResNet forward:
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        
        features = x # Shape: (B, 512, 4, 4) if 32x32 input and maxpool removed
        
        # GAP and FC
        out = self.resnet.avgpool(features)
        out = torch.flatten(out, 1)
        logits = self.resnet.fc(out)
        
        return features, logits

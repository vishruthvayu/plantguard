# import torch
# import torch.nn as nn

# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()
        
#         # Feature extractor - conv layers
#         self.features = nn.Sequential(
#             # Block 1: Conv → ReLU → MaxPool
#             nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             # Block 2: Conv → ReLU → MaxPool
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
            
#             # Block 3
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#         )
        
#         # Classifier - fully connected layers
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(128 * 28 * 28, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )
    
#     def forward(self, x):
#         x = self.features(x)
#         x = self.classifier(x)
#         return x
# we'll not use this simple CNN model, instead we'll use a pretrained model from torchvision.models and fine-tune it for our plant disease classification task.

import torch 
import torch.nn as nn
from  torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

def get_model(num_classes,pretrained=True):
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features,num_classes)
    return model
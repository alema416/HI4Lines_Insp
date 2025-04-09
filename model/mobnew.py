import torch
import torch.nn as nn
import torchvision.models as models

class MobileNetV3Wrapper(nn.Module):
    def __init__(self, num_classes=2):
        super(MobileNetV3Wrapper, self).__init__()
        self.model = models.mobilenet_v3_small(pretrained=True)  # Load pretrained model
        self.model.classifier[-1] = nn.Linear(self.model.classifier[-1].in_features, num_classes)  # Modify last layer

    def forward(self, x):
        return self.model(x)

def mobilenet(num_classes=2):
    return MobileNetV3Wrapper(num_classes)

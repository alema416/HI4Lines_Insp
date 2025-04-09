import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization

class MinimalEdgeTPUModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MinimalEdgeTPUModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.act1 = nn.ReLU6(inplace=True)  # Edge TPU supports only ReLU/ReLU6
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.act2 = nn.ReLU6(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Create Model
def get_model():
    model = MinimalEdgeTPUModel()
    # Convert to a Quantization-Aware Model
    model.qconfig = torch.quantization.get_default_qat_qconfig("fbgemm")
    torch.quantization.prepare_qat(model, inplace=True)
    return model

import torch.nn as nn
import torchvision.models as models


class CustomMobileNetV2(nn.Module):
    def __init__(self, num_classes=4):
        super(CustomMobileNetV2, self).__init__()
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier[1] = nn.Linear(1280, num_classes)

    def forward(self, x):
        return self.mobilenet(x)

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V2_Weights

class FishClassifier(nn.Module):
    def __init__(self, num_classes=8):
        super(FishClassifier, self).__init__()
        # Sử dụng MobileNetV2 thay vì ResNet
        self.mobilenetv2 = models.mobilenet_v2(weights = MobileNet_V2_Weights.IMAGENET1K_V1)
        
        # Thay thế Fully Connected Layer cuối cùng để có đúng số lượng lớp
        self.mobilenetv2.classifier[1] = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(self.mobilenetv2.classifier[1].in_features, num_classes)
)
        
    def forward(self, x):
        return self.mobilenetv2(x)
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


class TurnSpeedCNN(nn.Module):
    def __init__(self):
        super(TurnSpeedCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # 全局平均池化
        )
        self.fc = nn.Linear(64, 1)  # 输出Turn Speed

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def image_preprocessor(image, device):
    # image preprocessing
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Resize((50, 320)),  # 保持原尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet均值
    ])
    return transform(image).unsqueeze(0).to(device)

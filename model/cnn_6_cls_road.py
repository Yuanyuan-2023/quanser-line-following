import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import os
from PIL import Image


# ✅ 1️⃣ 重新定义 CNN 结构（改成 5 类分类）
class QBotCNN(torch.nn.Module):
    def __init__(self):
        super(QBotCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(2, 2)  # 2x2 池化
        self.fc1 = torch.nn.Linear(64 * 20 * 40, 128)  # 计算 64 * 20 * 40
        self.fc2 = torch.nn.Linear(128, 6)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = torch.flatten(x, start_dim=1)  # 展平
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def image_preprocessor(image, device="cpu"):
    # image preprocessing
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    # 图像预处理  
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((160, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0).to(device)


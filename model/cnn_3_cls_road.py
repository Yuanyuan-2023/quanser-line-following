import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


class ImageClassifier3Class(nn.Module):
    def __init__(self, in_channels=1):
        super(ImageClassifier3Class, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 25x160

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 12x80

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # 1x1
        )
        self.fc = nn.Linear(64, 3)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        return self.fc(x)

def image_preprocessor(image, device):
    # image preprocessing
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)

    transform = transforms.Compose([
        transforms.Grayscale(),                 
        transforms.Resize((64, 64)),            
        transforms.ToTensor(),              
    ])
    return transform(image).unsqueeze(0).to(device)


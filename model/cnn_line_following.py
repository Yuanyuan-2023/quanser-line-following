import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import cv2
import os

# # ✅ 设置数据目录
# DATA_DIR = r""
# CSV_PATH = os.path.join(DATA_DIR, "labels.csv")

# # ✅ 读取 CSV 并检查数据是否合理
# df = pd.read_csv(CSV_PATH)

# if "image_filename" not in df.columns or "normalized_label" not in df.columns:
#     raise ValueError("❌ CSV 文件缺少 `image_filename` 或 `normalized_label` 列")

# df.dropna(inplace=True)  # 移除 NaN 行
# df['image_path'] = df['image_filename'].apply(lambda x: os.path.join(DATA_DIR, x))

# # 🚨 确保 `normalized_label` 在 [-1, 1] 之间
# df['normalized_label'] = df['normalized_label'].clip(-1, 1)

# # ✅ 图像预处理
# transform = transforms.Compose([
#     transforms.ToPILImage(),
#     transforms.Resize((160, 320)),
#     transforms.Grayscale(num_output_channels=1),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# ✅ 创建自定义 Dataset
class QBotDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = row['image_path']
        label = torch.tensor(float(row['normalized_label']), dtype=torch.float32)

        # 🚨 确保图像文件存在
        if not os.path.exists(img_path):
            print(f"❌ 文件不存在: {img_path}, 跳过此样本")
            return None

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"❌ OpenCV 无法读取图像: {img_path}, 跳过此样本")
            return None
        
        if self.transform:
            img = self.transform(img)

        return img, label

# ✅ 定义 CNN 模型
class QBotCNN(nn.Module):
    def __init__(self):
        super(QBotCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 20 * 40, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1)  # 预测偏移量
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)

        # 🚨 防止 `NaN` 传播
        x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
        x = torch.clamp(x, min=-1, max=1)  # 限制范围 [-1, 1]
        
        return x

def image_preprocessor(image, device="cpu"):
    # 图像预处理  
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((160, 320)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return transform(image).unsqueeze(0).to(device)

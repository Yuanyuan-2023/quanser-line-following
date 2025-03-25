

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
from PIL import Image


class LineDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
        
        self.img_labels = [line.strip().split() for line in lines]
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = f"{self.img_labels[idx][0]}"
        if not os.path.exists(img_path):
            print(f"Path not exists: {img_path}")
            return 
        
        image = Image.open(img_path).convert("L")  # 假设是灰度图
        label = float(self.img_labels[idx][1])
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.float32)


class LineFollowerNet(nn.Module):
    def __init__(self):
        super(LineFollowerNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 1) 

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def regression_accuracy(y_pred, y_true, tolerance=0.05):
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()
    correct = sum(abs(p - t) <= tolerance for p, t in zip(y_pred, y_true))
    return correct / len(y_true)


def valid(model, dataloader, criterion, tolerance=0.05):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0

    with torch.no_grad():
        for images, labels in dataloader:
   
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)

            acc = regression_accuracy(outputs, labels, tolerance)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc = total_acc / total_samples

    return avg_loss, avg_acc


def train(train_dataloader, valid_dataloader):

    # Init
    model = LineFollowerNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    total_loss = 0.0
    total_acc = 0.0
    total_samples = 0
    best_acc = 0
    for epoch in range(10):
        for images, labels in train_dataloader:
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            acc = regression_accuracy(outputs.squeeze(), labels)
            
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_acc += acc * batch_size
            total_samples += batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        train_loss = total_loss / total_samples
        train_acc = total_acc / total_samples
        
        val_loss, val_acc = valid(model, valid_dataloader, criterion, tolerance=0.05)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Trian Accuracy: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f"ckpt/{val_acc:.4f}_line_follower_nn.pth")


def image_preprocessor(image, device):
    # image preprocessing
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    image = image.convert("L")

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0).to(device)


def main():
    train_path = "label/train.txt"
    valid_path = "label/valid.txt"


    transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                ])
   
    train_dataset = LineDataset(train_path,  transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    valid_dataset = LineDataset(valid_path,  transform=transform)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    train(train_dataloader, valid_dataloader)


if __name__ == "__main__":
    main()
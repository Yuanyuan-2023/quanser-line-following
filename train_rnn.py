import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import os
from PIL import Image
import numpy as np


class CNNEncoder(nn.Module):
    def __init__(self):
        super(CNNEncoder, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        return features.view(x.size(0), -1) 


class CNNRNNLineFollower(nn.Module):
    def __init__(self, cnn_embed_dim=512, hidden_dim=128):
        super(CNNRNNLineFollower, self).__init__()
        self.cnn = CNNEncoder()
        self.rnn = nn.LSTM(input_size=cnn_embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x_seq):
        B, T, C, H, W = x_seq.shape
        x_seq = x_seq.view(-1, C, H, W)  # (B*T, C, H, W)

        cnn_features = self.cnn(x_seq)         # (B*T, 512)
        cnn_features = cnn_features.view(B, T, -1)  # (B, T, 512)

        rnn_out, _ = self.rnn(cnn_features)    # (B, T, hidden_dim)
        out = self.fc(rnn_out[:, -1]) 
        return out


class SequenceLineDataset(Dataset):
    def __init__(self, txt_file, transform=None, seq_len=3):
        with open(txt_file, 'r') as f:
            self.data = [line.strip().split() for line in f if line.strip()]
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len + 1

    def __getitem__(self, idx):
        images = []
        for i in range(self.seq_len):
            img_path = self.data[idx + i][0]
            img = Image.open(img_path).convert("L")
            if self.transform:
                img = self.transform(img)
            images.append(img)

        label = float(self.data[idx + self.seq_len - 1][1])
        image_seq = torch.stack(images) 
        return image_seq, torch.tensor(label, dtype=torch.float32)


def evaluate_regression(model, dataloader, device, tolerance=0.05):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for image_seqs, labels in dataloader:
            image_seqs = image_seqs.to(device)
            labels = labels.cpu().numpy()
            outputs = model(image_seqs).squeeze().cpu().numpy()

            y_true.extend(labels)
            y_pred.extend(outputs)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    acc = np.mean(np.abs(y_true - y_pred) <= tolerance)

    return acc, mse, mae, r2


def train(train_loader, valid_loader, device="cpu", train_epoch=10, save_dir="ckpt"):
    os.makedirs(save_dir, exist_ok=True)
    model = CNNRNNLineFollower().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    best_acc = 0.0

    for epoch in range(train_epoch):
        model.train()
        total_loss = 0

        for image_seqs, labels in train_loader:
            image_seqs, labels = image_seqs.to(device), labels.to(device)
            outputs = model(image_seqs).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        acc, mse, mae, r2 = evaluate_regression(model, valid_loader, device)

        print(f"Epoch {epoch+1}/{train_epoch} Train Loss: {total_loss / len(train_loader):.6f}")
        print(f"Val MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}, Accuracy(±0.05): {acc*100:.2f}%")

        # ✅ 保存最优模型
        if acc > best_acc:
            best_acc = acc
            model_path = os.path.join(save_dir, f"{acc:.4f}_line_follower_rnn.pth")
            torch.save(model.state_dict(), model_path)


def main():
    train_path = "label/train.txt"
    valid_path = "label/valid.txt"

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])

    train_dataset = SequenceLineDataset(train_path, transform=transform)
    valid_dataset = SequenceLineDataset(valid_path, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train(train_loader, valid_loader, device=device, train_epoch=20, save_dir="ckpt")


if __name__ == "__main__":
    main()
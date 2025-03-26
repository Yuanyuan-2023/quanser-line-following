import cv2
import torch
from collections import deque
from model import *

# ====== Constants and Buffers ======
SEQ_LEN = 3  # Number of frames for RNN sequence
frame_buffer = deque(maxlen=SEQ_LEN)  # Frame queue for RNN input
device = torch.device("cpu")  # Change to "cuda" if GPU is available

# ====== Load Pretrained Models ======

# RNN-based line follower
rnn_path = "ckpt/0.8797_line_follower_rnn.pth"
rnn = CNNRNNLineFollower().to(device)
rnn.load_state_dict(torch.load(rnn_path, map_location=device))
rnn.eval()

# CNN-based line follower v1
cnn_path = "ckpt/qbot_line_follower_cnn.pth"
cnn = CNNLineFollower().to(device)
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

# CNN-based line follower v2
cnn2_path = "ckpt/0.8988_line_follower_cnn.pth"
cnn2 = CNNLineFollower2().to(device)
cnn2.load_state_dict(torch.load(cnn2_path, map_location=device))
cnn2.eval()

# CNN-based turn speed
tspd_path = "ckpt/best_turn_speed_model.pth"
tspd = TurnSpeedCNN().to(device)
tspd.load_state_dict(torch.load(tspd_path, map_location=device))
tspd.eval()

# RESNET-based offset
col_path = "ckpt/best_resnet_model_mps.pth"
col = load_resnet18(col_path, 1)
col.eval()

# Road classifier (5-class)
cnn_5_cls_path = "ckpt/classify_road_5_cnn.pth"
cnn_5_cls = CNN5ClassifyRoad().to(device)
cnn_5_cls.load_state_dict(torch.load(cnn_5_cls_path, map_location=device))
cnn_5_cls.eval()
CLASS_NAME_5 = ["Straight", "Turn", "Cross", "T-Junction", "Curve"]

# Road classifier (6-class)
cnn_6_cls_path = "ckpt/qbot_line_follower_cnn_6_classes.pth"
cnn_6_cls = CNN6ClassifyRoad().to(device)
cnn_6_cls.load_state_dict(torch.load(cnn_6_cls_path, map_location=device))
cnn_6_cls.eval()
CLASS_NAME_6 = ["Straight", "Turn", "Cross", "T-Junction", "Curve", "Black"]

# Road classifier (3-class)
cnn_3_cls_path2 = "ckpt/0.9846_cls_cnn.pth"
cnn_3_cls = CNN3ClassifyRoad().to(device)
cnn_3_cls.load_state_dict(torch.load(cnn_3_cls_path2, map_location=device))
cnn_3_cls.eval()
CLASS_NAME_3 = ["Blank", "Single Line", "Multiple Lines"]

# ====== Load Input Image ======
image = cv2.imread("../datasets/qbot_binary_images/binary_image_0.png")

# ====== Preprocess for Different Models ======
image_3_cls = load_3_cls_data(image, device)
image_5_cls = load_5_cls_data(image, device)
image_6_cls = load_6_cls_data(image, device)
image_cnn = load_cnn_data(image, device)
image_cnn2 = load_cnn_data2(image, device)
image_rnn = load_rnn_data(image, device)
image_resnet = load_resnet_data(image, device)
image_tspd = load_turnspd_data(image, device)

# ====== Make Predictions ======
# Classification (5-class and 3-class)
pred_3_cls = torch.argmax(cnn_3_cls(image_3_cls)[0]).item()
pred_5_cls = torch.argmax(cnn_5_cls(image_5_cls)[0]).item()
pred_6_cls = torch.argmax(cnn_6_cls(image_6_cls)[0]).item()

# CNN regression outputs
pred_cnn = cnn(image_cnn).item()
pred_cnn2 = cnn2(image_cnn2).item()
pred_tspd = tspd(image_tspd).item()

# ====== Prepare RNN Sequence Prediction ======
if len(frame_buffer) < SEQ_LEN:
    for _ in range(SEQ_LEN - len(frame_buffer)):
        frame_buffer.append(image_rnn.clone())
frame_buffer.append(image_rnn)

# Stack frames into (1, SEQ_LEN, C, H, W)
seq_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
pred_rnn = rnn(seq_tensor).item()

# ====== Prepare RESNET Prediction ======
pred_col = col(image_resnet).item()

# ====== Print Prediction Results ======
print("CNN1 Offset Prediction:", pred_cnn)
print("CNN2 Offset Prediction:", pred_cnn2)

print("CNN Turn Speed Prediction:", pred_tspd)

print("RNN offset Prediction:", pred_rnn)
print("ResNet Col Prediction:", pred_col)

print("5-Class Classification:", pred_5_cls, "-", CLASS_NAME_5[pred_5_cls])
print("6-Class Classification:", pred_6_cls, "-", CLASS_NAME_6[pred_6_cls])
print("3-Class Classification:", pred_3_cls, "-", CLASS_NAME_3[pred_3_cls])
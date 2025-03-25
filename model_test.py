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

# Road classifier (5-class)
cnn_classify_path = "ckpt/classify_road_5_cnn.pth"
cnn_classify = CNNClassifyRoad().to(device)
cnn_classify.load_state_dict(torch.load(cnn_classify_path, map_location=device))
cnn_classify.eval()
CLASS_NAME_5 = ["Straight", "Turn", "Cross", "T-Junction", "Curve"]

# Road classifier (3-class)
cnn_classify_path2 = "ckpt/0.9846_cls_cnn.pth"
cnn_classify2 = CNNClassifyRoad2().to(device)
cnn_classify2.load_state_dict(torch.load(cnn_classify_path2, map_location=device))
cnn_classify2.eval()
CLASS_NAME_3 = ["Blank", "Single Line", "Multiple Lines"]

# ====== Load Input Image ======
image = cv2.imread("../datasets/qbot_binary_images/binary_image_0.png")

# ====== Preprocess for Different Models ======
image_classify = load_classify_data(image, device)
image_classify2 = load_classify_data2(image, device)
image_cnn = load_cnn_data(image, device)
image_cnn2 = load_cnn_data2(image, device)
image_rnn = load_rnn_data(image, device)

# ====== Make Predictions ======
# Classification (5-class and 3-class)
pred_classify = torch.argmax(cnn_classify(image_classify)[0]).item()
pred_classify2 = torch.argmax(cnn_classify2(image_classify2)[0]).item()

# CNN regression outputs
pred_cnn = cnn(image_cnn).item()
pred_cnn2 = cnn2(image_cnn2).item()

# ====== Prepare RNN Sequence Prediction ======
if len(frame_buffer) < SEQ_LEN:
    for _ in range(SEQ_LEN - len(frame_buffer)):
        frame_buffer.append(image_rnn.clone())
frame_buffer.append(image_rnn)

# Stack frames into (1, SEQ_LEN, C, H, W)
seq_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(device)
pred_rnn = rnn(seq_tensor).item()

# ====== Print Prediction Results ======
print("CNN1 Prediction:", pred_cnn)
print("CNN2 Prediction:", pred_cnn2)
print("RNN  Prediction:", pred_rnn)
print("5-Class Classification:", pred_classify, "-", CLASS_NAME_5[pred_classify])
print("3-Class Classification:", pred_classify2, "-", CLASS_NAME_3[pred_classify2])
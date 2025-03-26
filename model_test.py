import cv2
import torch
from collections import deque
from model import *

# ====== Configuration ======
SEQ_LEN = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ====== Load Models ======
rnn = load_model(CNNRNNLineFollower, "ckpt/0.8797_line_follower_rnn.pth")
cnn  = load_model(CNNLineFollower, "ckpt/qbot_line_follower_cnn.pth")
cnn2 = load_model(CNNLineFollower2, "ckpt/0.8988_line_follower_cnn.pth")
tspd = load_model(TurnSpeedCNN, "ckpt/best_turn_speed_model.pth")
col  = load_resnet18("ckpt/best_resnet_model_mps.pth", output_dim=1)

cnn_5_cls = load_model(CNN5ClassifyRoad, "ckpt/classify_road_5_cnn.pth")
cnn_6_cls = load_model(CNN6ClassifyRoad, "ckpt/qbot_line_follower_cnn_6_classes.pth")
cnn_3_cls = load_model(CNN3ClassifyRoad, "ckpt/0.9846_cls_cnn.pth")

CLASS_NAME_3 = ["Blank", "Single Line", "Multiple Lines"]
CLASS_NAME_5 = ["Straight", "Turn", "Cross", "T-Junction", "Curve"]
CLASS_NAME_6 = ["Straight", "Turn", "Cross", "T-Junction", "Curve", "Black"]

# ====== Load and Preprocess Image ======
IMAGE_PATH = "../datasets/qbot_binary_images/binary_image_0.png"
image = cv2.imread(IMAGE_PATH)
data = preprocess_all(image, DEVICE)

# ====== Frame Buffer for RNN ======
frame_buffer = deque(maxlen=SEQ_LEN)
if len(frame_buffer) < SEQ_LEN:
    for _ in range(SEQ_LEN - len(frame_buffer)):
        frame_buffer.append(data["rnn"].clone())
frame_buffer.append(data["rnn"])
seq_tensor = torch.stack(list(frame_buffer)).unsqueeze(0).to(DEVICE)

# ====== Inference ======
pred_3_cls = torch.argmax(cnn_3_cls(data["3_cls"])[0]).item()
pred_5_cls = torch.argmax(cnn_5_cls(data["5_cls"])[0]).item()
pred_6_cls = torch.argmax(cnn_6_cls(data["6_cls"])[0]).item()

pred_cnn  = cnn(data["cnn"]).item()
pred_cnn2 = cnn2(data["cnn2"]).item()
pred_tspd = tspd(data["tspd"]).item()
pred_rnn  = rnn(seq_tensor).item()
pred_col  = col(data["resnet"]).item()

# ====== Output Results ======
print("\n===== Prediction Results =====")
print(f"CNN1 Offset Prediction: {pred_cnn:.4f}")
print(f"CNN2 Offset Prediction: {pred_cnn2:.4f}")
print(f"Turn Speed Prediction: {pred_tspd:.4f}")
print(f"RNN Offset Prediction: {pred_rnn:.4f}")
print(f"ResNet Offset Prediction: {pred_col:.4f}")

print(f"3-Class Road Type: {pred_3_cls} - {CLASS_NAME_3[pred_3_cls]}")
print(f"5-Class Road Type: {pred_5_cls} - {CLASS_NAME_5[pred_5_cls]}")
print(f"6-Class Road Type: {pred_6_cls} - {CLASS_NAME_6[pred_6_cls]}")
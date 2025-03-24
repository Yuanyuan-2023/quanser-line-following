import cv2
import torch
from collections import deque
from model import *

SEQ_LEN = 3
frame_buffer = deque(maxlen=SEQ_LEN)

device = torch.device("cpu")

rnn_path = "ckpt/0.8797_line_follower_rnn.pth"

rnn = CNNRNNLineFollower().to(device)
rnn.load_state_dict(torch.load(rnn_path, map_location=device))
rnn.eval()

cnn_path = "ckpt/qbot_line_follower_cnn.pth"
cnn = CNNLineFollower().to(device)
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

cnn_classify_path = "ckpt/classify_road_5_cnn.pth"
cnn_classify = CNNClassifyRoad().to(device)
cnn_classify.load_state_dict(torch.load(cnn_classify_path, map_location=device))
cnn_classify.eval()
CLASS_NAME = ["直线", "转弯", "十字", "T型", "小弯"]


image = cv2.imread("../datasets/qbot_binary_images/binary_image_0.png")

image_classify = load_classify_data(image, device)
image_cnn = load_cnn_data(image, device)
image_rnn = load_rnn_data(image, device)

pred_classify = torch.argmax(cnn_classify(image_classify)[0]).item()
pred_cnn = cnn(image_cnn).item()



if len(frame_buffer) < SEQ_LEN:
    for _ in range(SEQ_LEN - len(frame_buffer)):
        frame_buffer.append(image_rnn.clone())
frame_buffer.append(image_rnn)


seq_tensor = torch.stack(list(frame_buffer))
seq_tensor = seq_tensor.unsqueeze(0).to(device)
pred_rnn = rnn(seq_tensor).item()


print(pred_cnn, pred_classify, image_rnn.shape, pred_rnn)

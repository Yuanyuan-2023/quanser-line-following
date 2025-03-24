#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following---------------------------#
#-----------------------------------------------------------------------------#

# Imports
import time
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from collections import deque
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tk 后端（比 Qt 更稳定）
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, QBotPlatformCSICamera
from qbot_platform_functions import QBPVision, LineFollowingMetrics
from qlabs_setup import setup
from train_rnn import CNNRNNLineFollower
from PIL import Image

# Section A - Setup
metrics = LineFollowingMetrics()
setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)

ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype=np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
startTime = time.time()

SEQ_LEN = 3
frame_buffer = deque(maxlen=SEQ_LEN)

def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

# --------------- 1. 加载 RNN 序列模型 ---------------
device = torch.device("cpu")
model = CNNRNNLineFollower().to(device)
model.load_state_dict(torch.load("0.8797_line_follower_rnn.pth", map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

lineFollow = False
prev_k7 = False

try:
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    keyboard = Keyboard()
    vision = QBPVision()

    startTime = time.time()
    time.sleep(0.5)

    while noKill and not endFlag:
        t = elapsed_time()

        if not keyboard.read():
            continue

        arm = keyboard.k_space
        keyboardComand = keyboard.bodyCmd
        k7_pressed = keyboard.k_7
        if keyboard.k_u:
            noKill = False

        if k7_pressed and not prev_k7:
            lineFollow = not lineFollow
            print(f"切换模式: {'自动循线' if lineFollow else '手动控制'}")
            time.sleep(0.2)
        prev_k7 = k7_pressed

        if not lineFollow:
            commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            print(f"[手动模式] 速度指令: {commands}")
            myQBot.read_write_std(timestamp=elapsed_time(), arm=arm, commands=commands)
            continue

        newHIL = myQBot.read_write_std(timestamp=elapsed_time(), arm=arm, commands=commands)
        if newHIL:
            timeHIL = time.time()
            newDownCam = downCam.read()

            if newDownCam:
                counterDown += 1

                undistorted = vision.df_camera_undistort(downCam.imageData)
                binary = Image.fromarray(undistorted)
                gray_sm = cv2.resize(undistorted, (320, 200))
                binary0 = vision.subselect_and_threshold(image=gray_sm, rowStart=50, rowEnd=100,
                                                        minThreshold=180, maxThreshold=255)

                debug_image = cv2.cvtColor(binary0, cv2.COLOR_GRAY2BGR)
                

                frame_tensor = transform(binary)
                if len(frame_buffer) < SEQ_LEN:
                    for _ in range(SEQ_LEN - len(frame_buffer)):
                        frame_buffer.append(frame_tensor.clone())
                frame_buffer.append(frame_tensor)

                if len(frame_buffer) == SEQ_LEN:
                    seq_tensor = torch.stack(list(frame_buffer))
                    seq_tensor = seq_tensor.unsqueeze(0).to(device)

                    with torch.no_grad():
                        predicted_offset = model(seq_tensor).item()
                    print(f"🚀 预测偏移: {predicted_offset:.4f}")

                    cv2.putText(debug_image, f"Offset: {predicted_offset:.3f}", (10, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.imshow("Line Following", debug_image)
                    cv2.waitKey(1)

                    if predicted_offset is None or np.isnan(predicted_offset) or abs(predicted_offset) > 0.9:
                        print("⚠️ 偏移过大，后退调整")
                        forSpd = -0.1
                        turnSpd = 0.2 * (-1 if counterDown % 2 == 0 else 1)
                    else:
                        forSpd = 0.1
                        turnSpd = np.clip(predicted_offset * -0.5, -1, 1)

                    metrics.add_error(predicted_offset * 160)
                    commands = np.array([forSpd, turnSpd], dtype=np.float64)
                    print(f"[自动模式] 控制指令: {commands}")

            prevTimeHIL = timeHIL

    if not noKill:
        accuracy = metrics.calculate_accuracy()
        print(f"累计误差: {accuracy['Cumulative_Error']}")
        print(f"MAE: {accuracy['MAE']}")
        print(f"MSE: {accuracy['MSE']}")
        print(f"循线准确率: {accuracy['line_following_accuracy']:.2f}%")

except KeyboardInterrupt:
    print('User interrupted.')
finally:
    downCam.terminate()
    myQBot.terminate()
    keyboard.terminate()
    metrics.plot_errors(save_path="line_following_error.png")
    cv2.destroyAllWindows()
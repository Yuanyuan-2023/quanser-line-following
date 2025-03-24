# -------------------------------
# QBot CNN Line Following + Lidar Avoidance
# -------------------------------

import time
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, QBotPlatformCSICamera, QBotPlatformLidar
from qbot_platform_functions import QBPVision, QBPRanging, LineFollowingMetrics
from qlabs_setup import setup
from train_cnn import QBotCNN
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tk 后端（比 Qt 更稳定）
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

def elapsed_time():
    return time.time() - startTime

timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

# 加载 CNN 模型
device = torch.device("cpu")
model = QBotCNN().to(device)
model.load_state_dict(torch.load("./qbot_line_follower_cnn.pth", map_location=device))
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 320)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

lineFollow = False
prev_k7 = False

try:
    # 初始化硬件
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    lidar = QBotPlatformLidar()
    keyboard = Keyboard()
    vision = QBPVision()
    ranging = QBPRanging()
    startTime = time.time()
    time.sleep(1.5)

    # 主循环
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
            print(f"\n[切换模式] {'自动循线' if lineFollow else '手动控制'}")
            time.sleep(0.2)
        prev_k7 = k7_pressed

        if not lineFollow:
            commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            print(f"[手动模式] 指令: {commands}")
            myQBot.read_write_std(timestamp=time.time() - startTime, arm=arm, commands=commands)
            continue

        newHIL = myQBot.read_write_std(timestamp=time.time() - startTime, arm=arm, commands=commands)
        if newHIL:
            timeHIL = time.time()
            for i in range(3):
                newDownCam = downCam.read()
                if newDownCam:
                    break
                time.sleep(0.05)
            else:
                print("❌ 摄像头图像读取失败")
                continue           
            # Lidar reading (maximum 3 attempts)
            for i in range(3):
                newLidar = lidar.read()
                if newLidar and lidar.distances is not None and len(lidar.distances) > 0:
                    break
                time.sleep(0.05)  # Give it a moment to "catch its breath"
            else:
                print(" Invalid lidar frame, skipping this frame")
                continue

            
            rangesAdj, anglesAdj = ranging.adjust_and_subsample(lidar.distances, 
                                                         lidar.angles, 1260, 3)

            counterDown += 1
            undistorted = vision.df_camera_undistort(downCam.imageData)
            gray_sm = cv2.resize(undistorted, (320, 200))
            binary = vision.subselect_and_threshold(image=gray_sm, rowStart=50, rowEnd=100, minThreshold=180, maxThreshold=255)

            image_tensor = transform(binary).unsqueeze(0).to(device)

            with torch.no_grad():
                predicted_offset = model(image_tensor).item()
                print(f"CNN 偏移预测: {predicted_offset:.2f}")

            if predicted_offset is None or np.isnan(predicted_offset):
                forSpd = -0.1
                turnSpd = 0.2 * (-1 if counterDown % 2 == 0 else 1)
            else:
                # Use front window (±2°) calculated after angle correction of rangesAdj and anglesAdj）
                front_mask = np.logical_and(anglesAdj > -0.035, 
                                    anglesAdj < 0.035)  # about ±2°
                front_window = rangesAdj[front_mask]

# NaN protection & empty value protection
                if front_window.size == 0 or np.any(np.isnan(front_window)):
                    print("Radar window data is empty or contains NaN, skipping this frame")
                    continue

                # Calculate average distance
                front = np.mean(front_window)
                print(f"Average distance detected directly ahead:{front:.3f}m")

                 # 若距离小于阈值，则右转后恢复循线
                if front < 0.5:
                    print("T-junction detected: turning right")
                    forSpd = 0.05
                    turnSpd = -0.5

                 # Execute right turn for a period (non-blocking implementation)
                    turn_end_time = time.time() + 0.8
                    while time.time() < turn_end_time:
                        myQBot.read_write_std(timestamp=elapsed_time(), arm=arm,
                              commands=np.array([forSpd, turnSpd]))
                    print("Turn completed, returning to line-following mode")
                else:
                    forSpd = 0.15
                    turnSpd = np.clip(predicted_offset * -0.5, -1, 1)

            if predicted_offset is not None:
                metrics.add_error(predicted_offset * 160)

            commands = np.array([forSpd, turnSpd], dtype=np.float64)

        prevTimeHIL = timeHIL

    # 输出准确率报告
    if not noKill:
        accuracy = metrics.calculate_accuracy()
        print("\n========== 准确率报告 ==========")
        print(f"累计误差: {accuracy['Cumulative_Error']:.2f}")
        print(f"MAE: {accuracy['MAE']:.2f}")
        print(f"MSE: {accuracy['MSE']:.2f}")
        print(f"循线准确率: {accuracy['line_following_accuracy']:.2f}%")

except KeyboardInterrupt:
    print('用户中断程序。')

finally:
    downCam.terminate()
    if lidar is not None:
        lidar.terminate()
    myQBot.terminate()
    keyboard.terminate()
    metrics.plot_errors(save_path="line_following_error.png")

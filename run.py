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
from pal.products.qbot_platform import QBotPlatformDriver, Keyboard, QBotPlatformCSICamera, QBotPlatformLidar
from qbot_platform_functions import QBPVision,QBPRanging, LineFollowingMetrics
from qlabs_setup import setup
from model import *
from motion import get_motion_queque
import random

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
SEQ_LEN = 3
frame_buffer = deque(maxlen=SEQ_LEN)

device = torch.device("cpu")

rnn_path = "ckpt/0.8797_line_follower_rnn.pth"

rnn = CNNRNNLineFollower().to(device)
rnn.load_state_dict(torch.load(rnn_path, map_location=device))
rnn.eval()

# --------------- 2. 加载 CNN 模型 ---------------
cnn_path = "ckpt/qbot_line_follower_cnn.pth"
cnn = CNNLineFollower().to(device)
cnn.load_state_dict(torch.load(cnn_path, map_location=device))
cnn.eval()

cnn2_path = "ckpt/0.8988_line_follower_cnn.pth"
cnn2 = CNNLineFollower2().to(device)
cnn2.load_state_dict(torch.load(cnn2_path, map_location=device))
cnn2.eval()

# --------------- 3. 加载 CNN分类模型 ---------------
TURNSPD_LIST = []
cnn_classify_path = "ckpt/classify_road_5_cnn.pth"
cnn_classify = CNNClassifyRoad().to(device)
cnn_classify.load_state_dict(torch.load(cnn_classify_path, map_location=device))
cnn_classify.eval()
CLASS_NAME_5 = ["直线", "转弯", "十字", "T型", "小弯"]

# Road classifier (3-class)
cnn_classify_path2 = "ckpt/0.9846_cls_cnn.pth"
cnn_classify2 = CNNClassifyRoad2().to(device)
cnn_classify2.load_state_dict(torch.load(cnn_classify_path2, map_location=device))
cnn_classify2.eval()
CLASS_NAME_3 = ["Blank", "Single Line", "Multiple Lines"]

lineFollow = False
prev_k7 = False

try:
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    lidar = QBotPlatformLidar()
    keyboard = Keyboard()
    vision = QBPVision()
    ranging = QBPRanging()
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
            if newDownCam:
                counterDown += 1

                undistorted = vision.df_camera_undistort(downCam.imageData)
                gray_sm = cv2.resize(undistorted, (320, 200))
                binary = vision.subselect_and_threshold(image=gray_sm, rowStart=50, rowEnd=100,
                                                        minThreshold=180, maxThreshold=255)

                debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                
                image_cnn = load_cnn_data(binary, device)
                image_cnn2 = load_cnn_data2(binary, device)
                image_rnn = load_rnn_data(binary, device)
                
                pred_cnn = cnn(image_cnn).item()
                pred_cnn2 = cnn2(image_cnn2).item()
                if len(frame_buffer) < SEQ_LEN:
                    for _ in range(SEQ_LEN - len(frame_buffer)):
                        frame_buffer.append(image_rnn.clone())
                frame_buffer.append(image_rnn)
                seq_tensor = torch.stack(list(frame_buffer))
                seq_tensor = seq_tensor.unsqueeze(0).to(device)
                pred_rnn = rnn(seq_tensor).item()

                predicted_offset = pred_cnn2
                print(f"预测偏移: {predicted_offset:.4f} CNN: {pred_cnn:.4f}, CNN2:{pred_cnn2:.4f} RNN: {pred_rnn:.4f}")

                cv2.putText(debug_image, f"Offset: {predicted_offset:.3f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Line Following", debug_image)
                cv2.waitKey(1)

                image_classify = load_classify_data(binary, device)
                pred_classify_cnn = torch.argmax(cnn_classify(image_classify)[0]).item()
                road_class_5 = CLASS_NAME_5[pred_classify_cnn]

                image_classify2 = load_classify_data2(binary, device)
                pred_classify_cnn2 = torch.argmax(cnn_classify2(image_classify2)[0]).item()
                road_class_3 = CLASS_NAME_5[pred_classify_cnn2]

                print(f"路口 {road_class_5}, 线段 {road_class_3}")

                # Use front window (±2°) calculated after angle correction of rangesAdj and anglesAdj）
                front_mask = np.logical_and(anglesAdj > -0.035, 
                                anglesAdj < 0.035)  # about ±2°
                front_window = rangesAdj[front_mask]

                # 左右检测区间：±0.1 弧度 ≈ ±5~6°
                side_angle_margin = 0.1

                # 左侧窗口（大约 π/2 左右）
                left_mask = (anglesAdj > (np.pi / 2 - side_angle_margin)) & (anglesAdj < (np.pi / 2 + side_angle_margin))
                left_window = rangesAdj[left_mask]

                # 右侧窗口（大约 -π/2 左右）
                right_mask = (anglesAdj > (-np.pi / 2 - side_angle_margin)) & (anglesAdj < (-np.pi / 2 + side_angle_margin))
                right_window = rangesAdj[right_mask]
                # NaN protection & empty value protection
                if front_window.size == 0 or np.any(np.isnan(front_window)):
                    print("Radar window data is empty or contains NaN, skipping this frame")
                    continue
                
                # NaN 检查（确保不 crash）
                if left_window.size == 0 or right_window.size == 0 or \
                   np.any(np.isnan(left_window)) or np.any(np.isnan(right_window)):
                    print(" 左右距离窗口无效，跳过本帧")
                    continue

                # Calculate average distance
                front = np.mean(front_window)
                print(f"Average distance detected directly ahead:{front:.3f}m")
                # 求左右侧平均距离
                left_dist = np.mean(left_window)
                right_dist = np.mean(right_window)
                print(f" 左侧: {left_dist:.2f} m | 右侧: {right_dist:.2f} m")
                # predicted_offset = (pred_rnn + pred_cnn) / 2

                # Step 角落检测逻辑：前、左、右距离都小于阈值，认为卡住
                corner_threshold = 0.7  ##############可以调参
                in_corner = left_dist < corner_threshold and right_dist < corner_threshold

                if in_corner:
                    print("🧱⚠️ 检测到拐角卡死，执行脱困策略...")

                    # Step 1 - 后退一小段时间
                    forSpd = -0.05
                    turnSpd = 0.0
                    reverse_time = time.time() + 0.5
                    while time.time() < reverse_time:
                        myQBot.read_write_std(timestamp=elapsed_time(), arm=arm,
                                commands=np.array([forSpd, turnSpd]))

                    # Step 2 - 微向左转以脱离死角
                    forSpd = 0.03
                    turnSpd = 0.4
                    recover_time = time.time() + 0.7
                    while time.time() < recover_time:
                        myQBot.read_write_std(timestamp=elapsed_time(), arm=arm,
                                commands=np.array([forSpd, turnSpd]))

                    print("✅ 脱困完毕，返回 CNN 控制")
                    continue  # 跳过这一帧，重新进入循环
                #--------------------------------------------------------
    
                if TURNSPD_LIST:
                    forSpd = 0
                    turnSpd = TURNSPD_LIST[0]
                    TURNSPD_LIST.pop(0)
                # CLassify "十字", "T型",
                elif road_class_5 in ["十字", "T型"] and road_class_3 in ["Multiple Lines"]:
                    if front > 0.5:
                        print(f"转内弯")
                        _, TURNSPD_LIST = get_motion_queque("innerTurnRight") if random.random() < 0.5 else get_motion_queque("innerTurnLeft")
                    else:
                        print(f"转外弯")
                        _, TURNSPD_LIST = get_motion_queque("outerTurnRight")
                
                # # 若距离小于阈值，则右转后恢复循线
                # if front < 0.5:
                #     print("T-junction detected: turning right")
                #     forSpd = -0.1
                #     forSpd = 0.05
                #     turnSpd = -0.5
                #  # Execute right turn for a period (non-blocking implementation)
                #     turn_end_time = time.time() + 0.8
                #     while time.time() < turn_end_time:
                #         myQBot.read_write_std(timestamp=elapsed_time(), arm=arm,
                #                 commands=np.array([forSpd, turnSpd]))
                #     print("Turn completed, returning to line-following mode")
                
                #elif predicted_offset is None or np.isnan(predicted_offset) or abs(predicted_offset) > 0.9:
                #    print("⚠️ 偏移过大，后退调整")
                #    forSpd = -0.1
                #    turnSpd = 0.2 * (-1 if counterDown % 2 == 0 else 1)
                else:
                    forSpd = 0.1 ################可以调参
                    turnSpd = np.clip(predicted_offset * -0.5, -1, 1)

                    # 添加：侧墙微调逻辑（叠加在 CNN 输出的基础上）
                    side_threshold = 0.5
                    correction_turn = 0.2#####可以调参

                    if left_dist < side_threshold:
                        print("🔧 左侧过近 ➤ 微向右转")
                        forSpd = 0.05
                        turnSpd += -correction_turn
                    elif right_dist < side_threshold:
                         print("🔧 右侧过近 ➤ 微向左转")
                         forSpd = 0.05
                         turnSpd += correction_turn

                    # 防止超出最大转向
                    turnSpd = np.clip(turnSpd, -1, 1)
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
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

device = torch.device("cpu")

# --------------- Load the RNN sequence model ---------------
# SEQ_LEN = 3
# frame_buffer = deque(maxlen=SEQ_LEN)
# rnn = load_model(CNNRNNLineFollower, "ckpt/0.8797_line_follower_rnn.pth")

# --------------- Load the CNN regression model ---------------
cnn  = load_model(CNNLineFollower, "ckpt/qbot_line_follower_cnn.pth")
cnn2 = load_model(CNNLineFollower2, "ckpt/0.8988_line_follower_cnn.pth")
tspd = load_model(TurnSpeedCNN, "ckpt/best_turn_speed_model.pth")

# --------------- Load the CNN classification model ---------------
SMOOTH_NUM = 5
BACK_NUM = 20
FDSPD_LIST = []
TURNSPD_LIST = []
smooth_idx = 0
back_idx = 0
last_turn_time = 0
cold_time = 180
cnn_3_cls = load_model(CNN3ClassifyRoad, "ckpt/0.9846_cls_cnn.pth")
CLASS_NAME_3 = ["Blank", "Single Line", "Multiple Lines"]

# cnn_5_cls = load_model(CNN5ClassifyRoad, "ckpt/classify_road_5_cnn.pth")
# CLASS_NAME_5 = ["Straight", "Turn", "Cross", "T-Junction", "Curve"]

cnn_6_cls = load_model(CNN6ClassifyRoad, "ckpt/qbot_line_follower_cnn_6_classes.pth")
CLASS_NAME_6 = ["Straight", "Turn", "Cross", "T-Junction", "Curve", "Black"]

# --------------- Load the ResNet model ---------------
resnet_col  = load_resnet18("ckpt/best_resnet_model_mps.pth", output_dim=1)


# ----------------------- Start -------------------------
lineFollow = False
prev_k7 = False

try:
    myQBot = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam = QBotPlatformCSICamera(frameRate=frameRate, exposure=39.0, gain=17.0)
    lidar = QBotPlatformLidar()
    keyboard = Keyboard()
    vision = QBPVision()
    ranging = QBPRanging()
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
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
            lidar_get = 0
            time.sleep(0.2)
        prev_k7 = k7_pressed
        # print(f"pre k7 :{prev_k7}, current k7: {k7_pressed}")

        if not lineFollow:
            commands = np.array([keyboardComand[0], keyboardComand[1]], dtype=np.float64)
            print(f"[手动模式] 速度指令: {commands}")
            myQBot.read_write_std(timestamp=elapsed_time(), arm=arm, commands=commands)
            continue

        newHIL = myQBot.read_write_std(timestamp=elapsed_time(), arm=1, commands=commands)
        if newHIL:
            timeHIL = time.time()
            newDownCam = downCam.read()
            # Lidar reading (maximum 3 attempts)
            lidar_flag = 0
            for i in range(3):
                newLidar = lidar.read()
                if newLidar and lidar.distances is not None and len(lidar.distances) > 0:
                    break
                time.sleep(0.05)  # Give it a moment to "catch its breath"
                lidar_flag = 1
                rangesAdj, anglesAdj = ranging.adjust_and_subsample(lidar.distances, 
                                                         lidar.angles, 1260, 3)
            else:
                print(" Invalid lidar frame, skipping this frame")
            
            if newDownCam:
                counterDown += 1

                undistorted = vision.df_camera_undistort(downCam.imageData)
                gray_sm = cv2.resize(undistorted, (320, 200))
                binary = vision.subselect_and_threshold(image=gray_sm, rowStart=50, rowEnd=100,
                                                        minThreshold=180, maxThreshold=255)

                # Blob Detection via Connected Component Labeling
                col, row, area = vision.image_find_objects(image=binary, connectivity=8, minArea=500, maxArea=4000)

                # Section D.2 - Speed command from blob information
                forSpd_2, turnSpd_2 = line2SpdMap.send((col, 0.7, 100))  # kP=0.2, kD=0 可调整
                # forSpd *=0.2
                turnSpd_2 *=0.03
                
                image_cnn = load_cnn_data(binary, device)
                # image_cnn2 = load_cnn_data2(binary, device)
                # image_rnn = load_rnn_data(binary, device)
                
                pred_cnn = cnn(image_cnn).item()
                # pred_cnn2 = cnn2(image_cnn2).item()
                # if len(frame_buffer) < SEQ_LEN:
                #     for _ in range(SEQ_LEN - len(frame_buffer)):
                #         frame_buffer.append(image_rnn.clone())
                # frame_buffer.append(image_rnn)
                # seq_tensor = torch.stack(list(frame_buffer))
                # seq_tensor = seq_tensor.unsqueeze(0).to(device)
                # pred_rnn = rnn(seq_tensor).item()

                predicted_offset = pred_cnn
                
                # print(f"预测偏移: {predicted_offset:.4f} CNN: {pred_cnn:.4f}, CNN2:{pred_cnn2:.4f} RNN: {pred_rnn:.4f}")

                image_6_cls = load_6_cls_data(binary, device)
                pred_6_cls = torch.argmax(cnn_6_cls(image_6_cls)[0]).item()
                road_6_class = CLASS_NAME_6[pred_6_cls]

                image_3_cls = load_3_cls_data(binary, device)
                pred_3_cls = torch.argmax(cnn_3_cls(image_3_cls)[0]).item()
                road_3_class = CLASS_NAME_3[pred_3_cls]

                print(f"路口 {road_6_class}, 线段 {road_3_class} col {col}")

                debug_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                cv2.putText(debug_image, f"Col: {col} Offset: {predicted_offset:.3f}, {road_6_class}, {road_3_class} ", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Line Following", debug_image)
                cv2.waitKey(1)

                if lidar_flag:

                    # Use front window (±2°) calculated after angle correction of rangesAdj and anglesAdj）
                    front_mask = np.logical_and(anglesAdj > -0.035, 
                                    anglesAdj < 0.035)  # about ±2°
                    front_window = rangesAdj[front_mask]

                    # 左右检测区间：±0.1 弧度 ≈ ±5~6°
                    side_angle_margin = 0.1

                    # left_window（about π/2 ）
                    left_mask = (
                        (anglesAdj > (np.pi / 2 - side_angle_margin)) & 
                        (anglesAdj < (np.pi / 2 + side_angle_margin)))
                    left_window = rangesAdj[left_mask]

                    # right_window（about -π/2）
                    right_mask = (
                        (anglesAdj > (-np.pi / 2 - side_angle_margin)) & 
                        (anglesAdj < (-np.pi / 2 + side_angle_margin)))
                    right_window = rangesAdj[right_mask]
                    # NaN protection & empty value protection
                    if front_window.size == 0 or np.any(np.isnan(front_window)):
                        print("Radar window data is empty or contains NaN, skipping this frame")
                    
                    # NaN 检查（确保不 crash）
                    elif left_window.size == 0 or right_window.size == 0 or \
                    np.any(np.isnan(left_window)) or np.any(np.isnan(right_window)):
                        print(" 左右距离窗口无效，跳过本帧")
                    
                    else:
                        # Calculate average distance
                        front = np.mean(front_window)
                        print(f"Average distance detected directly ahead:{front:.3f}m")
                        # 求左右侧平均距离
                        left_dist = np.mean(left_window)
                        right_dist = np.mean(right_window)
                        print(f" 左侧: {left_dist:.2f} m | 右侧: {right_dist:.2f} m")
                        if not lidar_get:
                            if left_dist < 1 or right_dist < 1:
                                lidar_get = 1
                                # outer circle
                                print(f"outer circle, {right_dist:.2f},{left_dist:.2f}")
                            else:
                                lidar_get = 2
                                # inner circle
                                print(f"inner circle, {right_dist:.2f},{left_dist:.2f}")
                    # predicted_offset = (pred_rnn + pred_cnn) / 2

                    # # Step 角落检测逻辑：前、左、右距离都小于阈值，认为卡住
                    # corner_threshold = 0.7  ##############可以调参
                    # in_corner = left_dist < corner_threshold and right_dist < corner_threshold

                    # if in_corner:
                    #     print("🧱⚠️ 检测到拐角卡死，执行脱困策略...")

                    #     # Step 1 - 后退一小段时间
                    #     forSpd = -0.05
                    #     turnSpd = 0.0
                    #     reverse_time = time.time() + 0.5
                    #     while time.time() < reverse_time:
                    #         myQBot.read_write_std(timestamp=elapsed_time(), arm=arm,
                    #                 commands=np.array([forSpd, turnSpd]))

                    #     # Step 2 - 微向左转以脱离死角
                    #     forSpd = 0.03
                    #     turnSpd = 0.4
                    #     recover_time = time.time() + 0.7
                    #     while time.time() < recover_time:
                    #         myQBot.read_write_std(timestamp=elapsed_time(), arm=arm,
                    #                 commands=np.array([forSpd, turnSpd]))

                    #     print("✅ 脱困完毕，返回 CNN 控制")
                    #     # continue  # 跳过这一帧，重新进入循环
                    
                    # # # 若距离小于阈值，则右转后恢复循线
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
                    #--------------------------------------------------------
                
                # if lidar_get==2 and col is not None:
                #     metrics.add_error(col * 160)
                #     commands = np.array([forSpd_2, turnSpd_2], dtype=np.float64)
                #     print(f"[自动模式] PID 计算速度: {commands}")
                #     continue
                
                if TURNSPD_LIST:
                    if back_idx < BACK_NUM:
                        forSpd = -0.02
                        turnSpd = 0
                        back_idx +=1
                    else:
                        forSpd = 0.01
                        turnSpd = TURNSPD_LIST[0]
                        TURNSPD_LIST.pop(0)
                # 空白后退
                elif road_3_class in ["B"] or road_6_class in ["Black"]:
                    forSpd = -0.04
                    turnSpd = 0

                # CLassify "十字", "T型",
                elif road_6_class in ["T-shape", "Cross"] and road_3_class in ["M"] and time.time() - last_turn_time > cold_time and lidar_get==2:
                    if smooth_idx < SMOOTH_NUM:
                        smooth_idx += 1
                        forSpd = 0.01  # 正常前进速度
                        turnSpd = np.clip(predicted_offset * -0.5, -1, 1)  # 限制转向速度范围
                    elif lidar_get and left_dist > right_dist:
                        print(f"转左弯\n\n")
                        FDSPD_LIST, TURNSPD_LIST = get_motion_queque("turnLeft")
                        forSpd = -0.02
                        turnSpd = 0
                        last_turn_time = time.time()
                    else:
                        print(f"转右弯\n\n")
                        FDSPD_LIST, TURNSPD_LIST = get_motion_queque("turnRight")
                        forSpd = -0.02
                        turnSpd = 0
                        last_turn_time = time.time()
                elif road_6_class in ["T-shape", "Cross"] and road_3_class in ["M"] and lidar_flag:
                    if front < 1:
                        if lidar_get and left_dist > right_dist:
                            print(f"转左弯\n\n")
                            FDSPD_LIST, TURNSPD_LIST = get_motion_queque("turnLeft")
                            forSpd = -0.1
                            turnSpd = 0
                            last_turn_time = time.time()
                        else:
                            print(f"转右弯\n\n")
                            FDSPD_LIST, TURNSPD_LIST = get_motion_queque("turnRight")
                            forSpd = -0.1
                            turnSpd = 0
                            last_turn_time = time.time()
                else:
                    smooth_idx = 0
                    back_idx = 0
                    if lidar_get:
                        if lidar_get == 1:
                            # outer circle
                            print(f"outer circle, {right_dist:.2f},{left_dist:.2f}")
                            forSpd = 0.1  # 正常前进速度
                            turnSpd = np.clip(predicted_offset * -0.5, -1, 1)  # 限制转向速度范围
                        else:
                            # inner circle
                            print(f"inner circle, {right_dist:.2f},{left_dist:.2f}")
                            if lidar_get==2 and col is not None:
                                metrics.add_error(col * 160)
                                commands = np.array([forSpd_2, turnSpd_2], dtype=np.float64)
                                print(f"[自动模式] PID 计算速度: {commands}")
                                continue
                            else:
                                forSpd = 0.02  # 正常前进速度
                                turnSpd = np.clip(predicted_offset * -0.5, -1, 1)  # 限制转向速度范围
                    else:
                        print(f"no circle")
                        forSpd = 0.01  # 正常前进速度
                        turnSpd = np.clip(predicted_offset * -1, -1, 1)  # 限制转向速度范围
                
                # if predicted_offset is None or np.isnan(predicted_offset) or abs(predicted_offset) > 0.9:
                #    print("⚠️ 偏移过大，后退调整")
                #    forSpd = -0.1
                #    turnSpd = 0.2 * (-1 if counterDown % 2 == 0 else 1)
                # else:
                #     forSpd = 0.01 ################可以调参
                #     turnSpd = np.clip(predicted_offset * -1, -1, 1)

                #     if lidar_flag:
                #         # 添加：侧墙微调逻辑（叠加在 CNN 输出的基础上）
                #         side_threshold = 0.5
                #         correction_turn = 0.2#####可以调参

                #         if left_dist < side_threshold:
                #             print("🔧 左侧过近 ➤ 微向右转")
                #             forSpd = 0.05
                #             turnSpd += -correction_turn
                #         elif right_dist < side_threshold:
                #             print("🔧 右侧过近 ➤ 微向左转")
                #             forSpd = 0.05
                #             turnSpd += correction_turn
                #         # 防止超出最大转向
                #         turnSpd = np.clip(turnSpd, -1, 1)


                # # 🚀 进入“缓慢后退模式”
                # if predicted_offset is None or abs(predicted_offset) > 0.9:
                #     print("⚠️ 机器人偏离或找不到线，后退中...")
                #     forSpd = -0.1  # **缓慢后退**
                #     turnSpd = 0.2 * (-1 if counterDown % 2 == 0 else 1)  # **左右小幅调整**
                

                # 记录误差
                if predicted_offset is not None:
                    metrics.add_error(predicted_offset * 160)
                    commands = np.array([forSpd, turnSpd], dtype=np.float64)
                    print(f"[自动模式] CNN 计算速度: {commands}")


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
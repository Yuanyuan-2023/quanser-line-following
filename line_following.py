#-----------------------------------------------------------------------------#
#------------------Skills Progression 1 - Task Automation---------------------#
#-----------------------------------------------------------------------------#
#----------------------------Lab 3 - Line Following---------------------------#
#-----------------------------------------------------------------------------#

# Imports
from pal.products.qbot_platform import QBotPlatformDriver,Keyboard,\
    QBotPlatformCSICamera, QBotPlatformRealSense, QBotPlatformLidar
from qbot_platform_functions import QBPVision
from quanser.hardware import HILError
from pal.utilities.probe import Probe
from pal.utilities.gamepad import LogitechF710
from qbot_platform_functions import LineFollowingMetrics
import time
import numpy as np
from PIL import Image
import cv2
from qlabs_setup import setup
import matplotlib
matplotlib.use('TkAgg')  # 强制使用 Tk 后端（比 Qt 更稳定）

# Section A - Setup

metrics = LineFollowingMetrics()
setup(locationQBotP=[-1.35, 0.3, 0.05], rotationQBotP=[0, 0, 0], verbose=True)
time.sleep(2)
ipHost, ipDriver = 'localhost', 'localhost'
commands, arm, noKill = np.zeros((2), dtype = np.float64), 0, True
frameRate, sampleRate = 60.0, 1/60.0
counter, counterDown = 0, 0
endFlag, offset, forSpd, turnSpd = False, 0, 0, 0
startTime = time.time()
def elapsed_time():
    return time.time() - startTime
timeHIL, prevTimeHIL = elapsed_time(), elapsed_time() - 0.017

try:
    # Section B - Initialization
    myQBot       = QBotPlatformDriver(mode=1, ip=ipDriver)
    downCam      = QBotPlatformCSICamera(frameRate=frameRate, exposure = 39.0, gain=17.0)
    keyboard     = Keyboard()
    vision       = QBPVision()
    probe        = Probe(ip = ipHost)
    probe.add_display(imageSize = [200, 320, 1], scaling = True, scalingFactor= 2, name='Raw Image')
    probe.add_display(imageSize = [50, 320, 1], scaling = False, scalingFactor= 2, name='Binary Image')
    line2SpdMap = vision.line_to_speed_map(sampleRate=sampleRate, saturation=75)
    next(line2SpdMap)
    startTime = time.time()
    time.sleep(0.5)

    # 计数器用于给图像编号
    data_counter = 0
    save_frame_interval = 1  # 每5帧保存一次
    data_counter_limit = 2000  # 最多保存10张图像  # 每 10 帧保存一次图像
    frame_counter = 0  # 用于计数已处理的帧

    # Main loop
    while noKill and not endFlag:
        t = elapsed_time()
        frame_counter += 1  # 确保此语句存在！
        if not probe.connected:
            probe.check_connection()

        if probe.connected:

            # Keyboard Driver
            newkeyboard = keyboard.read()
            if newkeyboard:
                arm = keyboard.k_space
                lineFollow = keyboard.k_7
                keyboardComand = keyboard.bodyCmd
                if keyboard.k_u:
                    noKill = False

            # 修改后的误差采集部分
            if lineFollow:
                if col is not None:
                    error = col - 160  # 使用原始像素偏移量
                    metrics.add_error(error)  
                else:
                    error = 0  
                    metrics.add_error(error)

            # Section C - toggle line following
            if not lineFollow:
                commands = np.array([keyboardComand[0], keyboardComand[1]], dtype = np.float64) # robot spd command
            else:
                commands = np.array([forSpd, turnSpd], dtype = np.float64) # robot spd command

            # QBot Hardware
            newHIL = myQBot.read_write_std(timestamp = time.time() - startTime,
                                            arm = arm,
                                            commands = commands)
            if newHIL:
                timeHIL = time.time()
                newDownCam = downCam.read()
                if newDownCam:
                    counterDown += 1

                    # Section D - Image processing 
                    
                    # Section D.1 - Undistort and resize the image
                    undistorted = vision.df_camera_undistort(downCam.imageData)
                    gray_sm = cv2.resize(undistorted, (320, 200))
                    
                    #-------Replace the following line with your code---------#

                    # 应用高斯滤波去噪
                    #filtered_image = cv2.GaussianBlur(gray_sm, (5, 5), 0)  # 5x5 核心，高斯滤波

                    # 对图像应用中值滤波
                    #filtered_image = cv2.medianBlur(gray_sm, 5)  # 5x5的卷积核

                    # Subselect a part of the image and perform thresholding
                    binary = vision.subselect_and_threshold(
                        image=gray_sm, rowStart=50, rowEnd=100, minThreshold=180, maxThreshold=255
                    )

                    # 对图像应用中值滤波
                    #filtered_binary_image = cv2.medianBlur(binary, 7)  # 5x5的卷积核
                    #binary = filtered_binary_image
                    
                    # 定义一个3x3的内核
                    #kernel = np.ones((3, 3), np.uint8)
                    # 腐蚀操作：去除小噪点
                    #eroded_image = cv2.erode(binary, kernel, iterations=1)
                    # 膨胀操作：恢复白线形状
                    #dilated_image = cv2.dilate(eroded_image, kernel, iterations=1)
                    #binary = dilated_image 

                    # Blob Detection via Connected Component Labeling
                    col, row, area = vision.image_find_objects(image=binary, connectivity=8, minArea=500, maxArea=4000)

                    # Section D.2 - Speed command from blob information
                    forSpd, turnSpd = line2SpdMap.send((col, 0.7, 100))  # kP=0.2, kD=0 可调整
                    # forSpd *=0.2
                    turnSpd *=0.3
                    #---------------------------------------------------------#

                    if frame_counter >= save_frame_interval:
                        if data_counter < data_counter_limit:
                            if col is not None:
                                normalized_label = round((col - 160) / 160, 4)
                            else:
                                normalized_label = None
        
                    # 调用保存函数前打印调试信息
                            print(f"[调试] 准备保存第 {data_counter} 张图像，标签={normalized_label}")
        
                            vision.save_data_example(binary, "qbot_binary_images", data_counter, label=normalized_label)
                            data_counter += 1
                        frame_counter = 0
                    
                    # 每隔一定帧数（例如10帧）输出一次准确度
                    if counterDown % 10 == 0:
                        # 检查 col 是否为 None 或 0
                        if col == 0 or col is None:
                            print("未检测到白线")
                            accuracy = {"Cumulative_Error": None, "MAE": None, "MSE": None}  # 如果没有检测到白线，准确度为None
                        else:
                            # 计算准确度并输出
                            accuracy = metrics.calculate_accuracy()

                            # 将累计误差、MAE、MSE 打印在同一行
                            print(f"累计误差: {accuracy['Cumulative_Error']}, MAE: {accuracy['MAE']}, MSE: {accuracy['MSE']}")


                if counterDown%4 == 0:
                    sending = probe.send(name='Raw Image', imageData=gray_sm)
                    sending = probe.send(name='Binary Image', imageData=binary)
                prevTimeHIL = timeHIL


        # 任务结束后输出准确率
        if not noKill:
            # 获取并计算准确度指标
            accuracy = metrics.calculate_accuracy()

            # 输出相关指标
            print(f"累计误差: {accuracy['Cumulative_Error']}")
            print(f"MAE: {accuracy['MAE']}")
            print(f"MSE: {accuracy['MSE']}")
            if accuracy["line_following_accuracy"] is None:
                print("循线准确率: 无法计算")
            else:
                print(f"循线准确率: {accuracy['line_following_accuracy']:.2f}%")

except KeyboardInterrupt:
    print('User interrupted.')
except HILError as h:
    print(h.get_error_message())
finally:
    downCam.terminate()
    myQBot.terminate()
    probe.terminate()
    keyboard.terminate()
    # 新增绘图代码
    metrics.plot_errors(save_path="line_following_error.png")
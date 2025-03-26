import numpy as np
import cv2
from PIL import Image
import os
import csv
from pal.products.qbot_platform import QBotPlatformDriver
from pal.utilities.math import Calculus
from scipy.ndimage import median_filter
from pal.utilities.math import Calculus
from pal.utilities.stream import BasicStream
from quanser.common import Timeout
import matplotlib.pyplot as plt

class QBPMovement():
    """ This class contains the functions for the QBot Platform such as
    Forward/Inverse Differential Drive Kinematics etc. """

    def __init__(self):
        self.WHEEL_RADIUS = QBotPlatformDriver.WHEEL_RADIUS      # radius of the wheel (meters)
        self.WHEEL_BASE = QBotPlatformDriver.WHEEL_BASE          # distance between wheel contact points on the ground (meters)
        self.WHEEL_WIDTH = QBotPlatformDriver.WHEEL_WIDTH        # thickness of the wheel (meters)
        self.ENCODER_COUNTS = QBotPlatformDriver.ENCODER_COUNTS  # encoder counts per channel
        self.ENCODER_MODE = QBotPlatformDriver.ENCODER_MODE      # multiplier for a quadrature encoder

    def diff_drive_inverse_velocity_kinematics(self, forSpd, turnSpd):
        """This function is for the differential drive inverse velocity
        kinematics for the QBot Platform. It converts provided body speeds
        (forward speed in m/s and turn speed in rad/s) into corresponding
        wheel speeds (rad/s)."""

        #------------Replace the following lines with your code---------------#
        wL = 0
        wR = 0
        #---------------------------------------------------------------------#
        return wL, wR

    def diff_drive_forward_velocity_kinematics(self, wL, wR):
        """This function is for the differential drive forward velocity
        kinematics for the QBot Platform. It converts provided wheel speeds
        (rad/s) into corresponding body speeds (forward speed in m/s and
        turn speed in rad/s)."""
        #------------Replace the following lines with your code---------------#
        forSpd = 0
        turnSpd = 0
        #---------------------------------------------------------------------#
        return forSpd, turnSpd

class QBPVision():
    def __init__(self):
        self.imageCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def undistort_img(self,distImgs,cameraMatrix,distCoefficients):
        """
        This function undistorts a general camera, given the camera matrix and
        coefficients.
        """

        undist = cv2.undistort(distImgs,
                               cameraMatrix,
                               distCoefficients,
                               None,
                               cameraMatrix)
        return undist

    def df_camera_undistort(self, image):
        """
        This function undistorts the downward camera using the camera
        intrinsics and coefficients."""
        CSICamIntrinsics = np.array([[419.36179672, 0, 292.01381114],
                                     [0, 420.30767196, 201.61650657],
                                     [0, 0, 1]])
        CSIDistParam = np.array([-7.42983302e-01,
                                 9.24162996e-01,
                                 -2.39593372e-04,
                                 1.66230745e-02,
                                 -5.27787439e-01])
        undistortedImage = self.undistort_img(
                                                image,
                                                CSICamIntrinsics,
                                                CSIDistParam
                                                )
        return undistortedImage

    def subselect_and_threshold(self, image, rowStart, rowEnd, minThreshold, maxThreshold):
        """
        This function subselects a horizontal slice of the input image from
        rowStart to rowEnd for all columns, and then thresholds it based on the
        provided min and max thresholds. Returns the binary output from
        thresholding."""

        #------------Replace the following lines with your code---------------#
        subImage = image[rowStart:rowEnd, :]
        _, binary = cv2.threshold(subImage, minThreshold, maxThreshold, cv2.THRESH_BINARY)
        #---------------------------------------------------------------------#

        return binary
    
    def generate_normalized_label(self, col, image_width=320):
        """将col值归一化到[-1, 1]范围（假设图像中心为160）"""
        if col is None:
            return None  # 未检测到线时返回None
        center = image_width // 2  # 160
        normalized = (col - center) / center  # 范围[-1, 1]
        return round(normalized, 4)  # 保留4位小数

    def image_find_objects(self, image, connectivity, minArea, maxArea):
        """
        This function implements connected component labeling on the provided
        image with the desired connectivity. From the list of blobs detected,
        it returns the first blob that fits the desired area criteria based
        on minArea and maxArea provided. Returns the column and row location
        of the blob centroid, as well as the blob's area. """

        col = 0
        row = 0
        area = 0

        #------------Replace the following lines with your code---------------#

        # Step 1: 获取连通区域并检查是否有目标
        (labels, ids, values, centroids) = cv2.connectedComponentsWithStats(image, connectivity, cv2.CV_32S)

        if labels <= 1:
            print("Warning: No objects found in the binary image!")
            return None, None, 0  # 只有背景，没有目标
        # Step 2: Filter blobs by size
        max_area = 0
        largest_centroid = None

        for i in range(1, labels):  # Skipping label 0 (background)
            area = values[i, cv2.CC_STAT_AREA]

            if minArea <= area <= maxArea and area > max_area:
                max_area = area
                largest_centroid = tuple(centroids[i])  # (x, y)

    # 取消注释的代码段要求 `values` 变量，因此将 `stats` 赋值给 `values`
       #values = stats  
        #---------------------------------------------------------------------#

        #-------------Uncomment the following 12 lines of code----------------#

        for idx, val in enumerate(values):
             if val[4]>minArea and val[4] < maxArea:
                 value = val
                 centroid = centroids[idx]
                 col = centroid[0]
                 row = centroid[1]
                 area = value[4]
                 break
             else:
                 col = None
                 row = None
                 area = None
        #---------------------------------------------------------------------#

        return col, row, area

    def line_to_speed_map(self, sampleRate, saturation):

        integrator   = Calculus().integrator(dt = sampleRate, saturation=saturation)
        derivative   = Calculus().differentiator(dt = sampleRate)
        next(integrator)
        next(derivative)
        forSpd, turnSpd = 0, 0
        offset = 0

        while True:
            col, kP, kD = yield forSpd, turnSpd

            if col is not None:
                #-----------Complete the following lines of code--------------#
                error = (160 - col) + offset # 320 像素宽的图像，中心为 160
                angle = np.arctan2(error, 24)  # 50 是一个经验值，表示基准距离
                turnSpd = kP * angle + kD * derivative.send(angle)
                forSpd = 0.35 * np.cos(angle)
                #-------------------------------------------------------------#
                offset = integrator.send(turnSpd*0.8)
            

    def save_data_example(self, image, output_dir, data_counter, label=None):
        # 创建目录（如果不存在）
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 保存图像
        image_filename = f"binary_image_{data_counter}.png"
        image_path = os.path.join(output_dir, image_filename)
        cv2.imwrite(image_path, image)
        
        # 保存标签到CSV
        csv_path = os.path.join(output_dir, "labels.csv")
        try:
            # 检查文件是否存在，以决定是否写入表头
            file_exists = os.path.isfile(csv_path)
            
            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["image_filename", "normalized_label"])
                writer.writerow([image_filename, label])
            
            print(f"[调试] CSV文件已更新：{csv_path}")
        except Exception as e:
            print(f"[错误] 写入CSV失败：{str(e)}")
        
        return image_path


class LineFollowingMetrics:
    def __init__(self, error_threshold=25):
        # 新增原始误差记录
        self.raw_errors = []       # 原始误差值
        self.errors = []           # 归一化误差
        self.total_error = 0       
        self.frame_count = 0       
        self.max_error_value = 160 
        self.error_threshold = error_threshold

    def add_error(self, error):
        """记录原始误差和归一化误差"""
        self.raw_errors.append(error)  # 新增原始误差记录
        abs_error = abs(error)  
        normalized_error = abs_error / self.max_error_value  
        self.errors.append(normalized_error)
        self.total_error += normalized_error  
        self.frame_count += 1  

    def plot_errors(self, save_path="error_plot.png"):
        """绘制误差图表"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.raw_errors)), self.raw_errors, 
                label='Offset Error', color='blue')
        plt.axhline(0, color='red', linestyle='--', label='Center Line')
        plt.xlabel("Frame Number")
        plt.ylabel("Offset Error (pixels)")
        plt.title("Line Following Error vs Frame Number")
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()

    def calculate_accuracy(self):
        """计算并返回准确度相关指标，包括累计误差、MAE、MSE和循线准确率"""
        if self.frame_count == 0:
            return {"Cumulative_Error": None, "MAE": None, "MSE": None, "line_following_accuracy": None}

        # 计算累计误差（路径偏离误差的绝对值之和）
        cumulative_error = np.sum(self.errors)

        # 计算MAE (Mean Absolute Error)
        MAE = np.mean(self.errors)

        # 计算MSE (Mean Squared Error)
        MSE = np.mean(np.square(self.errors))

        # 计算循线准确率
        successful_frames = sum(1 for error in self.errors if abs(error * self.max_error_value) <= self.error_threshold)
        accuracy_percentage = (successful_frames / self.frame_count) * 100  # 转化为百分比
        #accuracy_percentage = 100 - (self.total_error / self.frame_count * 100)  # 计算准确率

        return {
            "Cumulative_Error": cumulative_error,
            "MAE": MAE,
            "MSE": MSE,
            "line_following_accuracy": accuracy_percentage
        }



class QBPRanging():
    def __init__(self):
        pass

    def adjust_and_subsample(self, ranges, angles,end=-1,step=4):

        # correct angles data
        angles_corrected = -1*angles + np.pi/2
        # return every 4th sample
        return ranges[0:end:step], angles_corrected[0:end:step]

    def correct_lidar(self, lidarPosition, ranges, angles):

        # Convert lidar data from polar into cartesian, and add lidar position
        # Then Convert back into polar coordinates

        #-------Replace the following line with your code---------#
        # Determine the start of the focus region 
        ranges_c=None
        angles_c=None
        #---------------------------------------------------------#

        return ranges_c, angles_c

    def detect_obstacle(self, ranges, angles, forSpd, forSpeedGain, turnSpd, turnSpeedGain, minThreshold, obstacleNumPoints):

        halfNumPoints = 205
        quarterNumPoints = round(halfNumPoints/2)

        # Grab the first half of ranges and angles representing 180 degrees
        frontRanges = ranges[0:halfNumPoints]
        frontAngles = angles[0:halfNumPoints]

        # Starting index in top half          1     West
        # Mid point in west quadrant         51     North-west
        # Center index in top half          102     North
        # Mid point in east quadrant     51+102     North-east
        # Ending index in top half          205     East

        ### Section 1 - Dynamic Focus Region ###
        
        #-------Replace the following line with your code---------#
        # Determine the start of the focus region 
        startingIndex = 0
        #---------------------------------------------------------#

        # Setting the upper and lower bound such that the starting index 
        # is always in the first quarant
        if startingIndex < 0:
            startingIndex = 0
        elif startingIndex > 102:
            startingIndex = 102

        # Pick quarterNumPoints in ranges and angles from the front half
        # this will be the region you monitor for obstacles
        monitorRanges = frontRanges[startingIndex:startingIndex+quarterNumPoints]
        monitorAngles = frontAngles[startingIndex:startingIndex+quarterNumPoints]

        ### Section 2 - Dynamic Stop Distance ###

        #-------Replace the following line with your code---------#
        # Determine safetyThreshold based on Forward Speed 
        safetyThreshold = 1
        
        #---------------------------------------------------------#

        
        # At angles corresponding to monitorAngles, pick uniform ranges based on
        # a safety threshold
        safetyAngles = monitorAngles
        safetyRanges = safetyThreshold*monitorRanges/monitorRanges

  
        ### Section 3 - Obstacle Detection ###

        
        #-------Replace the following line with your code---------#
        # Total number of obstacles detected between 
        # minThreshold & safetyThreshold
        # Then determine obstacleFlag based on obstacleNumPoints

        obstacleFlag = 0
        
        #---------------------------------------------------------#


        # Lidar Ranges and Angles for plotting (both scan & safety zone)
        plottingRanges = np.append(monitorRanges, safetyRanges)
        plottingAngles = np.append(monitorAngles, safetyAngles)

        return plottingRanges, plottingAngles, obstacleFlag

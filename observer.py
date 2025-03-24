# ------------------Skills Progression 1 - Task Automation---------------------#
# ----------------------------Lab 4 - Observer---------------------------#

from pal.utilities.probe import Observer

observer = Observer()

# 🚦 添加摄像头图像显示（下摄像头视图）
observer.add_display(
    imageSize=[200, 320, 1],     # 高x宽x通道数
    scalingFactor=2,
    name='Downward Camera View'
)

# 🛰️ 添加雷达图显示（激光雷达）
observer.add_plot(
    numMeasurements=1680,        # 雷达点数，一般是 1680
    frameSize=400,               # 图像窗口大小像素
    pixelsPerMeter=50,           # 每米多少像素（地图缩放比例）
    scalingFactor=4,             # 放大倍数
    name='Lidar View'
)

# ✅ 启动显示窗口
observer.launch()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

# 假设有一个二维图像（比如灰度图像），可以用 NumPy 创建一个简单的矩阵
# image = np.random.rand(10, 10)  # 生成一个 10x10 的随机浮动图像
image_path = r"/home/kasm-user/H2ASeg-main/save/visual_AutoPET/AutoPET_step_1_slice_38.png"
image = Image.open(image_path)

# 将图像转换为灰度图像（如果是彩色图像）
gray_image = image.convert('L')

# 将灰度图像转换为NumPy数组
image_array = np.array(gray_image)
# 创建一个热图
plt.imshow(image, cmap='hot')  # 'hot' 是热图的颜色映射方式，可以选择其他颜色映射
plt.colorbar()  # 显示颜色条
plt.show()

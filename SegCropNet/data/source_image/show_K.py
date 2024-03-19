#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         show_K
# Description:
# Author:       Ming_King
# Date:         2024/2/29
# -------------------------------------------------------------------------------
import numpy as np
import cv2
from sklearn.cluster import KMeans

# 读取图像
image = cv2.imread(r'D:\PythonProject\lanenet-lane-detection-pytorch-main\test_output\1_instance_image.jpg')

# 将图像转换为 RGB 格式
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 获取图像的形状
height, width, channels = image.shape

# 将图像数据转换为二维数组
image_data = image.reshape((-1, 3))

# 指定要分成的簇的数量
num_clusters = 5

# 使用 K-Means 聚类算法进行聚类
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(image_data)

# 获取聚类中心
cluster_centers = kmeans.cluster_centers_

# 获取每个像素所属的簇
labels = kmeans.labels_

# 创建一个新的图像，每个像素的颜色根据其所属的簇的中心颜色来决定
clustered_image = np.zeros_like(image)
for i in range(height):
    for j in range(width):
        cluster_index = labels[i * width + j]
        clustered_image[i, j] = cluster_centers[cluster_index]

# 显示原始图像和聚类后的图像
cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
cv2.imshow('Clustered Image', cv2.cvtColor(clustered_image, cv2.COLOR_RGB2BGR))
cv2.waitKey(0)
cv2.destroyAllWindows()

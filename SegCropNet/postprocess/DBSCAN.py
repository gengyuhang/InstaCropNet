#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         show_B
# Description:
# Author:       Ming_King
# Date:         2024/2/29
# -------------------------------------------------------------------------------
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import time
from test_Mutil import test


def DBSCAN(out_file,if_single_img=True):
    # 读取图像
    #image = cv2.imread(r'E:\A_trans\results\AH\3\Enet\279_instance_output.png')
    #mask = cv2.imread(r'E:\A_trans\results\AH\3\Enet\279_binary_output.png', cv2.IMREAD_GRAYSCALE)
    f1,f2,f3=test(if_single_img)
    image = cv2.imread(f2)
    mask = cv2.imread(f3, cv2.IMREAD_GRAYSCALE)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    # 将图像转换为 RGB 格式
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 提取第二张图像中像素值大于100的非零像素
    threshold_value = 100
    masked_pixels_indices = np.where(mask > threshold_value)
    masked_pixels = image[masked_pixels_indices]
    # masked_pixels = masked_pixels[np.where(masked_pixels[:, 0] > threshold_value)]

    start_time = time.time()
    # 使用 DBSCAN 聚类算法进行聚类
    dbscan = DBSCAN()  # 使用默认参数 or DBSCAN(eps=100, min_samples=50) 根据需要调整 eps 和 min_samples 参数
    dbscan.fit(masked_pixels)
    end_time = time.time()  # 记录结束时间
    elapsed_time_ms = (end_time - start_time) * 1000  # 计算消耗的时间（毫秒）
    print("消耗的时间（毫秒）DBSCAN:", elapsed_time_ms)
    # 获取每个像素所属的簇
    labels = dbscan.labels_

    # 获取聚类后的簇的数量
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # 将聚类后的像素重新放回原始图像的蒙版区域
    result_image = np.zeros_like(image)
    for i, (row, col) in enumerate(zip(masked_pixels_indices[0], masked_pixels_indices[1])):
        if i < len(labels):  # 确保索引有效
            cluster_index = labels[i]
            if cluster_index != -1:  # 如果不是噪声点
                result_image[row, col] = masked_pixels[i]

    # 显示原始图像、蒙版图像和处理后的图像
    cv2.imshow('Original Image', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imshow('Mask', mask)
    cv2.imshow('Result Image', cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    opening = cv2.morphologyEx(result_image, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
    # cv2.imwrite('D:/PythonProject/lanenet-lane-detection-pytorch-main/test_output/instance_image/279_E_B.png',
    #             cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_file,cv2.cvtColor(opening, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

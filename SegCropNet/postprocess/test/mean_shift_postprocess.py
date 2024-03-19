import numpy as np
from sklearn.cluster import MeanShift
import cv2
import matplotlib.pyplot as plt

def post_process_clusters(labels, min_cluster_size):
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # 将小于最小簇大小的簇标记为噪点
    noise_labels = unique_labels[label_counts < min_cluster_size]

    # 将噪点标记为零
    for noise_label in noise_labels:
        labels[labels == noise_label] = 0
    return labels


def lane_detection(binary_image_path, instance_image_path, delta_v, min_cluster_size=100):
    # 读取二值化图像和特征向量图像
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    # 设置阈值，将大于阈值的像素设为白色，小于等于阈值的像素设为黑色
    _, binary_smoothed = cv2.threshold(binary_image, 200, 255, cv2.THRESH_BINARY)

    # 使用中值滤波进行平滑处理
    binary_smoothed = cv2.medianBlur(binary_smoothed, 5)
    instance_image = cv2.imread(instance_image_path)

    # 获取非零像素的坐标
    non_zero_points = np.column_stack(np.where(binary_smoothed > 0))

    # 获取特征向量（颜色值）
    features = [instance_image[point[0], point[1]] for point in non_zero_points]

    # 转换为numpy数组
    features = np.array(features)

    # 使用MeanShift聚类
    clustering = MeanShift(bandwidth=delta_v)
    clustering.fit(features)

    # 获取簇中心和标签
    cluster_centers = clustering.cluster_centers_
    labels = clustering.labels_

    # 创建车道线分配字典
    lanes = {label: [] for label in set(labels)}
    # 后处理：去除小簇
    # labels = post_process_clusters(labels, min_cluster_size)
    # 根据聚类结果进行像素分配
    for i, point in enumerate(non_zero_points):
        label = labels[i]
        lanes[label].append(point)

    # 创建聚类后的实例结果图像
    instance_result = np.zeros_like(binary_image)

    # 根据车道线分配结果进行像素标记
    for label, lane_points in lanes.items():
        for point in lane_points:
            instance_result[point[0], point[1]] = label + 1

    # 保存实例结果图像
    # cv2.imwrite("instance_result.jpg", instance_result)

    # 显示实例结果图像
    plt.imshow(instance_result, cmap='jet')  # 使用jet colormap以显示不同的实例
    plt.title('Instance Segmentation Result')
    plt.colorbar()
    plt.show()


# 示例调用
binary_image_path = 'test_output/2_binary_image.jpg'
instance_image_path = 'test_output/2_instance_image.jpg'
delta_v = 50  # 你的半径参数

lane_detection(binary_image_path, instance_image_path, delta_v)

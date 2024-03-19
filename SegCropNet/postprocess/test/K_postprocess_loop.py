import numpy as np
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import time
import os


def post_process_clusters(labels, min_cluster_size):
    unique_labels, label_counts = np.unique(labels, return_counts=True)

    # 将小于最小簇大小的簇标记为噪点
    noise_labels = unique_labels[label_counts < min_cluster_size]

    # 将噪点标记为零
    for noise_label in noise_labels:
        labels[labels == noise_label] = 0

    return labels


def lane_detection(binary_image_path, instance_image_path, num_clusters, min_cluster_size=100):
    # 读取二值化图像和特征向量图像
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    # 设置阈值，将大于阈值的像素设为白色，小于等于阈值的像素设为黑色
    _, binary_smoothed = cv2.threshold(binary_image, 200, 255, cv2.THRESH_BINARY)

    # 使用中值滤波进行平滑处理
    binary_smoothed = cv2.medianBlur(binary_smoothed, 5)
    instance_image = cv2.imread(instance_image_path)

    # test(直接使用binary_image进行聚类)
    # binary_test = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)

    # 获取非零像素的坐标
    non_zero_points = np.column_stack(np.where(binary_smoothed > 0))

    # 获取特征向量
    features = [instance_image[point[0], point[1]] for point in non_zero_points]
    # features = binary_test

    # 转换为numpy数组
    features = np.array(features)

    # 使用K均值聚类
    clustering = KMeans(n_clusters=num_clusters, random_state=42)
    labels = clustering.fit_predict(features)

    # 创建作物行分配字典
    lanes = {label: [] for label in set(labels)}

    # 后处理：去除小簇
    labels = post_process_clusters(labels, min_cluster_size)

    # 创建聚类后的实例结果图像
    instance_result = np.zeros_like(binary_image)

    # 根据作物行分配结果进行像素标记
    for i, point in enumerate(non_zero_points):
    # for i, point in enumerate(binary_test):
        label = labels[i]
        lanes[label].append(point)
        instance_result[point[0], point[1]] = label + 1

    # 保存实例结果图像
    # cv2.imwrite("instance_result.jpg", instance_result)
    opening = cv2.morphologyEx(instance_result, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))

    # 显示实例结果图像
    # plt.imshow(opening, cmap='jet')  # 使用jet colormap以显示不同的实例
    # plt.title('Instance Segmentation Result')
    # plt.colorbar()
    # plt.show()
    labels = np.unique(labels, axis=0)
    return instance_result, labels
    # return opening, labels


def extract_centers(binary_image, labels, num_bars=9):
    # 划分长条
    bar_height = binary_image.shape[0] // num_bars

    centers_1 = []
    centers_2 = []
    centers_3 = []
    centers_4 = []

    centers_mid = []   # 临时保存中间列表

    for i in range(num_bars):
        # 提取每个长条
        bar = binary_image[i * bar_height: (i + 1) * bar_height, :]

        for l in labels:
            # 提取中心点
            D = np.sum(bar == (l + 1))
            if D == 0:
                continue
            center_x = np.sum(np.where(bar == (l + 1))[1]) / np.sum(bar == (l + 1))
            if np.isnan(center_x):
                continue
            center_y = i * bar_height + bar_height // 2
            if l + 1 == 1:
                centers_1.append((center_x, center_y))
            elif l + 1 == 2:
                centers_2.append((center_x, center_y))
            elif l + 1 == 3:
                centers_3.append((center_x, center_y))
            else:
                centers_4.append((center_x, center_y))
    centers_mid = adjust_centers_order(centers_1, centers_2, centers_3, centers_4)
    # return centers_1, centers_2, centers_3, centers_4
    return centers_mid


# 以类别1,2,3,4的顺序来排列检测的作物行
def adjust_centers_order(centers_1, centers_2, centers_3, centers_4):
    # 找到每个数组中center_y值最大的元素组
    max_centers = [
        max(centers_1, key=lambda x: x[1]),
        max(centers_2, key=lambda x: x[1]),
        max(centers_3, key=lambda x: x[1]),
        max(centers_4, key=lambda x: x[1])
    ]

    # 按照center_x的大小进行排序
    sorted_max_centers = sorted(max_centers, key=lambda x: x[0])

    # 调整centers_1, centers_2, centers_3, centers_4四个数组的顺序，并临时保存调整后的顺序到centers_mid数组中
    centers_mid = []
    for center in sorted_max_centers:
        if center in centers_1:
            centers_mid.append(centers_1)
        elif center in centers_2:
            centers_mid.append(centers_2)
        elif center in centers_3:
            centers_mid.append(centers_3)
        elif center in centers_4:
            centers_mid.append(centers_4)

    # 返回调整后的四个数组和临时保存的调整顺序数组
    return centers_mid


# Function to fit a straight line using least squares
def fit_line(x, y):
    if np.all(x == x[0]):  # Check if x-values are constant
        m = np.inf  # Slope is set to infinity for constant x-values
        c = np.mean(y)  # Intercept is the mean of y-values
    else:
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, c


# Function to find extended points for a line that starts from one edge and ends at the opposite edge
def find_extended_points(m, c, image_shape):
    x1 = 0
    y1 = c
    x2 = image_shape[1]
    y2 = m * x2 + c

    return x1, y1, x2, y2


# Function to plot data points and extended lines on an image
def plot_data_and_line(image, data_points, line_width=1.0, idx=0):
    plt.figure()  # 创建新的图像对象
    # whiteboard = np.ones_like(image) * 255
    # plt.imshow(whiteboard)
    plt.imshow(image)  # Display the image

    for i, (x, y) in enumerate(data_points):
        m, c = fit_line(x, y)
        if np.isinf(m):
            # plt.axvline(x[0], color=colors[i], label=f'Line {i + 1}', linewidth=line_width)  # with label
            plt.axvline(x[0], color=colors[i], linewidth=line_width)
        else:
            # plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}')
            # plt.scatter(x, y, color=colors[i])  # Scatter plot for data points

            # Find extended points for the line
            x1, y1, x2, y2 = find_extended_points(m, c, image.shape)

            # Plot the original line
            # plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}', linewidth=line_width)  # with label
            plt.plot(x, m * x + c, color=colors[i], linewidth=line_width)
            # plt.scatter(x, y, color=colors[i], facecolors='none')  # Scatter plot for data points

            # Plot the extended line
            # plt.plot([x1, x2], [y1, y2], linestyle='--', color=colors[i], linewidth=line_width)
            plt.plot([x1, x2], [y1, y2], color=colors[i], linewidth=line_width)

    # Set axis limits for lower left corner as origin
    plt.xlim((0, image.shape[1]))
    plt.ylim((image.shape[0], 0))

    # 设置坐标轴不可见
    plt.axis('off')

    # plt.legend()
    plt.savefig('E:/A_trans/results/AH/3/Unet/with_img/' + f'{idx}.png', format='png', dpi=400, bbox_inches='tight')
    # plt.show()
    plt.close()  # 关闭当前图像对象，以释放内存


# 将这个函数拆到draw_line中
def fit_lines_hough(data_points_old, idx):
    # 转换为新的 data_points 格式
    data_points_new = [(np.array([x for x, y in points]), np.array([y for x, y in points])) for points in
                       data_points_old]

    # Filter out sets with NaN values
    # Remove individual data points with NaN values
    # data_points_new = [
    #     (
    #         np.array([xi for xi, yi in zip(x, y) if not np.isnan(xi) and not np.isnan(yi)]),
    #         np.array([yi for xi, yi in zip(x, y) if not np.isnan(xi) and not np.isnan(yi)])
    #     )
    #     for x, y in data_points_new
    # ]
    # # Remove sets with empty arrays resulting from the removal of NaN values
    # data_points_new = [(x, y) for x, y in data_points_new if len(x) > 0 and len(y) > 0]

    image = IMG.copy()
    plot_data_and_line(image, data_points_new, line_width=2.0, idx=idx)


def draw_lines(instance_result, labels, idx=0):
    start_time = time.time()  # 记录开始时间
    # 提取中心点
    centers = extract_centers(instance_result, labels)
    # idx = 0
    # 进行霍夫变换拟合直线
    fit_lines_hough(centers, idx)
    end_time = time.time()  # 记录结束时间
    elapsed_time_ms = (end_time - start_time) * 1000  # 计算消耗的时间（毫秒）
    print("消耗的时间（毫秒）line fitting:", elapsed_time_ms)
    # 可视化结果
    # cv2.polylines(IMG, [np.int32(lines)], isClosed=False, color=(255, 0, 0), thickness=2)
    # 显示结果
    # cv2.imshow("Result", RGB_IMG)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


colors_map = [(33, 150, 243), (255, 152, 0), (244, 67, 54), (0, 150, 136), (63, 81, 181), (76, 175, 80)]
colors = ['#2196F3', '#FF9800', '#F44336', '#009688', '#3F51B5', '#4CAF50']
num_clusters = 4  # 簇的数量

# 循环调用
img_id = 106  # 起始ID
file_path = 'E:/A_trans/results/AH/3/Unet/'
for index in range(100):
    if os.path.exists(file_path + f'{img_id}' + '_input.png'):
        binary_image_path = file_path + f'{img_id}' + '_binary_output.png'
        instance_image_path = file_path + f'{img_id}' + '_instance_output.png'
        RGB_IMG = cv2.imread(file_path + f'{img_id}' + '_input.png')
        IMG = cv2.cvtColor(RGB_IMG, cv2.COLOR_RGB2BGR)

        instance_out, labels = lane_detection(binary_image_path, instance_image_path, num_clusters)
        draw_lines(instance_out, labels, img_id)
        img_id += 2
    else:
        while True:
            # 构造图片文件路径
            image_file = file_path + f'{img_id}' + '_input.png'

            # 判断文件是否存在
            if os.path.exists(image_file):
                print(f"图片文件 {image_file} 存在，结束循环")
                break
            else:
                print(f"图片文件 {image_file} 不存在，继续循环")
                img_id += 1
        binary_image_path = file_path + f'{img_id}' + '_binary_output.png'
        instance_image_path = file_path + f'{img_id}' + '_instance_output.png'
        RGB_IMG = cv2.imread(file_path + f'{img_id}' + '_input.png')
        IMG = cv2.cvtColor(RGB_IMG, cv2.COLOR_RGB2BGR)

        instance_out, labels = lane_detection(binary_image_path, instance_image_path, num_clusters)
        draw_lines(instance_out, labels, img_id)
        img_id += 2

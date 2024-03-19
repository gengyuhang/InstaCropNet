import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import cv2

# # 生成一些示例数据
# np.random.seed(0)
# X = 2 * np.random.rand(100, 1)
# y = 4 + 3 * X + np.random.randn(100, 1)
#
# # 使用最小二乘法进行线性回归
# model = LinearRegression()
# model.fit(X, y)
#
# # 获取拟合直线的斜率和截距
# slope = model.coef_[0].item()
# intercept = model.intercept_.item()
#
# # 绘制原始数据和拟合直线使用 matplotlib
# plt.scatter(X, y, label='Original Data')
# plt.plot(X, model.predict(X), color='red', label='Fitted Line (y = {:.2f}x + {:.2f})'.format(slope, intercept))
# # plt.legend()
# # plt.xlabel('X')
# # plt.ylabel('y')
# # plt.title('Linear Regression using Least Squares Method')
# plt.show()


# # Function to fit a straight line using least squares
# def fit_line(x, y):
#     mask = ~np.isnan(x) & ~np.isnan(y)  # Create a mask to filter out NaN values
#     x, y = x[mask], y[mask]
#
#     if np.all(x == x[0]):  # Check if x-values are constant
#         m = np.inf  # Slope is set to infinity for constant x-values
#         c = np.mean(y)  # Intercept is the mean of y-values
#     else:
#         A = np.vstack([x, np.ones_like(x)]).T
#         m, c = np.linalg.lstsq(A, y, rcond=None)[0]
#     return m, c
#
#
# # Function to plot data points and fitted line on an image
# def plot_data_and_line(image, data_points, colors):
#     plt.imshow(image)  # Display the image
#
#     for i, (x, y) in enumerate(data_points):
#         m, c = fit_line(x, y)
#         if np.isinf(m):
#             plt.axvline(x[0], color=colors[i], label=f'Line {i + 1}')
#         else:
#             plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}')
#         plt.scatter(x, y, color=colors[i])  # Scatter plot for data points
#
#     # Set axis limits for lower left corner as origin
#     plt.xlim((0, image.shape[1]))
#     plt.ylim((0, image.shape[0]))
#
#     plt.legend()
#     plt.show()
#
#
# # Example usage
# if __name__ == "__main__":
#     # Replace this with your actual image and data points
#     image = np.zeros((10, 10, 3))  # Placeholder for the image (replace with your image)
#
#     data_points = [
#         (np.array([1, 2.5, 3.4, 4, 5.8]), np.array([2, 3, 4, 5, 6])),
#         (np.array([1, 2, 3, 4, 5]), np.array([3, 4, 5, 6, 7])),
#         (np.array([2, 2, 2, 2, 2]), np.array([3, 4, 5, 6, 7])),
#         (np.array([1, 2, 3, 4, 5]), np.array([5, 6, 7, 8, 9])),
#     ]
#
#     # Filter out sets with NaN values
#     data_points = [(x, y) for x, y in data_points if not (np.any(np.isnan(x)) or np.any(np.isnan(y)))]
#     colors = ['red', 'blue', 'green', 'purple']
#
#     plot_data_and_line(image, data_points, colors)

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
def plot_data_and_line(image, data_points, colors, line_width=1.0):
    plt.imshow(image)  # Display the image

    for i, (x, y) in enumerate(data_points):
        m, c = fit_line(x, y)
        if np.isinf(m):
            plt.axvline(x[0], color=colors[i], label=f'Line {i + 1}', linewidth=line_width)
        else:
            # plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}')
            # plt.scatter(x, y, color=colors[i])  # Scatter plot for data points

            # Find extended points for the line
            x1, y1, x2, y2 = find_extended_points(m, c, image.shape)

            # Plot the original line
            plt.plot(x, m * x + c, color=colors[i], label=f'Line {i + 1}', linewidth=line_width)
            plt.scatter(x, y, color=colors[i], s=20, facecolors='none')  # Scatter plot for data points

            # Plot the extended line
            plt.plot([x1, x2], [y1, y2], linestyle='--', color=colors[i], linewidth=line_width)

    # Set axis limits for lower left corner as origin
    plt.xlim((0, image.shape[1]))
    plt.ylim((0, image.shape[0]))

    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Replace this with your actual image and data points
    image = np.zeros((10, 10, 3))  # Placeholder for the image (replace with your image)

    data_points = [
        (np.array([1, 2.5, 3.3, 3.6, 5]), np.array([2, 3, 4, 5, 6])),
        (np.array([1, 2, 3, 4, 5]), np.array([3, 4, 5, 6, 7])),
        (np.array([2, 2, 2, 2, 2]), np.array([3, 4, 5, 6, 7])),
        (np.array([1, 2, 3, 4, 5]), np.array([5, 6, 7, 8, 9])),
    ]

    # Filter out sets with NaN values
    data_points = [(x, y) for x, y in data_points if not (np.any(np.isnan(x)) or np.any(np.isnan(y)))]

    colors = ['red', 'blue', 'green', 'purple']

    plot_data_and_line(image, data_points, colors, line_width=2.0)




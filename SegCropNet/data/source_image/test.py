# import numpy as np
#
# # Example data_points format
# data_points = [
#     (np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6])),
#     (np.array([1, 2, np.nan, 4, 5]), np.array([3, 4, 5, 6, 7])),
#     (np.array([2, 2, 2, 2, np.nan]), np.array([3, 4, 5, 6, 7])),
#     (np.array([1, 2, 3, 4, 5]), np.array([5, 6, 7, 8, 9])),
# ]
#
# # Remove individual data points with NaN values
# data_points = [
#     (
#         np.array([xi for xi, yi in zip(x, y) if not np.isnan(xi) and not np.isnan(yi)]),
#         np.array([yi for xi, yi in zip(x, y) if not np.isnan(xi) and not np.isnan(yi)])
#     )
#     for x, y in data_points
# ]
#
# # Remove sets with empty arrays resulting from the removal of NaN values
# data_points = [(x, y) for x, y in data_points if len(x) > 0 and len(y) > 0]
#
# # Output the result
# print(data_points)
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# # Load the image
# image = cv2.imread(r'D:\PythonProject\lanenet-lane-detection-pytorch-main\test_output\2_binary_image.jpg')
#
# # Convert the image to a 1D array of pixels
# pixels = image.reshape((-1, 3))
#
# # Filter out non-zero pixels
# valid_pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
#
# # Apply K-means clustering
# kmeans = KMeans(n_clusters=4, random_state=42)  # You can adjust the number of clusters
# kmeans.fit(valid_pixels)
#
# # Get the labels assigned to each pixel
# labels = kmeans.labels_
#
# # Create a mask for non-zero pixels in the original image
# nonzero_mask = np.any(image != [0, 0, 0], axis=-1)
#
# # Create an array with the same size as the original image and fill with zeros
# segmented_image = np.zeros_like(nonzero_mask, dtype=np.uint8)
#
# # Assign the cluster labels to the corresponding non-zero pixels
# segmented_image[nonzero_mask] = labels
#
# # Display the result
# plt.imshow(segmented_image, cmap='jet')
# plt.colorbar()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

# Generate sample 4D data (x, y, z, time)
np.random.seed(42)
num_points = 100
data = np.random.rand(num_points, 4)

# Set up the figure and 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Initial scatter plot
scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=data[:, 3], cmap='viridis', marker='o')

# Update function for animation
def update(frame):
    # Update the scatter plot with new data for each frame
    scatter._offsets3d = (data[:, 0], data[:, 1], data[:, 2])
    scatter.set_array(data[:, 3])

# Create animation
animation = FuncAnimation(fig, update, frames=num_points, interval=200, blit=False)

# Show the plot
plt.show()




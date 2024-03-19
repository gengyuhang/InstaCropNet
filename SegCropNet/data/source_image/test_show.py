#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         test_show
# Description:
# Author:       Ming_King
# Date:         2024/2/29
# -------------------------------------------------------------------------------
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

# Load the image
image = cv2.imread(r'D:\PythonProject\lanenet-lane-detection-pytorch-main\test_output\2_instance_image.jpg')

# Convert the image to RGB (OpenCV loads images in BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Flatten the image to a 2D array of pixels
pixels = image_rgb.reshape((-1, 3))

# Define the number of clusters
num_clusters = 5

# Initialize KMeans model
kmeans = KMeans(n_clusters=num_clusters)

# Fit KMeans model to the pixel data
kmeans.fit(pixels)

# Get the cluster centroids and labels
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Plotting the clusters in the feature space (3D plot for RGB space)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each cluster's centroid and the pixels assigned to it
for i in range(num_clusters):
    cluster_pixels = pixels[labels == i]
    # ax.scatter(cluster_pixels[:, 0], cluster_pixels[:, 1], cluster_pixels[:, 2], label=f'Cluster {i}')
    ax.scatter(cluster_pixels[:, 0], cluster_pixels[:, 1], cluster_pixels[:, 2])

# Plot centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='black', marker='x', s=100, label='Centroids')

# ax.set_xlabel('Red')
# ax.set_ylabel('Green')
# ax.set_zlabel('Blue')
# ax.set_title('Pixel Clusters in RGB Space')
ax.legend()

plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
# Name:         model_info
# Description:
# Author:       Ming_King
# Date:         2024/1/17
# -------------------------------------------------------------------------------
import torch
from thop import profile

# net = Model()  # 定义好的网络模型
# inputs = torch.randn(1, 3, 112, 112)
# flops, params = profile(net, (inputs,))
# print('flops: ', flops, 'params: ', params)
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
#
# # Set a seed for reproducibility
# np.random.seed(42)
#
# # Generate data points for 4 clusters with some noise
# data = []
# centers = np.array([[2, 2], [8, 3], [3, 6], [6, 8]])
# for center in centers:
#     cluster_points = center + np.random.normal(scale=0.5, size=(250, 2))
#     data.extend(cluster_points)
#
# # Adding some noise points
# noise = np.random.uniform(low=-5, high=15, size=(100, 2))
# data.extend(noise)
#
# data = np.array(data)
#
# # Perform K-means clustering with 4 clusters
# kmeans = KMeans(n_clusters=4, random_state=42)
# kmeans.fit(data)
# labels = kmeans.labels_
# centers = kmeans.cluster_centers_
#
# # Plot the data points and cluster centers
# plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='w')
# plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200)
# plt.title('K-means Clustering Results')
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set a seed for reproducibility
np.random.seed(42)

# Generate data points for 4 clusters with some noise
data = []
centers = np.array([[2, 2], [8, 3], [4, 6], [9, 13]])
for center in centers:
    cluster_points = center + np.random.normal(scale=0.5, size=(300, 2))
    data.extend(cluster_points)

# Adding some noise points
noise = np.random.uniform(low=-5, high=15, size=(80, 2))
data.extend(noise)

data = np.array(data)

# Perform K-means clustering with 4 clusters
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(data)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

# Plot the data points and cluster centers with improved aesthetics
plt.figure(figsize=(10, 8))
scatter = plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='w', s=50)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Cluster Centers')

# Aesthetics improvements
plt.xticks([])  # Hide x-axis
plt.yticks([])  # Hide y-axis
plt.title('')   # Hide title

# Set font to New Roman
font = {'family': 'serif', 'serif': ['Times New Roman'], 'size': 14}
plt.rc('font', **font)

# Add legend with clear labels
legend_labels = [f'Cluster {i+1}' for i in range(4)]
plt.legend(handles=scatter.legend_elements()[0], title='Clusters', labels=legend_labels, fontsize=12)

# Remove spines (axes lines)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['bottom'].set_visible(False)
plt.gca().spines['left'].set_visible(False)

# Remove ticks
plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

# Save the figure for academic use
plt.savefig('kmeans_clustering_results.png', dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()




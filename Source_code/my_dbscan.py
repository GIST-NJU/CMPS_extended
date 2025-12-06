import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from sklearn.preprocessing import normalize


def cosine_distance(v1, v2):
    return 1 - np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def dbscan(X, eps, min_samples):
    labels = np.full(X.shape[0], -1)  # 初始化所有点的标签为 -1 (噪声)
    cluster_id = 0

    def region_query(point):
        neighbors = []
        for i in range(X.shape[0]):
            if cosine_distance(X[point], X[i]) < eps:
                neighbors.append(i)
        return neighbors

    def expand_cluster(point, neighbors):
        queue = deque(neighbors)
        labels[point] = cluster_id

        while queue:
            current_point = queue.popleft()
            if labels[current_point] == -1:  # 如果是噪声点，则归属当前簇
                labels[current_point] = cluster_id
            if labels[current_point] == 0:  # 如果是未分类点，则继续检查
                labels[current_point] = cluster_id
                current_neighbors = region_query(current_point)
                if len(current_neighbors) >= min_samples:
                    queue.extend(current_neighbors)

    for point in range(X.shape[0]):
        if labels[point] != -1:
            continue
        neighbors = region_query(point)
        if len(neighbors) < min_samples:
            labels[point] = -1  # 标记为噪声
        else:
            cluster_id += 1
            expand_cluster(point, neighbors)

    return labels

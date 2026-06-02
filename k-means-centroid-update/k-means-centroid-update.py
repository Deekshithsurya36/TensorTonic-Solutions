import numpy as np
def k_means_centroid_update(points, assignments, k):
    points = np.array(points, dtype=float)
    centroids = []
    for i in range(k):
        cluster = points[np.array(assignments) == i]
        if len(cluster) == 0:
            centroids.append([0.0] * points.shape[1])
        else:
            centroids.append(cluster.mean(axis=0).tolist())
    return centroids
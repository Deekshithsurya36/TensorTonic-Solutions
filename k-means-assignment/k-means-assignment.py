import numpy as np

def k_means_assignment(points, centroids):
    points = np.array(points)
    centroids = np.array(centroids)

    assignments = []

    for p in points:
        d = np.sum((centroids - p) ** 2, axis=1)
        assignments.append(int(np.argmin(d)))

    return assignments
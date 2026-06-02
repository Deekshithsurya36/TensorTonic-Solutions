import numpy as np
def decision_tree_split(X, y):
    X = np.array(X)
    y = np.array(y)
    def gini(labels):
        if len(labels) == 0:
            return 0
        _, counts = np.unique(labels, return_counts=True)
        p = counts / len(labels)
        return 1 - np.sum(p ** 2)
    parent_gini = gini(y)
    best_gain = -1
    best_feature = 0
    best_threshold = 0
    n_samples, n_features = X.shape
    for f in range(n_features):
        values = np.unique(X[:, f])
        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            left = y[X[:, f] <= threshold]
            right = y[X[:, f] > threshold]
            gini_split = (
                len(left) / n_samples * gini(left)
                + len(right) / n_samples * gini(right)
            )
            gain = parent_gini - gini_split
            if gain > best_gain:
                best_gain = gain
                best_feature = f
                best_threshold = threshold
    return [int(best_feature), float(best_threshold)]
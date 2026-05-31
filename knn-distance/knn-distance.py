import numpy as np

def knn_distance(X_train, X_test, k):

    X_train = np.array(X_train)
    X_test = np.array(X_test)

    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)

    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    distances = np.sqrt(np.sum((X_test[:, np.newaxis] - X_train) ** 2, axis=2))

    indices = np.argsort(distances, axis=1)

    result = np.full((len(X_test), k), -1, dtype=int)

    m = min(k, len(X_train))
    result[:, :m] = indices[:, :m]

    return result
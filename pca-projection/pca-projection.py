import numpy as np
def pca_projection(X, k):
    X = np.array(X, dtype=float)
    Xc = X - np.mean(X, axis=0)
    C = np.cov(Xc, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(C)
    idx = np.argsort(eigvals)[::-1]
    W = eigvecs[:, idx[:k]]
    return (Xc @ W).tolist()
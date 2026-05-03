import numpy as np

def entropy_node(y):
    y = np.array(y)

    if len(y) == 0:
        return 0.0

    values, counts = np.unique(y, return_counts=True)

    probabilities = counts / len(y)

    entropy = 0.0

    for p in probabilities:
        if p > 0:
            entropy -= p * np.log2(p)

    return float(entropy)
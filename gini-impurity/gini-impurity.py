import numpy as np

def gini_impurity(y_left, y_right):

    # Function to calculate gini of one node
    def gini(node):

        # Empty node
        if len(node) == 0:
            return 0.0

        node = np.array(node)
        values, counts = np.unique(node, return_counts=True)
        probabilities = counts / len(node)

        return 1 - np.sum(probabilities ** 2)

    n_left = len(y_left)
    n_right = len(y_right)
    total = n_left + n_right
    
    if total == 0:
        return 0.0

    weighted_gini = (
        (n_left / total) * gini(y_left)
        + (n_right / total) * gini(y_right)
    )

    return float(weighted_gini)
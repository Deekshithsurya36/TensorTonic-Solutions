import numpy as np

def _entropy(y):
    """
    Helper: Compute Shannon entropy (base 2) for labels y.
    """
    y = np.asarray(y)
    if y.size == 0:
        return 0.0
    vals, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum()) if p.size else 0.0

def information_gain(y, split_mask):
    """
    Compute Information Gain of a binary split on labels y.
    Use the _entropy() helper above.
    """
    # Write code here
    pass
def information_gain(y, split_mask):
    y = np.asarray(y)
    split_mask = np.asarray(split_mask)

    left = y[split_mask]
    right = y[~split_mask]

    if len(left) == 0 or len(right) == 0:
        return 0.0

    H = _entropy(y)
    HL = _entropy(left)
    HR = _entropy(right)

    return float(H - (len(left)/len(y))*HL - (len(right)/len(y))*HR)
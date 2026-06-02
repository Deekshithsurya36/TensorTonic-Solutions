import numpy as np
def random_forest_vote(predictions):
    predictions = np.array(predictions)
    result = []
    for col in predictions.T:
        votes = np.bincount(col)
        result.append(np.argmax(votes))
    return result
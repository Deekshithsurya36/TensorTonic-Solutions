import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)

    classes = np.sort(np.unique(y_train))
    n_classes = len(classes)

    log_probs = np.zeros((len(X_test), n_classes))

    for idx, c in enumerate(classes):
        X_c = X_train[y_train == c]

        prior = len(X_c) / len(X_train)
        theta = (np.sum(X_c, axis=0) + 1) / (len(X_c) + 2)

        log_prior = np.log(prior)

        log_likelihood = (
            X_test * np.log(theta) +
            (1 - X_test) * np.log(1 - theta)
        ).sum(axis=1)

        log_probs[:, idx] = log_prior + log_likelihood

    return np.round(log_probs, 3)
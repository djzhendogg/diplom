from scipy.stats import spearmanr


def spearman_correlation(y_true, y_pred):
    correlation, p_value = spearmanr(y_true, y_pred)
    return correlation

def spearman_correlation_with_p_val(y_true, y_pred):
    correlation, p_value = spearmanr(y_true, y_pred)
    return correlation, p_value

def spearman_scorer(estimator, X, y):
    y_pred = estimator.predict(X)
    return spearman_correlation(y, y_pred)
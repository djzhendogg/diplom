import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run_cv(target_type, regressor_type, x_df, y_df):
    cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)

    if regressor_type == 'linear':
        regressor = LinearRegression()
        use_scaler = True

    elif regressor_type == 'lasso':
        regressor = Lasso(alpha=0.01, random_state=42)
        use_scaler = True

    elif regressor_type == 'ridge':
        regressor = Ridge(alpha=1.0, random_state=42)
        use_scaler = True

    elif regressor_type == 'rf':
        regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42
        )
        use_scaler = False

    else:
        print("No such type of model")
        return

    X = x_df
    y = y_df[target_type]

    steps = []
    if use_scaler:
        steps.append(('scaler', StandardScaler()))

    steps.append(('pca', PCA(n_components=0.95)))
    steps.append(('regressor', regressor))
    pipeline = Pipeline(steps)

    spearman_scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        spearman = spearmanr(y_test, y_pred).correlation
        spearman_scores.append(spearman)

    print("\n===== RESULT (Repeated CV) =====")
    print(f"Spearman mean: {np.mean(spearman_scores):.4f}")
    print(f"Spearman std: {np.std(spearman_scores):.4f}")

    return {
        "spearman_mean": np.mean(spearman_scores),
        "spearman_std": np.std(spearman_scores),
        "all_scores": spearman_scores
    }

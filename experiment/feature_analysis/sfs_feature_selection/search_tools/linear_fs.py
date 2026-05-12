import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sp_scorer import spearman_scorer


def run_fs(target_type, regressor_type, x_df, y_df):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    if regressor_type == 'linear':
        regressor = LinearRegression()
    elif regressor_type == 'lasso':
        regressor = Lasso(alpha=0.01, random_state=42)
    elif regressor_type == 'ridge':
        regressor = Ridge(random_state=42)
    elif regressor_type == 'rf':
        regressor = RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=42
        )
    else:
        print("No such type of model")
        return

    X = x_df
    y = y_df[target_type]

    max_features = X.shape[1]

    outer_spearman_scores = []

    selected_features_per_fold = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\n===== OUTER FOLD {fold + 1} =====")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        best_score = -np.inf
        best_features = None

        # 🔁 перебор количества фич ТОЛЬКО на train
        # for n in range(5, max_features):
        sfs = SequentialFeatureSelector(
            Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', regressor)
            ]),
            direction='backward',
            scoring=spearman_scorer,
            cv=inner_cv,
            n_features_to_select=5
        )

        sfs.fit(X_train, y_train)
        features = X_train.columns[sfs.get_support()]

        X_train_fs = X_train[features]

        scores = cross_val_score(
            Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', regressor)
            ]),
            X_train_fs,
            y_train,
            cv=inner_cv,
            scoring=spearman_scorer
        )

        score = scores.mean()

        if score > best_score:
            best_score = score
            best_features = features

        print(f"Лучшее число фич: {len(best_features)}")

        # 🔒 финальная модель обучается только на train
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

        final_pipeline.fit(X_train[best_features], y_train)

        # 🎯 тестируем на НОВЫХ данных (outer test fold)
        y_pred = final_pipeline.predict(X_test[best_features])

        spearman = spearmanr(y_test, y_pred).correlation

        print(f"Spearman (test): {spearman:.4f}")

        outer_spearman_scores.append(spearman)
        selected_features_per_fold.append(list(best_features))

    print("\n===== ИТОГ (NESTED CV) =====")
    print(f"Spearman mean: {np.mean(outer_spearman_scores):.4f} (+/- {np.std(outer_spearman_scores) * 2:.4f})")

    return {
        "spearman_mean": np.mean(outer_spearman_scores),
        "spearman_std": np.std(outer_spearman_scores),
        "features_per_fold": selected_features_per_fold
    }

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .sp_scorer import spearman_scorer


def _nested_cv(X, y, get_features_func):
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    outer_scores = []
    selected_features = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
        print(f"\n===== OUTER FOLD {fold + 1} =====")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        features, model = get_features_func(X_train, y_train, inner_cv)

        model.fit(X_train[features], y_train)
        y_pred = model.predict(X_test[features])

        score = spearmanr(y_test, y_pred).correlation

        print(f"Features: {len(features)} | Spearman: {score:.4f}")

        outer_scores.append(score)
        selected_features.append(list(features))

    return {
        "spearman_mean": np.mean(outer_scores),
        "spearman_std": np.std(outer_scores),
        "features_per_fold": selected_features
    }


def _select_best_k_sfs(model, X_train, y_train, inner_cv, k_range):
    best_score = -np.inf
    best_features = None

    for k in k_range:
        sfs = SequentialFeatureSelector(
            model,
            direction='backward',
            scoring=spearman_scorer,
            cv=inner_cv,
            n_features_to_select=k
        )

        sfs.fit(X_train, y_train)
        features = X_train.columns[sfs.get_support()]

        scores = cross_val_score(
            model,
            X_train[features],
            y_train,
            cv=inner_cv,
            scoring=spearman_scorer
        )

        score = scores.mean()

        if score > best_score:
            best_score = score
            best_features = features

    return best_features


def run_linear_fs(X, y, k_range=None):
    def get_features(X_train, y_train, inner_cv):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])

        if k_range is None:
            max_k = min(20, X_train.shape[1])
            k_vals = range(5, max_k)
        else:
            k_vals = k_range

        features = _select_best_k_sfs(model, X_train, y_train, inner_cv, k_vals)

        return features, model

    return _nested_cv(X, y, get_features)


def run_ridge_fs(X, y, k_range=None):
    def get_features(X_train, y_train, inner_cv):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(random_state=42))
        ])

        if k_range is None:
            max_k = min(20, X_train.shape[1])
            k_vals = range(2, max_k)
        else:
            k_vals = k_range

        features = _select_best_k_sfs(model, X_train, y_train, inner_cv, k_vals)

        return features, model

    return _nested_cv(X, y, get_features)


def run_lasso_fs(X, y):
    def get_features(X_train, y_train, inner_cv):
        model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LassoCV(
                cv=inner_cv,
                random_state=42,
                n_alphas=100,
                max_iter=10000
            ))
        ])

        model.fit(X_train, y_train)

        lasso = model.named_steps['regressor']
        coef = lasso.coef_

        features = X_train.columns[coef != 0]

        # fallback если Lasso занулил всё
        if len(features) == 0:
            features = X_train.columns

        print(f"Alpha chosen: {lasso.alpha_:.6f} | Features: {len(features)}")

        return features, model

    return _nested_cv(X, y, get_features)


def run_rf_sfs(X, y, k_range=None):
    def get_features(X_train, y_train, inner_cv):
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42
        )

        if k_range is None:
            max_k = min(20, X_train.shape[1])
            k_vals = range(2, max_k)
        else:
            k_vals = k_range

        features = _select_best_k_sfs(model, X_train, y_train, inner_cv, k_vals)

        return features, model

    return _nested_cv(X, y, get_features)


def _select_best_k_rf(X_train, y_train, inner_cv, model, importance_getter, k_range):
    best_score = -np.inf
    best_features = None

    model.fit(X_train, y_train)

    importances = importance_getter(model, X_train, y_train)

    ranked_idx = np.argsort(importances)[::-1]

    for k in k_range:
        idx = ranked_idx[:k]
        features = X_train.columns[idx]

        scores = cross_val_score(
            model,
            X_train[features],
            y_train,
            cv=inner_cv,
            scoring=spearman_scorer
        )

        score = scores.mean()

        if score > best_score:
            best_score = score
            best_features = features

    return best_features


def rf_permutation_getter(model, X, y):
    result = permutation_importance(
        model,
        X,
        y,
        scoring=spearman_scorer,
        n_repeats=10,
        random_state=42
    )
    return result.importances_mean


def rf_importance_getter(model, X, y):
    return model.feature_importances_


def run_rf_importance(X, y, k_range=None):
    def get_features(X_train, y_train, inner_cv):
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42
        )

        if k_range is None:
            max_k = min(20, X_train.shape[1])
            k_vals = range(5, max_k)
        else:
            k_vals = k_range

        features = _select_best_k_rf(
            X_train,
            y_train,
            inner_cv,
            model,
            rf_importance_getter,
            k_vals
        )

        return features, model

    return _nested_cv(X, y, get_features)


def run_rf_permutation(X, y, k_range=None):
    def get_features(X_train, y_train, inner_cv):
        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_leaf=2,
            random_state=42
        )

        if k_range is None:
            max_k = min(20, X_train.shape[1])
            k_vals = range(5, max_k)
        else:
            k_vals = k_range

        features = _select_best_k_rf(
            X_train,
            y_train,
            inner_cv,
            model,
            rf_permutation_getter,
            k_vals
        )

        return features, model

    return _nested_cv(X, y, get_features)

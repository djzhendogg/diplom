from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from experiment.final_algorithm.utils import spearman_scorer


def run_cv(col_type, target_type, regressor_type, x_df, y_df):
    X = x_df[col_type[1]]
    y = y_df[target_type]

    if regressor_type == 'linear':
        regressor = LinearRegression()
    elif regressor_type == 'lasso':
        regressor = Lasso(alpha=0.01)
    elif regressor_type == 'tree':
        regressor = DecisionTreeRegressor(random_state=42, max_leaf_nodes=10)
    elif regressor_type == 'ridge':
        regressor = Ridge(alpha=1.0)
    else:
        print("No such type of model")
        return
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # mse_scores = -cross_val_score(pipeline, X, y, cv=cv,
    #                               scoring='neg_mean_squared_error')
    #
    # r2_scores = cross_val_score(pipeline, X, y, cv=cv,
    #                             scoring='r2')

    spearman_scores = cross_val_score(pipeline, X, y, cv=cv,
                                      scoring=spearman_scorer)

    # Выводим результаты кросс-валидации
    print("=== РЕЗУЛЬТАТЫ КРОСС-ВАЛИДАЦИИ (5-FOLD) ===")
    print(f"{col_type[0]}: {col_type[1]}")
    print(f'параметры: target_type: {target_type}, regressor_type: {regressor_type}')
    # print(f"MSE на CV: {mse_scores}")
    # print(f"Средний MSE на CV: {mse_scores.mean():.4f} (+/- {mse_scores.std() * 2:.4f})")
    # print(f"\nR2 на CV: {r2_scores}")
    # print(f"Средний R2 на CV: {r2_scores.mean():.4f} (+/- {r2_scores.std() * 2:.4f})")
    print(f"\nSpearman на CV: {spearman_scores}")
    print(f"Средний Spearman на CV: {spearman_scores.mean():.4f} (+/- {spearman_scores.std() * 2:.4f})")
    print("=" * 50)
    return spearman_scores.mean(), spearman_scores.std()

from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from experiment.feature_analysis.sfs_feature_selection.search_tools.linear_cv import run_cv
from experiment.feature_analysis.sfs_feature_selection.search_tools.sp_scorer import spearman_scorer


def run_fs(target_type, regressor_type, x_df, y_df):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
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
    X = x_df
    y = y_df[target_type]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', regressor)
    ])
    sfs = SequentialFeatureSelector(pipeline, direction='backward', scoring=spearman_scorer, cv=cv)
    sfs.fit(X, y)
    features_fs = sfs.get_feature_names_out()
    spearman_mean, spearman_std, r2_mean, r2_std = run_cv(("fs features", features_fs), target_type, regressor_type, x_df, y_df)
    return spearman_mean, spearman_std, r2_mean, r2_std, features_fs

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from experiment.final_algorithm.utils import spearman_scorer, spearman_correlation_with_p_val

models_aggregated_path = "../models/results/models_aggregated_mean.csv"
features_problexity_path = "../dc_problexity/results/problexity_significant.csv"
features_sd_path = "../dc_sequence_diversity/results/sequence_diversity_significant.csv"
features_fgw_path = "../ot_computation/results/fgw_small_for_model.csv"
# features_fugw_path = "../ot_computation/results/fgw_small_for_model.csv"

models_aggregated_df = pd.read_csv(models_aggregated_path, index_col='name')
target_column = models_aggregated_df.columns

features_problexity_df = pd.read_csv(features_problexity_path, index_col='name')
features_problexity_df.sort_index(ascending=False, inplace=True)
features_sd_df = pd.read_csv(features_sd_path, index_col='name')
features_fgw_df = pd.read_csv(features_fgw_path, index_col='name')

full_features = pd.concat([features_problexity_df, features_sd_df, features_fgw_df], axis=1)
full_df = pd.concat([full_features, models_aggregated_df], axis=1)

targets = full_df[target_column]
features = full_df.drop(target_column, axis=1)

target_type = 'f1_mean'
X = features
y = targets[target_type]
pipeline_fs = Pipeline([
    ('scaler', StandardScaler()),
    ('regressor', Ridge(alpha=1.0))
])

cv_fs = KFold(n_splits=5, shuffle=True, random_state=42)
sfs = SequentialFeatureSelector(pipeline_fs, direction='backward', scoring=spearman_scorer, cv=cv_fs)
sfs.fit(X, y)
features_fs = sfs.get_feature_names_out()

print(features_fs)

X_selected = features[features_fs]
cv_model = KFold(n_splits=5, shuffle=True, random_state=42)
pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', Ridge(alpha=1.0))
    ])
spearman_scores = cross_val_score(pipeline, X_selected, y, cv=cv_model,
                                      scoring=spearman_scorer)
print(f"\nSpearman на CV: {spearman_scores}")
print(f"Средний Spearman на CV: {spearman_scores.mean():.4f} (+/- {spearman_scores.std() * 2:.4f})")

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
with open(f'scaled_train/columns/{target_type}.txt', 'w', encoding='utf-8') as f:
    for col in X_train.columns:
        f.write(f"{col}\n")
sc = StandardScaler()
sc.fit(X_train)
X_train_sc = sc.transform(X_train)

np.save(f'scaled_train/x/{target_type}.npy', X_train_sc)
X_test_sc = sc.transform(X_test)

ridge = Ridge(alpha=1.0)
ridge.fit(X_train_sc, y_train)
y_test_pred_ridge = ridge.predict(X_test_sc)
y_train_pred_ridge = ridge.predict(X_train_sc)
sp, p_val = spearman_correlation_with_p_val(y_train, y_train_pred_ridge)
print(f"Train - Spearman: {sp}, p=value: {p_val}")
sp, p_val = spearman_correlation_with_p_val(y_test, y_test_pred_ridge)
print(f"Test - Spearman: {sp}, p=value: {p_val}")
print(ridge.coef_)

joblib.dump(ridge, f'ridge/{target_type}/model.joblib')
joblib.dump(sc, f'ridge/{target_type}/scaler.joblib')

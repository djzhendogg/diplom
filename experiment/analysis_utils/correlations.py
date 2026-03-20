from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def cross_correlation(df1, df2, method='pearson'):
    result = pd.DataFrame(index=df1.columns, columns=df2.columns)

    for col1 in df1.columns:
        for col2 in df2.columns:
            if method == 'pearson':
                result.loc[col1, col2] = df1[col1].corr(df2[col2])
            elif method == 'spearman':
                result.loc[col1, col2] = df1[col1].corr(df2[col2], method='spearman')
            else:
                raise ValueError("Method must be 'pearson' or 'spearman'")

    return result.astype(float)


def select_non_correlated_features_with_greedy(df1, df2, correlation_threshold=0.5):
    """
    Альтернативный подход: жадный алгоритм с учетом корреляции и важности

    На каждом шаге выбирается признак с максимальной средней корреляцией к df2,
    который не коррелирует сильно с уже выбранными признаками.
    """

    features_df1 = df1.columns.tolist()
    target_cols = df2.columns.tolist()

    feature_corr_matrix = pd.DataFrame(index=features_df1, columns=features_df1)

    for feat1, feat2 in combinations(features_df1, 2):
        mask = df1[feat1].notna() & df1[feat2].notna()
        if mask.sum() > 1:
            corr, _ = spearmanr(df1[feat1][mask], df1[feat2][mask])
            feature_corr_matrix.loc[feat1, feat2] = abs(corr)
            feature_corr_matrix.loc[feat2, feat1] = abs(corr)

    feature_target_corr = {}
    for feat in features_df1:
        target_corrs = []
        for target in target_cols:
            mask = df1[feat].notna() & df2[target].notna()
            if mask.sum() > 1:
                corr, _ = spearmanr(df1[feat][mask], df2[target][mask])
                target_corrs.append(abs(corr))
        feature_target_corr[feat] = np.mean(target_corrs) if target_corrs else 0

    sorted_by_importance = sorted(feature_target_corr.items(),
                                  key=lambda x: x[1],
                                  reverse=True)

    selected_features = []

    for feat, importance in sorted_by_importance:
        if not selected_features:
            selected_features.append(feat)
        else:
            max_corr_with_selected = 0
            for selected in selected_features:
                if pd.notna(feature_corr_matrix.loc[feat, selected]):
                    max_corr_with_selected = max(max_corr_with_selected,
                                                 feature_corr_matrix.loc[feat, selected])

            if max_corr_with_selected < correlation_threshold:
                selected_features.append(feat)

    return selected_features, sorted_by_importance, feature_corr_matrix

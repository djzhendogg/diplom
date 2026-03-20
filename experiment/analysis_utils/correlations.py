import pandas as pd


def cross_correlation(df1, df2, method='spearman'):
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


def select_non_correlated_features_with_greedy(features, targets, correlation_threshold):
    """
    жадный алгоритм для выбора наиболее релевантных для таргета некоррелирующих признаков
    """
    feature_corr_matrix = features.corr(method='spearman')
    feature_target_corr_matrix = cross_correlation(features, targets, method='spearman')

    feature_target_corr = feature_target_corr_matrix.mean(axis=1).to_dict()
    sorted_by_importance = sorted(feature_target_corr.items(),
                                  key=lambda x: abs(x[1]),
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

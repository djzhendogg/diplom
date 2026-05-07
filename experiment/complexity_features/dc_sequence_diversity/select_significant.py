import pandas as pd
from sklearn.feature_selection import VarianceThreshold

df = pd.read_csv('results/sequence_diversity.csv', index_col='name')
sel = VarianceThreshold()
sel.fit(df)
significant_variance_features = sel.get_feature_names_out()
to_drop = set(df.columns) - set(significant_variance_features)
print(f"Low variance features:{to_drop}")

df_significant = df[significant_variance_features]
df_significant.to_csv('results/sequence_diversity_significant.csv')

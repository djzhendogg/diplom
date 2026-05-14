import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE

from experiment.clustering.utils.prepare_clustering_data import scale_pca

benchmark_df = pd.read_csv("../results/ready_data/clustered_benchmarks.csv", index_col="name")

labels = benchmark_df["clusters"]

cluster_ranks = (benchmark_df.groupby(["clusters", "rank", "rank_weight"])["mcc"].mean().sort_values(ascending=False))

features = benchmark_df.drop(["clusters", "rank", "rank_weight"], axis=1)

X_pca = scale_pca(features)

tsne = TSNE(n_components=2, perplexity=50, learning_rate="auto", max_iter=1000, init="random", random_state=42)

X_tsne = tsne.fit_transform(X_pca)

unique_labels = np.unique(labels)
colors = cm.get_cmap("viridis", len(unique_labels))

x_min = X_tsne[:, 0].min() - 1
x_max = X_tsne[:, 0].max() + 1

y_min = X_tsne[:, 1].min() - 1
y_max = X_tsne[:, 1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

background = np.zeros(xx.shape)

for i, label in enumerate(unique_labels):
    mask = labels == label

    xy = X_tsne[mask].T

    kde = gaussian_kde(xy)

    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zz /= zz.max()

    background += zz * (i + 1)

plt.figure(figsize=(15, 10), dpi=1500)

plt.imshow(background, origin="lower", extent=(x_min, x_max, y_min, y_max), cmap="viridis", alpha=0.3)

for i, label in enumerate(unique_labels):
    mask = labels == label

    plt.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=70, color=colors(i),
        label=f"{label}, baseline={round(cluster_ranks.loc[i], 3)}")

# plt.legend(title="Clusters", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
plt.savefig("../results/plots/t-SNE.png")

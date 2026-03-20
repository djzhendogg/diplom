import matplotlib.pyplot as plt
import seaborn as sns


def pairwise_heatmap(correlation_matrix, save_path, figsize=None, fontsize=14, dpi=800):
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        correlation_matrix,
        xticklabels=False,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f"
    )
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def target_wise_heatmap(correlation_matrix, save_path, figsize=None, fontsize=14, dpi=800):
    plt.figure(figsize=figsize, dpi=dpi)
    sns.heatmap(
        correlation_matrix,
        vmin=-1, vmax=1,
        annot=True, fmt=".2f"
    )
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize, rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

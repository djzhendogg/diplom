import numpy as np
from Levenshtein import distance
from scipy.stats import entropy
from seqshannon import shannon_entropy

from levenshtein_distance_tools import (
    build_histograms_from_distance_matrix,
    compute_global_H_from_mixture
)


def general_characterize(df, name):
    levenshtein_distances_matrix = levenshtein_distances(df['sequence'].to_list())
    p_i, p_global, edges = build_histograms_from_distance_matrix(levenshtein_distances_matrix, n_bins=12, alpha=1e-9)

    df['len'] = df['sequence'].str.len()
    df['unique_trimers'] = df['sequence'].apply(unique_trimers)
    df['unique_trimers_on_all'] = df['sequence'].apply(unique_trimers_on_all_trimers)
    df['shannon_entropy'] = df['sequence'].apply(shannon_entropy)

    entropy_levenshtein = compute_global_H_from_mixture(p_global)
    mean_levenshtein = np.mean(levenshtein_distances_matrix)
    std_levenshtein = np.std(levenshtein_distances_matrix)
    median_levenshtein = np.median(levenshtein_distances_matrix)
    min_levenshtein = np.min(levenshtein_distances_matrix)
    max_levenshtein = np.max(levenshtein_distances_matrix)

    entropy_len = feature_entropy(df['len'])
    mean_len = df['len'].mean()
    std_len = np.std(df['len'])
    median_len = np.median(df['len'])
    min_len = int(df['len'].min())
    max_len = int(df['len'].max())

    entropy_shannon_entropy = feature_entropy(df['shannon_entropy'])
    mean_shannon_entropy = df['shannon_entropy'].mean()
    std_shannon_entropy = np.std(df['shannon_entropy'])
    median_shannon_entropy = np.median(df['shannon_entropy'])
    min_shannon_entropy = df['shannon_entropy'].min()
    max_shannon_entropy = df['shannon_entropy'].max()

    entropy_unique_trimers = feature_entropy(df['unique_trimers'])
    mean_unique_trimers = df['unique_trimers'].mean()
    std_unique_trimers = np.std(df['unique_trimers'])
    median_unique_trimers = np.median(df['unique_trimers'])
    min_unique_trimers = int(df['unique_trimers'].min())
    max_unique_trimers = int(df['unique_trimers'].max())

    entropy_unique_trimers_on_all = feature_entropy(df['unique_trimers_on_all'])
    mean_unique_trimers_on_all = df['unique_trimers_on_all'].mean()
    std_unique_trimers_on_all = np.std(df['unique_trimers_on_all'])
    median_unique_trimers_on_all = np.median(df['unique_trimers_on_all'])
    min_unique_trimers_on_all = df['unique_trimers_on_all'].min()
    max_unique_trimers_on_all = df['unique_trimers_on_all'].max()

    return {
        "name": name,
        "samples_num": df.shape[0],

        "entropy_levenshtein": entropy_levenshtein,
        "mean_levenshtein":mean_levenshtein,
        "std_levenshtein":std_levenshtein,
        "median_levenshtein":median_levenshtein,
        "min_levenshtein":min_levenshtein,
        "max_levenshtein":max_levenshtein,

        "entropy_len": entropy_len,
        "mean_len": mean_len,
        "std_len": std_len,
        "median_len": median_len,
        "min_len": min_len,
        "max_len": max_len,

        "entropy_shannon_entropy": entropy_shannon_entropy,
        "mean_shannon_entropy": mean_shannon_entropy,
        "std_shannon_entropy": std_shannon_entropy,
        "median_shannon_entropy": median_shannon_entropy,
        "min_shannon_entropy": min_shannon_entropy,
        "max_shannon_entropy": max_shannon_entropy,

        "entropy_unique_trimers": entropy_unique_trimers,
        "mean_unique_trimers": mean_unique_trimers,
        "std_unique_trimers": std_unique_trimers,
        "median_unique_trimers": median_unique_trimers,
        "min_unique_trimers": min_unique_trimers,
        "max_unique_trimers": max_unique_trimers,

        "entropy_unique_trimers_on_all": entropy_unique_trimers_on_all,
        "mean_unique_trimers_on_all": mean_unique_trimers_on_all,
        "std_unique_trimers_on_all": std_unique_trimers_on_all,
        "median_unique_trimers_on_all": median_unique_trimers_on_all,
        "min_unique_trimers_on_all": min_unique_trimers_on_all,
        "max_unique_trimers_on_all": max_unique_trimers_on_all,
    }


def full_dataset_metrics(df):
    levenshtein_distances_matrix = levenshtein_distances(df['seq'].to_list())
    p_i, p_global, edges = build_histograms_from_distance_matrix(levenshtein_distances_matrix, n_bins=12, alpha=1e-9)

    df['len'] = df['seq'].str.len()
    df['unique_trimers_on_all'] = df['seq'].apply(unique_trimers_on_all_trimers)
    df['shannon_entropy'] = df['seq'].apply(shannon_entropy)
    H_global = compute_global_H_from_mixture(p_global)
    return df, H_global


def calculate_entropy(series):
    value_counts = series.value_counts()
    probabilities = value_counts / len(series)
    return entropy(probabilities, base=2)


def levenshtein_distances(sequences: list) -> np.array:
    distances = np.zeros((len(sequences), len(sequences)))
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            distances[i, j] = distance(sequences[i], sequences[j])
            distances[j, i] = distances[i, j]
    return distances


def feature_entropy(values):
    values = np.asarray(values)

    # Min-Max нормализация
    vmin, vmax = values.min(), values.max()
    if vmax == vmin:
        return 0.0
    values_norm = (values - vmin) / (vmax - vmin)
    bins = optimal_bins_fd(values_norm)
    hist, edges = np.histogram(values_norm, bins=bins, range=(0, 1))
    p = hist / hist.sum()
    p = p[p > 0]

    return entropy(p, base=np.e)


def optimal_bins_fd(values):
    """
    Правило Фридмана-Диакониса: 2 * IQR * n^(-1/3)
    """
    q75, q25 = np.percentile(values, [75, 25])
    iqr = q75 - q25
    n = len(values)

    if iqr == 0:
        return 10

    bin_width = 2 * iqr / (n ** (1 / 3))
    bins = int((values.max() - values.min()) / bin_width)
    return max(10, min(100, bins))

def count_trimers(sequence):
    if len(sequence) < 3:
        return {}, 0
    trimer_counts = {}
    for i in range(len(sequence) - 2):
        trimer = sequence[i:i + 3]
        if trimer in trimer_counts:
            trimer_counts[trimer] += 1
        else:
            trimer_counts[trimer] = 1
    unique_trimers_count = len(trimer_counts)
    return trimer_counts, unique_trimers_count


def unique_trimers(sequence):
    _, unique = count_trimers(sequence)
    return unique


def unique_trimers_on_all_trimers(sequence):
    _, unique = count_trimers(sequence)
    if len(sequence) < 3:
        return 0
    else:
        return unique / (len(sequence) - 2)

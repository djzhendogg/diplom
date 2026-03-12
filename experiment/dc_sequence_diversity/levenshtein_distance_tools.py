import numpy as np

def build_histograms_from_distance_matrix(D, n_bins=None, bin_edges=None, alpha=1e-12):
    """
    D: (N,N) numpy array, distances, D[i,i] can be 0
    n_bins: number of common bins (if None -> use unique sorted values as bins)
    bin_edges: explicit bin edges (overrides n_bins if provided)
    alpha: additive smoothing (Laplace-like) added to counts to avoid zeros

    Returns:
      p_i : array shape (N, K) -- per-row distributions (normalized over N-1)
      p   : array shape (K,)   -- global mixture p = (1/N) sum_i p_i
      edges : bin_edges used (length K+1)
    """
    N = D.shape[0]
    assert D.shape[0] == D.shape[1], "D must be square"
    idx = np.arange(N)
    other_mask = ~np.eye(N, dtype=bool)
    rows = [D[i, other_mask[i]] for i in range(N)]
    # Convert to (N, N-1) array
    rows_arr = np.stack(rows, axis=0)  # shape (N, N-1)

    all_vals = rows_arr.ravel()

    if bin_edges is None:
        if n_bins is None:
            # default: unique sorted values -> for exact discrete distances
            uniq = np.unique(all_vals)
            # create edges so that each unique value falls into its own bin
            # if values are integers, we make integer-centered bins
            if np.all(np.floor(uniq) == uniq):
                # integer values
                edges = np.concatenate([uniq - 0.5, [uniq[-1] + 0.5]])
            else:
                # non-integer: use small epsilon around unique values (not ideal for many uniques)
                # fallback: use histogram_bin_edges 자동
                edges = np.histogram_bin_edges(all_vals, bins='auto')
        else:
            edges = np.histogram_bin_edges(all_vals, bins=n_bins)
    else:
        edges = np.asarray(bin_edges)

    K = len(edges) - 1
    # digitize rows_arr -> indices 0..K-1
    inds = np.digitize(rows_arr, bins=edges) - 1
    # Clip to valid range (edge cases)
    inds = np.clip(inds, 0, K-1)

    # Build counts per row
    counts = np.zeros((N, K), dtype=float)
    # vectorized accumulation
    rows_idx = np.repeat(np.arange(N), inds.shape[1])
    cols_idx = inds.ravel()
    np.add.at(counts, (rows_idx, cols_idx), 1)

    # Add smoothing alpha
    counts += alpha
    row_sums = counts.sum(axis=1, keepdims=True)  # shape (N,1) should be N-1 + K*alpha
    p_i = counts / row_sums  # normalized per-row distributions

    # global mixture p = (1/N) sum_i p_i
    p = p_i.mean(axis=0)
    # apply tiny renormalization to avoid rounding issues
    p = p / p.sum()
    return p_i, p, edges

def entropy_from_prob_vector(p_vec, base=np.e):
    """Shannon entropy of a probability vector p_vec (natural log default)."""
    p = np.asarray(p_vec, dtype=float)
    mask = p > 0
    return - np.sum(p[mask] * np.log(p[mask])) / (np.log(base) if base != np.e else 1.0)

def compute_Hi_and_KLi(p_i, p_global, apply_miller_madow=False, n_obs_per_row=50):
    """
    p_i: (N, K) per-row distributions
    p_global: (K,) global distribution
    apply_miller_madow: whether to apply Miller-Madow correction to each H_i
    n_obs_per_row: if provided, used for Miller-Madow (should be ~N-1 for each row)
    Returns:
      H_i: (N,) local entropies
      KL_i: (N,) local KL divergences D_KL(p_i || p_global)
    """
    N, K = p_i.shape
    H_i = np.zeros(N, dtype=float)
    KL_i = np.zeros(N, dtype=float)
    for i in range(N):
        pi = p_i[i]
        # entropy
        H = entropy_from_prob_vector(pi)
        # Miller-Madow correction: H_MM = H_emp + (K_pos - 1)/(2n)
        if apply_miller_madow:
            if n_obs_per_row is None:
                n = (pi.sum() * 0) + 1  # fallback (should not happen)
            else:
                n = n_obs_per_row
            K_pos = np.count_nonzero(pi > 0)
            H += (K_pos - 1) / (2.0 * n)
        H_i[i] = H
        # KL
        # avoid zeros by assuming p_i and p_global already smoothed (see build_histograms)
        mask = (pi > 0) & (p_global > 0)
        KL_i[i] = np.sum(pi[mask] * (np.log(pi[mask]) - np.log(p_global[mask])))
    return H_i, KL_i

def compute_global_H_from_mixture(p_global):
    return entropy_from_prob_vector(p_global)

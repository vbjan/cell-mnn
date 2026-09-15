import numpy as np
import ot  # pip install pot


def compute_cost_matrix(dist1: np.ndarray, dist2: np.ndarray, p: int = 1) -> np.ndarray:
    assert p in [1, 2], "Currently only p=1 and p=2 are supported"
    assert dist1.shape[1] == dist2.shape[
        1], f"Dimensionality mismatch: {dist1.shape[1]} != {dist2.shape[1]}"

    # For W2, use squared Euclidean distances
    metric = 'sqeuclidean' if p == 2 else 'euclidean'
    cost_matrix = ot.dist(dist1, dist2, metric=metric)
    return cost_matrix


def _subsample(dist: np.ndarray, n: int | None) -> np.ndarray:
    """`n` rows of `dist`, drawn without replacement; all of them if it already fits."""
    if n is None or dist.shape[0] <= n:
        return dist
    return dist[np.random.choice(dist.shape[0], size=n, replace=False)]


def compute_wasserstein(
        dist1: np.ndarray, 
        dist2: np.ndarray, 
        p: int = 1,
        num_iter_max: int = 200_000,
        n: int | None = None
        ) -> float:
    """
    Compute the Wasserstein-p Distance between two distributions.

    Args:
        dist1: First distribution as a numpy array of shape (n, d).
        dist2: Second distribution as a numpy array of shape (m, d).
        p: Order of the Wasserstein distance (default: 1 for W1, use 2 for W2).
        num_iter_max: Iteration cap for the network simplex.
        n: Samples drawn from each distribution before solving, without replacement.
            None uses every sample. The estimate is biased *upward* and the bias shrinks only like n^(-1/d), so
            results compare only across calls sharing the same `n`.

    Returns:
        The Wasserstein-p Distance (float) between the two distributions.
    """
    dist1 = _subsample(dist1, n)
    dist2 = _subsample(dist2, n)

    n_samples_1 = dist1.shape[0]
    n_samples_2 = dist2.shape[0]

    cost_matrix = compute_cost_matrix(dist1, dist2, p=p)

    # Define uniform distributions
    p_weights = np.ones((n_samples_1,)) / n_samples_1
    q_weights = np.ones((n_samples_2,)) / n_samples_2

    w_value = ot.emd2(p_weights, q_weights, cost_matrix,
                      numItermax=num_iter_max)

    # For W2, take square root of the result
    return np.sqrt(w_value) if p == 2 else w_value


def compute_ot_coupling(
        dist1: np.ndarray, 
        dist2: np.ndarray, 
        p: int = 1,
        num_itermax: int = 100_000
        ) -> np.ndarray:
    n_samples_1 = dist1.shape[0]
    n_samples_2 = dist2.shape[0]

    cost_matrix = compute_cost_matrix(dist1, dist2, p=p)

     # Define uniform distributions
    p_weights = np.ones((n_samples_1,)) / n_samples_1
    q_weights = np.ones((n_samples_2,)) / n_samples_2

    gamma = ot.emd(p_weights, q_weights, cost_matrix,
                      numItermax=num_itermax)
    
    row_prob = gamma / gamma.sum(axis=1, keepdims=True)  

    choices = [np.random.choice(n_samples_2, p=row_prob[i]) for i in range(n_samples_1)]

    paired_next = dist2[np.asarray(choices)]  
    return paired_next

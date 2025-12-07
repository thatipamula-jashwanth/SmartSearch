import numpy as np
from config import MAX_CANDIDATES

def build_candidate_pool_from_ids(candidate_ids, master_vectors, seed=None):
    if MAX_CANDIDATES < 0:
        raise ValueError("MAX_CANDIDATES must be >= 0")

    if master_vectors is None or master_vectors.ndim != 2:
        raise ValueError("master_vectors must be a 2D numpy array of shape (N, D)")


    if candidate_ids is None or len(candidate_ids) == 0:
        dim = master_vectors.shape[1]
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    if not isinstance(candidate_ids, (set, list, tuple, np.ndarray)):
        raise TypeError("candidate_ids must be an iterable of integers")

    if isinstance(candidate_ids, (set, list, tuple)):
        arr = np.fromiter(candidate_ids, dtype=np.int64, count=len(candidate_ids), casting="unsafe")
    else:
        arr = np.asarray(candidate_ids, dtype=np.int64)

    sorted_once = False
    if arr.size > 1_000_000:
        arr = np.sort(arr)
        arr = arr[np.concatenate(([True], arr[1:] != arr[:-1]))]
        sorted_once = True
    else:
        arr = np.unique(arr)

    max_id = master_vectors.shape[0]
    valid_mask = (arr >= 0) & (arr < max_id)
    if not np.any(valid_mask):
        dim = master_vectors.shape[1]
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    arr = np.ascontiguousarray(arr[valid_mask], dtype=np.int64)

    n = arr.size
    if MAX_CANDIDATES != 0 and n > MAX_CANDIDATES:
        if seed is None:
            seed = 42
        rng = np.random.default_rng(seed)
        arr = arr[rng.choice(n, MAX_CANDIDATES, replace=False)]

    if not sorted_once:
        arr = np.sort(arr)

    vectors = np.ascontiguousarray(master_vectors[arr], dtype=np.float32)
    return vectors, arr

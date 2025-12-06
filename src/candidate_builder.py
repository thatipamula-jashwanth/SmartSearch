import numpy as np
from config import MAX_CANDIDATES

def build_candidate_pool_from_ids(candidate_ids, master_vectors, seed=None):

    if MAX_CANDIDATES < 0:
        raise ValueError("MAX_CANDIDATES must be >= 0")

  
    if not candidate_ids:
        dim = master_vectors.shape[1]
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

  
    if isinstance(candidate_ids, (set, list, tuple)):
        arr = np.fromiter(candidate_ids, dtype=np.int64, count=len(candidate_ids))
    else:
        arr = np.asarray(candidate_ids, dtype=np.int64)

   
    if arr.size > 1_000_000:
        arr = np.sort(arr)
        arr = arr[np.concatenate(([True], arr[1:] != arr[:-1]))]
    else:
        arr = np.unique(arr)


    max_id = master_vectors.shape[0]
    arr = arr[(arr >= 0) & (arr < max_id)]
    if arr.size == 0:
        dim = master_vectors.shape[1]
        return np.zeros((0, dim), dtype=np.float32), np.zeros(0, dtype=np.int64)

    n = arr.size
    if MAX_CANDIDATES != 0 and n > MAX_CANDIDATES:
        rng = np.random.default_rng(seed)
        if n > 1_000_000:
            
            idx = rng.choice(n, MAX_CANDIDATES, replace=False, shuffle=True)
        else:
            
            idx = rng.permutation(n)[:MAX_CANDIDATES]
        arr = arr[idx]

    arr.sort()

    vectors = np.ascontiguousarray(master_vectors[arr], dtype=np.float32)

    return vectors, arr

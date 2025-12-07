import numpy as np
from numba import njit, prange


@njit(fastmath=True)
def _dot_serial(query, candidates):
    n = candidates.shape[0]
    dim = candidates.shape[1]
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        s = 0.0
        for j in range(dim):
            s += query[j] * candidates[i, j]
        out[i] = s
    return out


@njit(parallel=True, fastmath=True)
def _dot_parallel(query, candidates):
    n = candidates.shape[0]
    dim = candidates.shape[1]
    out = np.empty(n, dtype=np.float32)
    for i in prange(n):
        s = 0.0
        for j in range(dim):
            s += query[j] * candidates[i, j]
        out[i] = s
    return out


def compute_cosine_scores(query, candidates, batch=4096):

    query = np.asarray(query, dtype=np.float32)
    candidates = np.ascontiguousarray(candidates, dtype=np.float32)

    if query.ndim != 1:
        raise ValueError("query must be 1D")
    if candidates.ndim != 2:
        raise ValueError("candidates must be 2D")
    if query.shape[0] != candidates.shape[1]:
        raise ValueError("query dim mismatch candidates")

    n = candidates.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    norms = np.linalg.norm(candidates, axis=1)
    mask = norms > 1e-12
    valid = candidates[mask]

    scores_full = np.zeros(n, dtype=np.float32)
    if valid.shape[0] == 0:
        return scores_full

    scores_valid = np.empty(valid.shape[0], dtype=np.float32)

    for start in range(0, valid.shape[0], batch):
        end = min(start + batch, valid.shape[0])
        sliceC = valid[start:end]

        if sliceC.shape[0] >= 256:
            scores_valid[start:end] = _dot_parallel(query, sliceC)
        else:
            scores_valid[start:end] = _dot_serial(query, sliceC)

    scores_full[mask] = scores_valid
    return scores_full


def compute_weighted_cosine_scores(query, candidates, weights, batch=4096):

    query = np.asarray(query, dtype=np.float32)
    candidates = np.ascontiguousarray(candidates, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)

    if candidates.ndim != 2:
        raise ValueError("candidates must be 2D")
    if query.ndim != 1:
        raise ValueError("query must be 1D")
    if query.shape[0] != candidates.shape[1]:
        raise ValueError("query dim mismatch candidates")
    if query.shape[0] != weights.shape[0]:
        raise ValueError("query dim mismatch weights")

    q = query * weights
    C = candidates * weights

    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        raise ValueError("Query collapsed to zero after weighting")
    q = q / q_norm

    C_norms = np.linalg.norm(C, axis=1)
    mask = C_norms > 1e-12
    valid = C[mask]

    scores_full = np.zeros(C.shape[0], dtype=np.float32)
    if valid.shape[0] == 0:
        return scores_full

    valid = valid / C_norms[mask][:, None]
    valid = np.ascontiguousarray(valid, dtype=np.float32)

    scores_valid = np.empty(valid.shape[0], dtype=np.float32)

    for start in range(0, valid.shape[0], batch):
        end = min(start + batch, valid.shape[0])
        sliceC = valid[start:end]

        if sliceC.shape[0] >= 256:
            scores_valid[start:end] = _dot_parallel(q, sliceC)
        else:
            scores_valid[start:end] = _dot_serial(q, sliceC)

    scores_full[mask] = scores_valid
    return scores_full

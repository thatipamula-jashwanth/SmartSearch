import numpy as np
from distance_kernel import compute_cosine_scores, compute_weighted_cosine_scores

_warmed_up = False


def smart_search(
    query_lowdim,
    clustering_index,
    master_vectors,
    weights=None,
    K=10,
    topC=4,
    topF=2
):
    global _warmed_up

    q = np.asarray(query_lowdim, dtype=np.float32)
    if not np.all(np.isfinite(q)):
        return np.empty((0, 2), dtype=np.float32)

    dim = q.shape[0]
    if master_vectors.shape[1] != dim:
        raise ValueError(
            f"[smart_search] Dim mismatch — query:{dim}, vectors:{master_vectors.shape[1]}"
        )

    q_norm = np.linalg.norm(q)
    if q_norm < 1e-10:
        return np.empty((0, 2), dtype=np.float32)
    q = q / q_norm

    if not _warmed_up:
        dummy = np.zeros((1, dim), dtype=np.float32)
        try:
            if weights is None:
                compute_cosine_scores(q, dummy)
            else:
                compute_weighted_cosine_scores(q, dummy, weights)
        finally:
            _warmed_up = True

    coarse_ids = clustering_index.search_centroids(q, topC=topC)
    if coarse_ids is None or coarse_ids.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    ordered_ids = []
    for c in coarse_ids:
        fine_ids = clustering_index.search_fine(q, coarse_id=c, topF=topF)
        if fine_ids is None or len(fine_ids) == 0:
            continue

        postings = clustering_index.postings_l1[c]
        if postings is None:
            continue

        for f in fine_ids:
            if f < 0 or f >= len(postings):
                continue

            ids = postings[f]
            if ids is not None and len(ids) > 0:
                ordered_ids.extend(ids)

    if len(ordered_ids) == 0:
        return np.empty((0, 2), dtype=np.float32)

  
    ordered_ids_np = np.asarray(ordered_ids, dtype=np.int64)
    _, first_pos = np.unique(ordered_ids_np, return_index=True)
    dedup_ids = ordered_ids_np[np.sort(first_pos)]
  

    ids = dedup_ids
    vectors = master_vectors[ids]

    if weights is None:
        scores = compute_cosine_scores(q, vectors)
    else:
        if not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise ValueError("[smart_search] Invalid weights (NaN/Inf/negative)")
        if weights.shape[0] != dim:
            raise ValueError("[smart_search] Weight dimension mismatch query")
        if not np.any(weights):
            return np.empty((0, 2), dtype=np.float32)

        scores = compute_weighted_cosine_scores(q, vectors, weights)

    k_eff = min(K, len(scores))
    idx = np.argpartition(scores, -k_eff)[-k_eff:]
    idx = idx[np.argsort(scores[idx])[::-1]]

    return np.column_stack((ids[idx].astype(np.int64), scores[idx]))

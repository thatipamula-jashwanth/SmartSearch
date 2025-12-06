import numpy as np
from config import MAX_CANDIDATES


def collect_cluster_candidates(query_lowdim, clustering_index, topC=4, topF=2):
    if not isinstance(MAX_CANDIDATES, int) or MAX_CANDIDATES <= 0:
        raise ValueError("MAX_CANDIDATES must be a positive integer")

   
    q = np.asarray(query_lowdim, dtype=np.float32)
    if np.isnan(q).any() or np.isinf(q).any():
        raise ValueError("Query contains NaN or Inf")
    if q.ndim != 1:
        raise ValueError("Query must be a 1-D vector")

   
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        raise ValueError("Query is a zero vector")
    q = q / q_norm

    candidate_ids = set()


    C0 = getattr(clustering_index, "centroids_l0", None)
    C1_dict = getattr(clustering_index, "centroids_l1", None)
    P_dict = getattr(clustering_index, "postings_l1", None)

    if C0 is None or not isinstance(C0, np.ndarray) or C0.ndim != 2:
        raise ValueError("clustering_index.centroids_l0 must be (N, dim) ndarray")

    if not isinstance(C1_dict, dict) or not isinstance(P_dict, dict):
        raise ValueError("clustering_index.centroids_l1 and postings_l1 must be dicts")

    dim = q.shape[0]
    if C0.shape[1] != dim:
        raise ValueError("Dimension mismatch: centroid dim != query dim")

    
    C0_norm = C0 / (np.linalg.norm(C0, axis=1, keepdims=True) + 1e-12)
    sims0 = (C0_norm @ q).astype(np.float32)

    k0 = min(topC, sims0.shape[0])
    idx0 = np.argpartition(sims0, -k0)[-k0:]
    idx0 = idx0[np.argsort(sims0[idx0])[::-1]]  
    coarse_sel = idx0.tolist()

    for c in coarse_sel:
        C1 = C1_dict.get(c, None)
        postings = P_dict.get(c, None)

        if C1 is None or postings is None:
            continue
        if not isinstance(C1, np.ndarray) or C1.ndim != 2:
            continue
        if C1.shape[1] != dim:
            continue

        C1_norm = C1 / (np.linalg.norm(C1, axis=1, keepdims=True) + 1e-12)
        sims1 = (C1_norm @ q).astype(np.float32)

        k1 = min(topF, sims1.shape[0])
        idx1 = np.argpartition(sims1, -k1)[-k1:]
        idx1 = idx1[np.argsort(sims1[idx1])[::-1]]

        for f in idx1:
            posting = postings.get(f, None)
            if posting:
                candidate_ids.update(posting)
                if len(candidate_ids) >= MAX_CANDIDATES:
                    arr = np.fromiter(candidate_ids, dtype=np.int64)
                    arr.sort()  
                    return arr[:MAX_CANDIDATES]

    
    if not candidate_ids:
        return np.zeros(0, dtype=np.int64)

    arr = np.fromiter(candidate_ids, dtype=np.int64)
    arr.sort()  
    return arr[:MAX_CANDIDATES]

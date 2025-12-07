import numpy as np
from config import MAX_CANDIDATES


def collect_cluster_candidates(query_lowdim, clustering_index, topC=4, topF=2):

    if not isinstance(MAX_CANDIDATES, int) or MAX_CANDIDATES <= 0:
        raise ValueError("MAX_CANDIDATES must be a positive integer")


    if topC <= 0 or topF <= 0:
        return np.zeros(0, dtype=np.int64)

    q = np.asarray(query_lowdim, dtype=np.float32)
    if q.ndim != 1:
        raise ValueError("Query must be 1-D")
    if np.isnan(q).any() or np.isinf(q).any():
        raise ValueError("Query contains NaN/Inf")

    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        raise ValueError("Query is a zero vector")
    q = q / q_norm

    
    C0 = getattr(clustering_index, "centroids_l0", None)
    C1_list = getattr(clustering_index, "centroids_l1", None)
    P_list = getattr(clustering_index, "postings_l1", None)

    if not isinstance(C0, np.ndarray) or C0.ndim != 2:
        raise ValueError("centroids_l0 must be a (N, D) ndarray")
    if not isinstance(C1_list, (list, tuple)) or not isinstance(P_list, (list, tuple)):
        raise ValueError("centroids_l1 and postings_l1 must be list-like")


    if C0.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)

    dim = q.shape[0]
    if C0.shape[1] != dim:
        raise ValueError("Dimension mismatch: centroid dim != query dim")

    sims0 = C0 @ q
    k0 = min(topC, sims0.shape[0])
    idx0 = np.argpartition(sims0, -k0)[-k0:]
    idx0 = idx0[np.argsort(sims0[idx0])[::-1]]
    coarse_sel = idx0.tolist()

    cand = []

    for c in coarse_sel:
      
        if c < 0 or c >= len(C1_list) or c >= len(P_list):
            continue

        C1 = C1_list[c]
        postings = P_list[c]

        if C1 is None or postings is None or len(postings) == 0:
            continue
        if not isinstance(C1, np.ndarray) or C1.ndim != 2 or C1.shape[1] != dim:
            continue

        sims1 = C1 @ q
        k1 = min(topF, sims1.shape[0])
        idx1 = np.argpartition(sims1, -k1)[-k1:]
        idx1 = idx1[np.argsort(sims1[idx1])[::-1]]

        for f in idx1:
       
            if f < 0 or f >= len(postings):
                continue

            posting = postings[f]

            if not isinstance(posting, np.ndarray) or posting.size == 0:
                continue

            cand.extend(posting.tolist())

  
    if len(cand) == 0:
        return np.zeros(0, dtype=np.int64)

    arr = np.asarray(cand, dtype=np.int64)

    cap = MAX_CANDIDATES * 4
    if arr.size > cap:
        arr = arr[:cap]

    arr = np.unique(arr)
    if arr.size == 0:
        return np.zeros(0, dtype=np.int64)

    return arr[:MAX_CANDIDATES]

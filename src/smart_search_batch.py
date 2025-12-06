import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import shared_memory
from smart_search import smart_search



def _sanitize_vectors(vectors: np.ndarray) -> np.ndarray:
    mask = np.isfinite(vectors).all(axis=1)
    if not np.all(mask):
        print(f"[SmartSearch WARN] {np.sum(~mask)} vectors contain NaN/Inf → masked")
    return vectors * mask[:, None]


def _fits_in_ram(num_bytes: int) -> bool:
    try:
        import psutil
        return num_bytes < psutil.virtual_memory().available * 0.6
    except Exception:
       
        return True


def _smart_search_worker(
    idx_range,
    shm_q_name, q_shape,
    shm_v_name, v_shape,
    clustering_index_dict,
    weights,
    K
):
    from cluster_index import ClusteringIndex
    clustering_index = ClusteringIndex.from_serializable(clustering_index_dict)

    try:
        shm_q = shared_memory.SharedMemory(name=shm_q_name)
        shm_v = shared_memory.SharedMemory(name=shm_v_name)
    except FileNotFoundError:
        return idx_range[0], []

    queries = np.ndarray(q_shape, dtype=np.float32, buffer=shm_q.buf)
    vectors = np.ndarray(v_shape, dtype=np.float32, buffer=shm_v.buf)

    start, end = idx_range
    out = [None] * (end - start)

    for i_local, i_global in enumerate(range(start, end)):
        q = queries[i_global]

       
        if not np.all(np.isfinite(q)):
            out[i_local] = np.empty((0, 2), dtype=np.float32)
            continue

        try:
            res = smart_search(q, clustering_index, vectors, weights, K)
        except Exception as e:
            print(f"[Worker ERROR] {e}")
            res = np.empty((0, 2), dtype=np.float32)

        out[i_local] = res

    shm_q.close()
    shm_v.close()
    return start, out



def smart_search_batch_parallel(
    queries_lowdim,
    clustering_index,
    master_vectors,
    weights=None,
    K=10,
    num_workers=None,
    batch_size=32
):
    queries_lowdim = np.ascontiguousarray(queries_lowdim, dtype=np.float32)
    master_vectors = np.ascontiguousarray(master_vectors, dtype=np.float32)

    queries_lowdim = np.atleast_2d(queries_lowdim)
    B, dim = queries_lowdim.shape

    if master_vectors.shape[1] != dim:
        raise ValueError(
            f"Dimension mismatch — queries:{dim}, vectors:{master_vectors.shape[1]}"
        )

    master_vectors = _sanitize_vectors(master_vectors)


    if queries_lowdim.dtype != np.float32 or master_vectors.dtype != np.float32:
        print("[SmartSearch WARN] Casting inputs to float32")
        queries_lowdim = queries_lowdim.astype(np.float32)
        master_vectors = master_vectors.astype(np.float32)

    clustering_index_dict = clustering_index.to_serializable()

    if num_workers is None:
        num_workers = min(os.cpu_count(), max(1, B // max(1, batch_size)))

    if B <= batch_size or num_workers <= 1:
        return [
            smart_search(q, clustering_index, master_vectors, weights, K)
            for q in queries_lowdim
        ]

    total_shm = queries_lowdim.nbytes + master_vectors.nbytes
    shm_mode = _fits_in_ram(total_shm)

    if not shm_mode:
        print("[SmartSearch WARN] Shared memory too large → using local copies")
        return [
            smart_search(q, clustering_index, master_vectors, weights, K)
            for q in queries_lowdim
        ]

    shm_q = shared_memory.SharedMemory(create=True, size=queries_lowdim.nbytes)
    shm_v = shared_memory.SharedMemory(create=True, size=master_vectors.nbytes)

    np.copyto(np.ndarray(queries_lowdim.shape, dtype=np.float32, buffer=shm_q.buf), queries_lowdim)
    np.copyto(np.ndarray(master_vectors.shape, dtype=np.float32, buffer=shm_v.buf), master_vectors)

    idx_ranges = [(i, min(i + batch_size, B)) for i in range(0, B, batch_size)]
    results = [None] * B

    try:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _smart_search_worker,
                    r,
                    shm_q.name, queries_lowdim.shape,
                    shm_v.name, master_vectors.shape,
                    clustering_index_dict,
                    weights,
                    K
                )
                for r in idx_ranges
            ]

            for f in as_completed(futures):
                start_idx, batch_res = f.result()
                results[start_idx : start_idx + len(batch_res)] = batch_res

    finally:
        try: shm_q.close(); shm_q.unlink()
        except: pass
        try: shm_v.close(); shm_v.unlink()
        except: pass

    return results

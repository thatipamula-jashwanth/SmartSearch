import time
import numpy as np
from pathlib import Path
import sys

# === Path bootstrapping ===
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_dataset import generate_semantic_vectors
from projection_embedder import ProjectionEmbedder
from cluster_index import ClusteringIndex
from smart_search import smart_search

# === Optional ANN baselines ===
try: import faiss
except: faiss = None

try: from annoy import AnnoyIndex
except: AnnoyIndex = None

try: import hnswlib
except: hnswlib = None

from sklearn.neighbors import NearestNeighbors


# =======================================================
# BENCH CONFIG
# =======================================================
configs = [
    (8,   1_000_000),
    (16,  320_000),
    (64,  300_000),
    (128, 300_000),
    (256, 100_000),
    (512, 50_000),
]

Q_total = 1000
K = 50
gt_k = 50

def recall(gt, pred):
    if len(gt) == 0 or len(pred) == 0:
        return 0.0
    return len(set(gt) & set(pred)) / len(gt)


# =======================================================
# MAIN LOOP
# =======================================================
for DIM, N in configs:

    print(f"\n=== Benchmark: DIM={DIM}, N={N:,} ===")

    vectors, labels, _ = generate_semantic_vectors(n_samples=N, dim=DIM)
    queries = vectors[:min(Q_total, N)]
    base = vectors[min(Q_total, N):]

    routing_dim = max(1, min(192, DIM))
    proj = ProjectionEmbedder(original_dim=DIM, routing_dim=routing_dim, seed=42)
    routing_base = proj.transform(base)
    routing_queries = proj.transform(queries)

    clustering_index = ClusteringIndex(
        n_coarse=min(256, max(1, routing_base.shape[0] // 1000)),
        n_fine=8,
        batch_size=4096,
        seed=42
    ).build(routing_base)

    # === ANN BASELINE MODELS ===
    # FAISS IP
    if faiss:
        faiss_index = faiss.IndexFlatIP(routing_base.shape[1])
        faiss_index.add(routing_base.astype(np.float32))

    # Annoy
    if AnnoyIndex:
        annoy_index = AnnoyIndex(routing_base.shape[1], metric="angular")
        for i, v in enumerate(routing_base):
            annoy_index.add_item(i, v.tolist())
        annoy_index.build(50)

    # HNSWlib (⚠ Skip if dataset too big)
    hnsw = None
    if hnswlib:
        if routing_base.shape[0] <= 300_000:
            hnsw = hnswlib.Index(space='cosine', dim=routing_base.shape[1])
            hnsw.init_index(max_elements=routing_base.shape[0],
                            ef_construction=120, M=64)
            hnsw.add_items(routing_base)
            hnsw.set_ef(200)
        else:
            print("⚠ Skipping HNSW build — too large for CPU benchmark")

    # Sklearn brute force (⚠ Skip if too large)
    sklearn_enabled = routing_base.shape[0] <= 300_000
    if sklearn_enabled:
        nn_sklearn = NearestNeighbors(metric="cosine", algorithm="brute")
        nn_sklearn.fit(routing_base)

    # =======================================================
    # Benchmark loop
    # =======================================================
    rec = dict(smart=[0,0,0], faiss=[0,0,0], annoy=[0,0,0], hnsw=[0,0,0], sklearn=[0,0,0])
    lat = dict(smart=[], faiss=[], annoy=[], hnsw=[], sklearn=[])

    for q in routing_queries:
        # Fast truth top-K
        scores_gt = routing_base @ q
        idx = np.argpartition(scores_gt, -gt_k)[-gt_k:]
        gt = idx[np.argsort(scores_gt[idx])[::-1]]

        # SmartSearch
        t = time.perf_counter()
        res = smart_search(q, clustering_index, routing_base,
                           K=K, topC=8, topF=16)
        lat["smart"].append((time.perf_counter() - t) * 1000)
        nn = res[np.argsort(res[:,1])[::-1], 0].astype(int) if res.size else []
        rec["smart"][0] += recall(gt[:1], nn[:1])
        rec["smart"][1] += recall(gt[:10], nn[:10])
        rec["smart"][2] += recall(gt[:50], nn[:50])

        # FAISS
        if faiss:
            t = time.perf_counter()
            _, ids = faiss_index.search(q.reshape(1,-1), K)
            ids = ids[0]
            lat["faiss"].append((time.perf_counter() - t) * 1000)
            rec["faiss"][0] += recall(gt[:1], ids[:1])
            rec["faiss"][1] += recall(gt[:10], ids[:10])
            rec["faiss"][2] += recall(gt[:50], ids[:50])

        # Annoy
        if AnnoyIndex:
            t = time.perf_counter()
            ids = annoy_index.get_nns_by_vector(q.tolist(), K)
            lat["annoy"].append((time.perf_counter() - t) * 1000)
            rec["annoy"][0] += recall(gt[:1], ids[:1])
            rec["annoy"][1] += recall(gt[:10], ids[:10])
            rec["annoy"][2] += recall(gt[:50], ids[:50])

        # HNSW
        if hnsw is not None:
            t = time.perf_counter()
            ids, _ = hnsw.knn_query(q, k=K)
            ids = ids[0]
            lat["hnsw"].append((time.perf_counter() - t) * 1000)
            rec["hnsw"][0] += recall(gt[:1], ids[:1])
            rec["hnsw"][1] += recall(gt[:10], ids[:10])
            rec["hnsw"][2] += recall(gt[:50], ids[:50])

        # Sklearn brute force
        if sklearn_enabled:
            t = time.perf_counter()
            _, ids = nn_sklearn.kneighbors(q.reshape(1,-1), n_neighbors=K)
            ids = ids[0]
            lat["sklearn"].append((time.perf_counter() - t) * 1000)
            rec["sklearn"][0] += recall(gt[:1], ids[:1])
            rec["sklearn"][1] += recall(gt[:10], ids[:10])
            rec["sklearn"][2] += recall(gt[:50], ids[:50])

    # =======================================================
    # Print results
    # =======================================================
    def fmt(r, l):
        return f"{np.mean(l):.3f}ms | QPS={1000/np.mean(l):.1f} | " \
               f"R@1:{r[0]/len(routing_queries):.3f} " \
               f"R@10:{r[1]/len(routing_queries):.3f} " \
               f"R@50:{r[2]/len(routing_queries):.3f}"

    print("\n★ RESULTS")
    print("SmartSearch :", fmt(rec["smart"], lat["smart"]))
    if faiss: print("FAISS       :", fmt(rec["faiss"], lat["faiss"]))
    if AnnoyIndex: print("Annoy       :", fmt(rec["annoy"], lat["annoy"]))
    if hnsw is not None: print("HNSWlib     :", fmt(rec["hnsw"], lat["hnsw"]))
    if sklearn_enabled:
        print("SklearnBF   :", fmt(rec["sklearn"], lat["sklearn"]))
    else:
        print("SklearnBF   : Skipped — too slow for large dataset")

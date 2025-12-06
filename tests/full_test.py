import time
import numpy as np
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from semantic_dataset import generate_semantic_vectors
from projection_embedder import ProjectionEmbedder
from cluster_index import ClusteringIndex
from smart_search import smart_search

# Optional FAISS
try:
    import faiss
except Exception:
    faiss = None


configs = [
    (8,   1_000_000),
    (16,  32_000),
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

summary = []

for DIM, N in configs:
    print(f"\n=== Benchmark: DIM={DIM}, N={N:,} ===")

  
    vectors, labels, _ = generate_semantic_vectors(n_samples=N, dim=DIM)
    queries = vectors[:min(Q_total, N)]
    base = vectors[min(Q_total, N):]

    routing_dim = max(1, min(192, DIM))
    proj = ProjectionEmbedder(original_dim=DIM, routing_dim=routing_dim, seed=42)


    routing_base = proj.transform(base, normalize_input=True, chunk=base.shape[0])
    routing_queries = proj.transform(queries, normalize_input=True, chunk=queries.shape[0])


    clustering_index = ClusteringIndex(
        n_coarse=min(256, max(1, routing_base.shape[0] // 1000)),
        n_fine=8,
        batch_size=4096,
        seed=42
    ).build(routing_base)

    if faiss:
        index_faiss = faiss.IndexFlatIP(routing_base.shape[1])
        index_faiss.add(routing_base.astype(np.float32))

  
    rec_smart_1 = rec_smart_10 = rec_smart_50 = 0
    rec_faiss_1 = rec_faiss_10 = rec_faiss_50 = 0
    lat_smart = []
    lat_faiss = []


    for i, q in enumerate(routing_queries):
        scores_gt = routing_base @ q
        gt = np.argsort(scores_gt)[::-1][:gt_k]


        t0 = time.perf_counter()
        res = smart_search(q, clustering_index, routing_base, K=K, topC=8, topF=16)
        lat_smart.append((time.perf_counter() - t0) * 1000)
        if res.size > 0:
            nn = res[np.argsort(res[:, 1])[::-1], 0].astype(int)
        else:
            nn = np.zeros(0, dtype=int)
        rec_smart_1  += recall(gt[:1], nn[:1])
        rec_smart_10 += recall(gt[:10], nn[:10])
        rec_smart_50 += recall(gt[:50], nn[:50])


        if faiss:
            t1 = time.perf_counter()
            _, ids = index_faiss.search(q.reshape(1, -1).astype(np.float32), K)
            lat_faiss.append((time.perf_counter() - t1) * 1000)
            ids = ids[0]
            rec_faiss_1  += recall(gt[:1], ids[:1])
            rec_faiss_10 += recall(gt[:10], ids[:10])
            rec_faiss_50 += recall(gt[:50], ids[:50])

    avg_lat_smart = np.mean(lat_smart)
    qps_smart = 1000 / avg_lat_smart
    if faiss:
        avg_lat_faiss = np.mean(lat_faiss)
        qps_faiss = 1000 / avg_lat_faiss
    else:
        avg_lat_faiss = qps_faiss = None


    print(f"\nSMARTSEARCH  | latency={avg_lat_smart:.3f} ms/query | QPS={qps_smart:.1f}")
    print(f"Recall@1  : {rec_smart_1/len(routing_queries):.4f}")
    print(f"Recall@10 : {rec_smart_10/len(routing_queries):.4f}")
    print(f"Recall@50 : {rec_smart_50/len(routing_queries):.4f}")

    if faiss:
        print(f"\nFAISS  | latency={avg_lat_faiss:.3f} ms/query | QPS={qps_faiss:.1f}")
        print(f"Recall@1  : {rec_faiss_1/len(routing_queries):.4f}")
        print(f"Recall@10 : {rec_faiss_10/len(routing_queries):.4f}")
        print(f"Recall@50 : {rec_faiss_50/len(routing_queries):.4f}")

    summary.append({
        "DIM": DIM,
        "N": N,
        "SmartSearch_ms": avg_lat_smart,
        "SmartSearch_QPS": qps_smart,
        "SmartSearch_R@1": rec_smart_1/len(routing_queries),
        "SmartSearch_R@10": rec_smart_10/len(routing_queries),
        "SmartSearch_R@50": rec_smart_50/len(routing_queries),
        "FAISS_ms": avg_lat_faiss,
        "FAISS_QPS": qps_faiss,
        "FAISS_R@1": rec_faiss_1/len(routing_queries) if faiss else None,
        "FAISS_R@10": rec_faiss_10/len(routing_queries) if faiss else None,
        "FAISS_R@50": rec_faiss_50/len(routing_queries) if faiss else None,
    })


print("\n=== SUMMARY TABLE ===")
print("DIM | N | SmartSearch(ms/QPS/R@1/R@10/R@50) | FAISS(ms/QPS/R@1/R@10/R@50)")
for row in summary:
    faiss_str = (f"{row['FAISS_ms']:.2f}/{row['FAISS_QPS']:.1f}/"
                 f"{row['FAISS_R@1']:.3f}/{row['FAISS_R@10']:.3f}/{row['FAISS_R@50']:.3f}"
                 if faiss else "N/A")
    print(f"{row['DIM']:3d} | {row['N']:,} | "
          f"{row['SmartSearch_ms']:.2f}/{row['SmartSearch_QPS']:.1f}/"
          f"{row['SmartSearch_R@1']:.3f}/{row['SmartSearch_R@10']:.3f}/{row['SmartSearch_R@50']:.3f} | "
          f"{faiss_str}")

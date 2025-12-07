import time
import numpy as np
import psutil
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from projection_embedder import ProjectionEmbedder
from cluster_index import ClusteringIndex
from smart_search import smart_search


try: import faiss
except: faiss = None

try: from annoy import AnnoyIndex
except: AnnoyIndex = None

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except:
    HF_AVAILABLE = False

proc = psutil.Process()
def ram_mb():
    return proc.memory_info().rss / (1024 * 1024)
def print_ram(msg):
    print(f"{msg} — RAM: {ram_mb():.1f} MB\n")

print_ram("Start")


SAFE_LIMIT_MB = 650
VECTOR_DIM = 300
TARGET = 250_000  
DATASET_NAME = "glue"
DATASET_CONFIG = "sst2"


def load_vectors_public(max_rows=250_000, dim=300):
    if not HF_AVAILABLE:
        print("⚠ HuggingFace not installed — using synthetic vectors")
        return np.random.randn(max_rows, dim).astype(np.float32)

    print(f"Loading {DATASET_NAME}/{DATASET_CONFIG} embeddings...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")

    vectors = np.zeros((min(max_rows, len(ds)), dim), dtype=np.float32)
    for i, row in enumerate(ds):
        if i >= max_rows:
            break

        text = row["sentence"]
        seed = abs(hash(text)) % (2 ** 32)
        rng = np.random.default_rng(seed)
        vectors[i] = rng.standard_normal(dim)

        if i % 5000 == 0:
            print(f"Loaded {i} vectors — RAM {ram_mb():.1f} MB")
        if ram_mb() > SAFE_LIMIT_MB:
            print("\n⚠ SAFE RAM LIMIT REACHED – stopped early.")
            return vectors[:i]

    try: ds.cleanup_cache_files()
    except: pass
    return vectors

vectors = load_vectors_public(TARGET, VECTOR_DIM)
N, DIM = vectors.shape
print_ram(f"Loaded {N:,} vectors (public dataset)")

vectors /= (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
print_ram("Vectors normalized")


Q = min(200, N)
queries = vectors[:Q]
base = vectors      
print_ram("Query/Base split done")


routing_dim = 96 if DIM >= 128 else DIM
proj = ProjectionEmbedder(DIM, routing_dim=routing_dim, seed=42)
routing_base = proj.transform(base)
routing_queries = proj.transform(queries)
print_ram("Projection done")


n_coarse = min(128, max(4, N // 2000))
n_fine = 8
clustering_index = ClusteringIndex(
    n_coarse=n_coarse,
    n_fine=n_fine,
    batch_size=2048,
    seed=42
).build(routing_base)
print_ram("SmartSearch index built")


use_faiss_ivf = faiss is not None and N <= 600_000
use_faiss_hnsw = faiss is not None and N <= 500_000
use_annoy = AnnoyIndex is not None and N <= 200_000


if use_faiss_ivf:
    nlist = max(64, N // 2000)
    quantizer = faiss.IndexFlatIP(routing_dim)
    faiss_ivf_index = faiss.IndexIVFFlat(quantizer, routing_dim, nlist, faiss.METRIC_INNER_PRODUCT)
    faiss_ivf_index.train(routing_base.astype(np.float32))
    faiss_ivf_index.add(routing_base.astype(np.float32))
    print_ram("FAISS IVF index built")


if use_faiss_hnsw:
    faiss_hnsw_index = faiss.IndexHNSWFlat(routing_dim, 32)
    faiss_hnsw_index.hnsw.efConstruction = 40
    faiss_hnsw_index.add(routing_base.astype(np.float32))
    print_ram("FAISS HNSW index built")


if use_annoy:
    annoy_index = AnnoyIndex(routing_dim, metric="angular")
    for i, vec in enumerate(routing_base):
        annoy_index.add_item(i, vec.tolist())
    annoy_index.build(20)
    print_ram("Annoy index built")


def recall(gt, pred):
    return len(set(gt) & set(pred)) / len(gt) if len(gt) else 0

K = 20
gt_k = 20

times = {"smart": [], "ivf": [], "hnsw": [], "annoy": []}
recall_sum = {"smart": np.zeros(3), "ivf": np.zeros(3), "hnsw": np.zeros(3), "annoy": np.zeros(3)}

for i, (q_full, q_route) in enumerate(zip(queries, routing_queries)):


    scores = base @ q_full
    idx = np.argpartition(scores, -gt_k)[-gt_k:]
    gt = idx[np.argsort(scores[idx])[::-1]]

    t = time.perf_counter()
    res = smart_search(q_route, clustering_index, routing_base, K=K)
    times["smart"].append((time.perf_counter() - t) * 1000)
    nn = res[:, 0].astype(int) if res.size else []
    recall_sum["smart"][0] += recall(gt[:1], nn[:1])
    recall_sum["smart"][1] += recall(gt[:5], nn[:5])
    recall_sum["smart"][2] += recall(gt[:20], nn[:20])


    if use_faiss_ivf:
        t = time.perf_counter()
        _, ids = faiss_ivf_index.search(q_route.reshape(1, -1), K)
        times["ivf"].append((time.perf_counter() - t) * 1000)
        ids = ids[0]
        recall_sum["ivf"][0] += recall(gt[:1], ids[:1])
        recall_sum["ivf"][1] += recall(gt[:5], ids[:5])
        recall_sum["ivf"][2] += recall(gt[:20], ids[:20])


    if use_faiss_hnsw:
        t = time.perf_counter()
        _, ids = faiss_hnsw_index.search(q_route.reshape(1,-1), K)
        times["hnsw"].append((time.perf_counter() - t) * 1000)
        ids = ids[0]
        recall_sum["hnsw"][0] += recall(gt[:1], ids[:1])
        recall_sum["hnsw"][1] += recall(gt[:5], ids[:5])
        recall_sum["hnsw"][2] += recall(gt[:20], ids[:20])

  
    if use_annoy:
        t = time.perf_counter()
        ids = annoy_index.get_nns_by_vector(q_route.tolist(), K)
        times["annoy"].append((time.perf_counter() - t) * 1000)
        recall_sum["annoy"][0] += recall(gt[:1], ids[:1])
        recall_sum["annoy"][1] += recall(gt[:5], ids[:5])
        recall_sum["annoy"][2] += recall(gt[:20], ids[:20])


print("\n=== SAFE RESULTS (Public Dataset) ===")
print(f"SmartSearch : {np.mean(times['smart']):.3f}ms | "
      f"R@1={recall_sum['smart'][0]/Q:.3f} | R@5={recall_sum['smart'][1]/Q:.3f} | R@20={recall_sum['smart'][2]/Q:.3f}")

if use_faiss_ivf:
    print(f"FAISS IVF   : {np.mean(times['ivf']):.3f}ms | "
          f"R@1={recall_sum['ivf'][0]/Q:.3f} | R@5={recall_sum['ivf'][1]/Q:.3f} | R@20={recall_sum['ivf'][2]/Q:.3f}")

if use_faiss_hnsw:
    print(f"FAISS HNSW  : {np.mean(times['hnsw']):.3f}ms | "
          f"R@1={recall_sum['hnsw'][0]/Q:.3f} | R@5={recall_sum['hnsw'][1]/Q:.3f} | R@20={recall_sum['hnsw'][2]/Q:.3f}")

if use_annoy:
    print(f"Annoy       : {np.mean(times['annoy']):.3f}ms | "
          f"R@1={recall_sum['annoy'][0]/Q:.3f} | R@5={recall_sum['annoy'][1]/Q:.3f} | R@20={recall_sum['annoy'][2]/Q:.3f}")

print_ram("End of benchmark")

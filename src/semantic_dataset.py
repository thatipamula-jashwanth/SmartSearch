import numpy as np

def generate_semantic_vectors(
    n_samples=200_000,
    dim=256,
    n_clusters=50,
    sub_noise=0.10,
    global_noise=0.02,
):
    
    sub_noise = float(sub_noise)
    global_noise = float(global_noise)

    
    cluster_centers = np.random.randn(n_clusters, dim).astype(np.float32)
    cluster_centers /= np.linalg.norm(cluster_centers, axis=1, keepdims=True)

  
    cluster_ids = np.random.randint(0, n_clusters, size=n_samples)

   
    vectors = cluster_centers[cluster_ids] + sub_noise * np.random.randn(n_samples, dim)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)

    vectors += global_noise * np.random.randn(n_samples, dim)
    vectors = vectors.astype(np.float32)

    return vectors, cluster_ids, cluster_centers

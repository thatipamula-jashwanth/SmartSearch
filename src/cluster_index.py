import numpy as np
from sklearn.cluster import MiniBatchKMeans
import warnings


class ClusteringIndex:

    def __init__(self, n_coarse=256, n_fine=8, batch_size=4096, seed=42, max_postings=None):
        if n_coarse <= 0 or n_fine <= 0:
            raise ValueError("n_coarse and n_fine must be positive integers")

        self.n_coarse = int(n_coarse)
        self.n_fine = int(n_fine)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.max_postings = max_postings  

        self.centroids_l0 = None
        self.centroids_l1 = {}
        self.postings_l1 = {}

    def build(self, routing_vectors):
        X = np.asarray(routing_vectors, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("routing_vectors must be a 2-D array")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("routing_vectors contains NaN / Inf")

        N, D = X.shape
        if N < self.n_coarse:
            raise ValueError(f"Dataset too small for requested n_coarse: N={N} < n_coarse={self.n_coarse}")

        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms < 1e-12] = 1.0 
        X = X / norms

        if self.n_coarse * self.n_fine > 500_000:
            warnings.warn(
                "n_coarse × n_fine is extremely large — memory usage may be high",
                RuntimeWarning
            )

       
        k0 = MiniBatchKMeans(
            n_clusters=self.n_coarse,
            batch_size=self.batch_size,
            random_state=self.seed
            
        )
        labels0 = k0.fit_predict(X)
        C0 = k0.cluster_centers_.astype(np.float32)
        C0 /= np.linalg.norm(C0, axis=1, keepdims=True) + 1e-12
        self.centroids_l0 = C0

        
        self.postings_l1 = {c: {} for c in range(self.n_coarse)}

       
        for c in range(self.n_coarse):
            ids = np.where(labels0 == c)[0]
            if len(ids) == 0:
                continue

            Xc = X[ids]
            nf = min(self.n_fine, len(Xc))

           
            if nf == 1:
                self.centroids_l1[c] = Xc.copy()
                self.postings_l1[c][0] = ids.tolist()
                continue

            k1 = MiniBatchKMeans(
                n_clusters=nf,
                batch_size=self.batch_size,
                random_state=self.seed + c
               
            )
            labels1 = k1.fit_predict(Xc)
            C1 = k1.cluster_centers_.astype(np.float32)
            C1 /= np.linalg.norm(C1, axis=1, keepdims=True) + 1e-12
            self.centroids_l1[c] = C1

            for f in range(nf):
                mask = labels1 == f
                posting_ids = ids[mask]
                if self.max_postings is not None and len(posting_ids) > self.max_postings:
                    posting_ids = posting_ids[:self.max_postings]
                self.postings_l1[c][f] = posting_ids.tolist()

        return self

  
    def search_centroids(self, query, topC=4):
        if self.centroids_l0 is None:
            raise RuntimeError("ClusteringIndex.build() must be called first")

        q = np.asarray(query, dtype=np.float32)
        if q.ndim != 1:
            raise ValueError("query must be 1-D")
        if np.isnan(q).any() or np.isinf(q).any():
            raise ValueError("query contains NaN/Inf")

        q /= (np.linalg.norm(q) + 1e-12)

        sims = self.centroids_l0 @ q
        k = min(topC, sims.shape[0])
        idx = np.argpartition(sims, -k)[-k:]
        return idx[np.argsort(sims[idx])[::-1]]

    
    def search_fine(self, query, coarse_id, topF=2):
        C1 = self.centroids_l1.get(coarse_id)
        if C1 is None or C1.size == 0:
            return []

        q = np.asarray(query, dtype=np.float32)
        q /= np.linalg.norm(q) + 1e-12

        sims = C1 @ q

        k = min(topF, sims.shape[0])
        idx = np.argpartition(sims, -k)[-k:]
        return idx[np.argsort(sims[idx])[::-1]]

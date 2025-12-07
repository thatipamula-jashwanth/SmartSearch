import numpy as np

class ProjectionEmbedder:

    def __init__(self, original_dim, routing_dim=None, seed=42):
        self.original_dim = int(original_dim)
        self.seed = int(seed)

        if routing_dim is None:
            od = self.original_dim
            if od <= 64:
                routing_dim = od
            elif od <= 256:
                routing_dim = 96
            elif od <= 512:
                routing_dim = 128
            elif od <= 1024:
                routing_dim = 192
            else:
                routing_dim = 256

        self.routing_dim = int(routing_dim)


        rng = np.random.default_rng(self.seed)
        R = rng.standard_normal((self.original_dim, self.routing_dim), dtype=np.float32)
        Q, _ = np.linalg.qr(R)
        self.R = Q[:, :self.routing_dim].astype(np.float32, copy=False)

    def transform(self, X, normalize_input=True, return_fp16=False, chunk=200_000):
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=np.float32)
        else:
            X = X.astype(np.float32, copy=False)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.original_dim:
            raise ValueError(
                f"[ProjectionEmbedder] Expected dim={self.original_dim}, got {X.shape[1]}"
            )


        if normalize_input:
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            mask = norms > 1e-12
            if not np.all(mask):
               
                X = X.copy()
                X[mask] = X[mask] / norms[mask]
            else:
                X = X / (norms + 1e-12)

        N = X.shape[0]
        Y = np.empty((N, self.routing_dim), dtype=np.float32)


        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            Y[start:end] = X[start:end] @ self.R

   
        Yn = np.linalg.norm(Y, axis=1, keepdims=True)
        mask = Yn > 1e-12
        if not np.all(mask):
          
            Y[mask] = Y[mask] / Yn[mask]
       
        else:
            Y = Y / (Yn + 1e-12)

        if return_fp16:
            Y = Y.astype(np.float16)

        return Y

    def save(self, path):
        np.savez(
            path,
            R=self.R,
            original_dim=self.original_dim,
            routing_dim=self.routing_dim,
            seed=self.seed,
            version=5,
            allow_pickle=False,
        )

    @staticmethod
    def load(path):
        data = np.load(path, allow_pickle=False)

        ver = int(data.get("version", 0))
        if ver != 5:
            raise ValueError(
                f"ProjectionEmbedder version mismatch: expected 5, got {ver}"
            )

        embedder = ProjectionEmbedder(
            original_dim=int(data["original_dim"]),
            routing_dim=int(data["routing_dim"]),
            seed=int(data.get("seed", 42)),
        )
        embedder.R = data["R"].astype(np.float32, copy=False)
        return embedder

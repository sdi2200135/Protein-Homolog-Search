import numpy as np
from collections import defaultdict

class EuclideanLSH:
    def __init__(self, vectors, k=10, L=2, w=4.0, seed=42):
        self.vectors = vectors
        self.ids = list(vectors.keys())
        self.data = np.array(list(vectors.values()))
        self.n,self.dim = self.data.shape

        self.k = k
        self.L = L
        self.w = w
        self.rng = np.random.default_rng(seed)

        #για καθε table κραταμε hash table + random vectors/shifts
        self.hash_tables = []
        self.random_vectors = []
        self.random_shifts = []

        self._initialize()
        self._build_tables()

    def _initialize(self):
        for _ in range(self.L):
            v = self.rng.normal(0, 1, size=(self.k, self.dim))
            t = self.rng.uniform(0, self.w, self.k)
            self.random_vectors.append(v)
            self.random_shifts.append(t)
            self.hash_tables.append(defaultdict(list))
    
    def _hash(self, x, table_idx):
        v = self.random_vectors[table_idx]
        t = self.random_shifts[table_idx]
        h = np.floor((v @ x + t) / self.w).astype(int)
        return tuple(h)

    def _build_tables(self):
        for i,x in enumerate(self.data):
            for l in range(self.L):
                h = self._hash(x,l)
                self.hash_tables[l][h].append(i)

    def query(self,query_vector,N):
        candidates = set()
        for l in range(self.L):
            h = self._hash(query_vector,l)
            candidates.update(self.hash_tables[l].get(h,[]))

        results = []
        for idx in candidates:
            dist = np.linalg.norm(query_vector - self.data[idx])
            results.append((self.ids[idx], dist))
        
        results.sort(key=lambda x: x[1])
        return results[:N]
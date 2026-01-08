import numpy as np
import itertools
from collections import defaultdict

class Hypercube:
    def __init__(self, vectors, k=10, M=1000,probes=5, w= 4.0, seed =42):
        
        self.k = k 
        self.M = M
        self.probes = probes
        self.w = w
        self.seed = seed

        self.ids = list(vectors.keys())
        self.data = np.array(list(vectors.values()))
        self.n,self.dim = self.data.shape

        
        rng = np.random.default_rng(seed)

        #για καθε table κραταμε hash table + random vectors/shifts
        self.v = rng.normal(0, 1, size=(k, self.dim)) 
        self.t =  rng.uniform(0, w, size=k)
        
        self.cube =  defaultdict(list)
        self._build_index()

    def _hash(self,x):
        dots = (self.v @ x + self.t) / self.w
        return tuple((np.floor(dots) % 2).astype(int))
    
    def _build_index(self):
        for idx, vec in enumerate(self.data):
            vertex = self._hash(vec)
            self.cube[vertex].append(idx)
    
    def _hamming_vertices(self, vertex, max_vertices):
        yield vertex
        count = 1
        for dist in range(1, self.k + 1):
            for positions in itertools.combinations(range(self.k), dist):
                if count >= max_vertices:
                    return
                v = list(vertex)
                for p in positions:
                    v[p] ^= 1
                yield tuple(v)
                count += 1

    def query(self, query_vector, N):
        query_vertex = self._hash(query_vector)

        candidates = []
        checked = 0
        visited_vertices = 0

        for vertex in self._hamming_vertices(query_vertex, self.probes):
            if visited_vertices >= self.probes:
                break

            if vertex not in self.cube:
                continue

            for idx in self.cube[vertex]:
                if checked >= self.M:
                    break

                dist = np.linalg.norm(self.data[idx] - query_vector)
                candidates.append((self.ids[idx], dist))
                checked += 1

            visited_vertices += 1

            if checked >= self.M:
                break

        # fallback αν δεν βρεθεί τίποτα
        if not candidates:
            for idx in range(len(self.data)):
                dist = np.linalg.norm(self.data[idx] - query_vector)
                candidates.append((self.ids[idx], dist))

        candidates.sort(key=lambda x: x[1])
        return candidates[:N]
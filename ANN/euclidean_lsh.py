import numpy as np
from collections import defaultdict
import itertools

class EuclideanLSH:
    def __init__(self, vectors, k=10, L=2, w=4.0, seed=42, probe_radius=1):
        self.vectors = vectors
        self.ids = list(vectors.keys())
        self.data = np.array(list(vectors.values()))
        self.n, self.dim = self.data.shape
        
        # LSH parameters
        self.k = k
        self.L = L
        self.w = w
        self.probe_radius = probe_radius  # Απόσταση Hamming για πολλαπλά buckets
        self.rng = np.random.default_rng(seed)
        
        # Data structures
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
    
    def _get_nearby_buckets(self, hash_code, radius=1):
        """Παράγει όλα τα hash codes μέσα σε Hamming distance radius"""
        nearby = [hash_code]
        if radius > 0:
            # Για κάθε θέση στο hash code
            for i in range(len(hash_code)):
                # Αλλαγή της τιμής στη θέση i
                base_list = list(hash_code)
                
                base_list[i] = hash_code[i] + 1
                nearby.append(tuple(base_list))
                
                base_list[i] = hash_code[i] - 1
                nearby.append(tuple(base_list))
                
                # Μπορούμε να προσθέσουμε περισσότερες τιμές για μεγαλύτερο radius
                if radius > 1:
                    for delta in range(2, radius + 1):
                        base_list[i] = hash_code[i] + delta
                        nearby.append(tuple(base_list))
                        base_list[i] = hash_code[i] - delta
                        nearby.append(tuple(base_list))
        
        return list(set(nearby))  # Αφαίρεση διπλοτύπων
    
    def _build_tables(self):
        for i, x in enumerate(self.data):
            for l in range(self.L):
                h = self._hash(x, l)
                self.hash_tables[l][h].append(i)
    
    def query(self, query_vector, N):
        candidates = set()
        
        # 1. Συλλογή υποψηφίων από hash tables με multi-probe
        for l in range(self.L):
            h = self._hash(query_vector, l)
            
            # Πάρτε το ακριβές bucket
            candidates.update(self.hash_tables[l].get(h, []))
            
            # Ψάξτε σε παρακείμενα buckets
            if self.probe_radius > 0:
                nearby_buckets = self._get_nearby_buckets(h, self.probe_radius)
                for bucket in nearby_buckets:
                    if bucket != h:  # Το ακριβές bucket το έχουμε ήδη
                        candidates.update(self.hash_tables[l].get(bucket, []))
        
        # 2. Αν δεν βρήκαμε αρκετούς υποψηφίους, ψάξτε σε περισσότερα buckets
        if len(candidates) < N * 2:  # Χρειαζόμαστε τουλάχιστον 2*N υποψηφίους
            print(f"  Προειδοποίηση: Μόνο {len(candidates)} υποψήφιοι. Επέκταση αναζήτησης...")
            # Ψάξτε σε όλα τα buckets των πρώτων L/2 tables
            for l in range(min(self.L // 2, self.L)):
                for bucket in self.hash_tables[l].keys():
                    candidates.update(self.hash_tables[l][bucket])
                    if len(candidates) >= N * 10:  # Σταματήστε αν έχουμε αρκετούς
                        break
        
        # 3. Υπολογισμός αποστάσεων
        results = []
        for idx in candidates:
            dist = np.linalg.norm(query_vector - self.data[idx])
            results.append((self.ids[idx], dist))
        
        # 4. Ταξινόμηση και επιστροφή Top-N
        results.sort(key=lambda x: x[1])
        return results[:N]
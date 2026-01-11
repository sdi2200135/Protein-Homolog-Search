import numpy as np
from collections import defaultdict

class EuclideanLSH:
    def __init__(self, vectors, k=10, L=2, w=4.0, seed=42):
        
        self.vectors = vectors # λεξικο embeddings (id -> vector)
        self.ids = list(vectors.keys()) # λιστα με τα portein ids (index -> protein_id)
        self.data = np.array(list(vectors.values())) # πινακας NxD με ολα τα embeddings
        self.n,self.dim = self.data.shape #πληθος vectors , διασταση

        # παραμετροι LSH
        self.k = k    # hash function ανα table
        self.L = L    # πληθος hash tables
        self.w = w    #bucket width
        self.rng = np.random.default_rng(seed)

        #για καθε table κραταμε hash table + random vectors/shifts
        self.hash_tables = []       # λιστα απο defaultdicts
        self.random_vectors = []    # λιστα απο (k x dim) πίνακες
        self.random_shifts = []     # λιστα απο k-dimensional vectors

        self._initialize()  #αρχικοποιηση δομων 
        self._build_tables()    # γεμισμα  hash tables με τα δεδομενα

    # Δημιουργει τυχαιες hash functions για καθε table
    def _initialize(self):
        for _ in range(self.L):
            v = self.rng.normal(0, 1, size=(self.k, self.dim))
            t = self.rng.uniform(0, self.w, self.k)
            self.random_vectors.append(v)
            self.random_shifts.append(t)
            self.hash_tables.append(defaultdict(list))
    
    # Hash function για συγκεκριμενο table
    def _hash(self, x, table_idx):
        v = self.random_vectors[table_idx]
        t = self.random_shifts[table_idx]
        h = np.floor((v @ x + t) / self.w).astype(int)
        return tuple(h)

    # Κατασκευη hash tables
    def _build_tables(self):
        for i,x in enumerate(self.data):
            for l in range(self.L):
                h = self._hash(x,l)
                self.hash_tables[l][h].append(i)

    # Query: ευρεση ANN
    def query(self,query_vector,N):

        # συνολο υποψηφιων indices (χωρίς διπλοτυπα)
        candidates = set()
        # 1. Συλλογη υποψηφιων απο hash tables 
        for l in range(self.L):
            h = self._hash(query_vector,l)  ## Παίρνουμε όλα τα vectors που έπεσαν στο ίδιο bucket
            candidates.update(self.hash_tables[l].get(h,[]))

        # 2. Ακριβης υπολογισμος αποστασεων μονο στους υποψηφιους
        results = []
        for idx in candidates:
            dist = np.linalg.norm(query_vector - self.data[idx])  # Ευκλειδεια αποσταση
            results.append((self.ids[idx], dist))   # Αποθηκευουμε (protein_id, αποσταση)
        
         # 3. Ταξινομηση και επιστροφη Top-N
        results.sort(key=lambda x: x[1])
        return results[:N]
    
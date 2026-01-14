# ivfpq.py
import numpy as np
from typing import List, Tuple, Dict, Optional
import heapq
from sklearn.cluster import KMeans

# Υλοποίηση του IVFPQ index για αναζήτηση σε διανύσματα πρωτεϊνών
class IVFPQ:    
    def __init__(self, n_clusters: int = 100, n_subvectors: int = 8, 
                 n_bits: int = 8, random_state: int = 42):
        self.n_clusters = n_clusters
        self.M = n_subvectors
        self.n_bits = n_bits
        self.random_state = random_state
        self.n_pq_centroids = 1 << n_bits  # 2^n_bits
        
        # Αποθήκευση δεδομένων
        self.ivf_centroids = None
        self.inverted_lists = None
        self.pq_codebooks = None
        self.pq_codes = None
        self.vector_ids = None
        self.vector_dim = None
        self.sub_dim = None
        
        print(f"IVFPQ initialized: n_clusters={n_clusters}, M={n_subvectors}, n_bits={n_bits}")
    
    def _compute_residual(self, vector: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        return vector - centroid
    
    # Υπολογισμός Ευκλείδειας απόστασης 
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        # Μετατροπή σε float arrays
        a = np.asarray(a, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32)
        return float(np.sqrt(np.sum((a - b) ** 2)))
    
    def _find_nearest_centroids(self, query: np.ndarray, n_probe: int) -> List[int]:
        if self.ivf_centroids is None:
            raise ValueError("Index not built!")
        
        distances = np.sum((self.ivf_centroids - query) ** 2, axis=1)
        return np.argsort(distances)[:n_probe].tolist()
    
    # Δημιουργία Look-Up Table 
    def _build_LUT(self, query_residual: np.ndarray) -> np.ndarray:
        if self.pq_codebooks is None:
            raise ValueError("Index not built!")
        
        # Δημιουργία LUT με float dtype
        LUT = np.zeros((self.M, self.n_pq_centroids), dtype=np.float32)
        
        # Βεβαιωθείτε ότι το query_residual είναι float array
        query_residual = np.asarray(query_residual, dtype=np.float32)
        
        for m in range(self.M):
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            
            # Βεβαιωθείτε για τα όρια
            if start_idx >= len(query_residual):
                break
                
            end_idx = min(end_idx, len(query_residual))
            sub_query = query_residual[start_idx:end_idx]
            
            # Αν το sub_query είναι μικρότερο από sub_dim, κάντε padding
            if len(sub_query) < self.sub_dim:
                padded = np.zeros(self.sub_dim, dtype=np.float32)
                padded[:len(sub_query)] = sub_query
                sub_query = padded
            
            # Βεβαιωθείτε ότι το codebook είναι float array
            codebook = np.asarray(self.pq_codebooks[m], dtype=np.float32)
            
            # Υπολογισμός αποστάσεων
            for c in range(min(len(codebook), self.n_pq_centroids)):
                LUT[m, c] = float(np.sqrt(np.sum((sub_query - codebook[c]) ** 2)))
        
        return LUT
    
    # Κατασκευή IVFPQ index
    def build(self, vectors: Dict[str, np.ndarray]):
        if not vectors:
            raise ValueError("Empty vectors!")
        
        self.vector_ids = list(vectors.keys())
        vector_list = list(vectors.values())
        X = np.array(vector_list)
        
        n_vectors, self.vector_dim = X.shape
        print(f"Building IVFPQ index for {n_vectors} vectors of dimension {self.vector_dim}")
        
        # Έλεγχος ότι η διάσταση διαιρείται με M
        if self.vector_dim % self.M != 0:
            # Αντί για error, προσαρμόζουμε το M
            # Βρες τον μεγαλύτερο κοινό διαιρέτη
            possible_M = []
            for m in [1, 2, 4, 8, 16, 32, 64]:
                if self.vector_dim % m == 0:
                    possible_M.append(m)
            
            if possible_M:
                new_M = max(possible_M)
                print(f"Warning: Dimension {self.vector_dim} not divisible by M={self.M}. Using M={new_M} instead.")
                self.M = new_M
            else:
                raise ValueError(f"Dimension {self.vector_dim} has no common divisor with typical M values!")
        
        self.sub_dim = self.vector_dim // self.M
        
        # 1. IVF Clustering
        print("Step 1: IVF clustering...")
        actual_n_clusters = min(self.n_clusters, n_vectors // 10)
        if actual_n_clusters < 2:
            actual_n_clusters = 2
        
        kmeans_ivf = KMeans(
            n_clusters=actual_n_clusters,
            random_state=self.random_state,
            n_init=3  # Μείωσε για ταχύτητα
        )
        ivf_labels = kmeans_ivf.fit_predict(X)
        self.ivf_centroids = kmeans_ivf.cluster_centers_
        self.n_clusters = actual_n_clusters  # Ενημέρωσε το πραγματικό αριθμό
        
        # 2. Δημιουργία αναστραμμένων λιστών
        print("Step 2: Creating inverted lists...")
        self.inverted_lists = [[] for _ in range(self.n_clusters)]
        for idx, label in enumerate(ivf_labels):
            self.inverted_lists[label].append(idx)
        
        # 3. Υπολογισμός residuals
        print("Step 3: Computing residuals...")
        residuals = np.zeros_like(X)
        for idx in range(n_vectors):
            centroid_idx = ivf_labels[idx]
            residuals[idx] = X[idx] - self.ivf_centroids[centroid_idx]
        
        # 4. Product Quantization Training - ΔΙΟΡΘΩΜΕΝΟ
        print(f"Step 4: PQ training with M={self.M} subspaces...")
        self.pq_codebooks = []
        
        for m in range(self.M):
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            subspace_data = residuals[:, start_idx:end_idx]
    
            if self.sub_dim < 4:
                print(f"Warning: sub_dim={self.sub_dim} is too small. Adjusting M...")
               
                self.M = self.vector_dim // 4
                self.sub_dim = 4
                print(f"New M: {self.M}, new sub_dim: {self.sub_dim}")
            elif self.sub_dim < 10:
          
                actual_pq_centroids = 16
            elif self.sub_dim < 20:
           
                actual_pq_centroids = 64
            else:
       
                actual_pq_centroids = 256
            
            actual_pq_centroids = min(actual_pq_centroids, len(subspace_data))
            
            if len(subspace_data) >= actual_pq_centroids:
                kmeans_pq = KMeans(
                    n_clusters=actual_pq_centroids,
                    random_state=self.random_state + m,
                    n_init=2
                )
                kmeans_pq.fit(subspace_data)
                codebook = kmeans_pq.cluster_centers_
            else:   
                idxs = np.random.choice(len(subspace_data), actual_pq_centroids, replace=True)
                codebook = subspace_data[idxs] + np.random.randn(actual_pq_centroids, self.sub_dim) * 0.01
            
            full_codebook = np.zeros((self.n_pq_centroids, self.sub_dim))
            full_codebook[:actual_pq_centroids] = codebook
            self.pq_codebooks.append(full_codebook)
        
        # 5. Κωδικοποίηση διανυσμάτων
        print("Step 5: Encoding vectors...")
        self.pq_codes = np.zeros((n_vectors, self.M), dtype=np.uint8)
        
        for idx in range(n_vectors):
            centroid_idx = ivf_labels[idx]
            residual = residuals[idx]
            
            for m in range(self.M):
                start_idx = m * self.sub_dim
                end_idx = start_idx + self.sub_dim
                subvector = residual[start_idx:end_idx]
                
                # Χρησιμοποιούμε μόνο τους πραγματικούς centroids
                codebook = self.pq_codebooks[m][:actual_pq_centroids]
                distances = np.sum((codebook - subvector) ** 2, axis=1)
                best_code = np.argmin(distances)
                self.pq_codes[idx, m] = best_code
        
        print(f"IVFPQ index built successfully! {n_vectors} vectors encoded.")
        print(f"Final parameters: n_clusters={self.n_clusters}, M={self.M}, sub_dim={self.sub_dim}")
    
    # Αναζήτηση
    def query(self, query_vector: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[str, float]]:
        if self.ivf_centroids is None or self.pq_codebooks is None:
            raise ValueError("Index not built!")
         
        query_vector = np.asarray(query_vector, dtype=np.float32)
        
        # 1. Εύρεση πλησιέστερων centroids
        nearest_centroids = self._find_nearest_centroids(query_vector, n_probe)
        
        # 2. Αναζήτηση σε επιλεγμένα clusters
        candidates = []
        
        for centroid_idx in nearest_centroids:
            centroid = self.ivf_centroids[centroid_idx]
            query_residual = query_vector - centroid
            
            # Δημιουργία LUT
            LUT = self._build_LUT(query_residual)
            
            # Έλεγχος διανυσμάτων στο cluster
            for vector_idx in self.inverted_lists[centroid_idx]:
                # Υπολογισμός απόστασης
                dist_sq = 0.0
                for m in range(self.M):
                    code = int(self.pq_codes[vector_idx, m])  
                    dist_sq += float(LUT[m, code]) ** 2
                
                dist = float(np.sqrt(dist_sq))
                vector_id = self.vector_ids[vector_idx]
                candidates.append((dist, vector_id))
        
        # Ταξινόμηση και επιστροφή
        candidates.sort(key=lambda x: float(x[0]))  
        return [(id_, float(dist)) for dist, id_ in candidates[:k]]  # Επιστροφή ως floats

# Wrapper κλάση για χρήση στο protein_search.py
class IVFPQSearch:
    
    def __init__(self, vectors: Dict[str, np.ndarray], nlist: int = 100, nprobe: int = 10, m: int = 8, seed: int = 42):
        embedding_dim = list(vectors.values())[0].shape[0]
        
        # Βρες κατάλληλο m αν το δοσμένο δεν διαιρεί
        if embedding_dim % m != 0:
            print(f"Warning: Embedding dimension {embedding_dim} not divisible by m={m}")
            # Βρες τον μεγαλύτερο διαιρέτη που είναι <= m
            divisors = []
            for d in range(1, min(m, embedding_dim) + 1):
                if embedding_dim % d == 0:
                    divisors.append(d)
            
            if divisors:
                m = max(divisors)
                print(f"Using m={m} instead")
            else:
                m = 1
                print(f"Using m={m} as fallback")
        
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.seed = seed
        
        # Δημιουργία IVFPQ index
        self.index = IVFPQ(
            n_clusters=nlist,
            n_subvectors=m,
            n_bits=8,  # Χρησιμοποιούμε 8 bits (256 centroids)
            random_state=seed
        )
        
        # Κατασκευή index
        self.index.build(vectors)
        
        print(f"IVF-PQ initialized (nlist={nlist}, m={m})")
    
    def query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        return self.index.query(query_vector, k=k, n_probe=self.nprobe)
    
    def get_stats(self) -> Dict:
        return self.index.get_statistics()
# ivfpq.py
import numpy as np
from typing import List, Tuple, Dict, Optional
import heapq
from sklearn.cluster import KMeans

class KMeansClustering:
    """Υλοποίηση του k-means για IVFPQ"""
    
    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        
    def fit(self, X: np.ndarray) -> np.ndarray:
        """Εκπαίδευση k-means"""
        n_samples, n_features = X.shape
        
        # Αρχικοποίηση με k-means++
        np.random.seed(self.random_state)
        
        # 1ο centroid τυχαίο
        centroids = [X[np.random.randint(n_samples)]]
        
        for _ in range(1, self.n_clusters):
            # Υπολογισμός αποστάσεων
            distances = np.zeros(n_samples)
            for i, x in enumerate(X):
                min_dist = np.inf
                for c in centroids:
                    dist = np.sum((x - c) ** 2)
                    min_dist = min(min_dist, dist)
                distances[i] = min_dist
            
            # Επιλογή επόμενου centroid
            probabilities = distances / distances.sum()
            cumulative_probs = probabilities.cumsum()
            r = np.random.rand()
            next_idx = np.searchsorted(cumulative_probs, r)
            centroids.append(X[next_idx])
        
        self.centroids = np.array(centroids)
        
        # Κύριος αλγόριθμος k-means
        for iteration in range(self.max_iter):
            # Αντιστοίχιση σημείων στα centroids
            labels = np.argmin(np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2), axis=1)
            
            # Ενημέρωση centroids
            new_centroids = np.zeros_like(self.centroids)
            for i in range(self.n_clusters):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    # Επαναρχικοποίηση κενού cluster
                    new_centroids[i] = X[np.random.randint(n_samples)]
            
            # Έλεγχος σύγκλισης
            centroid_shift = np.sqrt(np.sum((new_centroids - self.centroids) ** 2))
            self.centroids = new_centroids
            
            if centroid_shift < self.tol:
                break
        
        return self.centroids
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Πρόβλεψη labels για νέα δεδομένα"""
        if self.centroids is None:
            raise ValueError("Το μοντέλο δεν έχει εκπαιδευτεί!")
        
        distances = np.sum((X[:, np.newaxis] - self.centroids) ** 2, axis=2)
        return np.argmin(distances, axis=1)


class IVFPQ:
    """Υλοποίηση του IVFPQ index για αναζήτηση σε διανύσματα πρωτεϊνών"""
    
    def __init__(self, n_clusters: int = 100, n_subvectors: int = 8, 
                 n_bits: int = 8, random_state: int = 42):
        """
        Αρχικοποίηση IVFPQ
        
        Παράμετροι:
        -----------
        n_clusters : int
            Αριθμός clusters για το IVF
        n_subvectors : int
            Αριθμός υποδιανυσμάτων για Product Quantization (M)
        n_bits : int
            Αριθμός bits για κωδικοποίηση (2^n_bits centroids ανά υποχώρο)
        random_state : int
            Seed για αναπαραγωγιμότητα
        """
        self.n_clusters = n_clusters
        self.M = n_subvectors
        self.n_bits = n_bits
        self.random_state = random_state
        self.n_pq_centroids = 1 << n_bits  # 2^n_bits
        
        # Αποθήκευση δεδομένων
        self.ivf_centroids = None  # IVF centroids
        self.inverted_lists = None  # Αναστραμμένες λίστες
        self.pq_codebooks = None    # PQ codebooks
        self.pq_codes = None        # Κωδικοποιημένα διανύσματα
        self.vector_ids = None      # IDs διανυσμάτων
        self.vector_dim = None      # Διάσταση διανυσμάτων
        self.sub_dim = None         # Διάσταση υποδιανύσματος
        
        print(f"IVFPQ initialized: n_clusters={n_clusters}, "
              f"M={n_subvectors}, n_bits={n_bits}")
    
    def _compute_residual(self, vector: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Υπολογισμός υπολοίπου (διαφοράς από centroid)"""
        return vector - centroid
    
    def _euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Υπολογισμός Ευκλείδειας απόστασης"""
        return np.sqrt(np.sum((a - b) ** 2))
    
    def _find_nearest_centroids(self, query: np.ndarray, n_probe: int) -> List[int]:
        """Εύρεση n_probe πλησιέστερων centroids"""
        if self.ivf_centroids is None:
            raise ValueError("Το index δεν έχει κατασκευαστεί!")
        
        # Υπολογισμός αποστάσεων από όλα τα centroids
        distances = np.sqrt(np.sum((self.ivf_centroids - query) ** 2, axis=1))
        
        # Επιλογή των n_probe πλησιέστερων
        if n_probe >= self.n_clusters:
            return list(range(self.n_clusters))
        else:
            return np.argsort(distances)[:n_probe].tolist()
    
    def _build_LUT(self, query_residual: np.ndarray) -> np.ndarray:
        """
        Δημιουργία Look-Up Table για γρήγορους υπολογισμούς
        
        Επιστρέφει πίνακα μεγέθους M x n_pq_centroids
        """
        if self.pq_codebooks is None:
            raise ValueError("Το index δεν έχει κατασκευαστεί!")
        
        LUT = np.zeros((self.M, self.n_pq_centroids))
        
        for m in range(self.M):
            # Εξαγωγή υποδιανύσματος από το query
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            sub_query = query_residual[start_idx:end_idx]
            
            # Υπολογισμός αποστάσεων από όλα τα PQ centroids
            for c in range(self.n_pq_centroids):
                LUT[m, c] = self._euclidean_distance(sub_query, self.pq_codebooks[m][c])
        
        return LUT
    
    def build(self, vectors: Dict[str, np.ndarray]):
        """
        Κατασκευή IVFPQ index
        
        Παράμετροι:
        -----------
        vectors : Dict[str, np.ndarray]
            Λεξικό με ID διανυσμάτων ως κλειδιά και διανύσματα ως τιμές
        """
        if not vectors:
            raise ValueError("Η λίστα διανυσμάτων είναι κενή!")
        
        # Μετατροπή σε πίνακα και αποθήκευση IDs
        self.vector_ids = list(vectors.keys())
        vector_list = list(vectors.values())
        X = np.array(vector_list)
        
        n_vectors, self.vector_dim = X.shape
        print(f"Building IVFPQ index for {n_vectors} vectors of dimension {self.vector_dim}")
        
        # Έλεγχος ότι η διάσταση διαιρείται με M
        if self.vector_dim % self.M != 0:
            raise ValueError(f"Διάσταση {self.vector_dim} δεν διαιρείται με M={self.M}!")
        
        self.sub_dim = self.vector_dim // self.M
        
        # 1. IVF Clustering
        print("Step 1: IVF clustering...")
        kmeans_ivf = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        ivf_labels = kmeans_ivf.fit_predict(X)
        self.ivf_centroids = kmeans_ivf.cluster_centers_
        
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
            residuals[idx] = self._compute_residual(X[idx], self.ivf_centroids[centroid_idx])
        
        # 4. Product Quantization Training
        print(f"Step 4: PQ training with M={self.M} subspaces...")
        self.pq_codebooks = []
        
        for m in range(self.M):
            # Εξαγωγή υποδιανυσμάτων
            start_idx = m * self.sub_dim
            end_idx = start_idx + self.sub_dim
            subspace_data = residuals[:, start_idx:end_idx]
            
            # Κάθε subspace έχει 2^n_bits centroids
            n_pq_clusters = min(self.n_pq_centroids, len(subspace_data))
            
            if len(subspace_data) >= n_pq_clusters:
                kmeans_pq = KMeans(
                    n_clusters=n_pq_clusters,
                    random_state=self.random_state + m,
                    n_init=3
                )
                kmeans_pq.fit(subspace_data)
                codebook = kmeans_pq.cluster_centers_
            else:
                # Αν δεν έχουμε αρκετά δεδομένα, χρησιμοποιούμε τυχαία σημεία
                codebook = subspace_data[np.random.choice(len(subspace_data), n_pq_clusters, replace=True)]
            
            self.pq_codebooks.append(codebook)
        
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
                
                # Εύρεση πλησιέστερου centroid
                distances = np.sqrt(np.sum((self.pq_codebooks[m] - subvector) ** 2, axis=1))
                best_code = np.argmin(distances)
                self.pq_codes[idx, m] = best_code
        
        print(f"IVFPQ index built successfully! {n_vectors} vectors encoded.")
    
    def query(self, query_vector: np.ndarray, k: int = 10, n_probe: int = 10) -> List[Tuple[str, float]]:
        """
        Αναζήτηση k πλησιέστερων γειτόνων
        
        Παράμετροι:
        -----------
        query_vector : np.ndarray
            Διάνυσμα ερώτημα
        k : int
            Αριθμός πλησιέστερων γειτόνων
        n_probe : int
            Αριθμός clusters για έλεγχο
        
        Επιστρέφει:
        -----------
        List[Tuple[str, float]]
            Λίστα με (ID γείτονα, απόσταση)
        """
        if self.ivf_centroids is None or self.pq_codebooks is None:
            raise ValueError("Το index δεν έχει κατασκευαστεί!")
        
        # 1. Εύρεση πλησιέστερων centroids
        nearest_centroids = self._find_nearest_centroids(query_vector, n_probe)
        
        # 2. Αναζήτηση σε επιλεγμένα clusters
        candidates = []
        
        for centroid_idx in nearest_centroids:
            # Υπολογισμός residual
            centroid = self.ivf_centroids[centroid_idx]
            query_residual = self._compute_residual(query_vector, centroid)
            
            # Δημιουργία LUT
            LUT = self._build_LUT(query_residual)
            
            # Έλεγχος όλων των διανυσμάτων στο cluster
            for vector_idx in self.inverted_lists[centroid_idx]:
                # Υπολογισμός απόστασης χρησιμοποιώντας LUT
                dist = 0.0
                for m in range(self.M):
                    code = self.pq_codes[vector_idx, m]
                    dist += LUT[m, code] ** 2  # Χρήση τετραγωνικής απόστασης
                
                dist = np.sqrt(dist)  # Μετατροπή σε Ευκλείδεια απόσταση
                vector_id = self.vector_ids[vector_idx]
                candidates.append((dist, vector_id))
        
        # 3. Ταξινόμηση και επιστροφή των k πλησιέστερων
        candidates.sort(key=lambda x: x[0])
        return candidates[:k]
    
    def query_batch(self, query_vectors: Dict[str, np.ndarray], k: int = 10, 
                    n_probe: int = 10) -> Dict[str, List[Tuple[str, float]]]:
        """
        Αναζήτηση για πολλαπλά ερωτήματα
        
        Παράμετροι:
        -----------
        query_vectors : Dict[str, np.ndarray]
            Λεξικό με ερωτήματα
        k : int
            Αριθμός πλησιέστερων γειτόνων
        n_probe : int
            Αριθμός clusters για έλεγχο
        
        Επιστρέφει:
        -----------
        Dict[str, List[Tuple[str, float]]]
            Αποτελέσματα για κάθε ερώτημα
        """
        results = {}
        for query_id, query_vector in query_vectors.items():
            results[query_id] = self.query(query_vector, k, n_probe)
        
        return results
    
    def get_statistics(self) -> Dict:
        """Επιστροφή στατιστικών για το index"""
        if self.inverted_lists is None:
            return {}
        
        cluster_sizes = [len(lst) for lst in self.inverted_lists]
        
        return {
            'n_vectors': len(self.vector_ids) if self.vector_ids else 0,
            'n_clusters': self.n_clusters,
            'vector_dim': self.vector_dim,
            'sub_dim': self.sub_dim,
            'n_pq_centroids': self.n_pq_centroids,
            'cluster_sizes': {
                'min': min(cluster_sizes) if cluster_sizes else 0,
                'max': max(cluster_sizes) if cluster_sizes else 0,
                'avg': np.mean(cluster_sizes) if cluster_sizes else 0,
                'std': np.std(cluster_sizes) if cluster_sizes else 0
            }
        }


# Κλάση συμβατή με το protein_search.py
class IVFPQSearch:
    """Wrapper κλάση για χρήση στο protein_search.py"""
    
    def __init__(self, vectors: Dict[str, np.ndarray], nlist: int = 100, 
                 nprobe: int = 10, m: int = 8, seed: int = 42):
        """
        Αρχικοποίηση
        
        Παράμετροι:
        -----------
        vectors : Dict[str, np.ndarray]
            Διανύσματα βάσης δεδομένων
        nlist : int
            Αριθμός clusters (IVF)
        nprobe : int
            Αριθμός clusters για έλεγχο
        m : int
            Αριθμός υποδιανυσμάτων (M)
        seed : int
            Seed για αναπαραγωγιμότητα
        """
        self.nlist = nlist
        self.nprobe = nprobe
        self.m = m
        self.seed = seed
        
        # Δημιουργία IVFPQ index
        self.index = IVFPQ(
            n_clusters=nlist,
            n_subvectors=m,
            n_bits=8,  # Προκαθορισμένο
            random_state=seed
        )
        
        # Κατασκευή index
        self.index.build(vectors)
        
        print(f"IVF-PQ initialized (nlist={nlist}, m={m})")
    
    def query(self, query_vector: np.ndarray, k: int) -> List[Tuple[str, float]]:
        """
        Αναζήτηση k πλησιέστερων γειτόνων
        
        Παράμετροι:
        -----------
        query_vector : np.ndarray
            Διάνυσμα ερώτημα
        k : int
            Αριθμός πλησιέστερων γειτόνων
        
        Επιστρέφει:
        -----------
        List[Tuple[str, float]]
            Λίστα με (ID γείτονα, απόσταση)
        """
        return self.index.query(query_vector, k=k, n_probe=self.nprobe)
    
    def get_stats(self) -> Dict:
        """Επιστροφή στατιστικών"""
        return self.index.get_statistics()
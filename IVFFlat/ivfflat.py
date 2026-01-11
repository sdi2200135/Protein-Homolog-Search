# ivfflat.py
import numpy as np
from sklearn.cluster import KMeans
import heapq
import time
import random
from typing import List, Tuple, Dict, Set

class IVFFlat:
    def __init__(self, vectors: Dict[str, np.ndarray], nlist: int = 100, nprobe: int = 10, seed: int = 42):

        self.vectors = vectors
        self.nlist = nlist
        self.nprobe = nprobe
        self.seed = seed
        
        # Μετατροπη embeddings σε numpy array
        self.ids = list(vectors.keys())
        self.data = np.array([vectors[pid] for pid in self.ids])     # Διαστασιμοτητα embeddings
        self.dim = self.data.shape[1] if len(self.data.shape) > 1 else 1
        
        self.centroids = None               # κεντρα clusters
        self.cluster_assignments = None     # cluster id ανα vector
        self.cluster_members = None         # indices vectors ανα cluster
        self.index_built = False
        
        # random seed
        np.random.seed(seed)
        random.seed(seed)
    
    # Κατασκευη IVF index   
    def build_index(self):
        print(f"Building IVF-Flat index with {self.nlist} clusters...")
        
        # 1. k-means clustering
        kmeans = KMeans(n_clusters=self.nlist, random_state=self.seed, n_init=10)
        self.cluster_assignments = kmeans.fit_predict(self.data)            # cluster id για καθε vector
        self.centroids = kmeans.cluster_centers_                 # centroids των clusters
        
        # 2. Δημιουργία inverted lists
        self.cluster_members = [[] for _ in range(self.nlist)]
        for idx, cluster_id in enumerate(self.cluster_assignments):
            self.cluster_members[cluster_id].append(idx)
        
        self.index_built = True
        print(f"IVF-Flat index built. Cluster sizes: {[len(c) for c in self.cluster_members[:5]]}...")

    # Ευκλειδεια αποσταση    
    def euclidean_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.linalg.norm(v1 - v2)
    
    # Ευρεση κοντινοτερων centroids
    def find_nearest_centroids(self, query: np.ndarray, nprobe_count: int = None) -> List[int]:
        if nprobe_count is None:
            nprobe_count = self.nprobe
        
        # Αποσταση query -> καθε centroid
        distances = []
        for i, centroid in enumerate(self.centroids):
            dist = self.euclidean_distance(query, centroid)
            distances.append((dist, i))
        
        # Ταξινομηση και επιλογη nprobe
        distances.sort(key=lambda x: x[0])
        return [idx for _, idx in distances[:nprobe_count]]
    
    # Query k-NN
    def query(self, query_vec: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
      
        if not self.index_built:
            self.build_index()
        
        # 1. επιλογη clusters
        nearest_centroids = self.find_nearest_centroids(query_vec, self.nprobe)
        
        # 2. συλλογη υποψηφίων
        candidates = []
        for cluster_id in nearest_centroids:
            candidates.extend(self.cluster_members[cluster_id])
        
        # Αφαιρεση διπλοτυπων
        candidates = list(set(candidates))
        
        # 3. ακριβης υπολογισμος αποστασεων
        distances = []
        for idx in candidates:
            dist = self.euclidean_distance(query_vec, self.data[idx])
            distances.append((dist, idx))
        
        # k μικροτερες αποστασεις
        distances.sort(key=lambda x: x[0])
        top_k = distances[:k]
        
        #επιστροψη (protein_id, distance)
        return [(self.ids[idx], dist) for dist, idx in top_k]
    
    # Range search
    def range_search(self, query_vec: np.ndarray, radius: float) -> List[Tuple[str, float]]:

        if not self.index_built:
            self.build_index()
        
        # 1. βρισκουμε κοντινοτερα centroids
        nearest_centroids = self.find_nearest_centroids(query_vec, self.nprobe)
        
        # 2. συλλογη υποψηφιων απο τα επιλεγμενα clusters
        candidates = []
        for cluster_id in nearest_centroids:
            candidates.extend(self.cluster_members[cluster_id])
        
        # Αφαιρεση διπλοτυπων
        candidates = list(set(candidates))
        
        results = []
        for idx in candidates:
            dist = self.euclidean_distance(query_vec, self.data[idx])
            if dist <= radius:
                results.append((self.ids[idx], dist))
    
        results.sort(key=lambda x: x[1])
        return results
    
    # Silhouette score (ποιοτητα clustering)
    def compute_silhouette_score(self, sample_size: int = 100) -> float:
        if not self.index_built:
            self.build_index()
        
        n_samples = min(sample_size, len(self.data))
        sample_indices = np.random.choice(len(self.data), n_samples, replace=False)
        
        total_silhouette = 0.0
        valid_samples = 0
        
        for idx in sample_indices:
            point = self.data[idx]
            cluster_id = self.cluster_assignments[idx]
            
            # a(i): μεση αποσταση στο ιδιο cluster
            same_cluster_indices = [i for i in self.cluster_members[cluster_id] if i != idx]
            
            if len(same_cluster_indices) == 0:
                continue
            
            a_i = np.mean([self.euclidean_distance(point, self.data[i]) 
                          for i in same_cluster_indices])
            
            # b(i): κοντινοτερο αλλο cluster
            b_i = float('inf')
            
            for other_cluster in range(self.nlist):
                if other_cluster == cluster_id:
                    continue
                
                other_indices = self.cluster_members[other_cluster]
                if not other_indices:
                    continue
                
                sample_count = min(10, len(other_indices))
                sample_idx = np.random.choice(other_indices, sample_count, replace=False)
                
                avg_dist = np.mean([self.euclidean_distance(point, self.data[i]) 
                                   for i in sample_idx])
                b_i = min(b_i, avg_dist)
            
            if b_i == float('inf'):
                continue
      
            if max(a_i, b_i) > 0:
                silhouette = (b_i - a_i) / max(a_i, b_i)
                total_silhouette += silhouette
                valid_samples += 1
        
        return total_silhouette / valid_samples if valid_samples > 0 else 0.0
    
    # Στατιστικα clusters
    def get_cluster_stats(self) -> Dict:
        if not self.index_built:
            self.build_index()
        
        cluster_sizes = [len(members) for members in self.cluster_members]
        
        return {
            'n_clusters': self.nlist,
            'total_points': len(self.data),
            'min_cluster_size': min(cluster_sizes),
            'max_cluster_size': max(cluster_sizes),
            'avg_cluster_size': np.mean(cluster_sizes),
            'std_cluster_size': np.std(cluster_sizes),
            'empty_clusters': sum(1 for size in cluster_sizes if size == 0)
        }
    
    # Αποθηκευση index
    def save_index(self, filepath: str):
        """Save the IVF index to disk."""
        import pickle
        
        index_data = {
            'centroids': self.centroids,
            'cluster_assignments': self.cluster_assignments,
            'cluster_members': self.cluster_members,
            'ids': self.ids,
            'data_shape': self.data.shape,
            'nlist': self.nlist,
            'nprobe': self.nprobe,
            'seed': self.seed
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
        
        print(f"IVF index saved to {filepath}")
    
    # Φορτωση index
    def load_index(self, filepath: str):
        """Load IVF index from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.centroids = index_data['centroids']
        self.cluster_assignments = index_data['cluster_assignments']
        self.cluster_members = index_data['cluster_members']
        self.ids = index_data['ids']
        self.nlist = index_data['nlist']
        self.nprobe = index_data['nprobe']
        self.seed = index_data['seed']
        
        # Recreate data array from vectors
        self.data = np.array([self.vectors[pid] for pid in self.ids])
        self.dim = self.data.shape[1] if len(self.data.shape) > 1 else 1
        
        self.index_built = True
        print(f"IVF index loaded from {filepath}")

def create_ivfflat_index(vectors: Dict[str, np.ndarray], nlist: int = 100, nprobe: int = 10) -> IVFFlat:
  
    ivf = IVFFlat(vectors, nlist=nlist, nprobe=nprobe)
    ivf.build_index()
    return ivf


if __name__ == "__main__":
   
    n_points = 1000 # πληθος vectors
    dim = 320   # διασταση embeddings 
    
    # Δημιουργια τυχαιων embeddings
    vectors = {f"prot_{i}": np.random.randn(dim) for i in range(n_points)}
    
    # Δημιουργια IVF-Flat index
    ivf = IVFFlat(vectors, nlist=50, nprobe=5)
    ivf.build_index()  # Κατασκευη του index 
    
    # Εμφανιση στατιστικων για τα clusters
    stats = ivf.get_cluster_stats()
    print(f"Cluster stats: {stats}")
    
    # Υπολογισμος silhouette score
    silhouette = ivf.compute_silhouette_score()
    print(f"Silhouette score: {silhouette:.4f}")
    
    # Δημιουργια τυχαιου query vector
    query_vec = np.random.randn(dim)
    results = ivf.query(query_vec, k=5)  # αναζητηση k κοντινοτερων γειτονων (k-NN)
    
    print(f"\nTop 5 neighbors:")
    for i, (prot_id, dist) in enumerate(results):
        print(f"{i+1}. {prot_id}: distance = {dist:.4f}")
    
    radius = 10.0
    range_results = ivf.range_search(query_vec, radius)
    print(f"\nPoints within radius {radius}: {len(range_results)}")
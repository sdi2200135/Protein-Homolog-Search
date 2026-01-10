# ivfflat.py
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import time

@dataclass
class IVFFlatConfig:
    """Configuration for IVF-Flat algorithm"""
    n_clusters: int = 100
    n_probe: int = 10
    random_state: int = 42
    max_iter: int = 100
    tol: float = 1e-4

class IVFFlat:
    """
    Inverted File with Flat storage (IVF-Flat) for approximate nearest neighbor search.
    """
    
    def __init__(self, config: Optional[IVFFlatConfig] = None):
        self.config = config or IVFFlatConfig()
        self.is_trained = False
        self.centroids = None
        self.inverted_lists = None
        self.data_vectors = None
        self.data_ids = None
        
    def fit(self, vectors: Dict[str, np.ndarray]) -> 'IVFFlat':
        """
        Train the IVF-Flat index on protein embeddings.
        
        Args:
            vectors: Dictionary mapping protein IDs to their embeddings
            
        Returns:
            self: Trained IVFFlat instance
        """
        print(f"Training IVF-Flat with {self.config.n_clusters} clusters...")
        
        # Store data
        self.data_ids = list(vectors.keys())
        self.data_vectors = np.array([vectors[pid] for pid in self.data_ids])
        
        # Step 1: Cluster the data using K-Means
        kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.config.random_state,
            n_init=10
        )
        
        start_time = time.time()
        cluster_labels = kmeans.fit_predict(self.data_vectors)
        self.centroids = kmeans.cluster_centers_
        print(f"K-Means clustering completed in {time.time() - start_time:.2f}s")
        
        # Step 2: Build inverted lists
        self.inverted_lists = [[] for _ in range(self.config.n_clusters)]
        
        for idx, label in enumerate(cluster_labels):
            self.inverted_lists[label].append(idx)
        
        # Calculate cluster statistics
        cluster_sizes = [len(lst) for lst in self.inverted_lists]
        print(f"Cluster sizes: min={min(cluster_sizes)}, max={max(cluster_sizes)}, "
              f"avg={np.mean(cluster_sizes):.1f}")
        
        self.is_trained = True
        return self
    
    def _find_nearest_clusters(self, query_vector: np.ndarray, n_probe: int) -> List[int]:
        """
        Find the n_probe nearest clusters to the query vector.
        
        Args:
            query_vector: Query embedding
            n_probe: Number of clusters to probe
            
        Returns:
            List of cluster indices sorted by proximity
        """
        # Calculate distances to all centroids
        query_expanded = query_vector.reshape(1, -1)
        distances = np.linalg.norm(self.centroids - query_expanded, axis=1)
        
        # Get indices of n_probe closest clusters
        nearest_indices = np.argsort(distances)[:n_probe]
        return nearest_indices.tolist()
    
    def query(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Query the index for k nearest neighbors.
        
        Args:
            query_vector: Query embedding
            k: Number of neighbors to return
            
        Returns:
            List of tuples (protein_id, distance)
        """
        if not self.is_trained:
            raise RuntimeError("IVF-Flat index not trained. Call fit() first.")
        
        # Step 1: Find nearest clusters to probe
        cluster_indices = self._find_nearest_clusters(query_vector, self.config.n_probe)
        
        # Step 2: Collect candidates from selected clusters
        candidate_indices = []
        for cluster_idx in cluster_indices:
            candidate_indices.extend(self.inverted_lists[cluster_idx])
        
        if not candidate_indices:
            return []
        
        # Step 3: Compute exact distances for candidates
        candidate_vectors = self.data_vectors[candidate_indices]
        distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
        
        # Step 4: Sort by distance and return top-k
        sorted_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in sorted_indices:
            data_idx = candidate_indices[idx]
            protein_id = self.data_ids[data_idx]
            distance = distances[idx]
            results.append((protein_id, float(distance)))
        
        return results
    
    def query_batch(self, query_vectors: List[np.ndarray], k: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Query multiple vectors at once.
        
        Args:
            query_vectors: List of query embeddings
            k: Number of neighbors to return per query
            
        Returns:
            List of results for each query
        """
        return [self.query(qv, k) for qv in query_vectors]
    
    def save_index(self, filepath: str):
        """Save the index to disk."""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'config': self.config,
                'centroids': self.centroids,
                'inverted_lists': self.inverted_lists,
                'data_ids': self.data_ids,
                'data_vectors': self.data_vectors,
                'is_trained': self.is_trained
            }, f)
    
    @classmethod
    def load_index(cls, filepath: str) -> 'IVFFlat':
        """Load index from disk."""
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        ivf = cls(config=data['config'])
        ivf.centroids = data['centroids']
        ivf.inverted_lists = data['inverted_lists']
        ivf.data_ids = data['data_ids']
        ivf.data_vectors = data['data_vectors']
        ivf.is_trained = data['is_trained']
        
        return ivf
    
    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        if not self.is_trained:
            return {}
        
        cluster_sizes = [len(lst) for lst in self.inverted_lists]
        
        return {
            'n_clusters': self.config.n_clusters,
            'n_probe': self.config.n_probe,
            'total_vectors': len(self.data_vectors),
            'cluster_size_min': min(cluster_sizes),
            'cluster_size_max': max(cluster_sizes),
            'cluster_size_avg': np.mean(cluster_sizes),
            'cluster_size_std': np.std(cluster_sizes),
            'centroid_shape': self.centroids.shape
        }


# Alternative implementation with custom K-Means (if sklearn is not available)
class SimpleIVFFlat:
    """Simplified IVF-Flat implementation with custom K-Means"""
    
    def __init__(self, n_clusters: int = 100, n_probe: int = 10, seed: int = 42):
        self.n_clusters = n_clusters
        self.n_probe = n_probe
        self.seed = seed
        self.centroids = None
        self.inverted_lists = None
        self.data = None
        self.data_ids = None
        
    def _kmeans_plus_plus(self, vectors: np.ndarray, k: int) -> np.ndarray:
        """K-Means++ initialization"""
        np.random.seed(self.seed)
        n_samples = vectors.shape[0]
        
        # First centroid
        centroids = [vectors[np.random.randint(n_samples)]]
        
        for _ in range(1, k):
            # Calculate distances to nearest centroid
            distances = np.array([
                min([np.linalg.norm(vec - cent)**2 for cent in centroids])
                for vec in vectors
            ])
            
            # Choose next centroid with probability proportional to distance^2
            probabilities = distances / distances.sum()
            cumulative_prob = probabilities.cumsum()
            r = np.random.rand()
            
            for i, cp in enumerate(cumulative_prob):
                if r <= cp:
                    centroids.append(vectors[i])
                    break
        
        return np.array(centroids)
    
    def _kmeans(self, vectors: np.ndarray, k: int, max_iter: int = 100) -> Tuple[np.ndarray, List[int]]:
        """Simple K-Means implementation"""
        # Initialize centroids
        centroids = self._kmeans_plus_plus(vectors, k)
        
        for iteration in range(max_iter):
            # Assign points to nearest centroid
            distances = np.linalg.norm(vectors[:, np.newaxis] - centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                cluster_points = vectors[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]  # Keep old centroid if empty
            
            # Check convergence
            if np.allclose(centroids, new_centroids, rtol=1e-4):
                print(f"K-Means converged after {iteration + 1} iterations")
                break
            
            centroids = new_centroids
        
        return centroids, labels.tolist()
    
    def fit(self, vectors: Dict[str, np.ndarray]) -> 'SimpleIVFFlat':
        """Train the index"""
        self.data_ids = list(vectors.keys())
        data_list = [vectors[pid] for pid in self.data_ids]
        self.data = np.array(data_list)
        
        print(f"Training SimpleIVFFlat with {self.n_clusters} clusters...")
        start_time = time.time()
        
        self.centroids, labels = self._kmeans(self.data, self.n_clusters)
        
        # Build inverted lists
        self.inverted_lists = [[] for _ in range(self.n_clusters)]
        for idx, label in enumerate(labels):
            self.inverted_lists[label].append(idx)
        
        print(f"Training completed in {time.time() - start_time:.2f}s")
        return self
    
    def query(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """Query the index"""
        # Find nearest clusters
        distances_to_centroids = np.linalg.norm(self.centroids - query_vector, axis=1)
        nearest_clusters = np.argsort(distances_to_centroids)[:self.n_probe]
        
        # Collect candidates
        candidate_indices = []
        for cluster_idx in nearest_clusters:
            candidate_indices.extend(self.inverted_lists[cluster_idx])
        
        if not candidate_indices:
            return []
        
        # Compute distances
        candidates = self.data[candidate_indices]
        distances = np.linalg.norm(candidates - query_vector, axis=1)
        
        # Get top-k
        top_indices = np.argsort(distances)[:k]
        
        results = []
        for idx in top_indices:
            data_idx = candidate_indices[idx]
            protein_id = self.data_ids[data_idx]
            distance = distances[idx]
            results.append((protein_id, float(distance)))
        
        return results


# Usage example for your protein_search.py
def create_ivfflat_index(vectors_dict: Dict[str, np.ndarray], 
                        n_clusters: int = 100, 
                        n_probe: int = 10,
                        use_simple: bool = False) -> IVFFlat:
    """
    Helper function to create and train an IVF-Flat index.
    
    Args:
        vectors_dict: Dictionary of protein embeddings
        n_clusters: Number of clusters
        n_probe: Number of clusters to probe during search
        use_simple: Use simple implementation (if sklearn not available)
        
    Returns:
        Trained IVFFlat index
    """
    if use_simple:
        ivf = SimpleIVFFlat(n_clusters=n_clusters, n_probe=n_probe)
    else:
        config = IVFFlatConfig(n_clusters=n_clusters, n_probe=n_probe)
        ivf = IVFFlat(config)
    
    return ivf.fit(vectors_dict)


# Integration with your existing code
if __name__ == "__main__":
    # Example usage
    import json
    
    # Create some dummy data
    n_proteins = 1000
    dim = 320
    
    dummy_vectors = {
        f"prot_{i}": np.random.randn(dim)
        for i in range(n_proteins)
    }
    
    # Create and train index
    print("Creating IVF-Flat index...")
    ivf_index = create_ivfflat_index(dummy_vectors, n_clusters=50, n_probe=5)
    
    # Test query
    test_query = np.random.randn(dim)
    results = ivf_index.query(test_query, k=5)
    
    print("\nTop 5 results:")
    for i, (pid, dist) in enumerate(results):
        print(f"{i+1}. {pid}: {dist:.4f}")
    
    # Get statistics
    stats = ivf_index.get_stats() if hasattr(ivf_index, 'get_stats') else {}
    print("\nIndex statistics:", json.dumps(stats, indent=2, default=str))
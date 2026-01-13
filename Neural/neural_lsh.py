# # Neural/neural_lsh.py
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import os
# import pickle
# import time
# from typing import Dict, List, Tuple
# from sklearn.neighbors import NearestNeighbors
# from kahip import kaffpa

# class NeuralLSH:
#     def __init__(self, vectors: Dict[str, np.ndarray], 
#                  index_dir: str = "neural_index", 
#                  rebuild: bool = False,
#                  k: int = 10,
#                  m: int = 100,
#                  T: int = 5,
#                  epochs: int = 10,
#                  seed: int = 42):
#         """
#         Neural LSH για πρωτεϊνικά embeddings.
        
#         Args:
#             vectors: Λεξικό {protein_id: embedding_vector}
#             index_dir: Φάκελος για αποθήκευση index
#             rebuild: Αν True, ξαναχτίζει το index
#             k: Αριθμός γειτόνων για k-NN graph
#             m: Αριθμός partitions
#             T: Αριθμός partitions προς έλεγχο
#             epochs: Epochs για εκπαίδευση MLP
#             seed: Seed για αναπαραγωγή
#         """
#         self.vectors = vectors
#         self.ids = list(vectors.keys())
#         self.data = np.array(list(vectors.values()))
#         self.n, self.d = self.data.shape
#         self.index_dir = index_dir
#         self.k = k
#         self.m = m
#         self.T = T
#         self.seed = seed
#         self.epochs = epochs
        
#         os.makedirs(index_dir, exist_ok=True)
        
#         # Διαδρομές αρχείων
#         self.model_path = os.path.join(index_dir, "model.pth")
#         self.inv_path = os.path.join(index_dir, "inverted_file.npy")
#         self.partition_path = os.path.join(index_dir, "partition.npy")
#         self.ids_path = os.path.join(index_dir, "ids.txt")
        
#         if rebuild or not os.path.exists(self.model_path):
#             print("Building Neural LSH index...")
#             self._build_index()
#         else:
#             print("Loading existing Neural LSH index...")
#             self._load_index()
    
#     # def _build_index(self):
#     #     """Χτίζει το Neural LSH index (k-NN + KaHIP + MLP)"""
#     #     # 1. Κατασκευή k-NN graph
#     #     print("  Building k-NN graph...")
#     #     knn_model = NearestNeighbors(n_neighbors=self.k+1, metric='euclidean')
#     #     knn_model.fit(self.data)
#     #     distances, indices = knn_model.kneighbors(self.data)
        
#     #     # 2. Δημιουργία adjacency list (αγνοούμε τον εαυτό)
#     #     adj = {i: [] for i in range(self.n)}
#     #     for i in range(self.n):
#     #         for j in indices[i, 1:]:  # Αγνοούμε τον εαυτό
#     #             adj[i].append(j)
        
#     #     # 3. Μετατροπή σε weighted undirected
#     #     print("  Making graph undirected...")
#     #     from graph_tools.symmetric import make_weighted_undirected
#     #     adj_weighted = make_weighted_undirected(adj)
        
#     #     # 4. Μετατροπή σε CSR για KaHIP
#     #     print("  Converting to CSR format...")
#     #     from graph_tools.csr import to_csr
#     #     vwgt, xadj, adjncy, adjcwgt = to_csr(adj_weighted, self.n)
        
#     #     # 5. Partitioning με KaHIP
#     #     print(f"  Running KaHIP partitioning into {self.m} parts...")
#     #     try:
#     #         edgecut, partition = kaffpa(
#     #             vwgt, xadj, adjncy, adjcwgt,
#     #             self.m,           # αριθμός partitions
#     #             0.03,            # imbalance
#     #             True,            # suppress output
#     #             self.seed,       # seed
#     #             2                # mode (STRONG)
#     #         )
#     #         self.partition = np.array(partition, dtype=np.int32)
#     #         print(f"  KaHIP completed. Edgecut: {edgecut}")
#     #     except Exception as e:
#     #         print(f"  KaHIP failed: {e}. Using random partitioning.")
#     #         np.random.seed(self.seed)
#     #         self.partition = np.random.randint(0, self.m, size=self.n)
        
#     #     # 6. Εκπαίδευση MLP
#     #     print("  Training MLP...")
#     #     self.model = self._train_mlp()
        
#     #     # 7. Δημιουργία inverted file
#     #     print("  Building inverted file...")
#     #     self.inverted_file = {}
#     #     for idx, part in enumerate(self.partition):
#     #         if part not in self.inverted_file:
#     #             self.inverted_file[part] = []
#     #         self.inverted_file[part].append(idx)
        
#     #     # 8. Αποθήκευση
#     #     self._save_index()

#     # Στο _build_index() μέθοδο του neural_lsh.py
# def _build_index(self):
#     """Χτίζει το Neural LSH index (k-NN + KaHIP + MLP)"""
#     # 1. Κατασκευή k-NN graph
#     print("  Building k-NN graph...")
#     from sklearn.neighbors import NearestNeighbors
#     knn_model = NearestNeighbors(n_neighbors=self.k+1, metric='euclidean')
#     knn_model.fit(self.data)
#     distances, indices = knn_model.kneighbors(self.data)
    
#     # 2. Δημιουργία adjacency list (αγνοούμε τον εαυτό)
#     adj = {i: [] for i in range(self.n)}
#     for i in range(self.n):
#         for j in indices[i, 1:]:  # Αγνοούμε τον εαυτό
#             adj[i].append(j)
    
#     # 3. Μετατροπή σε weighted undirected (ΜΕ graph_tools)
#     print("  Making graph undirected...")
#     from Neural.graph_tools.symmetric import make_weighted_undirected
#     from Neural.graph_tools.check import check_graph_consistency
#     from Neural.graph_tools.csr import to_csr
    
#     adj_weighted = make_weighted_undirected(adj)
#     check_graph_consistency(adj_weighted)
    
#     # 4. Μετατροπή σε CSR για KaHIP
#     print("  Converting to CSR format...")
#     vwgt, xadj, adjncy, adjcwgt = to_csr(adj_weighted, self.n)
    
#     # 5. Partitioning με KaHIP (αν είναι διαθέσιμο)
#     print(f"  Partitioning into {self.m} parts...")
#     try:
#         from kahip import kaffpa
#         edgecut, partition = kaffpa(
#             vwgt, xadj, adjncy, adjcwgt,
#             self.m,           # αριθμός partitions
#             0.03,            # imbalance
#             True,            # suppress output
#             self.seed,       # seed
#             2                # mode (STRONG)
#         )
#         self.partition = np.array(partition, dtype=np.int32)
#         print(f"  KaHIP completed. Edgecut: {edgecut}")
#     except ImportError:
#         print("  KaHIP not available, using K-means clustering...")
#         from sklearn.cluster import KMeans
#         kmeans = KMeans(n_clusters=self.m, 
#                        random_state=self.seed, 
#                        n_init=10,
#                        max_iter=100)
#         self.partition = kmeans.fit_predict(self.data)
    
    
#     def _train_mlp(self):
#         """Εκπαίδευση MLP για πρόβλεψη partitions"""
#         class MLP(nn.Module):
#             def __init__(self, input_dim, hidden_dim, output_dim):
#                 super(MLP, self).__init__()
#                 self.fc1 = nn.Linear(input_dim, hidden_dim)
#                 self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#                 self.fc3 = nn.Linear(hidden_dim, output_dim)
#                 self.dropout = nn.Dropout(0.2)
            
#             def forward(self, x):
#                 x = F.relu(self.fc1(x))
#                 x = self.dropout(x)
#                 x = F.relu(self.fc2(x))
#                 x = self.fc3(x)
#                 return x
        
#         # Δημιουργία μοντέλου
#         model = MLP(self.d, 256, self.m)
        
#         # Προετοιμασία δεδομένων
#         X = torch.FloatTensor(self.data)
#         y = torch.LongTensor(self.partition)
        
#         # Dataset και DataLoader
#         dataset = torch.utils.data.TensorDataset(X, y)
#         train_size = int(0.8 * len(dataset))
#         val_size = len(dataset) - train_size
#         train_dataset, val_dataset = torch.utils.data.random_split(
#             dataset, [train_size, val_size]
#         )
        
#         train_loader = torch.utils.data.DataLoader(
#             train_dataset, batch_size=128, shuffle=True
#         )
#         val_loader = torch.utils.data.DataLoader(
#             val_dataset, batch_size=128, shuffle=False
#         )
        
#         # Ορισμός loss και optimizer
#         criterion = nn.CrossEntropyLoss()
#         optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
#         # Εκπαίδευση
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model.to(device)
#         model.train()
        
#         for epoch in range(self.epochs):
#             train_loss = 0.0
#             for batch_X, batch_y in train_loader:
#                 batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#                 optimizer.zero_grad()
#                 outputs = model(batch_X)
#                 loss = criterion(outputs, batch_y)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss += loss.item()
            
#             # Validation
#             model.eval()
#             val_loss = 0.0
#             correct = 0
#             total = 0
#             with torch.no_grad():
#                 for batch_X, batch_y in val_loader:
#                     batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#                     outputs = model(batch_X)
#                     loss = criterion(outputs, batch_y)
#                     val_loss += loss.item()
                    
#                     _, predicted = torch.max(outputs, 1)
#                     total += batch_y.size(0)
#                     correct += (predicted == batch_y).sum().item()
            
#             val_acc = 100 * correct / total
#             print(f"    Epoch {epoch+1}/{self.epochs}: "
#                   f"Train Loss: {train_loss/len(train_loader):.4f}, "
#                   f"Val Loss: {val_loss/len(val_loader):.4f}, "
#                   f"Val Acc: {val_acc:.2f}%")
#             model.train()
        
#         model.eval()
#         return model
    
#     def _save_index(self):
#         """Αποθηκεύει το index"""
#         # Αποθήκευση MLP
#         torch.save({
#             'model_state_dict': self.model.state_dict(),
#             'input_dim': self.d,
#             'output_dim': self.m,
#             'hidden_dim': 256,
#             'num_layers': 3
#         }, self.model_path)
        
#         # Αποθήκευση inverted file
#         np.save(self.inv_path, self.inverted_file)
        
#         # Αποθήκευση partition
#         np.save(self.partition_path, self.partition)
        
#         # Αποθήκευση IDs
#         with open(self.ids_path, 'w') as f:
#             for id in self.ids:
#                 f.write(id + "\n")
        
#         print(f"✓ Index saved to {self.index_dir}")
    
#     def _load_index(self):
#         """Φορτώνει αποθηκευμένο index"""
#         # Φόρτωση IDs
#         with open(self.ids_path, 'r') as f:
#             self.ids = [line.strip() for line in f]
        
#         # Φόρτωση MLP
#         checkpoint = torch.load(self.model_path, map_location='cpu')
        
#         class MLP(nn.Module):
#             def __init__(self, input_dim, hidden_dim, output_dim):
#                 super(MLP, self).__init__()
#                 self.fc1 = nn.Linear(input_dim, hidden_dim)
#                 self.fc2 = nn.Linear(hidden_dim, hidden_dim)
#                 self.fc3 = nn.Linear(hidden_dim, output_dim)
#                 self.dropout = nn.Dropout(0.2)
            
#             def forward(self, x):
#                 x = F.relu(self.fc1(x))
#                 x = self.dropout(x)
#                 x = F.relu(self.fc2(x))
#                 x = self.fc3(x)
#                 return x
        
#         self.model = MLP(checkpoint['input_dim'], 256, checkpoint['output_dim'])
#         self.model.load_state_dict(checkpoint['model_state_dict'])
#         self.model.eval()
        
#         # Φόρτωση inverted file
#         self.inverted_file = np.load(self.inv_path, allow_pickle=True).item()
        
#         # Φόρτωση partition
#         self.partition = np.load(self.partition_path)
        
#         print(f"✓ Index loaded from {self.index_dir}")
    
#     def query(self, query_vector: np.ndarray, N: int) -> List[Tuple[str, float]]:
#         """
#         Αναζήτηση k πλησιέστερων γειτόνων.
        
#         Args:
#             query_vector: Το embedding του query
#             N: Αριθμός γειτόνων προς επιστροφή
        
#         Returns:
#             Λίστα με (protein_id, απόσταση) για τους top-N γείτονες
#         """
#         # 1. Πρόβλεψη partitions με MLP
#         query_tensor = torch.FloatTensor(query_vector).unsqueeze(0)
        
#         with torch.no_grad():
#             outputs = self.model(query_tensor)
#             probs = F.softmax(outputs, dim=1)
#             top_probs, top_partitions = torch.topk(probs, self.T)
        
#         top_partitions = top_partitions.numpy().flatten()
        
#         # 2. Συλλογή υποψηφίων
#         candidates = set()
#         for part in top_partitions:
#             if part in self.inverted_file:
#                 candidates.update(self.inverted_file[part])
        
#         if not candidates:
#             return []
        
#         # 3. Υπολογισμός αποστάσεων
#         results = []
#         for idx in candidates:
#             dist = np.linalg.norm(query_vector - self.data[idx])
#             results.append((self.ids[idx], dist))
        
#         # 4. Ταξινόμηση και επιστροφή top-N
#         results.sort(key=lambda x: x[1])
#         return results[:N]



# Neural/neural_lsh.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import time
from typing import Dict, List, Tuple
from sklearn.neighbors import NearestNeighbors

class NeuralLSH:
    def __init__(self, vectors: Dict[str, np.ndarray], 
                 index_dir: str = "neural_index", 
                 rebuild: bool = False,
                 k: int = 10,
                 m: int = 100,
                 T: int = 5,
                 epochs: int = 10,
                 seed: int = 42):
        """
        Neural LSH για πρωτεϊνικά embeddings.
        
        Args:
            vectors: Λεξικό {protein_id: embedding_vector}
            index_dir: Φάκελος για αποθήκευση index
            rebuild: Αν True, ξαναχτίζει το index
            k: Αριθμός γειτόνων για k-NN graph
            m: Αριθμός partitions
            T: Αριθμός partitions προς έλεγχο
            epochs: Epochs για εκπαίδευση MLP
            seed: Seed για αναπαραγωγή
        """
        self.vectors = vectors
        self.ids = list(vectors.keys())
        self.data = np.array(list(vectors.values()))
        self.n, self.d = self.data.shape
        self.index_dir = index_dir
        self.k = k
        self.m = m
        self.T = T
        self.seed = seed
        self.epochs = epochs
        
        os.makedirs(index_dir, exist_ok=True)
        
        # Διαδρομές αρχείων
        self.model_path = os.path.join(index_dir, "model.pth")
        self.inv_path = os.path.join(index_dir, "inverted_file.pkl")
        self.partition_path = os.path.join(index_dir, "partition.npy")
        self.ids_path = os.path.join(index_dir, "ids.txt")
        
        if rebuild or not os.path.exists(self.model_path):
            print("Building Neural LSH index...")
            self._build_index()
        else:
            print("Loading existing Neural LSH index...")
            self._load_index()
    
    def _build_index(self):
        """Χτίζει το Neural LSH index (k-NN + partitioning + MLP)"""
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 1. Κατασκευή k-NN graph
        print("  Building k-NN graph...")
        knn_model = NearestNeighbors(n_neighbors=min(self.k+1, self.n), 
                                     metric='euclidean')
        knn_model.fit(self.data)
        distances, indices = knn_model.kneighbors(self.data)
        
        # 2. Δημιουργία adjacency list (αγνοούμε τον εαυτό)
        print("  Creating adjacency matrix...")
        adj = {i: [] for i in range(self.n)}
        for i in range(self.n):
            for j in indices[i, 1:]:  # Αγνοούμε τον εαυτό
                adj[i].append(j)
        
        # 3. Partitioning με K-means (απλοποιημένο)
        print(f"  Partitioning into {self.m} clusters...")
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(self.m, self.n),  # Μην ζητήσεις περισσότερα clusters από δείγματα
                       random_state=self.seed, 
                       n_init=10,
                       max_iter=100)
        self.partition = kmeans.fit_predict(self.data)
        
        print(f"  Partitioning complete. Shape: {self.partition.shape}")
        
        # 4. Εκπαίδευση MLP
        print("  Training MLP...")
        self.model = self._train_mlp()
        
        # 5. Δημιουργία inverted file
        print("  Building inverted file...")
        self.inverted_file = {}
        for idx, part in enumerate(self.partition):
            if part not in self.inverted_file:
                self.inverted_file[part] = []
            self.inverted_file[part].append(idx)
        
        # 6. Αποθήκευση
        self._save_index()
    
    def _train_mlp(self):
        """Εκπαίδευση MLP για πρόβλεψη partitions"""
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        # Δημιουργία μοντέλου
        hidden_dim = min(256, self.d * 2)
        output_dim = len(np.unique(self.partition))
        model = MLP(self.d, hidden_dim, output_dim)
        
        # Προετοιμασία δεδομένων
        X = torch.FloatTensor(self.data)
        y = torch.LongTensor(self.partition)
        
        # Dataset και DataLoader
        dataset = torch.utils.data.TensorDataset(X, y)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=128, shuffle=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=128, shuffle=False
        )
        
        # Ορισμός loss και optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Εκπαίδευση
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        best_val_acc = 0
        for epoch in range(self.epochs):
            # Training
            model.train()
            train_loss = 0.0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            train_loss = train_loss / len(train_loader)
            val_loss = val_loss / len(val_loader)
            val_acc = 100 * correct / total
            
            print(f"    Epoch {epoch+1}/{self.epochs}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        model.eval()
        print(f"  Best validation accuracy: {best_val_acc:.2f}%")
        return model
    
    def _save_index(self):
        """Αποθηκεύει το index"""
        # Αποθήκευση MLP
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_dim': self.d,
            'output_dim': len(np.unique(self.partition)),
            'hidden_dim': min(256, self.d * 2),
        }, self.model_path)
        
        # Αποθήκευση inverted file
        with open(self.inv_path, 'wb') as f:
            pickle.dump(self.inverted_file, f)
        
        # Αποθήκευση partition
        np.save(self.partition_path, self.partition)
        
        # Αποθήκευση IDs
        with open(self.ids_path, 'w') as f:
            for id in self.ids:
                f.write(id + "\n")
        
        print(f"✓ Index saved to {self.index_dir}")
    
    def _load_index(self):
        """Φορτώνει αποθηκευμένο index"""
        # Φόρτωση IDs
        with open(self.ids_path, 'r') as f:
            self.ids = [line.strip() for line in f]
        
        # Φόρτωση MLP
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(input_dim, hidden_dim)
                self.fc2 = nn.Linear(hidden_dim, hidden_dim)
                self.fc3 = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(0.2)
            
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = self.dropout(x)
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        hidden_dim = checkpoint.get('hidden_dim', min(256, checkpoint['input_dim'] * 2))
        self.model = MLP(checkpoint['input_dim'], 
                        hidden_dim, 
                        checkpoint['output_dim'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Φόρτωση inverted file
        with open(self.inv_path, 'rb') as f:
            self.inverted_file = pickle.load(f)
        
        # Φόρτωση partition
        self.partition = np.load(self.partition_path)
        
        print(f"✓ Index loaded from {self.index_dir}")
    
    def query(self, query_vector: np.ndarray, N: int) -> List[Tuple[str, float]]:
        """
        Αναζήτηση k πλησιέστερων γειτόνων.
        
        Args:
            query_vector: Το embedding του query
            N: Αριθμός γειτόνων προς επιστροφή
        
        Returns:
            Λίστα με (protein_id, απόσταση) για τους top-N γείτονες
        """
        # 1. Πρόβλεψη partitions με MLP
        query_tensor = torch.FloatTensor(query_vector).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(query_tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_partitions = torch.topk(probs, min(self.T, probs.shape[1]))
        
        top_partitions = top_partitions.numpy().flatten()
        
        # 2. Συλλογή υποψηφίων
        candidates = set()
        for part in top_partitions:
            if part in self.inverted_file:
                candidates.update(self.inverted_file[part])
        
        if not candidates:
            return []
        
        # 3. Υπολογισμός αποστάσεων
        results = []
        candidate_list = list(candidates)
        candidate_vectors = self.data[candidate_list]
        
        # Διανυσματικός υπολογισμός αποστάσεων
        distances = np.linalg.norm(candidate_vectors - query_vector, axis=1)
        
        # 4. Ταξινόμηση και επιστροφή top-N
        sorted_indices = np.argsort(distances)[:N]
        
        for idx in sorted_indices:
            candidate_idx = candidate_list[idx]
            results.append((self.ids[candidate_idx], distances[idx]))
        
        return results
# Neural/neural_lsh.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
from sklearn.cluster import KMeans

class NeuralLSH:
    def __init__(self, vectors, index_dir="neural_index", rebuild=False, 
                 m=100, T=5, neural_epochs=10, seed=42):
        self.vectors = vectors
        self.ids = list(vectors.keys())
        self.data = np.array(list(vectors.values()))
        self.index_dir = index_dir
        self.m = m  # αριθμος partitions
        self.T = T  # αριθμος partitions προς έλεγχο
        self.seed = seed
        
        os.makedirs(index_dir, exist_ok=True)
        
        if rebuild or not os.path.exists(os.path.join(index_dir, "model.pth")):
            print("Building Neural LSH index...")
            self._build_index()
        else:
            print("Loading existing Neural LSH index...")
            self._load_index()
    
    # Δημιουργια index με k-means clustering και MLP
    def _build_index(self):
        
        # 1. K-means clustering
        print("Running k-means clustering...")
        kmeans = KMeans(n_clusters=self.m, random_state=self.seed, n_init=10)
        self.partition = kmeans.fit_predict(self.data)
        
        # 2. Εκπαιδευση MLP
        print("Training MLP...")
        self.model = self._train_mlp(self.partition)
        
        # 3. Inverted file
        print("Building inverted file...")
        self.inverted_file = {}
        for idx, part in enumerate(self.partition):
            if part not in self.inverted_file:
                self.inverted_file[part] = []
            self.inverted_file[part].append(idx)
        
        # 4. Αποθηκευση
        self._save_index()
    
    def _train_mlp(self, labels):
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
        
        model = MLP(self.data.shape[1], 256, self.m)
        X = torch.FloatTensor(self.data)
        y = torch.LongTensor(labels)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(10):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
        return model
    
    # Αναζήτηση k πλησιέστερων γειτόνων
    def query(self, query_vector, N):
       
        # Προβλεψη partitions
        query_tensor = torch.FloatTensor(query_vector).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.model(query_tensor)
            probs = F.softmax(outputs, dim=1)
            top_probs, top_partitions = torch.topk(probs, self.T)
        
        top_partitions = top_partitions.numpy().flatten()
        
        # Συλλογη υποψηφιων
        candidates = []
        for part in top_partitions:
            if part in self.inverted_file:
                candidates.extend(self.inverted_file[part])
        
        # Υπολογισμος αποστασεων
        results = []
        for idx in set(candidates):
            dist = np.linalg.norm(query_vector - self.data[idx])
            results.append((self.ids[idx], dist))
        
        # Ταξινομηση και επιστροφη top-N
        results.sort(key=lambda x: x[1])
        return results[:N]
    

# class NeuralLSH:
    # Αρχικοποιηση Neural LSH.
    # def __init__(self, vectors, index_dir="neural_index", rebuild=False, 
    #              k=10, m=100, neural_epochs=10, batch_size=128, seed=42):
      
    #     self.vectors = vectors
    #     self.ids = list(vectors.keys())
    #     self.data = np.array(list(vectors.values()))
    #     self.index_dir = index_dir
    #     self.k = k  # k για kNN graph
    #     self.m = m  # αριθμος partitions
    #     self.seed = seed
        
    #     os.makedirs(index_dir, exist_ok=True)
        
    #     # Ελεγχος αν υπαρχει ηδη index
    #     model_path = os.path.join(index_dir, "model.pth")
    #     inv_file_path = os.path.join(index_dir, "inverted_file.npy")
    #     partition_path = os.path.join(index_dir, "partition.npy")
        
    #     if rebuild or not os.path.exists(model_path) or not os.path.exists(inv_file_path):
    #         print("Building Neural LSH index...")
    #         self._build_index()
    #     else:
    #         print("Loading existing Neural LSH index...")
    #         self._load_index()
    # # Δημιουργει index (kNN graph + KaHIP + MLP)
    # def _build_index(self):
       
    #     # 1. Χτιζουμε kNN graph (χρησιμοποιωντας IVF-Flat )
    #     print("Building kNN graph...")
    #     from IVFFlat.ivfflat import IVFFlat
    #     ivf = IVFFlat(self.vectors, nlist=100, nprobe=10)
    #     ivf.build_index()
        
    #     # 2. Δημιουργουμε adjacency matrix απο kNN
    #     print("Creating adjacency matrix...")
    #     n = len(self.data)
    #     adj = {i: [] for i in range(n)}
        
    #     for i in range(n):
    #         neighbors = ivf.query(self.data[i], k=self.k+1)  # +1 για να αγνοήσουμε τον εαυτο
    #         for neighbor_id, _ in neighbors[1:]:  # Παραβλεψη πρωτου (ιδιος)
    #             idx = self.ids.index(neighbor_id)
    #             adj[i].append(idx)
    #             adj[idx].append(i)
        
    #     # 3. Τρεχουμε KaHIP για partitioning
    #     print("Running KaHIP partitioning...")
    #     try:
    #         from kahip import kaffpa
    #         # Μετατροπη γραφου σε μορφή KaHIP
    #         edgecut, partition = kaffpa(...)
    #         self.partition = np.array(partition, dtype=np.int32)
    #     except ImportError:
    #         print("KaHIP not available, using random partitioning")
    #         np.random.seed(self.seed)
    #         self.partition = np.random.randint(0, self.m, size=n)
        
    #     # 4. Εκπαιδευτουμε MLP
    #     print("Training MLP...")
    #     self.model = self._train_mlp(self.partition)
        
    #     # 5. Δημιουργουμε inverted file
    #     print("Building inverted file...")
    #     self.inverted_file = {}
    #     for idx, part in enumerate(self.partition):
    #         if part not in self.inverted_file:
    #             self.inverted_file[part] = []
    #         self.inverted_file[part].append(idx)
        
    #     # 6. Αποθηκευση
    #     self._save_index()
    
    # def _train_mlp(self, labels):
    #     """Εκπαίδευση MLP για πρόβλεψη partitions"""
    #     class MLP(nn.Module):
    #         def __init__(self, input_dim, hidden_dim, output_dim):
    #             super(MLP, self).__init__()
    #             self.fc1 = nn.Linear(input_dim, hidden_dim)
    #             self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    #             self.fc3 = nn.Linear(hidden_dim, output_dim)
    #             self.dropout = nn.Dropout(0.2)
            
    #         def forward(self, x):
    #             x = F.relu(self.fc1(x))
    #             x = self.dropout(x)
    #             x = F.relu(self.fc2(x))
    #             x = self.fc3(x)
    #             return x
        
    #     model = MLP(self.data.shape[1], 256, self.m)
        
    #     # Μετατροπη δεδομενων
    #     X = torch.FloatTensor(self.data)
    #     y = torch.LongTensor(labels)
        
    #     dataset = torch.utils.data.TensorDataset(X, y)
    #     train_loader = torch.utils.data.DataLoader(
    #         dataset, batch_size=128, shuffle=True
    #     )
        
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
    #     model.train()
    #     for epoch in range(10):
    #         total_loss = 0
    #         for batch_X, batch_y in train_loader:
    #             optimizer.zero_grad()
    #             outputs = model(batch_X)
    #             loss = criterion(outputs, batch_y)
    #             loss.backward()
    #             optimizer.step()
    #             total_loss += loss.item()
    #         print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
        
    #     return model
    
    # Αποθηκευση index
    def _save_index(self):
    
        torch.save(self.model.state_dict(), 
                  os.path.join(self.index_dir, "model.pth"))
        
        with open(os.path.join(self.index_dir, "inverted_file.pkl"), 'wb') as f:
            pickle.dump(self.inverted_file, f)
        
        np.save(os.path.join(self.index_dir, "partition.npy"), self.partition)
        
        with open(os.path.join(self.index_dir, "ids.txt"), 'w') as f:
            for id in self.ids:
                f.write(id + "\n")

    # Φορτωση index
    def _load_index(self):
        
        # Φορτωση IDs
        with open(os.path.join(self.index_dir, "ids.txt"), 'r') as f:
            self.ids = [line.strip() for line in f]
        
        # Φορτωση MLP
        class MLP(nn.Module):
            pass
        
        self.model = MLP(self.data.shape[1], 256, self.m)
        self.model.load_state_dict(
            torch.load(os.path.join(self.index_dir, "model.pth"))
        )
        self.model.eval()
        
        # Φορτωση inverted file
        with open(os.path.join(self.index_dir, "inverted_file.pkl"), 'rb') as f:
            self.inverted_file = pickle.load(f)
        
        self.partition = np.load(
            os.path.join(self.index_dir, "partition.npy")
        )

    # Αναζήτηση με Neural LSH.
    # def query(self, query_vector, N, T=5):
    #     # Προβλεψη partitions με MLP
    #     query_tensor = torch.FloatTensor(query_vector).unsqueeze(0)
        
    #     with torch.no_grad():
    #         outputs = self.model(query_tensor)
    #         probs = F.softmax(outputs, dim=1)
    #         top_probs, top_partitions = torch.topk(probs, T)
        
    #     top_partitions = top_partitions.numpy().flatten()
        
    #     # Συλλογη υποψηφιων
    #     candidates = []
    #     for part in top_partitions:
    #         if part in self.inverted_file:
    #             candidates.extend(self.inverted_file[part])
        
    #     # Υπολογισμος αποστασεων
    #     results = []
    #     for idx in set(candidates):  # Αφαίρεση διπλοτυπων
    #         dist = np.linalg.norm(query_vector - self.data[idx])
    #         results.append((self.ids[idx], dist))
        
    #     # Ταξινομηση και επιστροφη top-N
    #     results.sort(key=lambda x: x[1])
    #     return results[:N]
# import matplotlib.pyplot as plt
# import networkx as nx

# def visualize_graph(adj, max_nodes=70000):
#     """Οπτικοποίηση γράφου (προαιρετικό)"""
#     if len(adj) > max_nodes:
#         print(f">> Graph too large for visualization (> {max_nodes} nodes)")
#         return
        
#     G = nx.Graph()
#     for u, neighbors in adj.items():
#         for v, w in neighbors:
#             G.add_edge(u, v, weight=w)
#     plt.figure(figsize=(10, 10))
#     pos = nx.spring_layout(G)
#     nx.draw(G, pos, with_labels=True, node_size=300, node_color="skyblue")
#     edge_labels = nx.get_edge_attributes(G, 'weight')
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
#     plt.savefig('graph_visualization.png')
#     print(">> Graph visualization saved as 'graph_visualization.png'")
#     plt.close()


import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from sklearn.manifold import TSNE
import warnings

def visualize_graph(adj, labels=None, max_nodes=70000, sample=True, save_path=None):
    """
    Οπτικοποίηση υποσυνόλου του γράφου με έγχρωμη κωδικοποίηση βάσει partition.
    
    Args:
        adj: adjacency dictionary {node: [(neighbor, weight), ...]}
        labels: array με partition labels (προαιρετικό)
        max_nodes: μέγιστος αριθμός κόμβων για οπτικοποίηση
        sample: αν True, δείγματειάζει τυχαία κόμβους
        save_path: διαδρομή για αποθήκευση εικόνας
    """
    n_nodes = len(adj)
    
    if n_nodes > max_nodes:
        if sample:
            print(f">> Graph too large ({n_nodes} nodes). Sampling {max_nodes} random nodes...")
            # Δειγματοληψία τυχαίων κόμβων
            all_nodes = list(adj.keys())
            sampled_nodes = np.random.choice(all_nodes, size=min(max_nodes, n_nodes), replace=False)
            
            # Δημιουργία υπογράφου
            G = nx.Graph()
            for u in sampled_nodes:
                if u in adj:
                    for v, w in adj[u]:
                        if v in sampled_nodes:
                            G.add_edge(u, v, weight=w)
        else:
            print(f">> Graph too large for visualization ({n_nodes} > {max_nodes} nodes)")
            return
    else:
        # Χρήση ολόκληρου του γράφου
        G = nx.Graph()
        for u, neighbors in adj.items():
            for v, w in neighbors:
                G.add_edge(u, v, weight=w)
    
    # Χρωματική κωδικοποίηση βάσει labels (αν δίνονται)
    if labels is not None:
        # Κρατάμε labels μόνο για τους κόμβους του υπογράφου
        node_colors = []
        valid_nodes = []
        for node in G.nodes():
            if node < len(labels):
                node_colors.append(labels[node])
                valid_nodes.append(node)
        # Χρήση colormap
        cmap = plt.cm.tab20  # ή tab10, Set3, κλπ.
    else:
        node_colors = 'skyblue'
        cmap = None
    
    print(f">> Visualizing graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Εναλλακτική τοποθέτηση κόμβων (γρηγορότερη από spring_layout)
    plt.figure(figsize=(12, 10))
    
    if G.number_of_nodes() < 500:
        # Χρήση spring_layout μόνο για μικρούς γράφους
        pos = nx.spring_layout(G, seed=42, k=0.3)
    else:
        # Χρήση spectral_layout ή random_layout για μεγαλύτερους
        pos = nx.spectral_layout(G) if G.number_of_nodes() < 2000 else nx.random_layout(G)
    
    # Σχεδίαση
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=30 if G.number_of_nodes() > 100 else 80,
        node_color=node_colors if labels is None else [cmap(c % 20) for c in node_colors],
        alpha=0.8,
        cmap=cmap
    )
    
    # Σχεδίαση ακμών (μόνο αν δεν είναι πολλές)
    if G.number_of_edges() < 1000:
        nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5)
    
    plt.title(f"k-NN Graph Visualization (n={G.number_of_nodes()}, e={G.number_of_edges()})")
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f">> Graph visualization saved as '{save_path}'")
    else:
        plt.savefig('graph_visualization.png', dpi=150, bbox_inches='tight')
        print(">> Graph visualization saved as 'graph_visualization.png'")
    
    plt.close()
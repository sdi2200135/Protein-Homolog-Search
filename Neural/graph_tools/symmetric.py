# Βήμα: Προετοιμασία Γράφου

#συνάρτηση που μετατρέπει έναν directed knn γράφο σε συμμετρικο undirected γράφο με βάρη 1 ή 2 
def make_weighted_undirected(graph):
    cleaned_graph = {}  #δημιουργείται νέο κενό λεξικό
    for u in graph: #για κάθε κόμβο του αρχικού γράφου 
        #αφαίρεση self-loops και διπλοεγγραφών, διατήρηση μόνο μοναδικών γειτόνων που να μην είναι ο ίδιος ο κόμβος 
        unique_neighbors = list(set([nb for nb in graph[u] if nb != u]))
        cleaned_graph[u] = unique_neighbors     #αποθήκευση των "καθαρων" γειτόνων
    
    edge_weights = {}                   #λεξικό που μετράει πόσες φορές υπάρχει η κάθε ακμή 
    for u in cleaned_graph:             #για κάθε κόμβο στο νέο γράφο 
        for v in cleaned_graph[u]:      #για κάθε γείτονα του 
            edge = tuple(sorted((u, v)))    #κάνει ταξινόμηση έτσι ώστε ακμές όπως (3,7) και (7,3) να θεωρούνται ίδια ακμή
            edge_weights[edge] = edge_weights.get(edge, 0) + 1  #αύξηση μετρητή εμφ΄άνισης ακμής 
    
    #δημιουργία τελικού συμμετρικού γράφου
    adj = {}    #τελικό λεξικό adjacency list
    for (u, v), count in edge_weights.items():  #για κάθε ακμή και τη συχνότητά της
        weight = 2 if count == 2 else 1 #αν η ακμή υπάρχει και από τις 2 πλευρές τότε έχουμε βάρος 2, αλλιώς βάρος 1
        
        if u not in adj:    #αν ο κόμβος δεν υπάρχει στο adjacency list, τότε τον δημιουργούμε
            adj[u] = []
        if v not in adj:
            adj[v] = []
            
        #προσθήκη και από τις δύο πλευρές με το ίδιο βάρος
        adj[u].append((v, weight))
        adj[v].append((u, weight))
    
    return adj  #επιστρέφει τον τελικό συμμετρικό γράφο με βάρη 
# Βήμα: Προετοιμασία Γράφου

#συνάρτηση που μετατρέπει έναν adjacency list γράφο σε CSR format για KaHIP
def to_csr(adj, n):
    print(">> Converting graph to CSR format...")   #ενημερωτικό μήνυμα 
    xadj = [0]      #csr: δείκτης έναρξης γειτόνων κάθε κόμβου (offsets)
    adjncy = []     #csr: λίστα με γείτονες όλων των κόμβων σε συνεχόμενο πίνακα 
    adjcwgt = []    #csr: βάρη για κάθε ακμή 
    vwgt = [1] * n  #csr: βάρη κορυφών (εδώ όλα 1)

    for i in range(n):                  #για κάθε κόμβο από 0 έως n-1
        neighbors = adj.get(i, [])      #παίρνει τους γείτονες του i, αν δεν υπάρχουν παίρνει άδεια λίστα
        for (v, w) in neighbors:        #για κάθε γείτονα v με βάρος w
            adjncy.append(v)            #προσθέτει το κόμβο γείτονα στη λίστα adjncy 
            adjcwgt.append(w)           #προσθέτει το βάρος της ακμής
        xadj.append(len(adjncy))        #τέλος, προσθέτει στη λίστα των γειτόνων για το κόμβο i τo offset

    return vwgt, xadj, adjncy, adjcwgt  #επιστρέφει τις 4 λίστες που απαιτεί η KaHIP
                                        # vwgt: βάρη κορυφών
                                        # xadj: offsets γειτόνων για κάθε κόμβο
                                        # adjncy: λίστα γειτόνων σε σειρά
                                        # adjcwgt: βάρη ακμών
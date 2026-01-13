# Βήμα: Προετοιμασία Γράφου

#συνάρτηση που ελέγχει ότι ο γράφος είναι συμμετρικός
def check_graph_consistency(adj):
    print(">> Checking graph consistency...")   #ενημερωτικό μήνυμα 
    
    inconsistencies = 0                         #μετρητής για μη συμμετρικές ακμές 
    for u in adj:                               #για κάθε κόμβο u στο adjacency  list
        for (v, w1) in adj[u]:                  #για κάθε γείτονα v και βαρος w1
            found = False                       #flag αν βρεθεί η αντίστροφη ακμή 
            reverse_weight = None               #για να αποθηκεύσει το βάρος την αντίστροφης ακμής 
            if v in adj:                        #αν ο γείτονας υπάρχει στο γράφο 
                for (u2, w2) in adj[v]:         #για κάθε γείτονα u2 του v 
                    if u2 == u:                 #αν βρούμε τον αρχικό κόμβο u
                        found = True            #το flag -> true
                        reverse_weight = w2     #αποθηκεύουμε το βάρος και βγαίνομε από το for loop των γειτόνων
                        break
            
            if not found:                       #αν δε βρέθηκε η αντίστροφη ακμή, εμφανίζουμε μήνυμα σφάλματος
                print(f"ERROR: Edge {u}→{v} exists but {v}→{u} missing!")
                inconsistencies += 1            #αυξάνουμε το μετρητή
            elif w1 != reverse_weight:          #αν υπάρχει αλλά με διαφορετικό βάρος, εμφανίζουμε μήνυμα σφάλματος
                print(f"ERROR: Edge {u}→{v} has weight {w1} but {v}→{u} has weight {reverse_weight}")
                inconsistencies += 1            #αυξάνουμε το μετρητή
    
    if inconsistencies == 0:    #αν δεν υπάρχει κάποια ασυμφωνία, εμφανίζουμε μήνυμα με επιτυχή έλεγχο
        print("✓ Graph is perfectly symmetric!")
    else:                       #αλλιώς εμφανίζουμε το πλήθος των ασυμφωνιών 
        print(f"✗ Found {inconsistencies} inconsistencies")
    
    return inconsistencies == 0 #επιστρέφει true αν ο γράφος είναι συμμετρικός και false αν δεν είναι 
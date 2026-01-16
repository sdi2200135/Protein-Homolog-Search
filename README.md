# Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα - 3η Προγραμματιστική Εργασία - Αναζήτηση Απομακρυσμένων ομόλογων με Προσεγγιστικές Μεθόδους ESM


## Στοιχεία Φοιτητών
**Ομάδα:**
1. Παπαθανασίου Ελένη - 1115202200135
2. Τόντου Αλτάνη-Δάφνη - 1115202200288

## Περιγρφή 
Η παρούσα εργασία υλοποιεί ένα σύστημα για την αναζήτηση απομακρυσμένων ομόλογων πρωτεϊνών χρησιμοποιώντας προσεγγιστικές μεθόδους ANN (Approximate Nearest Neighbor) και διανυσματικές αναπαραστάσεις από το μοντέλο ESM-2.

Οι μέθοδοι που υλοποιήθηκαν και αξιολογήθηκαν συγκριτικά είναι:

    Euclidean LSH - Locality Sensitive Hashing με Ευκλείδεια απόσταση

    Hypercube Projection - Ταχύτατη αναζήτηση σε υπερκύβο

    IVF-Flat - Inverted File με ακριβείς αποστάσεις

    IVF-PQ - Product Quantization για εξοικονόμηση μνήμης

    Neural LSH - Νευρωνικό δίκτυο για βελτιστοποιημένη προβολή

Το project αποτελείται από δύο κύρια προγράμματα:

**protein_embed.py**: Μετατροπή πρωτεϊνικών ακολουθιών σε embeddings με ESM-2
**protein_search.py**: Σύγκριση των 5 μεθόδων ANN και αξιολόγηση με βάση τα αποτελέσματα BLAST

Κύρια βήματα αλγορίθμου:
1. Κατασκευή γράφου k-NN από το σύνολο δεδομένων
2. Μετατροπή σε μη κατευθυνόμενο ζυγισμένο γράφο
3. Ισοκατανεμημένη διαμέριση KaHIP σε m μέρη
4. Εκπαίδευση MLP ταξινομητή για πρόβλεψη partition labels 
5. Αποθήκευση ευρετηρίου (μοντέλο + inverted file)
6. Αναζήτηση με multi-probe τεχνική


## Κατάλογος Αρχείων
## Κύρια Προγράμματα

1. **protein_embed.py** : Μετατροπή FASTA αρχείων σε ESM-2 embeddings
2. **protein_search.py** : Αναζήτηση και αξιολόγηση με 5 μεθόδους ANN

## ANN Μέθοδοι
Euclidean LSH
**euclidean_lsh.py** : LSH με Ευκλείδειες αποστάσεις και multi-probe

Hypercube
**hypercube.py** :Προβολή σε υπερκύβο με Hamming απόσταση

IVF Μέθοδοι
**ivfflat.py** :IVF με ακριβείς αποστάσεις και K-means clustering
**ivfpq.py** :IVF με Product Quantization για συμπίεση

Neural LSH
**neural_lsh.py** :Υβριδική μέθοδος με MLP και partitioning

## Αρχεία Δεδομένων (παραδείγματα)
**swissprot.fasta** : Βάση δεδομένων πρωτεϊνών
**targets.fasta** : Πρωτεΐνες-στόχοι για αναζήτηση
**blast_results.txt** : Αποτελέσματα BLAST για αξιολόγηση


### Βοηθητικά modules -> **knn_graph/**
1. **ANN/euclidean_lsh.py**   :
2. **Hypercube/hypercube.py**    : 
3. **IVFFlat/ivfflat.py**   :
4. **IVFPQ/ivfpq.py**   :
5. **Neural/neural_lsh.py** : 
           **/graph_tools**:



## Οδηγίες Εγκατάστασης

# Δημιουργία και ενεργοποίηση virtual environment
    python3 -m venv venv
    source venv/bin/activate
# Εγκατάσταση βασικών εξαρτήσεων
    pip install torch numpy biopython tqdm scikit-learn pandas
    pip install fair-esm  # ESM-2 μοντέλο
    pip install -r requirements.txt  # Όλες οι εξαρτήσεις
# Εγκατάσταση KaHIP για διαμέριση γράφων
    pip install kahip
# Εγκατάσταση ESM-2 Μοντέλου
    python -c "import esm; print('ESM εγκατεστημένο')"



## Οδηγίες Χρήσης
# Σενάριο 1: Κατασκευή Ευρετηρίου (Build)
python nlsh_build.py -d <input_file> -i <index_path> -type <sift|mnist>

# Σενάριο 2: Αναζήτηση (Search)
python nlsh_search.py -d <dataset> -q <query_file> -i <index_path> -o <output_file> -type <sift|mnist>

## Συμβατότητα με 1η Εργασία
1. Χρησιμοποιούνται τα ίδια datasets (MNIST, SIFT)
2. Η ίδια μορφή αρχείων εισόδου/εξόδου
3. Δυνατότητα άμεσης σύγκρισης με LSH, Hypercube, IVFFlat, IVFPQ


## Οδηγίες Μεταγλώττισης και Εκτέλεσης
Αρχικά, για την εγκατάσταση των απαραίτητων βιβλιοθηκών εκτελούμε την εντολή **make setup**.
Στη συνέχεια, για να εκτελέσουμε πλήρως τον αλγόριθμο Neural LSH για το SIFT dataset χρησιμοποιούμε την εντολή **make run_sift**.
Αν θέλουμε να εκτελέσουμε πλήρως τον αλγόριθμο Neural LSH για το MNIST dataset χρησιμοποιούμε την εντολή **make run_mnist**.
Για να εκτελέσουμε την πειραματική ανάλυση με τις ακριβείς παραμέτρους από την εκφώνηση για το SIFT, χρησιμοποιούμε την εντολή **make run_exact_sift**.
Για να εκτελέσουμε την πειραματική ανάλυση με τις ακριβείς παραμέτρους από την εκφώνηση για το MNIST, χρησιμοποιούμε την εντολή **make run_exact_mnist**.
Για να εκτελέσουμε μόνο τη φάση της κατασκευής του ευρετηρίου για το SIFT χρησιμοποιούμε την εντολή **make build_sift**.
Για να εκτελέσουμε μόνο τη φάση της αναζήτησης για το SIFT χρησιμοποιούμε την εντολή **make search_sift**.
Για να εκτελέσουμε μόνο τη φάση της κατασκευής του ευρετηρίου για το MNIST χρησιμοποιούμε την εντολή **make build_mnist**.
Για να εκτελέσουμε μόνο τη φάση της αναζήτησης για το MNIST χρησιμοποιούμε την εντολή **make search_mnist**.
Για να διαγράψουμε τα παραγόμενα αρχεία χρησιμοποιούμε την εντολή **make clean_all**.
Για να εμφανίσουμε όλες τις διαθέσιμες εντολές του Makefile χρησιμοποιούμε την εντολή **make help**



## Απαιτήσεις Συστήματος
1. **Python 3.8+** με pip package manager
2. **PyTorch (>= 1.9.0)** για νευρωνικά δίκτυα
3. **KaHIP Python** bindings για διαμέριση γράφων
4. **NumPy, SciPy, scikit-learn** για επεξεργασία δεδομένων
5. **Matplotlib, Seaborn** για οπτικοποίηση
6. Επαρκής μνήμη **RAM** (16GB+ για μεγάλα datasets)
7. Προαιρετικά: **CUDA-enabled GPU** για επιτάχυνση εκπαίδευσης

pip install -r requirements.txt

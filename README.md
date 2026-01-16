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

## Κατάλογος Αρχείων
## Κύρια Προγράμματα

1. **protein_embed.py**  : Μετατροπή FASTA αρχείων σε ESM-2 embeddings
2. **protein_search.py** : Αναζήτηση και αξιολόγηση με 5 μεθόδους ANN

### Μέθοδοι Αναζήτησης 
1. **ANN/euclidean_lsh.py**   : LSH με Ευκλείδειες αποστάσεις και multi-probe
2. **Hypercube/hypercube.py** : Προβολή σε υπερκύβο με Hamming απόσταση
3. **IVFFlat/ivfflat.py**     : IVF με ακριβείς αποστάσεις και K-means clustering
4. **IVFPQ/ivfpq.py**         : IVF με Product Quantization για συμπίεση
5. **Neural/neural_lsh.py**   : Υβριδική μέθοδος με MLP και partitioning
6. **Neural/graph_tools**     : Βοηθητικά αρχεία για την υλοποίηση του Neural LSH

## Αρχεία Δεδομένων (παραδείγματα)
**swissprot.fasta** : Βάση δεδομένων πρωτεϊνών
**targets.fasta** : Πρωτεΐνες-στόχοι για αναζήτηση
**blast_results.txt** : Αποτελέσματα BLAST για αξιολόγηση


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
# Βήμα 1: Παραγωγή Embeddings
    python protein_embed.py -i swissprot.fasta -o protein_vectors.dat \ --model esm2_t6_8M_UR50D --batch-size 16

    Παράμετροι:

    -i: Είσοδος FASTA αρχείο

    -o: Έξοδος δυαδικό αρχείο embeddings

    --model: Προεκπαιδευμένο μοντέλο ESM-2

    --batch-size: Μέγεθος batch για επεξεργασία

# Βήμα 2: Αναζήτηση και Αξιολόγηση
    python protein_search.py -d protein_vectors.dat -q targets.fasta \ -blast blast_results.txt -o results.txt -method all \ --recall_N 50 --N 10
    
    Παράμετροι:

    -d: Αρχείο embeddings βάσης δεδομένων

    -q: Query FASTA αρχείο

    -blast: Αποτελέσματα BLAST για αξιολόγηση

    -o: Αρχείο εξόδου με αποτελέσματα

    -method: Μέθοδος ANN (all, lsh, hypercube, neural, ivfflat, ivfpq)

    --recall_N: N για υπολογισμό Recall@N

    --N: Αριθμός γειτόνων για εμφάνιση

# Βήμα 3: Τρέξιμο Μεθόδων
# Euclidean LSH
    python protein_search.py -method lsh --lsh_k 10 --lsh_L 5 --lsh_w 4.0

# Hypercube
    python protein_search.py -method hypercube --hypercube_k 10 --hypercube_M 1000

# IVF-Flat
    python protein_search.py -method ivfflat --ivfflat_nlist 100 --ivfflat_nprobe 10

# IVF-PQ
    python protein_search.py -method ivfpq --ivfpq_nlist 100 --ivfpq_m 8

# Neural LSH
    python protein_search.py -method neural --neural_epochs 10 --neural_k 10


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

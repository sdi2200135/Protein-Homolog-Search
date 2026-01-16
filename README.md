# Ανάπτυξη Λογισμικού για Αλγοριθμικά Προβλήματα - 3η Προγραμματιστική Εργασία - Αναζήτηση Απομακρυσμένων ομόλογων με Προσεγγιστικές Μεθόδους ESM

## Στοιχεία Φοιτητών
**Ομάδα:**
1. Παπαθανασίου Ελένη - 1115202200135
2. Τόντου Αλτάνη-Δάφνη - 1115202200288

## Περιγραφή 
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


## Δομή Αποτελεσμάτων
[1] Συνοπτική σύγκριση μεθόδων
Method            | Time/query (s) | QPS     | Recall@N vs BLAST Top-N
Euclidean LSH     | 0.020          | 50.0    | 0.92
Hypercube         | 0.030          | 33.3    | 0.88
Neural LSH        | 0.010          | 100.0   | 0.95
IVF-Flat          | 0.008          | 125.0   | 0.93
IVF-PQ            | 0.005          | 200.0   | 0.90

# Αναλυτικοί Γείτονες
[2] Top-10 γείτονες ανά μέθοδο
Method: Euclidean LSH
Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment
1    | P14181      | 1.685   | 0.0%           | No              | REMOTE HOMOLOG CANDIDATE (0.0%)

## Οδηγίες Μεταγλώττισης και Εκτέλεσης
# Πλήρης Εκτέλεση
make all
1. make embeddings
2. make blast
3. make blast-search
# Για LSH Search
    make lsh
# Για  Hypercube Search
    make hypercube
# Για IVFFlat Search
    make ivfflat
# Για IVFPQ Search
    make ivfpq
# Για Neural LSH Search
    make neural
# Για όλες τις μεθοδους 
make ann-methods

## Απαιτήσεις Συστήματος

1. Python 3.10+ με pip package manager
2. PyTorch 2.0+ για ESM-2 embeddings
3. CUDA-capable GPU (προτεινόμενο) για επιτάχυνση
4. RAM 8GB+ για επεξεργασία embeddings
5. Δίσκος 5GB+ για αποθήκευση δεδομένων
6. Python 3.11 για βελτιστοποιημένη απόδοση
7. PyTorch με CUDA 11.8+ για GPU υπολογισμούς
8. RAM 16GB+ για μεγάλα datasets
9. SSD/NVMe για γρήγορη ανάγνωση δεδομένων
10. Multi-core CPU για παράλληλη επεξεργασία

# Εγκατάσταση με Conda 
conda create -n protein_search python=3.11
conda activate protein_search
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy scipy pandas scikit-learn biopython tqdm
pip install fair-esm
# Project_3

python protein_embed.py -i "data-query-sets&pfam-info/swissprot_50k.fasta" -o output.dat

import numpy as np

# φορτώνουμε τα embeddings που αποθηκεύσαμε
emb = np.load("protein_vectors.dat", allow_pickle=True).item()

# τυπώνουμε το σχήμα του πρώτου vector για έλεγχο
print(next(iter(emb.values())).shape)

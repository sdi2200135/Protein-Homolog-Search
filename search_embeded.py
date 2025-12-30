import argparse
import numpy as np
from Bio import SeqIO
import time
# from ann.lsh import EuclideanLSH
# from ann.hypercube import Hypercube
# from ann.ivf import IVFFlat, IVFPQ
# from ann.neural_lsh import NeuralLSH

def parse_args():
    parser = argparse.ArgumentParser(description="Protein search using embeddings and ANN")
    parser.add_argument("-d", "--database", required=True, help="Protein vectors (.npy)")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-method", default="all", help="Method: all, lsh, hypercube, neural, ivf")
    parser.add_argument("-N", type=int, default=10, help="Top-N neighbors")
    return parser.parse_args()

def load_queries(fasta_file):
    return [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")]

def load_embeddings(file):
    return np.load(file, allow_pickle=True).item()

def main():
    args = parse_args()
    
    #φορτωνουμε τα embeddings
    db_vectors = load_embeddings(args.database)
    print(f"Loaded {len(db_vectors)} protein embeddings.")

    # φορτωνουμε τα queries
    queries = load_queries(args.query)
    print(f"Loaded {len(queries)} queries.")

     # Initialize ANN methods (συμπληρωση)
    methods = {}
    if args.method in ["all", "lsh"]:
        methods["Euclidean LSH"] = None # αντικατάστηση με ANN index
    if args.method in ["all", "hypercube"]:
        methods["Hypercube"] = None
    if args.method in ["all", "neural"]:
        methods["Neural LSH"] = None
    if args.method in ["all", "ivf"]:
        methods["IVF-Flat"] = None
        methods["IVFPQ"] = None

    # Αποθήκευση αποτελεσμάτων
    output_lines = []

if __name__ == "__main__":
    main()

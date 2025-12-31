import argparse
import numpy as np
from Bio import SeqIO
import time
import pandas as pd
from ANN.euclidean_lsh import EuclideanLSH
# from ann.hypercube import Hypercube
# from ann.ivf import IVFFlat, IVFPQ
# from ann.neural_lsh import NeuralLSH

def parse_args():
    parser = argparse.ArgumentParser(description="Protein search using embeddings and ANN")
    parser.add_argument("-d", "--database", required=True, help="Protein vectors (.npy)")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-blast", "--blast_file", required=True, help="BLAST output file (tabular, -outfmt 6)")
    parser.add_argument("-method", default="all", help="Method: all, lsh, hypercube, neural, ivf")
    parser.add_argument("-N", type=int, default=10, help="Top-N neighbors")
    return parser.parse_args()

def load_queries(fasta_file):
    return [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")]

def load_embeddings(file):
    return np.load(file, allow_pickle=True).item()

def load_blast_results(blast_file, N):
    columns = ["query_id","subject_id","pident","length","mismatch","gapopen",
               "qstart","qend","sstart","send","evalue","bitscore"]
    df = pd.read_csv(blast_file, sep="\t", names=columns)
    df = df[df['evalue'] < 0.01]  # φίλτρο evalue
    topN = df.sort_values(["query_id","bitscore"], ascending=[True, False]).groupby("query_id").head(N)
    return {q: list(g["subject_id"]) for q, g in topN.groupby("query_id")}

def main():
    args = parse_args()
    
    #φορτωνουμε τα embeddings
    db_vectors = load_embeddings(args.database)
    print(f"Loaded {len(db_vectors)} protein embeddings.")
    query_vectors = load_embeddings("target_vectors.dat.npy")

    # φορτωνουμε τα queries
    queries = load_queries(args.query)
    print(f"Loaded {len(queries)} queries.")

    #φορτωνουμς BLAST αποτελέσματα
    blast_topN_dict = load_blast_results(args.blast_file, args.N)
    
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
    lsh = EuclideanLSH(db_vectors, k=10, L=20, w=4.0)

    # loop για καθε query
    for q_id, _ in queries:
        query_vector = query_vectors[q_id]
        if query_vector is None:
            print(f"Query {q_id} not found in embeddings, skipping...")
            continue

        for method_name in methods.keys():
            start_time = time.time()

            if method_name == "Euclidean LSH":
                # Euclidean LSH
                top_N_lsh = lsh.query(query_vector,args.N)
            elapsed = time.time() - start_time

            #υπολογισμος Recall
            lsh_ids = [pid for pid, _ in top_N_lsh]
            blast_ids = blast_topN_dict.get(q_id, [])
            hits = sum([1 for pid in lsh_ids if pid in blast_ids])
            recall = hits / min(len(blast_ids), args.N) if blast_ids else 0.0


            #Αποθηκευση αποτελεσματων
            output_lines.append(f"Query Protein: {q_id}")
            output_lines.append(f"Method: {method_name} | Time: {elapsed:.4f}s | Top-{args.N}  | Recall@N: {recall:.2f}")
            output_lines.append("Rank\tNeighbor ID\tL2 Distance")
            for rank, (pid, dist) in enumerate(top_N_lsh, 1):
                output_lines.append(f"{rank}\t{pid}\t{dist:.4f}")
            output_lines.append("\n")
    #γραφουμε σε αρχειο
    with open(args.output, "w") as f:
        f.write("\n".join(output_lines))
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()

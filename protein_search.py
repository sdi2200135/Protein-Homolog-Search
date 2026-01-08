import argparse
import numpy as np
from Bio import SeqIO
import time
import pandas as pd
import torch
import esm
from tqdm import tqdm
import os
from ANN.euclidean_lsh import EuclideanLSH
from Hypercube.hypercube import Hypercube
from IVFFlat.ivfflat import IVFFlat
from IVFPQ.ivfpq import IVFPQ
from Neural.neural_lsh import NeuralLSH
  

def parse_args():
    parser = argparse.ArgumentParser(description="Protein search using embeddings and ANN")
    parser.add_argument("-d", "--database", required=True, help="Protein vectors (.npy)")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-blast", "--blast_file", required=True, help="BLAST output file (tabular, -outfmt 6)")
    parser.add_argument("-N", type=int, default=10, help="Top-N neighbors to display")
    parser.add_argument("--recall_N", type=int, default=50, help="N for Recall@N calculation")
    # parser.add_argument("--lsh_k", type=int, default=10, help="LSH parameter k")
    # parser.add_argument("--lsh_L", type=int, default=5, help="LSH parameter L")
    # parser.add_argument("--lsh_w", type=float, default=4.0, help="LSH parameter w")
    parser.add_argument("--query_embeddings", help="Pre-computed query embeddings (.npy)")
    parser.add_argument("-method",choices=["all", "lsh", "hypercube", "neural", "ivfflat", "ivfpq"], default="lsh",help="ANN method to use")
    parser.add_argument("--max_length", type=int, default=1022, help="Max sequence length for embedding")
    # parser.add_argument("--ivf_nlist", type=int, default=100, help="IVF: number of clusters")
    # parser.add_argument("--ivf_nprobe", type=int, default=10, help="IVF: number of clusters to probe")
    # parser.add_argument("--ivfpq_m", type=int, default=8, help="IVFPQ: number of subvectors")
    # parser.add_argument("--neural_epochs", type=int, default=10, help="Neural LSH training epochs")
    
    return parser.parse_args()

def load_esm_model():
    """Load ESM-2 model for embedding queries"""
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def embed_sequences(sequences, model, batch_converter, max_length=1022):
    """Embed a list of (id, sequence) pairs"""
    embeddings = {}
    
    # Process in batches
    batch_size = 32
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        
        # Prepare batch
        data = [(pid, seq[:max_length]) for pid, seq in batch]
        labels, strs, tokens = batch_converter(data)
        
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        with torch.no_grad():
            output = model(tokens, repr_layers=[6])
            token_reps = output["representations"][6].cpu()
        
        # Process each sequence in batch
        for j, (pid, seq) in enumerate(batch):
            seq_len = min(len(seq), max_length)
            reps = token_reps[j, 1:seq_len+1]
            embedding = reps.mean(dim=0).numpy()
            embeddings[pid] = embedding
    
    return embeddings

def load_embeddings(file):
    return np.load(file, allow_pickle=True).item()

def load_queries(fasta_file):
    return [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")]

def load_blast_results(blast_file, recall_N=50):
    cols = ["query", "subject", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
    
    try:
        df = pd.read_csv(blast_file, sep="\t", names=cols)
    except Exception as e:
        print(f"Error loading BLAST file: {e}")
        return {}, {}
    
    # Filter by evalue
    df = df[df["evalue"] < 0.01]
    
    # Get top-N for recall calculation (sorted by bitscore)
    blast_topN = {}
    for query, group in df.groupby("query"):
        top_hits = group.nlargest(recall_N, "bitscore")
        blast_topN[query] = set(top_hits["subject"])
    
    # Create identity dictionary for all hits
    blast_identity = {}
    for _, row in df.iterrows():
        blast_identity.setdefault(row["query"], {})[row["subject"]] = row["pident"]
    
    return blast_topN, blast_identity

def main():
    args = parse_args()
    
    #φορτωνουμε τα embeddings
    db_embeddings  = load_embeddings(args.database)
    print(f"Loaded {len(db_embeddings )} protein embeddings.")
    
    # Query embeddings
    if args.query_embeddings and os.path.exists(args.query_embeddings):
        print(f"Loading pre-computed query embeddings from {args.query_embeddings}...")
        query_embeddings = load_embeddings(args.query_embeddings)
    else:
        print(f"Loading and embedding queries from {args.query}...")
        model, batch_converter = load_esm_model()
        queries = []
        for record in SeqIO.parse(args.query, "fasta"):
            queries.append((record.id, str(record.seq)))
        query_embeddings = embed_sequences(queries, model, batch_converter, args.max_length)
    print(f"Loaded {len(query_embeddings)} query embeddings")
  
    #φορτωνουμς BLAST αποτελέσματα
    blast_topN, blast_identity = load_blast_results(args.blast_file, args.recall_N)
   
    # Initialize methods
    methods = {}
    if args.method in ["all", "lsh"]:
        methods["Euclidean LSH"] = EuclideanLSH(
            vectors=db_embeddings,
            k=10,
            L=5,
            w=4.0,
            seed=42
        )
    if args.method in ["all", "hypercube"]:
        methods["Hypercube"] = Hypercube(
            vectors=db_embeddings,
            k=10,
            M=1000,
            probes= 5,
            w= 4.0,
            seed=42
        )
    if args.method in ["all", "ivfflat"]:
        methods["IVF-Flat"] = IVFFlat(
            vectors=db_embeddings,
            nlist=100,
            nprobe=10
        )
    if args.method in ["all", "ivfpq"]:
        methods["IVF-PQ"] = IVFPQ(
            vectors=db_embeddings,
            nlist=100,
            nprobe=10,
            m=8
        )
    if args.method in ["all", "neural"]:
        methods["Neural LSH"] = NeuralLSH(
            vectors=db_embeddings,
            epochs=10
        )

  
    # Αποτελέσματα
    results = []
    total_queries = len(query_embeddings)
    total_time = 0
    total_recall = 0

    # loop για καθε query
    for query_id, query_vec in tqdm(query_embeddings.items(), desc="Processing queries"):
        
        results.append(f"Query Protein: {query_id}")
        results.append(f"N = {args.recall_N} (size of Top-N list for Recall@N calculation)")
        results.append("")

        blast_hits = blast_topN.get(query_id, set())
        identities = blast_identity.get(query_id, {})

        # Loop για κάθε μέθοδο
        results.append("[1] Συνοπτική σύγκριση μεθόδων")
        results.append("-" * 70)
        results.append("Method            | Time/query (s) | QPS     | Recall@N vs BLAST Top-N")
        results.append("-" * 70)

        method_neighbors = {}
        for method_name, method_obj in methods.items():
            if method_obj is None:
                continue  # skip αν δεν έχει υλοποιηθεί 

            start_time = time.time()
            neighbors = method_obj.query(query_vec, args.recall_N * 2)  # Get more for safety
            elapsed = time.time() - start_time
            
            
            # Calculate QPS
            qps = 1.0 / elapsed if elapsed > 0 else 0

            # Υπολογισμός recall
            neighbor_ids = [pid for pid, _ in neighbors[:args.recall_N]]
            hits = sum(1 for pid in neighbor_ids if pid in blast_hits)
            recall = hits / args.recall_N
            total_recall += recall
            
            total_time += elapsed
            results.append(f"{method_name:16} | {elapsed:12.4f} | {qps:7.1f} | {recall:.3f}")
            method_neighbors[method_name] = neighbors[:args.N]
       
        # Προσθήκη BLAST αναφοράς
        if blast_hits:
            results.append(f"{'BLAST (Ref)':16} | {'-':12} | {'-':7} | 1.000")
        results.append("")
        
        # Λεπτομερή αποτελέσματα ανά μέθοδο
        for method_name, neighbors in method_neighbors.items():
            results.append(f"[2] Top-{args.N} γείτονες ({method_name})")
            results.append("-" * 90)
            results.append("Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment")
            results.append("-" * 90)
            for rank, (neighbor_id, distance) in enumerate(neighbors, 1):
                identity = identities.get(neighbor_id, 0.0)
                in_blast = "Yes" if neighbor_id in blast_hits else "No"
                results.append(f"{rank:4} | {neighbor_id:11} | {distance:7.4f} | {identity:13.1f}% | {in_blast:15} | --")
            results.append("")
        results.append("="*80)
        results.append("")
    
    # Αποθήκευση  σε αρχειο
    with open(args.output, "w") as f:
        f.write("\n".join(results))
    print(f"Results saved to {args.output}")

    # 8. Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total queries processed: {total_queries}")
    print(f"Database size: {len(db_embeddings)}")
    print(f"Average query time: {total_time/total_queries:.4f}s")
    print(f"Average QPS: {total_queries/total_time:.1f}" if total_time > 0 else "Average QPS: 0")
    print(f"Average Recall@{args.recall_N}: {total_recall/total_queries:.3f}")

if __name__ == "__main__":
    main()

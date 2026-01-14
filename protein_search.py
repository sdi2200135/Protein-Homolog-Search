import argparse
import numpy as np
from Bio import SeqIO
import time
import pandas as pd
import torch
import esm
from tqdm import tqdm
import os
import json
from typing import Dict, List, Tuple, Set
import warnings
warnings.filterwarnings('ignore')

# Import ANN methods - όλα μαζί όπως στα προηγούμενα
from ANN.euclidean_lsh import EuclideanLSH
from Hypercube.hypercube import Hypercube
from IVFFlat.ivfflat import IVFFlat
from IVFPQ.ivfpq import IVFPQSearch
from Neural.neural_lsh import NeuralLSH
from protein_embed import load_embeddings_single_file 

#Ορισμος και αναγνωση arguments απο τη γραμμη εντολων
def parse_args():
    parser = argparse.ArgumentParser(description="Protein search using embeddings and ANN")
    parser.add_argument("-d", "--database", required=True, help="Protein vectors (.npy)")
    parser.add_argument("-q", "--query", required=True, help="Query FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Output file")
    parser.add_argument("-blast", "--blast_file", required=True, help="BLAST output file (tabular, -outfmt 6)")
    parser.add_argument("-N", type=int, default=10, help="Top-N neighbors to display")
    parser.add_argument("--recall_N", type=int, default=50, help="N for Recall@N calculation")
    parser.add_argument("--query_embeddings", help="Pre-computed query embeddings (.npy)")
    parser.add_argument("-method", choices=["all", "lsh", "hypercube", "neural", "ivfflat", "ivfpq"], 
                       default="all", help="ANN method to use")
    parser.add_argument("--max_length", type=int, default=1022, help="Max sequence length for embedding")
    
    # Method παραμετροι
    parser.add_argument("--lsh_k", type=int, default=10, help="LSH: number of hash functions")
    parser.add_argument("--lsh_L", type=int, default=5, help="LSH: number of hash tables")
    parser.add_argument("--lsh_w", type=float, default=4.0, help="LSH: bucket width")
    parser.add_argument("--hypercube_k", type=int, default=10, help="Hypercube: dimension")
    parser.add_argument("--hypercube_M", type=int, default=1000, help="Hypercube: max candidates")
    parser.add_argument("--hypercube_probes", type=int, default=5, help="Hypercube: probes")
    parser.add_argument("--ivfflat_nlist", type=int, default=100, help="IVF-Flat: number of clusters")
    parser.add_argument("--ivfflat_nprobe", type=int, default=10, help="IVF-Flat: clusters to probe")
    parser.add_argument("--ivfpq_nlist", type=int, default=100, help="IVFPQ: number of clusters")
    parser.add_argument("--ivfpq_nprobe", type=int, default=10, help="IVFPQ: clusters to probe")
    parser.add_argument("--ivfpq_m", type=int, default=8, help="IVFPQ: subvectors")
    parser.add_argument("--neural_epochs", type=int, default=10, help="Neural LSH: epochs")
    parser.add_argument("--neural_k", type=int, default=10, help="Neural LSH: k for k-NN graph")
    parser.add_argument("--neural_m", type=int, default=100, help="Neural LSH: number of partitions")
    parser.add_argument("--neural_T", type=int, default=5, help="Neural LSH: probes")
    
    # παραμετρος UniProt annotations (προαιρετικα)
    parser.add_argument("--uniprot_info", help="JSON file with UniProt annotations (optional)")
    
    return parser.parse_args()

# Φορτωνει UniProt annotations 
def load_uniprot_info(filepath):
    if not filepath or not os.path.exists(filepath): #διαβασμα αρχειου json ,αν δεν υπαρχει επιστροφη κενου
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def is_remote_homolog(identity, l2_distance=None, threshold=0.3):
    """Ελέγχει αν 2 πρωτεΐνες είναι απομακρυσμένοι ομόλογοι"""
    if l2_distance is None:
        return False
    return identity < 30 and l2_distance < threshold  # Twilight Zone threshold


def get_bio_comment(uniprot_info, query_id, neighbor_id, identity, l2_distance=None):
    if is_remote_homolog(identity, l2_distance):
        comment = f"REMOTE HOMOLOG CANDIDATE ({identity:.1f}%)"
    elif identity > 30:
        comment = "High similarity"
    elif identity > 20:
        comment = f"Twilight Zone ({identity:.1f}%)"
    elif identity > 0:
        comment = f"Remote homolog candidate ({identity:.1f}%)"
    else:
        comment = "No BLAST match"
    
    # Αν υπαρχουν UniProt annotations
    if uniprot_info:
        query_annot = uniprot_info.get(query_id, {})
        neighbor_annot = uniprot_info.get(neighbor_id, {})
        
        # ελεγχος κοινων domains
        if query_annot and neighbor_annot:
            query_domains = set(query_annot.get('domains', []))
            neighbor_domains = set(neighbor_annot.get('domains', []))
            common_domains = query_domains.intersection(neighbor_domains)
            
            if common_domains:
                comment += f" [Common domains: {', '.join(list(common_domains)[:2])}]"
            
            # ελεγχος EC numbers
            query_ec = set(query_annot.get('ec_numbers', []))
            neighbor_ec = set(neighbor_annot.get('ec_numbers', []))
            common_ec = query_ec.intersection(neighbor_ec)
            
            if common_ec:
                comment += f" [Common EC: {', '.join(list(common_ec))}]"
    
    return comment

# Φορτωνει ESM μοντελο (μονο για queries)
def load_esm_model():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

# Μετατρεπει ακολουθιες σε embeddings
def embed_sequences(sequences, model, batch_converter, max_length=1022):
    embeddings = {}
    batch_size = 32

    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i+batch_size]
        
        # περιορισμος μηκους ακολουθιας
        data = [(pid, seq[:max_length]) for pid, seq in batch]
        labels, strs, tokens = batch_converter(data)
        
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        
        with torch.no_grad():
            output = model(tokens, repr_layers=[6])
            token_reps = output["representations"][6].cpu()
        
        # Mean pooling
        for j, (pid, seq) in enumerate(batch):
            seq_len = min(len(seq), max_length)
            reps = token_reps[j, 1:seq_len+1]
            embedding = reps.mean(dim=0).numpy()
            embeddings[pid] = embedding
    
    return embeddings

# Φορτωνει embeddings απο δυαδικο αρχειο και μετατρεπει σε dict 
def load_embeddings(file):
    embeddings, ids = load_embeddings_single_file(file)
    return {pid: emb for pid, emb in zip(ids, embeddings)}

# Φορτωνει BLAST αποτελεσματα
def load_blast_results(blast_file, recall_N=50):
    cols = ["query", "subject", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
    
    try:
        df = pd.read_csv(blast_file, sep="\t", names=cols)
    except Exception as e:
        print(f"Error loading BLAST file: {e}")
        return {}, {}
    
    # φιλτραρισμα evalue 
    df = df[df["evalue"] < 0.01]
    
    # Παιρνουμε τα top-N για recall calculation
    blast_topN = {}
    for query, group in df.groupby("query"):
        top_hits = group.nlargest(recall_N, "bitscore")
        blast_topN[query] = set(top_hits["subject"].tolist())
    
    # Δημιουργια identity dictionary για ολα τα  hits
    blast_identity = {}
    for _, row in df.iterrows():
        blast_identity.setdefault(row["query"], {})[row["subject"]] = row["pident"]
    
    print(f"Loaded BLAST results: {len(blast_topN)} queries, {len(df)} total hits")
    return blast_topN, blast_identity

#Δημιουργει ολες τις επιλεγμενες ANN μεθοδους
def initialize_methods(args, db_embeddings):
    methods = {}
    
    # Euclidean LSH
    if args.method in ["all", "lsh"]:
        methods["Euclidean LSH"] = EuclideanLSH(
            vectors=db_embeddings,
            k=args.lsh_k,
            L=args.lsh_L,
            w=args.lsh_w,
            seed=42
        )
        print(f"Euclidean LSH initialized (k={args.lsh_k}, L={args.lsh_L}, w={args.lsh_w})")
    
    # Hypercube
    if args.method in ["all", "hypercube"]:
        methods["Hypercube"] = Hypercube(
            vectors=db_embeddings,
            k=args.hypercube_k,
            M=args.hypercube_M,
            probes=args.hypercube_probes,
            w=args.lsh_w,
            seed=42
        )
        print(f"Hypercube initialized (k={args.hypercube_k}, M={args.hypercube_M})")
    
    # IVF-Flat
    if args.method in ["all", "ivfflat"]:
        methods["IVF-Flat"] = IVFFlat(
            vectors=db_embeddings,
            nlist=args.ivfflat_nlist,
            nprobe=args.ivfflat_nprobe,
            seed=42
        )
        print(f"IVF-Flat initialized (nlist={args.ivfflat_nlist}, nprobe={args.ivfflat_nprobe})")
    
    # IVF-PQ
    if args.method in ["all", "ivfpq"]:
        methods["IVF-PQ"] = IVFPQSearch(
            vectors=db_embeddings,
            nlist=args.ivfpq_nlist,
            nprobe=args.ivfpq_nprobe,
            m=args.ivfpq_m,
            seed=42
        )
        print(f"IVF-PQ initialized (nlist={args.ivfpq_nlist}, m={args.ivfpq_m})")
    
    # Neural LSH
    if args.method in ["all", "neural"]:
        methods["Neural LSH"] = NeuralLSH(
            vectors=db_embeddings,
            index_dir="neural_lsh_index",
            rebuild=False,
            k=args.neural_k,
            m=args.neural_m,
            T=args.neural_T,
            epochs=args.neural_epochs,
            seed=42
        )
        
        print(f"Neural LSH initialized (k=10, m=100, T=5, epochs={args.neural_epochs})")
    
    return methods

#Μετατρεπει αποσταση σε float
def format_distance(distance):
    if isinstance(distance, (str, np.str_)):
        try:
            return float(distance)
        except (ValueError, TypeError):
            return 0.0
    else:
        try:
            return float(distance)
        except (ValueError, TypeError):
            return 0.0

def main():
    # Αναγνωση παραμετρων
    args = parse_args()
    
    # Δημιουργια φακελου εξοου αν υπαρχει
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # Φορτωση UniProt annotations
    uniprot_info = load_uniprot_info(args.uniprot_info) if args.uniprot_info else {}
    
    # 1. Φορτωση embeddings βασης δεδομενων
    print(f"Loading database embeddings from {args.database}...")
    db_embeddings = load_embeddings(args.database)
    
    if not isinstance(db_embeddings, dict):
        # Μετατροπη σε λεξικο
        db_embeddings = {f"prot_{i}": emb for i, emb in enumerate(db_embeddings)}
    print(f"Loaded {len(db_embeddings)} protein embeddings")
    
    # 2. Φορτωση ή υπολογισμος query embeddings
    if args.query_embeddings and os.path.exists(args.query_embeddings):
        # αν υπαρχουν προυπολογισμενα embeddings
        print(f"Loading pre-computed query embeddings from {args.query_embeddings}...")
        query_embeddings = load_embeddings(args.query_embeddings)
    else:
        # διαφορετικα υπολογιζονται απο FASTA 
        print(f"Loading and embedding queries from {args.query}...")
        model, batch_converter = load_esm_model()
        
        queries = [(record.id, str(record.seq)) for record in SeqIO.parse(args.query, "fasta")]
        query_embeddings = embed_sequences(queries, model, batch_converter, args.max_length)
        
        if args.query_embeddings:
            np.save(args.query_embeddings, query_embeddings)
            print(f"✓ Saved query embeddings to {args.query_embeddings}")
    
    print(f"Loaded {len(query_embeddings)} query embeddings")
    
    # 3. Φορτωση BLAST αποτελεσματων (Ground Truth)
    print(f"Loading BLAST results from {args.blast_file}...")
    blast_topN, blast_identity = load_blast_results(args.blast_file, args.recall_N)
    
    # 4. Αρχικοποιηση ANN μεθοδων
    print("\nInitializing ANN methods...")
    methods = initialize_methods(args, db_embeddings)
    
    if not methods:
        print("Error: No ANN methods could be initialized!")
        return
    
    print(f"\nSuccessfully initialized {len(methods)} methods")
    
    # 5. Επεξεργασια queries
    print(f"\nProcessing {len(query_embeddings)} queries...")
    
    results = [] # λιστα με γραμμες αποτελεσματων

    # στατιστικα ανα μεθοδο
    method_stats = {name: {'total_time': 0.0, 'total_recall': 0.0, 'qps': []} 
                   for name in methods}
    
    # εναρξη χρονου για QPS
    overall_start_time = time.time()
    
    # Loop  πανω σε καθε query
    for query_id, query_vec in tqdm(query_embeddings.items(), desc="Queries"):
        query_output = []
        
        # Query header
        query_output.append(f"Query Protein: {query_id}")
        query_output.append(f"N = {args.recall_N} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)")
        query_output.append("")
        
        # BLAST hits για το συγκεκιμενο query
        blast_hits = blast_topN.get(query_id, set())
        identities = blast_identity.get(query_id, {})
        
        # Summary comparison tabl
        query_output.append("[1] Συνοπτική σύγκριση μεθόδων")
        query_output.append("-" * 70)
        query_output.append("Method            | Time/query (s) | QPS     | Recall@N vs BLAST Top-N")
        query_output.append("-" * 70)
        
        method_neighbors = {}
        
        # Εκτελεση καθε ΑΝΝ μεθοδου
        for method_name, method_obj in methods.items():
            try:
                # χρονος για search
                start_time = time.time()
                #Αναζητηση γειτονων
                neighbors = method_obj.query(query_vec, args.recall_N)
                print(f"\nDEBUG: Method {method_name} found {len(neighbors)} neighbors")
                if neighbors:
                    print(f"\nDEBUG: Method {method_name} found {len(neighbors)} neighbors for query {query_id}")
                    # Τύπωσε τους πρώτους 5 γείτονες και τις αποστάσεις τους
                    print(f"Top 5 neighbors:")
                    for i, (neighbor_id, distance) in enumerate(neighbors[:5]):
                        print(f"  {i+1}. {neighbor_id}: distance={distance:.4f}")
                    
                    # Έλεγξε αν κάποιος είναι στα BLAST top-N
                    found_in_blast = [pid for pid, _ in neighbors[:args.recall_N] if pid in blast_hits]
                    print(f"Found {len(found_in_blast)}/{min(args.recall_N, len(neighbors))} in BLAST top-{args.recall_N}")
                else:
                    print(f"\nDEBUG: Method {method_name} found 0 neighbors for query {query_id}")
                elapsed = time.time() - start_time
                
                # Υπολογισμος recall
                neighbor_ids = [pid for pid, _ in neighbors[:args.recall_N]]
                hits = sum(1 for pid in neighbor_ids if pid in blast_hits)
                recall = hits / min(args.recall_N, len(blast_hits)) if blast_hits else 0
                
                # Υπολογισμος QPS για αυτο το query
                qps = 1.0 / elapsed if elapsed > 0 else 0
                
                # Ενημερωση στατιστικων
                method_stats[method_name]['total_time'] += elapsed
                method_stats[method_name]['total_recall'] += recall
                method_stats[method_name]['qps'].append(qps)
                
                # καταγραφη γραμμης αποτελεσματος
                query_output.append(f"{method_name:16} | {elapsed:12.3f} | {qps:7.1f} | {recall:.3f}")
                
                # αποθηκευση γειτονων
                method_neighbors[method_name] = neighbors[:args.N]
                
            except Exception as e:
                query_output.append(f"{method_name:16} | ERROR        | ERROR  | ERROR")
                print(f"Error in {method_name} for query {query_id}: {e}")
                method_neighbors[method_name] = []
        
        # BLAST reference
        query_output.append(f"{'BLAST (Ref)':16} | {'-':12} | {'-':7} | 1.000")
        query_output.append("-" * 70)
        query_output.append("")
        
        # Detailed neighbors για καθε μεθοδο
        query_output.append(f"[2] Top-{args.N} γείτονες ανά μέθοδο (εδώ π.χ. N = {args.N} για εκτύπωση)")
        query_output.append("")
        
        for method_name, neighbors in method_neighbors.items():
            if not neighbors:
                continue
            
            query_output.append(f"Method: {method_name}")
            query_output.append("-" * 90)
            query_output.append("Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment")
            query_output.append("-" * 90)
            
            for rank, (neighbor_id, distance) in enumerate(neighbors, 1):
                # Βεβαιώσου ότι η απόσταση είναι float
                distance = format_distance(distance)
                
                identity = identities.get(neighbor_id, 0.0)
                in_blast = "Yes" if neighbor_id in blast_hits else "No"
                # comment = get_bio_comment(uniprot_info, query_id, neighbor_id, identity)
                comment = get_bio_comment(uniprot_info, query_id, neighbor_id, identity, distance)
                
                query_output.append(f"{rank:4} | {neighbor_id:11} | {distance:7.3f} | "
                                  f"{identity:13.1f}% | {in_blast:15} | {comment}")
            
            query_output.append("")
        
        query_output.append("=" * 80)
        query_output.append("")
        results.extend(query_output)
    
    # Τελικη συνοψη
    overall_elapsed = time.time() - overall_start_time
    overall_qps = len(query_embeddings) / overall_elapsed if overall_elapsed > 0 else 0
  
    results.append("\n=== ΣΥΝΟΠΤΙΚΗ ΣΤΑΤΙΣΤΙΚΗ ===")
    results.append(f"Συνολικός αριθμός queries: {len(query_embeddings)}")
    results.append(f"Μέγεθος βάσης δεδομένων: {len(db_embeddings)}")
    results.append(f"Συνολικός χρόνος εκτέλεσης: {overall_elapsed:.2f}s")
    results.append(f"Συνολικό QPS: {overall_qps:.1f}")
    results.append("")
    
    for method_name, stats in method_stats.items():
        if len(query_embeddings) > 0:
            avg_time = stats['total_time'] / len(query_embeddings)
            avg_recall = stats['total_recall'] / len(query_embeddings)
            avg_qps = np.mean(stats['qps']) if stats['qps'] else 0
            
            results.append(f"{method_name}:")
            results.append(f"  Μέσος χρόνος αναζήτησης: {avg_time:.4f}s")
            results.append(f"  Μέσος QPS: {avg_qps:.1f}")
            results.append(f"  Μέσος Recall@{args.recall_N}: {avg_recall:.3f}")
            results.append("")
    
    # Αποθηκευση αποτελεσματων
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(results))
    
    print(f"\nResults saved to {args.output}")
    print(f"\n=== ΣΥΝΟΠΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ===")
    print(f"Συνολικός αριθμός queries: {len(query_embeddings)}")
    print(f"Συνολικός χρόνος: {overall_elapsed:.2f}s")
    print(f"Συνολικό QPS: {overall_qps:.1f}")
    

    for method_name, stats in method_stats.items():
        if len(query_embeddings) > 0:
            avg_recall = stats['total_recall'] / len(query_embeddings)
            print(f"{method_name}: Recall@{args.recall_N} = {avg_recall:.3f}")

if __name__ == "__main__":
    main()
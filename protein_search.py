# import argparse
# import numpy as np
# from Bio import SeqIO
# import time
# import pandas as pd
# import torch
# import esm
# from tqdm import tqdm
# import os
# from ANN.euclidean_lsh import EuclideanLSH
# from Hypercube.hypercube import Hypercube
# from IVFFlat.ivfflat import IVFFlat
# from IVFPQ.ivfpq import IVFPQ
# from Neural.neural_lsh import NeuralLSH
  

# def parse_args():
#     parser = argparse.ArgumentParser(description="Protein search using embeddings and ANN")
#     parser.add_argument("-d", "--database", required=True, help="Protein vectors (.npy)")
#     parser.add_argument("-q", "--query", required=True, help="Query FASTA file")
#     parser.add_argument("-o", "--output", required=True, help="Output file")
#     parser.add_argument("-blast", "--blast_file", required=True, help="BLAST output file (tabular, -outfmt 6)")
#     parser.add_argument("-N", type=int, default=10, help="Top-N neighbors to display")
#     parser.add_argument("--recall_N", type=int, default=50, help="N for Recall@N calculation")
#     # parser.add_argument("--lsh_k", type=int, default=10, help="LSH parameter k")
#     # parser.add_argument("--lsh_L", type=int, default=5, help="LSH parameter L")
#     # parser.add_argument("--lsh_w", type=float, default=4.0, help="LSH parameter w")
#     parser.add_argument("--query_embeddings", help="Pre-computed query embeddings (.npy)")
#     parser.add_argument("-method",choices=["all", "lsh", "hypercube", "neural", "ivfflat", "ivfpq"], default="lsh",help="ANN method to use")
#     parser.add_argument("--max_length", type=int, default=1022, help="Max sequence length for embedding")
#     # parser.add_argument("--ivf_nlist", type=int, default=100, help="IVF: number of clusters")
#     # parser.add_argument("--ivf_nprobe", type=int, default=10, help="IVF: number of clusters to probe")
#     # parser.add_argument("--ivfpq_m", type=int, default=8, help="IVFPQ: number of subvectors")
#     # parser.add_argument("--neural_epochs", type=int, default=10, help="Neural LSH training epochs")
    
#     return parser.parse_args()

# def load_esm_model():
#     """Load ESM-2 model for embedding queries"""
#     model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#     model.eval()
#     if torch.cuda.is_available():
#         model = model.cuda()
#     batch_converter = alphabet.get_batch_converter()
#     return model, batch_converter

# def embed_sequences(sequences, model, batch_converter, max_length=1022):
#     """Embed a list of (id, sequence) pairs"""
#     embeddings = {}
    
#     # Process in batches
#     batch_size = 32
#     for i in range(0, len(sequences), batch_size):
#         batch = sequences[i:i+batch_size]
        
#         # Prepare batch
#         data = [(pid, seq[:max_length]) for pid, seq in batch]
#         labels, strs, tokens = batch_converter(data)
        
#         if torch.cuda.is_available():
#             tokens = tokens.cuda()
        
#         with torch.no_grad():
#             output = model(tokens, repr_layers=[6])
#             token_reps = output["representations"][6].cpu()
        
#         # Process each sequence in batch
#         for j, (pid, seq) in enumerate(batch):
#             seq_len = min(len(seq), max_length)
#             reps = token_reps[j, 1:seq_len+1]
#             embedding = reps.mean(dim=0).numpy()
#             embeddings[pid] = embedding
    
#     return embeddings

# def load_embeddings(file):
#     return np.load(file, allow_pickle=True).item()

# def load_queries(fasta_file):
#     return [(record.id, str(record.seq)) for record in SeqIO.parse(fasta_file, "fasta")]

# def load_blast_results(blast_file, recall_N=50):
#     cols = ["query", "subject", "pident", "length", "mismatch", "gapopen",
#             "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
    
#     try:
#         df = pd.read_csv(blast_file, sep="\t", names=cols)
#     except Exception as e:
#         print(f"Error loading BLAST file: {e}")
#         return {}, {}
    
#     # Filter by evalue
#     df = df[df["evalue"] < 0.01]
    
#     # Get top-N for recall calculation (sorted by bitscore)
#     blast_topN = {}
#     for query, group in df.groupby("query"):
#         top_hits = group.nlargest(recall_N, "bitscore")
#         blast_topN[query] = set(top_hits["subject"])
    
#     # Create identity dictionary for all hits
#     blast_identity = {}
#     for _, row in df.iterrows():
#         blast_identity.setdefault(row["query"], {})[row["subject"]] = row["pident"]
    
#     return blast_topN, blast_identity

# def main():
#     args = parse_args()
    
#     #φορτωνουμε τα embeddings
#     db_embeddings  = load_embeddings(args.database)
#     print(f"Loaded {len(db_embeddings )} protein embeddings.")
    
#     # Query embeddings
#     if args.query_embeddings and os.path.exists(args.query_embeddings):
#         print(f"Loading pre-computed query embeddings from {args.query_embeddings}...")
#         query_embeddings = load_embeddings(args.query_embeddings)
#     else:
#         print(f"Loading and embedding queries from {args.query}...")
#         model, batch_converter = load_esm_model()
#         queries = []
#         for record in SeqIO.parse(args.query, "fasta"):
#             queries.append((record.id, str(record.seq)))
#         query_embeddings = embed_sequences(queries, model, batch_converter, args.max_length)
#     print(f"Loaded {len(query_embeddings)} query embeddings")
  
#     #φορτωνουμς BLAST αποτελέσματα
#     blast_topN, blast_identity = load_blast_results(args.blast_file, args.recall_N)
   
#     # Initialize methods
#     methods = {}
#     if args.method in ["all", "lsh"]:
#         methods["Euclidean LSH"] = EuclideanLSH(
#             vectors=db_embeddings,
#             k=10,
#             L=5,
#             w=4.0,
#             seed=42
#         )
#     if args.method in ["all", "hypercube"]:
#         methods["Hypercube"] = Hypercube(
#             vectors=db_embeddings,
#             k=10,
#             M=1000,
#             probes= 5,
#             w= 4.0,
#             seed=42
#         )
#     if args.method in ["all", "ivfflat"]:
#         methods["IVF-Flat"] = IVFFlat(
#             vectors=db_embeddings,
#             nlist=100,
#             nprobe=10
#         )
#     if args.method in ["all", "ivfpq"]:
#         methods["IVF-PQ"] = IVFPQ(
#             vectors=db_embeddings,
#             nlist=100,
#             nprobe=10,
#             m=8
#         )
#     if args.method in ["all", "neural"]:
#         methods["Neural LSH"] = NeuralLSH(
#             vectors=db_embeddings,
#             epochs=10
#         )

  
#     # Αποτελέσματα
#     results = []
#     total_queries = len(query_embeddings)
#     total_time = 0
#     total_recall = 0

#     # loop για καθε query
#     for query_id, query_vec in tqdm(query_embeddings.items(), desc="Processing queries"):
        
#         results.append(f"Query Protein: {query_id}")
#         results.append(f"N = {args.recall_N} (size of Top-N list for Recall@N calculation)")
#         results.append("")

#         blast_hits = blast_topN.get(query_id, set())
#         identities = blast_identity.get(query_id, {})

#         # Loop για κάθε μέθοδο
#         results.append("[1] Συνοπτική σύγκριση μεθόδων")
#         results.append("-" * 70)
#         results.append("Method            | Time/query (s) | QPS     | Recall@N vs BLAST Top-N")
#         results.append("-" * 70)

#         method_neighbors = {}
#         for method_name, method_obj in methods.items():
#             if method_obj is None:
#                 continue  # skip αν δεν έχει υλοποιηθεί 

#             start_time = time.time()
#             neighbors = method_obj.query(query_vec, args.recall_N * 2)  # Get more for safety
#             elapsed = time.time() - start_time
            
            
#             # Calculate QPS
#             qps = 1.0 / elapsed if elapsed > 0 else 0

#             # Υπολογισμός recall
#             neighbor_ids = [pid for pid, _ in neighbors[:args.recall_N]]
#             hits = sum(1 for pid in neighbor_ids if pid in blast_hits)
#             recall = hits / args.recall_N
#             total_recall += recall
            
#             total_time += elapsed
#             results.append(f"{method_name:16} | {elapsed:12.4f} | {qps:7.1f} | {recall:.3f}")
#             method_neighbors[method_name] = neighbors[:args.N]
       
#         # Προσθήκη BLAST αναφοράς
#         if blast_hits:
#             results.append(f"{'BLAST (Ref)':16} | {'-':12} | {'-':7} | 1.000")
#         results.append("")
        
#         # Λεπτομερή αποτελέσματα ανά μέθοδο
#         for method_name, neighbors in method_neighbors.items():
#             results.append(f"[2] Top-{args.N} γείτονες ({method_name})")
#             results.append("-" * 90)
#             results.append("Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment")
#             results.append("-" * 90)
#             for rank, (neighbor_id, distance) in enumerate(neighbors, 1):
#                 identity = identities.get(neighbor_id, 0.0)
#                 in_blast = "Yes" if neighbor_id in blast_hits else "No"
#                 results.append(f"{rank:4} | {neighbor_id:11} | {distance:7.4f} | {identity:13.1f}% | {in_blast:15} | --")
#             results.append("")
#         results.append("="*80)
#         results.append("")
    
#     # Αποθήκευση  σε αρχειο
#     with open(args.output, "w") as f:
#         f.write("\n".join(results))
#     print(f"Results saved to {args.output}")

#     # 8. Print summary statistics
#     print(f"\n=== Summary Statistics ===")
#     print(f"Total queries processed: {total_queries}")
#     print(f"Database size: {len(db_embeddings)}")
#     print(f"Average query time: {total_time/total_queries:.4f}s")
#     print(f"Average QPS: {total_queries/total_time:.1f}" if total_time > 0 else "Average QPS: 0")
#     print(f"Average Recall@{args.recall_N}: {total_recall/total_queries:.3f}")

# if __name__ == "__main__":
#     main()

# protein_search.py - Updated with IVFFlat support
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
from typing import Dict, List, Tuple, Set, Optional
import warnings
import sys

# Add the parent directory to path to import ANN methods
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Import ANN methods
from ANN.euclidean_lsh import EuclideanLSH
from Hypercube.hypercube import Hypercube
from Neural.neural_lsh import NeuralLSH
from IVFPQ.ivfpq import IVFPQ
from IVFFlat.ivfflat import IVFFlat, IVFFlatConfig

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
    
    # Method parameters
    parser.add_argument("--lsh_k", type=int, default=10, help="LSH: number of hash functions")
    parser.add_argument("--lsh_L", type=int, default=5, help="LSH: number of hash tables")
    parser.add_argument("--lsh_w", type=float, default=4.0, help="LSH: bucket width")
    parser.add_argument("--hypercube_k", type=int, default=10, help="Hypercube: dimension")
    parser.add_argument("--hypercube_M", type=int, default=1000, help="Hypercube: max candidates")
    parser.add_argument("--hypercube_probes", type=int, default=5, help="Hypercube: probes")
    
    # IVF-Flat parameters (new)
    parser.add_argument("--ivfflat_nlist", type=int, default=100, help="IVF-Flat: number of clusters")
    parser.add_argument("--ivfflat_nprobe", type=int, default=10, help="IVF-Flat: clusters to probe")
    parser.add_argument("--ivfflat_use_simple", action="store_true", help="Use simple IVFFlat (no sklearn)")
    
    # IVF-PQ parameters
    parser.add_argument("--ivfpq_nlist", type=int, default=100, help="IVFPQ: number of clusters")
    parser.add_argument("--ivfpq_nprobe", type=int, default=10, help="IVFPQ: clusters to probe")
    parser.add_argument("--ivfpq_m", type=int, default=8, help="IVFPQ: subvectors")
    
    # Neural LSH parameters
    parser.add_argument("--neural_epochs", type=int, default=10, help="Neural LSH: epochs")
    
    # Additional parameters for biological evaluation
    parser.add_argument("--uniprot_info", help="JSON file with UniProt annotations (optional)")
    parser.add_argument("--save_indices", action="store_true", help="Save trained indices to disk")
    parser.add_argument("--load_indices", help="Load pre-trained indices from directory")
    
    return parser.parse_args()

def load_uniprot_info(filepath: str) -> Dict:
    """Load UniProt annotations from JSON file"""
    if not filepath or not os.path.exists(filepath):
        return {}
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading UniProt info: {e}")
        return {}

def get_bio_comment(uniprot_info: Dict, query_id: str, neighbor_id: str, identity: float) -> str:
    """Generate biological comment based on annotations"""
    if identity > 30:
        return "High similarity"
    elif identity > 20:
        comment = f"Twilight Zone ({identity:.1f}%)"
    elif identity > 0:
        comment = f"Remote homolog candidate ({identity:.1f}%)"
    else:
        comment = "No BLAST match"
    
    # Add annotation info if available
    if uniprot_info:
        query_annot = uniprot_info.get(query_id, {})
        neighbor_annot = uniprot_info.get(neighbor_id, {})
        
        # Check for common domains
        if query_annot and neighbor_annot:
            query_domains = set(query_annot.get('domains', []))
            neighbor_domains = set(neighbor_annot.get('domains', []))
            common_domains = query_domains.intersection(neighbor_domains)
            
            if common_domains:
                domain_list = list(common_domains)[:2]
                comment += f" [Common domains: {', '.join(domain_list)}]"
            
            # Check for similar EC numbers
            query_ec = set(query_annot.get('ec_numbers', []))
            neighbor_ec = set(neighbor_annot.get('ec_numbers', []))
            common_ec = query_ec.intersection(neighbor_ec)
            
            if common_ec:
                comment += f" [Common EC: {', '.join(list(common_ec))}]"
            
            # Check for common GO terms
            query_go = set(query_annot.get('go_terms', []))
            neighbor_go = set(neighbor_annot.get('go_terms', []))
            common_go = query_go.intersection(neighbor_go)
            
            if common_go:
                go_list = list(common_go)[:2]
                comment += f" [Common GO: {', '.join(go_list)}]"
    
    return comment

def load_esm_model():
    """Load ESM-2 model for embedding queries"""
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def embed_sequences(sequences: List[Tuple[str, str]], model, batch_converter, max_length: int = 1022) -> Dict[str, np.ndarray]:
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
            reps = token_reps[j, 1:seq_len+1]  # Skip [CLS] token
            embedding = reps.mean(dim=0).numpy()
            embeddings[pid] = embedding
    
    return embeddings

def load_embeddings(file: str) -> Dict[str, np.ndarray]:
    """Load embeddings from .npy file"""
    try:
        data = np.load(file, allow_pickle=True)
        
        if isinstance(data, np.ndarray):
            # If it's a numpy array, check if it's a dictionary
            if data.dtype == object and data.shape == ():
                embeddings = data.item()
            else:
                # Convert array to dict with indices as keys
                embeddings = {f"prot_{i}": emb for i, emb in enumerate(data)}
        elif isinstance(data, dict):
            embeddings = data
        else:
            embeddings = {}
            
        # Ensure all values are numpy arrays
        for key in list(embeddings.keys()):
            if not isinstance(embeddings[key], np.ndarray):
                embeddings[key] = np.array(embeddings[key])
                
        return embeddings
        
    except Exception as e:
        print(f"Error loading embeddings from {file}: {e}")
        return {}

def load_blast_results(blast_file: str, recall_N: int = 50) -> Tuple[Dict[str, Set[str]], Dict[str, Dict[str, float]]]:
    """Load and parse BLAST results"""
    cols = ["query", "subject", "pident", "length", "mismatch", "gapopen",
            "qstart", "qend", "sstart", "send", "evalue", "bitscore"]
    
    try:
        df = pd.read_csv(blast_file, sep="\t", names=cols)
        print(f"Loaded {len(df)} BLAST hits from {blast_file}")
    except Exception as e:
        print(f"Error loading BLAST file: {e}")
        return {}, {}
    
    # Filter by evalue (as specified in assignment)
    df_filtered = df[df["evalue"] < 0.01]
    print(f"After filtering (evalue < 0.01): {len(df_filtered)} hits")
    
    # Get top-N for recall calculation (sorted by bitscore)
    blast_topN = {}
    blast_identity = {}
    
    for query, group in df_filtered.groupby("query"):
        # Sort by bitscore (descending)
        group_sorted = group.sort_values("bitscore", ascending=False)
        
        # Get top-N hits
        top_hits = group_sorted.head(recall_N)
        blast_topN[query] = set(top_hits["subject"].tolist())
        
        # Store identities for all hits
        blast_identity[query] = {}
        for _, row in group_sorted.iterrows():
            blast_identity[query][row["subject"]] = row["pident"]
    
    print(f"Processed {len(blast_topN)} queries for BLAST ground truth")
    return blast_topN, blast_identity

def initialize_methods(args, db_embeddings: Dict[str, np.ndarray], load_dir: Optional[str] = None):
    """Initialize all selected ANN methods"""
    methods = {}
    method_info = {}
    
    # Convert embeddings to appropriate format
    vector_dict = {}
    for pid, vec in db_embeddings.items():
        if not isinstance(vec, np.ndarray):
            vec = np.array(vec)
        vector_dict[pid] = vec
    
    # Euclidean LSH
    if args.method in ["all", "lsh"]:
        print(f"\nInitializing Euclidean LSH...")
        try:
            methods["Euclidean LSH"] = EuclideanLSH(
                vectors=vector_dict,
                k=args.lsh_k,
                L=args.lsh_L,
                w=args.lsh_w,
                seed=42
            )
            method_info["Euclidean LSH"] = f"k={args.lsh_k}, L={args.lsh_L}, w={args.lsh_w}"
            print(f"✓ Euclidean LSH initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Euclidean LSH: {e}")
    
    # Hypercube
    if args.method in ["all", "hypercube"]:
        print(f"\nInitializing Hypercube...")
        try:
            methods["Hypercube"] = Hypercube(
                vectors=vector_dict,
                k=args.hypercube_k,
                M=args.hypercube_M,
                probes=args.hypercube_probes,
                w=args.lsh_w,
                seed=42
            )
            method_info["Hypercube"] = f"k={args.hypercube_k}, M={args.hypercube_M}, probes={args.hypercube_probes}"
            print(f"✓ Hypercube initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Hypercube: {e}")
    
    # IVF-Flat
    if args.method in ["all", "ivfflat"]:
        print(f"\nInitializing IVF-Flat...")
        try:
            if load_dir and os.path.exists(os.path.join(load_dir, "ivfflat_index.pkl")):
                # Load pre-trained index
                methods["IVF-Flat"] = IVFFlat.load_index(os.path.join(load_dir, "ivfflat_index.pkl"))
                print(f"✓ IVF-Flat loaded from file")
            else:
                # Train new index
                config = IVFFlatConfig(
                    n_clusters=args.ivfflat_nlist,
                    n_probe=args.ivfflat_nprobe,
                    random_state=42
                )
                methods["IVF-Flat"] = IVFFlat(config)
                
                # Train the model
                start_time = time.time()
                methods["IVF-Flat"].fit(vector_dict)
                train_time = time.time() - start_time
                
                # Save if requested
                if args.save_indices:
                    os.makedirs("indices", exist_ok=True)
                    methods["IVF-Flat"].save_index("indices/ivfflat_index.pkl")
                    print(f"✓ IVF-Flat index saved")
                
                method_info["IVF-Flat"] = f"nlist={args.ivfflat_nlist}, nprobe={args.ivfflat_nprobe}"
                print(f"✓ IVF-Flat trained in {train_time:.2f}s")
        except Exception as e:
            print(f"✗ Failed to initialize IVF-Flat: {e}")
    
    # IVF-PQ
    if args.method in ["all", "ivfpq"]:
        print(f"\nInitializing IVF-PQ...")
        try:
            methods["IVF-PQ"] = IVFPQ(
                vectors=vector_dict,
                nlist=args.ivfpq_nlist,
                nprobe=args.ivfpq_nprobe,
                m=args.ivfpq_m,
                seed=42
            )
            method_info["IVF-PQ"] = f"nlist={args.ivfpq_nlist}, nprobe={args.ivfpq_nprobe}, m={args.ivfpq_m}"
            print(f"✓ IVF-PQ initialized")
        except Exception as e:
            print(f"✗ Failed to initialize IVF-PQ: {e}")
    
    # Neural LSH
    if args.method in ["all", "neural"]:
        print(f"\nInitializing Neural LSH...")
        try:
            methods["Neural LSH"] = NeuralLSH(
                vectors=vector_dict,
                epochs=args.neural_epochs,
                seed=42
            )
            method_info["Neural LSH"] = f"epochs={args.neural_epochs}"
            print(f"✓ Neural LSH initialized")
        except Exception as e:
            print(f"✗ Failed to initialize Neural LSH: {e}")
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Successfully initialized {len(methods)} methods:")
    for method_name, info in method_info.items():
        print(f"  - {method_name}: {info}")
    print(f"{'='*50}")
    
    return methods

def process_query(query_id: str, query_vec: np.ndarray, methods: Dict, 
                  blast_hits: Set[str], identities: Dict[str, float], 
                  uniprot_info: Dict, recall_N: int, display_N: int) -> List[str]:
    """Process a single query and return formatted output lines"""
    output_lines = []
    
    # Query header (EXACTLY as specified in assignment)
    output_lines.append(f"Query Protein: {query_id}")
    output_lines.append(f"N = {recall_N} (μέγεθος λίστας Top-N για την αξιολόγηση Recall@N)")
    output_lines.append("")
    
    # [1] Summary comparison table
    output_lines.append("[1] Συνοπτική σύγκριση μεθόδων")
    output_lines.append("-" * 70)
    output_lines.append("Method            | Time/query (s) | QPS     | Recall@N vs BLAST Top-N")
    output_lines.append("-" * 70)
    
    method_results = {}
    
    # Evaluate each method for this query
    for method_name, method_obj in methods.items():
        try:
            # Time the search
            start_time = time.time()
            neighbors = method_obj.query(query_vec, recall_N)
            elapsed = time.time() - start_time
            
            # Calculate recall
            neighbor_ids = [pid for pid, _ in neighbors[:recall_N]]
            hits = sum(1 for pid in neighbor_ids if pid in blast_hits)
            
            if blast_hits:
                recall = hits / min(recall_N, len(blast_hits))
            else:
                recall = 0.0
            
            # Calculate QPS for this query
            qps = 1.0 / elapsed if elapsed > 0 else 0
            
            # Format output exactly as in assignment example
            method_line = f"{method_name:16} | {elapsed:12.3f} | {qps:7.1f} | {recall:.3f}"
            output_lines.append(method_line)
            
            # Store results for detailed output
            method_results[method_name] = {
                'neighbors': neighbors[:display_N],
                'time': elapsed,
                'recall': recall,
                'qps': qps
            }
            
        except Exception as e:
            error_line = f"{method_name:16} | ERROR        | ERROR  | ERROR"
            output_lines.append(error_line)
            print(f"Error in {method_name} for query {query_id}: {e}")
            method_results[method_name] = {'neighbors': [], 'time': 0, 'recall': 0, 'qps': 0}
    
    # Add BLAST reference line (as in assignment example)
    output_lines.append(f"{'BLAST (Ref)':16} | {'-':12} | {'-':7} | 1.000")
    output_lines.append("-" * 70)
    output_lines.append("")
    
    # [2] Detailed neighbors per method
    output_lines.append(f"[2] Top-{display_N} γείτονες ανά μέθοδο (εδώ π.χ. N = {display_N} για εκτύπωση)")
    output_lines.append("")
    
    for method_name, results in method_results.items():
        neighbors = results['neighbors']
        if not neighbors:
            continue
        
        output_lines.append(f"Method: {method_name}")
        output_lines.append("-" * 90)
        output_lines.append("Rank | Neighbor ID | L2 Dist | BLAST Identity | In BLAST Top-N? | Bio comment")
        output_lines.append("-" * 90)
        
        for rank, (neighbor_id, distance) in enumerate(neighbors, 1):
            identity = identities.get(neighbor_id, 0.0)
            in_blast = "Yes" if neighbor_id in blast_hits else "No"
            comment = get_bio_comment(uniprot_info, query_id, neighbor_id, identity)
            
            # Format exactly as in assignment example
            output_lines.append(f"{rank:4} | {neighbor_id:11} | {distance:7.3f} | "
                              f"{identity:13.1f}% | {in_blast:15} | {comment}")
        
        output_lines.append("")
    
    output_lines.append("=" * 80)
    output_lines.append("")
    
    return output_lines, method_results

def identify_remote_homologs(query_id: str, method_results: Dict, 
                            identities: Dict[str, float], uniprot_info: Dict,
                            identity_threshold: float = 30.0) -> List[Dict]:
    """Identify potential remote homologs from results"""
    remote_homologs = []
    
    # Collect all unique neighbors from all methods
    all_neighbors = {}
    for method_name, results in method_results.items():
        for neighbor_id, distance in results['neighbors']:
            if neighbor_id not in all_neighbors:
                all_neighbors[neighbor_id] = {
                    'distance': distance,
                    'methods': [method_name],
                    'identity': identities.get(neighbor_id, 0.0)
                }
            else:
                # Keep the smallest distance
                if distance < all_neighbors[neighbor_id]['distance']:
                    all_neighbors[neighbor_id]['distance'] = distance
                all_neighbors[neighbor_id]['methods'].append(method_name)
    
    # Check each neighbor for remote homology criteria
    for neighbor_id, info in all_neighbors.items():
        identity = info['identity']
        
        # Criteria for remote homolog:
        # 1. Low BLAST identity (Twilight Zone)
        # 2. Small distance in embedding space
        # 3. Biological evidence from annotations
        if identity < identity_threshold and info['distance'] < 0.5:  # Distance threshold
            # Check for biological evidence
            query_annot = uniprot_info.get(query_id, {})
            neighbor_annot = uniprot_info.get(neighbor_id, {})
            
            common_domains = False
            common_ec = False
            common_go = False
            
            if query_annot and neighbor_annot:
                # Check domains
                query_domains = set(query_annot.get('domains', []))
                neighbor_domains = set(neighbor_annot.get('domains', []))
                if query_domains.intersection(neighbor_domains):
                    common_domains = True
                
                # Check EC numbers
                query_ec = set(query_annot.get('ec_numbers', []))
                neighbor_ec = set(neighbor_annot.get('ec_numbers', []))
                if query_ec.intersection(neighbor_ec):
                    common_ec = True
                
                # Check GO terms
                query_go = set(query_annot.get('go_terms', []))
                neighbor_go = set(neighbor_annot.get('go_terms', []))
                if query_go.intersection(neighbor_go):
                    common_go = True
            
            # Consider it a potential remote homolog if there's some biological evidence
            if common_domains or common_ec or common_go:
                remote_homologs.append({
                    'id': neighbor_id,
                    'identity': identity,
                    'distance': info['distance'],
                    'methods': info['methods'],
                    'common_domains': common_domains,
                    'common_ec': common_ec,
                    'common_go': common_go
                })
    
    return remote_homologs

def main():
    args = parse_args()
    
    print(f"{'='*60}")
    print(f"PROTEIN SEARCH - Remote Homolog Detection")
    print(f"{'='*60}")
    print(f"Database: {args.database}")
    print(f"Queries: {args.query}")
    print(f"BLAST results: {args.blast_file}")
    print(f"Method: {args.method}")
    print(f"Recall N: {args.recall_N}")
    print(f"Display N: {args.N}")
    print(f"{'='*60}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # 1. Load UniProt annotations if provided
    uniprot_info = {}
    if args.uniprot_info:
        print(f"\nLoading UniProt annotations from {args.uniprot_info}...")
        uniprot_info = load_uniprot_info(args.uniprot_info)
        print(f"✓ Loaded annotations for {len(uniprot_info)} proteins")
    
    # 2. Load database embeddings
    print(f"\nLoading database embeddings from {args.database}...")
    db_embeddings = load_embeddings(args.database)
    if not db_embeddings:
        print(f"Error: Could not load embeddings from {args.database}")
        return
    
    print(f"✓ Loaded {len(db_embeddings)} protein embeddings")
    
    # 3. Load or compute query embeddings
    query_embeddings = {}
    if args.query_embeddings and os.path.exists(args.query_embeddings):
        print(f"\nLoading pre-computed query embeddings from {args.query_embeddings}...")
        query_embeddings = load_embeddings(args.query_embeddings)
    else:
        print(f"\nLoading and embedding queries from {args.query}...")
        try:
            model, batch_converter = load_esm_model()
            queries = [(record.id, str(record.seq)) for record in SeqIO.parse(args.query, "fasta")]
            print(f"Found {len(queries)} query sequences")
            
            query_embeddings = embed_sequences(queries, model, batch_converter, args.max_length)
            
            if args.query_embeddings:
                np.save(args.query_embeddings, query_embeddings)
                print(f"✓ Saved query embeddings to {args.query_embeddings}")
        except Exception as e:
            print(f"Error embedding queries: {e}")
            return
    
    print(f"✓ Loaded {len(query_embeddings)} query embeddings")
    
    # 4. Load BLAST results (Ground Truth)
    print(f"\nLoading BLAST results from {args.blast_file}...")
    blast_topN, blast_identity = load_blast_results(args.blast_file, args.recall_N)
    
    # 5. Initialize ANN methods
    print(f"\nInitializing ANN methods...")
    methods = initialize_methods(args, db_embeddings, args.load_indices)
    
    if not methods:
        print("Error: No ANN methods could be initialized!")
        return
    
    # 6. Process queries
    print(f"\nProcessing {len(query_embeddings)} queries...")
    
    all_results = []
    method_stats = {name: {'total_time': 0.0, 'total_recall': 0.0, 'qps': []} 
                   for name in methods}
    
    # For remote homolog analysis
    all_remote_homologs = []
    
    # Start timing for overall QPS calculation
    overall_start_time = time.time()
    
    # Process each query with progress bar
    query_items = list(query_embeddings.items())
    for query_id, query_vec in tqdm(query_items, desc="Processing queries"):
        # Get BLAST hits for this query
        blast_hits = blast_topN.get(query_id, set())
        identities = blast_identity.get(query_id, {})
        
        # Process the query
        query_output, query_results = process_query(
            query_id, query_vec, methods, blast_hits, identities, 
            uniprot_info, args.recall_N, args.N
        )
        
        all_results.extend(query_output)
        
        # Update statistics
        for method_name, results in query_results.items():
            method_stats[method_name]['total_time'] += results['time']
            method_stats[method_name]['total_recall'] += results['recall']
            method_stats[method_name]['qps'].append(results['qps'])
        
        # Identify remote homologs for this query
        remote_homs = identify_remote_homologs(
            query_id, query_results, identities, uniprot_info
        )
        
        if remote_homs:
            all_remote_homologs.append({
                'query': query_id,
                'remote_homologs': remote_homs
            })
    
    # Calculate overall QPS
    overall_elapsed = time.time() - overall_start_time
    overall_qps = len(query_embeddings) / overall_elapsed if overall_elapsed > 0 else 0
    
    # Add final summary
    all_results.append("\n=== ΣΥΝΟΠΤΙΚΗ ΣΤΑΤΙΣΤΙΚΗ ===")
    all_results.append(f"Συνολικός αριθμός queries: {len(query_embeddings)}")
    all_results.append(f"Μέγεθος βάσης δεδομένων: {len(db_embeddings)}")
    all_results.append(f"Συνολικός χρόνος εκτέλεσης: {overall_elapsed:.2f}s")
    all_results.append(f"Συνολικό QPS: {overall_qps:.1f}")
    all_results.append("")
    
    # Method-wise statistics
    for method_name, stats in method_stats.items():
        if len(query_embeddings) > 0:
            avg_time = stats['total_time'] / len(query_embeddings)
            avg_recall = stats['total_recall'] / len(query_embeddings)
            avg_qps = np.mean(stats['qps']) if stats['qps'] else 0
            
            all_results.append(f"{method_name}:")
            all_results.append(f"  Μέσος χρόνος αναζήτησης: {avg_time:.4f}s")
            all_results.append(f"  Μέσος QPS: {avg_qps:.1f}")
            all_results.append(f"  Μέσος Recall@{args.recall_N}: {avg_recall:.3f}")
            all_results.append("")
    
    # Add remote homolog analysis
    if all_remote_homologs:
        all_results.append("\n=== ΑΝΑΛΥΣΗ ΑΠΟΜΑΚΡΥΣΜΕΝΩΝ ΟΜΟΛΟΓΩΝ ===")
        
        total_remote = sum(len(item['remote_homologs']) for item in all_remote_homologs)
        all_results.append(f"Συνολικός αριθμός υποψήφιων απομακρυσμένων ομόλογων: {total_remote}")
        all_results.append(f"Αριθμός queries με υποψήφιους ομόλογους: {len(all_remote_homologs)}")
        all_results.append("")
        
        # Show top 5 most promising examples
        all_results.append("Παραδείγματα υποψήφιων απομακρυσμένων ομόλογων:")
        all_results.append("-" * 80)
        
        example_count = 0
        for item in all_remote_homologs:
            if example_count >= 5:
                break
                
            query_id = item['query']
            for homolog in item['remote_homologs'][:3]:  # Show up to 3 per query
                all_results.append(f"Query: {query_id} -> {homolog['id']}")
                all_results.append(f"  BLAST Identity: {homolog['identity']:.1f}%")
                all_results.append(f"  Εμβύθιση απόσταση: {homolog['distance']:.3f}")
                all_results.append(f"  Μέθοδοι ανίχνευσης: {', '.join(homolog['methods'])}")
                
                evidence = []
                if homolog['common_domains']:
                    evidence.append("κοινά domains")
                if homolog['common_ec']:
                    evidence.append("κοινά EC numbers")
                if homolog['common_go']:
                    evidence.append("κοινά GO terms")
                
                if evidence:
                    all_results.append(f"  Βιολογικές ενδείξεις: {', '.join(evidence)}")
                
                all_results.append("")
                example_count += 1
    
    # 7. Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_results))
    
    print(f"✓ Results saved to {args.output}")
    
    # 8. Print final summary to console
    print(f"\n{'='*60}")
    print("ΣΥΝΟΠΤΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ")
    print(f"{'='*60}")
    print(f"Συνολικός αριθμός queries: {len(query_embeddings)}")
    print(f"Συνολικός χρόνος: {overall_elapsed:.2f}s")
    print(f"Συνολικό QPS: {overall_qps:.1f}")
    print(f"Μέση απόδοση:")
    
    for method_name, stats in method_stats.items():
        if len(query_embeddings) > 0:
            avg_recall = stats['total_recall'] / len(query_embeddings)
            avg_qps = np.mean(stats['qps']) if stats['qps'] else 0
            print(f"  {method_name}: Recall@{args.recall_N} = {avg_recall:.3f}, QPS = {avg_qps:.1f}")
    
    if all_remote_homologs:
        total_remote = sum(len(item['remote_homologs']) for item in all_remote_homologs)
        print(f"\nΑνίχνευση απομακρυσμένων ομόλογων:")
        print(f"  Υποψήφιοι ομόλογοι: {total_remote}")
        print(f"  Queries με ομόλογους: {len(all_remote_homologs)}")
    
    print(f"\nΤα πλήρη αποτελέσματα έχουν αποθηκευτεί στο: {args.output}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
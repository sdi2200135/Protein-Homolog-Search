# import argparse
# import torch
# import numpy as np
# from Bio import SeqIO
# from tqdm import tqdm
# import esm
# import os

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Generate protein embeddings with ESM-2"
#     )
#     parser.add_argument("-i", "--input", required=True,
#                         help="Input FASTA file")
#     parser.add_argument("-o", "--output", required=True,
#                         help="Output embeddings file (.npy)")
#     parser.add_argument("--batch_size", type=int, default=32,
#                         help="Batch size for processing (default: 32)")
#     parser.add_argument("--max_length", type=int, default=1022,
#                         help="Maximum sequence length (truncate if longer)")
#     return parser.parse_args()

# def load_model():
#     """Load ESM-2 model and batch converter"""
#     print("Loading ESM-2 model (esm2_t6_8M_UR50D)...")
#     model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#     model.eval()
    
#     # Move model to GPU if available
#     if torch.cuda.is_available():
#         print("Using GPU for inference")
#         model = model.cuda()
#     else:
#         print("Using CPU for inference")
    
#     batch_converter = alphabet.get_batch_converter()
#     return model, batch_converter

# def process_batch(batch_data, model, batch_converter, embeddings_dict, max_length):
#     """Process a batch of sequences"""
#     # Prepare batch
#     labels, strs, tokens = batch_converter(batch_data)
    
#     # Move to GPU if available
#     if torch.cuda.is_available():
#         tokens = tokens.cuda()
    
#     # Forward pass
#     with torch.no_grad():
#         output = model(tokens, repr_layers=[6])
#         token_reps = output["representations"][6]  # (batch_size, seq_len, 320)
    
#     # Move back to CPU for numpy conversion
#     token_reps = token_reps.cpu()
    
#     # Process each sequence in the batch
#     for i, (protein_id, original_seq) in enumerate(batch_data):
#         # Handle truncated sequences
#         seq_len = len(original_seq)
#         if seq_len > max_length:
#             seq_len = max_length  # We truncated earlier
        
#         # Get embeddings for actual amino acids (exclude special tokens)
#         # tokens: [CLS] + sequence + [EOS]
#         reps = token_reps[i, 1:seq_len+1]  # Remove CLS token
        
#         # Mean pooling
#         embedding = reps.mean(dim=0).numpy()
        
#         # Store with protein ID
#         embeddings_dict[protein_id] = embedding

# def main():
#     args = parse_args()
    
#     # Check input file exists
#     if not os.path.exists(args.input):
#         print(f"Error: Input file '{args.input}' not found")
#         return
    
#     # Load model
#     model, batch_converter = load_model()
    
#     # Parse FASTA file
#     print(f"Reading sequences from {args.input}...")
#     records = list(SeqIO.parse(args.input, "fasta"))
#     print(f"Loaded {len(records)} proteins")
    
#     # Dictionary to store embeddings {protein_id: embedding_vector}
#     embeddings = {}
    
#     # Prepare batches
#     batch_data = []
    
#     print(f"Generating embeddings (truncating sequences longer than {args.max_length} AA)...")
#     for record in tqdm(records, desc="Processing proteins"):
#         protein_id = record.id
#         sequence = str(record.seq)
        
#         # Truncate if too long (for memory efficiency)
#         if len(sequence) > args.max_length:
#             sequence = sequence[:args.max_length]
#             print(f"  Warning: Truncated {protein_id} from {len(str(record.seq))} to {args.max_length} AA")
        
#         # Add to current batch
#         batch_data.append((protein_id, sequence))
        
#         # Process batch when full
#         if len(batch_data) >= args.batch_size:
#             process_batch(batch_data, model, batch_converter, embeddings, args.max_length)
#             batch_data = []
    
#     # Process remaining sequences
#     if batch_data:
#         process_batch(batch_data, model, batch_converter, embeddings, args.max_length)
    
#     # Save embeddings
#     print(f"Saving {len(embeddings)} embeddings to {args.output}...")
    
#     # Save as numpy dictionary
#     np.save(args.output, embeddings)
    
#     # Also save metadata
#     metadata_file = args.output.replace('.npy', '_metadata.txt')
#     with open(metadata_file, 'w') as f:
#         f.write(f"Total proteins: {len(embeddings)}\n")
#         f.write(f"Embedding dimension: {list(embeddings.values())[0].shape[0]}\n")
#         f.write(f"Max sequence length used: {args.max_length}\n")
#         f.write("\nProtein IDs:\n")
#         for protein_id in embeddings.keys():
#             f.write(f"{protein_id}\n")
    
#     print(f"Embeddings saved to {args.output}")
#     print(f"Metadata saved to {metadata_file}")
    
#     # Print statistics
#     emb_array = np.array(list(embeddings.values()))
#     print("\n=== Embedding Statistics ===")
#     print(f"Shape: {emb_array.shape}")
#     print(f"Mean: {np.mean(emb_array):.4f}")
#     print(f"Std: {np.std(emb_array):.4f}")
#     print(f"Min: {np.min(emb_array):.4f}")
#     print(f"Max: {np.max(emb_array):.4f}")

# if __name__ == "__main__":
#     main()


# protein_embed.py
import argparse
import os
import sys
import time
import warnings
import struct
from pathlib import Path

import numpy as np
import torch
from Bio import SeqIO
from tqdm import tqdm

# Προσπάθεια εισαγωγής του ESM
try:
    import esm
    from esm import pretrained
except ImportError:
    print("Σφάλμα: Το πακέτο esm δεν είναι εγκατεστημένο.")
    print("Εγκατάσταση: pip install fair-esm")
    sys.exit(1)

# Απενεργοποίηση προειδοποιήσεων
warnings.filterwarnings("ignore")

class ProteinEmbedder:
    def __init__(self, model_name="esm2_t6_8M_UR50D", device=None, batch_size=8, max_length=1022):
        """
        Αρχικοποίηση του embedder πρωτεϊνών.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        
        # Αυτόματη επιλογή συσκευής
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Χρήση συσκευής: {self.device}")
        print(f"Φόρτωση μοντέλου {model_name}...")
        
        # Φόρτωση μοντέλου
        self.model, self.alphabet = pretrained.load_model_and_alphabet_hub(model_name)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.batch_converter = self.alphabet.get_batch_converter()
        
        # Πληροφορίες μοντέλου
        self.num_layers = len(self.model.layers)
        self.embedding_dim = self.model.embed_tokens.embedding_dim
        print(f"Μοντέλο φορτώθηκε: {model_name}")
        print(f"Αριθμός επιπέδων: {self.num_layers}")
        print(f"Διάσταση embeddings: {self.embedding_dim}")
    
    def truncate_sequence(self, sequence):
        """Περικόπτει την ακολουθία αν είναι πολύ μεγάλη."""
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return sequence
    
    def embed_batch(self, batch_data):
        """
        Εξάγει embeddings για ένα batch ακολουθιών.
        """
        labels, strs, tokens = self.batch_converter(batch_data)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[self.num_layers])
        
        token_embeddings = results["representations"][self.num_layers]
        
        # Mean pooling με mask για padding
        mask = (tokens != self.alphabet.padding_idx).float()
        mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size())
        
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        token_counts = torch.sum(mask, dim=1, keepdim=True)
        embeddings = sum_embeddings / token_counts.clamp(min=1e-9)
        
        return embeddings.cpu().numpy(), [data[0] for data in batch_data]
    
    def embed_fasta(self, fasta_path):
        """
        Εξάγει embeddings για όλες τις ακολουθίες σε ένα FASTA αρχείο.
        """
        print(f"Ανάγνωση FASTA αρχείου: {fasta_path}")
        
        # Ανάγνωση ακολουθιών
        sequences = []
        total_seqs = 0
        
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_id = record.id
            seq_str = str(record.seq)
            
            # Φιλτράρισμα μη έγκυρων αμινοξέων
            valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
            seq_str = ''.join([aa for aa in seq_str if aa in valid_aa])
            
            if len(seq_str) > 0:
                sequences.append((seq_id, seq_str))
                total_seqs += 1
        
        print(f"Βρέθηκαν {total_seqs} έγκυρες ακολουθίες")
        
        if total_seqs == 0:
            raise ValueError("Δεν βρέθηκαν έγκυρες ακολουθίες")
        
        # Ταξινόμηση κατά μήκος για βέλτιστο padding
        sequences.sort(key=lambda x: len(x[1]))
        
        all_embeddings = []
        all_ids = []
        
        print("Εξαγωγή embeddings...")
        start_time = time.time()
        
        # Batch processing
        for i in tqdm(range(0, len(sequences), self.batch_size), 
                     desc="Processing", unit="batch"):
            batch = sequences[i:i + self.batch_size]
            
            processed_batch = []
            for seq_id, seq_str in batch:
                truncated_seq = self.truncate_sequence(seq_str)
                processed_batch.append((seq_id, truncated_seq))
            
            batch_embeddings, batch_ids = self.embed_batch(processed_batch)
            
            all_embeddings.append(batch_embeddings)
            all_ids.extend(batch_ids)
        
        # Συνένωση
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
        else:
            all_embeddings = np.array([])
        
        elapsed_time = time.time() - start_time
        print(f"Ολοκληρώθηκε σε {elapsed_time:.2f} δευτερόλεπτα")
        print(f"Παράχθηκαν {len(all_ids)} embeddings")
        
        return all_embeddings, all_ids
    
    def save_single_file(self, embeddings, ids, output_path):
        """
        Αποθήκευση embeddings και IDs σε ΕΝΑ δυαδικό αρχείο.
        
        Δομή αρχείου:
        1. Header:
           - Magic number (4 bytes): "ESM2"
           - Version (1 byte): 1
           - Num proteins (4 bytes): N
           - Embedding dim (4 bytes): D
           - Id length (4 bytes): max_id_len
        
        2. IDs section:
           - Για κάθε ID: μήκος (2 bytes) + ID string (UTF-8)
        
        3. Embeddings section:
           - Για κάθε embedding: D floats (32-bit)
        """
        print(f"Αποθήκευση σε ενιαίο αρχείο: {output_path}")
        
        N, D = embeddings.shape
        max_id_len = max(len(id.encode('utf-8')) for id in ids)
        
        print(f"  Αριθμός πρωτεϊνών: {N}")
        print(f"  Διάσταση embedding: {D}")
        print(f"  Μέγεθος embeddings: {embeddings.nbytes / (1024**2):.2f} MB")
        
        with open(output_path, 'wb') as f:
            # 1. HEADER
            f.write(b'ESM2')                     # Magic number (4 bytes)
            f.write(struct.pack('B', 1))         # Version (1 byte)
            f.write(struct.pack('I', N))         # Num proteins (4 bytes)
            f.write(struct.pack('I', D))         # Embedding dim (4 bytes)
            f.write(struct.pack('I', max_id_len)) # Max ID length (4 bytes)
            
            # 2. IDs SECTION
            for seq_id in ids:
                id_bytes = seq_id.encode('utf-8')
                id_len = len(id_bytes)
                f.write(struct.pack('H', id_len))  # Μήκος ID (2 bytes)
                f.write(id_bytes)                  # ID string
                # Padding για σταθερό μήκος (για ευκολία ανάγνωσης)
                if id_len < max_id_len:
                    f.write(b'\0' * (max_id_len - id_len))
            
            # 3. EMBEDDINGS SECTION
            # Αποθήκευση ως float32 για εξοικονόμηση χώρου
            embeddings_float32 = embeddings.astype(np.float32)
            embeddings_float32.tofile(f)
        
        print(f"Αποθηκεύτηκε επιτυχώς στο: {output_path}")
        print(f"Συνολικό μέγεθος αρχείου: {os.path.getsize(output_path) / (1024**2):.2f} MB")

def load_embeddings_single_file(file_path):
    """
    Φόρτωση embeddings από το ενιαίο αρχείο.
    Χρήσιμο για debugging και για το search script.
    """
    print(f"Φόρτωση embeddings από: {file_path}")
    
    with open(file_path, 'rb') as f:
        # Διάβασμα header
        magic = f.read(4)
        if magic != b'ESM2':
            raise ValueError("Μη έγκυρο αρχείο embeddings")
        
        version = struct.unpack('B', f.read(1))[0]
        N = struct.unpack('I', f.read(4))[0]      # Αριθμός πρωτεϊνών
        D = struct.unpack('I', f.read(4))[0]      # Διάσταση
        max_id_len = struct.unpack('I', f.read(4))[0]  # Μέγιστο μήκος ID
        
        print(f"  Αρχείο έκδοσης: {version}")
        print(f"  Πρωτεΐνες: {N}, Διάσταση: {D}")
        
        # Διάβασμα IDs
        ids = []
        for i in range(N):
            id_len = struct.unpack('H', f.read(2))[0]
            id_bytes = f.read(id_len)
            seq_id = id_bytes.decode('utf-8')
            ids.append(seq_id)
            
            # Skip padding
            if id_len < max_id_len:
                f.seek(max_id_len - id_len, 1)
        
        # Διάβασμα embeddings
        embeddings = np.fromfile(f, dtype=np.float32).reshape(N, D)
    
    return embeddings, ids

def main():
    parser = argparse.ArgumentParser(
        description="Εξαγωγή embeddings πρωτεϊνών με ESM-2 - Ένα ενιαίο αρχείο εξόδου"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Είσοδος FASTA αρχείο (π.χ. swissprot.fasta)"
    )
    parser.add_argument(
        "-o", "--output", 
        default="vectors.dat",
        help="Έξοδος δυαδικό αρχείο embeddings (default: vectors.dat)"
    )
    parser.add_argument(
        "--model", 
        default="esm2_t6_8M_UR50D",
        help="Όνομα προεκπαιδευμένου μοντέλου ESM-2 (default: esm2_t6_8M_UR50D)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=8,
        help="Μέγεθος batch (default: 8)"
    )
    parser.add_argument(
        "--max-length", 
        type=int, 
        default=1022,
        help="Μέγιστο μήκος ακολουθίας (default: 1022)"
    )
    parser.add_argument(
        "--device", 
        choices=["cpu", "cuda"], 
        default=None,
        help="Συσκευή (default: auto)"
    )
    parser.add_argument(
        "--test-load",
        action="store_true",
        help="Δοκιμαστική φόρτωση του αρχείου μετά την αποθήκευση"
    )
    
    args = parser.parse_args()
    
    # Έλεγχος αρχείου εισόδου
    if not os.path.exists(args.input):
        print(f"Σφάλμα: Το αρχείο '{args.input}' δεν υπάρχει.")
        sys.exit(1)
    
    # Δημιουργία καταλόγου εξόδου
    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Αρχικοποίηση embedder
        embedder = ProteinEmbedder(
            model_name=args.model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length
        )
        
        # Εξαγωγή embeddings
        embeddings, ids = embedder.embed_fasta(args.input)
        
        # Αποθήκευση σε ενιαίο αρχείο
        embedder.save_single_file(embeddings, ids, args.output)
        
        print("\n" + "="*60)
        print("ΣΥΝΟΨΗ ΑΠΟΘΗΚΕΥΣΗΣ")
        print("="*60)
        print(f"Είσοδος FASTA: {args.input}")
        print(f"Έξοδος ενιαίου αρχείου: {args.output}")
        print(f"Πρωτεΐνες που επεξεργάστηκαν: {len(ids)}")
        print(f"Διάσταση embeddings: {embeddings.shape[1]}")
        print(f"Μέγεθος ενιαίου αρχείου: {os.path.getsize(args.output) / (1024**2):.2f} MB")
        
        # Δοκιμαστική φόρτωση αν ζητήθηκε
        if args.test_load:
            print("\n" + "="*60)
            print("ΔΟΚΙΜΑΣΤΙΚΗ ΦΟΡΤΩΣΗ ΑΡΧΕΙΟΥ")
            print("="*60)
            loaded_embeddings, loaded_ids = load_embeddings_single_file(args.output)
            print(f"Φορτώθηκαν {len(loaded_ids)} embeddings")
            print(f"Πρώτο ID: {loaded_ids[0]}")
            print(f"Πρώτο embedding (5 πρώτες τιμές): {loaded_embeddings[0][:5]}")
            
            # Έλεγχος ορθότητας
            if np.allclose(embeddings, loaded_embeddings, rtol=1e-5):
                print("✓ Τα embeddings φορτώθηκαν σωστά!")
            else:
                print("✗ Υπάρχει διαφορά στα embeddings!")
        
    except Exception as e:
        print(f"Σφάλμα: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
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

# Προσπάθεια εισαγωγής του ESM,αν δεν υπάρχει το πρόγραμμα τερματίζει
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
    # Αρχικοποίηση του embedder πρωτεϊνών
    def __init__(self, model_name="esm2_t6_8M_UR50D", device=None, batch_size=8, max_length=1022):    
        self.model_name = model_name # ονομα εκπαιδευμένου μοντέλου ΕΣΜ-2
        self.batch_size = batch_size # ποσες ακολουθιες επεξεργαζονται ταυτοχρονα 
        self.max_length = max_length #μεγιστο μηκος ακολουθιας
        
        # Αυτόματη επιλογή συσκευής,αν cpu->cuda αλλιως cpu
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Χρήση συσκευής: {self.device}")
        print(f"Φόρτωση μοντέλου {model_name}...")
        
      
        self.model, self.alphabet = pretrained.load_model_and_alphabet_hub(model_name)  # Φόρτωση προεκπαιδευμενου μοντέλου και αλφαβητου
        self.model = self.model.to(self.device) #μεταγορα μοντελου στη συσκευη
        self.model.eval() #μοντελο σε evaluation mode,απενεργοποιει dropout
        
        self.batch_converter = self.alphabet.get_batch_converter() # μετατροπη conventer -> tokens
        
        # Πληροφορίες μοντέλου
        self.num_layers = len(self.model.layers)
        self.embedding_dim = self.model.embed_tokens.embedding_dim
        print(f"Μοντέλο φορτώθηκε: {model_name}")
        print(f"Αριθμός επιπέδων: {self.num_layers}")
        print(f"Διάσταση embeddings: {self.embedding_dim}")
    
    # Περικόπτει ακολουθιες που ξεπερνουν το max_length
    def truncate_sequence(self, sequence):
        if len(sequence) > self.max_length:
            return sequence[:self.max_length]
        return sequence
   
    # Εξάγει embeddings για ένα batch ακολουθιών.  
    def embed_batch(self, batch_data):
        
        labels, strs, tokens = self.batch_converter(batch_data) #μετατροπη ακολουθιων σε tokens
        tokens = tokens.to(self.device)
        
        with torch.no_grad(): #απενεργοποιηση gradients
            results = self.model(tokens, repr_layers=[self.num_layers]) # represantation απο το τελευταιο layer
        
        token_embeddings = results["representations"][self.num_layers]  # tensor διαστασεις(batch,seq_len, embedding_dim) 
        
        # Mean pooling με mask αγνοουμε padding tokens
        mask = (tokens != self.alphabet.padding_idx).float()
        mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size())
        
        sum_embeddings = torch.sum(token_embeddings * mask_expanded, dim=1)
        token_counts = torch.sum(mask, dim=1, keepdim=True)
        embeddings = sum_embeddings / token_counts.clamp(min=1e-9)
        
        return embeddings.cpu().numpy(), [data[0] for data in batch_data] #επιστροφη σε numpy
    
    # Εξάγει embeddings για όλες τις ακολουθίες σε ένα FASTA αρχείο.
    def embed_fasta(self, fasta_path):
        
        print(f"Ανάγνωση FASTA αρχείου: {fasta_path}")
        
        # Ανάγνωση ακολουθιών
        sequences = []
        total_seqs = 0
        
        #αναγνωση FASTA με BioPython
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
        
        # Συνένωση ολων των batches
        if all_embeddings:
            all_embeddings = np.vstack(all_embeddings)
        else:
            all_embeddings = np.array([])
        
        elapsed_time = time.time() - start_time
        print(f"Ολοκληρώθηκε σε {elapsed_time:.2f} δευτερόλεπτα")
        print(f"Παράχθηκαν {len(all_ids)} embeddings")
        
        return all_embeddings, all_ids
    
    # Αποθήκευση embeddings και IDs σε ΕΝΑ δυαδικό αρχείο.
    def save_single_file(self, embeddings, ids, output_path):
        """ 
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
            f.write(struct.pack('I', N))         # αριθμος πρωτεινων (4 bytes)
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

# Φόρτωση embeddings από το ενιαίο αρχείο.
def load_embeddings_single_file(file_path):
   
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
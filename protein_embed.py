# import argparse
# import torch
# import numpy as np
# from Bio import SeqIO
# from tqdm import tqdm
# import esm

# def parse_args():
#     parser = argparse.ArgumentParser(
#         description="Generate protein embeddings with ESM-2"
#     )
#     parser.add_argument("-i", "--input", required=True,
#                         help="Input FASTA file")
#     parser.add_argument("-o", "--output", required=True,
#                         help="Output embeddings file")
#     return parser.parse_args()

# def load_model():
#     model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
#     model.eval()
#     batch_converter = alphabet.get_batch_converter()
#     return model, batch_converter

# def main():
#     args = parse_args()

#     model, batch_converter = load_model()

#     records = list(SeqIO.parse(args.input, "fasta"))
#     print(f"Loaded {len(records)} proteins")

#     embeddings = {}
#     count = 0
#     for i in tqdm(records,desc = "Embedding protein"):
       
#         protein_id = i.id #id πρωτεινης πχ. Q7RY68
#         sequence = str(i.seq) #αλληλουχια MSYE....

#         data = [(protein_id,sequence)]
#         _, _, tokens = batch_converter(data) #μετατροπή αλληλουχιας σε tokens,Μετατρέπει γράμματα → αριθμούς
#         # count +=1
#         # if count == 1 : 
#         #     print(tokens.shape)

#         #περναμε στο μοντελο 
#         with torch.no_grad(): #κανουμε inference ,δεν εκπαιδευουμε
#             output = model(tokens,repr_layers=[6]) #
#             token_reps = output["representations"][6] #(1,252,320)

#         token_reps = token_reps[0,1:len(sequence)+1] #βγαζουμε το CLS + EOS
#         embedding = token_reps.mean(dim=0).cpu().numpy()  #παιρνουμε το vector,mean=παιρνουμε τον μεσο ορο ολων των αμινοξεων,cpu = φερνουμε τα δεδομενα στη μνημη του υπολογιστη, numpy = μετατροπη tensor σε numpy array  
#         embeddings[protein_id]= embedding
    
#     np.save(args.output,embeddings)
#     print(f"Saved {len(embeddings)} embeddings to {args.output}")

# if __name__ == "__main__":
#     main()

# #embeddings = {
# #    "Q7RY68": array([0.12, -0.88, ..., 0.34]),
# #    ...
# #}

import argparse
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import esm
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings with ESM-2"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output embeddings file (.npy)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for processing (default: 32)")
    parser.add_argument("--max_length", type=int, default=1022,
                        help="Maximum sequence length (truncate if longer)")
    return parser.parse_args()

def load_model():
    """Load ESM-2 model and batch converter"""
    print("Loading ESM-2 model (esm2_t6_8M_UR50D)...")
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        print("Using GPU for inference")
        model = model.cuda()
    else:
        print("Using CPU for inference")
    
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def process_batch(batch_data, model, batch_converter, embeddings_dict, max_length):
    """Process a batch of sequences"""
    # Prepare batch
    labels, strs, tokens = batch_converter(batch_data)
    
    # Move to GPU if available
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    
    # Forward pass
    with torch.no_grad():
        output = model(tokens, repr_layers=[6])
        token_reps = output["representations"][6]  # (batch_size, seq_len, 320)
    
    # Move back to CPU for numpy conversion
    token_reps = token_reps.cpu()
    
    # Process each sequence in the batch
    for i, (protein_id, original_seq) in enumerate(batch_data):
        # Handle truncated sequences
        seq_len = len(original_seq)
        if seq_len > max_length:
            seq_len = max_length  # We truncated earlier
        
        # Get embeddings for actual amino acids (exclude special tokens)
        # tokens: [CLS] + sequence + [EOS]
        reps = token_reps[i, 1:seq_len+1]  # Remove CLS token
        
        # Mean pooling
        embedding = reps.mean(dim=0).numpy()
        
        # Store with protein ID
        embeddings_dict[protein_id] = embedding

def main():
    args = parse_args()
    
    # Check input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found")
        return
    
    # Load model
    model, batch_converter = load_model()
    
    # Parse FASTA file
    print(f"Reading sequences from {args.input}...")
    records = list(SeqIO.parse(args.input, "fasta"))
    print(f"Loaded {len(records)} proteins")
    
    # Dictionary to store embeddings {protein_id: embedding_vector}
    embeddings = {}
    
    # Prepare batches
    batch_data = []
    
    print(f"Generating embeddings (truncating sequences longer than {args.max_length} AA)...")
    for record in tqdm(records, desc="Processing proteins"):
        protein_id = record.id
        sequence = str(record.seq)
        
        # Truncate if too long (for memory efficiency)
        if len(sequence) > args.max_length:
            sequence = sequence[:args.max_length]
            print(f"  Warning: Truncated {protein_id} from {len(str(record.seq))} to {args.max_length} AA")
        
        # Add to current batch
        batch_data.append((protein_id, sequence))
        
        # Process batch when full
        if len(batch_data) >= args.batch_size:
            process_batch(batch_data, model, batch_converter, embeddings, args.max_length)
            batch_data = []
    
    # Process remaining sequences
    if batch_data:
        process_batch(batch_data, model, batch_converter, embeddings, args.max_length)
    
    # Save embeddings
    print(f"Saving {len(embeddings)} embeddings to {args.output}...")
    
    # Save as numpy dictionary
    np.save(args.output, embeddings)
    
    # Also save metadata
    metadata_file = args.output.replace('.npy', '_metadata.txt')
    with open(metadata_file, 'w') as f:
        f.write(f"Total proteins: {len(embeddings)}\n")
        f.write(f"Embedding dimension: {list(embeddings.values())[0].shape[0]}\n")
        f.write(f"Max sequence length used: {args.max_length}\n")
        f.write("\nProtein IDs:\n")
        for protein_id in embeddings.keys():
            f.write(f"{protein_id}\n")
    
    print(f"Embeddings saved to {args.output}")
    print(f"Metadata saved to {metadata_file}")
    
    # Print statistics
    emb_array = np.array(list(embeddings.values()))
    print("\n=== Embedding Statistics ===")
    print(f"Shape: {emb_array.shape}")
    print(f"Mean: {np.mean(emb_array):.4f}")
    print(f"Std: {np.std(emb_array):.4f}")
    print(f"Min: {np.min(emb_array):.4f}")
    print(f"Max: {np.max(emb_array):.4f}")

if __name__ == "__main__":
    main()
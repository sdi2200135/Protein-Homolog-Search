import argparse
import torch
import numpy as np
from Bio import SeqIO
from tqdm import tqdm
import esm

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate protein embeddings with ESM-2"
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input FASTA file")
    parser.add_argument("-o", "--output", required=True,
                        help="Output embeddings file")
    return parser.parse_args()

def load_model():
    model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter

def main():
    args = parse_args()

    model, batch_converter = load_model()

    records = list(SeqIO.parse(args.input, "fasta"))
    print(f"Loaded {len(records)} proteins")

    embeddings = {}
    count = 0
    for i in tqdm(records,desc = "Embedding protein"):
       
        protein_id = i.id #id πρωτεινης πχ. Q7RY68
        sequence = str(i.seq) #αλληλουχια MSYE....

        data = [(protein_id,sequence)]
        _, _, tokens = batch_converter(data) #μετατροπή αλληλουχιας σε tokens,Μετατρέπει γράμματα → αριθμούς
        # count +=1
        # if count == 1 : 
        #     print(tokens.shape)

        #περναμε στο μοντελο 
        with torch.no_grad(): #κανουμε inference ,δεν εκπαιδευουμε
            output = model(tokens,repr_layers=[6]) #
            token_reps = output["representations"][6] #(1,252,320)

        token_reps = token_reps[0,1:len(sequence)+1] #βγαζουμε το CLS + EOS
        embedding = token_reps.mean(dim=0).cpu().numpy()  #παιρνουμε το vector,mean=παιρνουμε τον μεσο ορο ολων των αμινοξεων,cpu = φερνουμε τα δεδομενα στη μνημη του υπολογιστη, numpy = μετατροπη tensor σε numpy array  
        embeddings[protein_id]= embedding
    
    np.save(args.output,embeddings)
    print(f"Saved {len(embeddings)} embeddings to {args.output}")

if __name__ == "__main__":
    main()

#embeddings = {
#    "Q7RY68": array([0.12, -0.88, ..., 0.34]),
#    ...
#}

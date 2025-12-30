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

if __name__ == "__main__":
    main()

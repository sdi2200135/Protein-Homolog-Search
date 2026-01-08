# Project_3

python protein_embed.py -i "data-query-sets&pfam-info/swissprot_50k.fasta" -o output.dat

 python search_embeded.py \
  -d output.dat.npy \
  -q "./data-query-sets&pfam-info/targets.fasta" \
  -o results.txt \
  -N 10

# δημιουργια blast database
  makeblastdb -in "./data-query-sets&pfam-info/swissprot_50k.fasta" -dbtype prot -out swissprot_db

# τρεξιμο BLAST για τα queries
 blastp \
  -query "./data-query-sets&pfam-info/targets.fasta" \
  -db swissprot_db \
  -out blast_results.tsv \
  -outfmt 6 \
  -evalue 0.01

# script με LSH+Recall
python ./search_embeded.py   -d protein_vectors.npy   -q "./data-que
ry-sets&pfam-info/targets.fasta"   -o results_lsh.txt   -blast blast_results.tsv   -method lsh   --lsh_k 10  
 --lsh_L 5   --lsh_w 4.0   -N 10   --recall_N 50

# script με Hypercube+Recall
 python ./search_embeded.py \
    -d protein_vectors.npy \
    -q "./data-query-sets&pfam-info/targets.fasta" \
    -o results_hypercube.txt \
    -blast blast_results.tsv \
    -method hypercube \
    -N 10 \
    --recall_N 50


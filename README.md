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
blastp -query "./data-query-sets&pfam-info/targets.fasta" -db swissprot_db -outfmt 6 -evalue 0.01 -out blast_results.txt

# script με LSH+Recall
python search_embeded.py \
  -d output.dat.npy \
  -q "./data-query-sets&pfam-info/targets.fasta" \
  -blast blast_results.txt \
  -o results.txt \
  -N 10

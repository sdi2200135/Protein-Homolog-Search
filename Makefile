.PHONY: all embeddings blast lsh hypercube ivfflat ivfpq neural clean

# Default target - run everything
all: embeddings blast lsh hypercube ivfflat ivfpq neural

# -------------------------------
# 1. Generate Embeddings
# -------------------------------
embeddings:
	python protein_embed.py -i "data-query-sets&pfam-info/swissprot_50k.fasta" -o output.dat
	
# -------------------------------
# 2. Create BLAST database
# -------------------------------
blast:
	mkdir -p blast
	makeblastdb -in "data-query-sets&pfam-info/swissprot_50k.fasta" -dbtype prot -out blast/swissprot_db
	
# -------------------------------
# 3. Run BLAST search
# -------------------------------
blast-search: 
	blastp -query "data-query-sets&pfam-info/targets.fasta" -db blast/swissprot_db -out blast_results.txt -outfmt 6 -evalue 0.01
	
# -------------------------------
# 4. LSH Search
# -------------------------------
lsh: 
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_lsh.txt -blast blast_results.txt \
		-method lsh --lsh_k 10 --lsh_L 5 --lsh_w 4.0 -N 10 --recall_N 50
	
# -------------------------------
# 5. Hypercube Search
# -------------------------------
hypercube: 
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_hypercube.txt \
		-blast blast_results.txt -method hypercube --hypercube_k 10 --hypercube_M 1000 --hypercube_probes 5 -N 10 --recall_N 50

# -------------------------------
# 6. IVFFlat Search
# -------------------------------
ivfflat: 
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_ivfflat.txt -blast blast_results.txt \
		-method ivfflat --ivfflat_nlist 100 --ivfflat_nprobe 10 -N 10 --recall_N 50

# -------------------------------
# 7. IVFPQ Search
# -------------------------------
ivfpq: 
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_ivfpq.txt -blast blast_results.txt \
		-method ivfpq --ivfpq_nlist 100 --ivfpq_nprobe 10 --ivfpq_m 8 -N 10 --recall_N 50

# -------------------------------
# 8. Neural LSH Search
# -------------------------------
neural:
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_neural_m300.txt -blast blast_results.txt \
		-method neural --neural_epochs 20 --neural_k 20 --neural_m 300 --neural_T 10 -N 10 --recall_N 50

# -------------------------------
# 9. Run all ANN methods
# -------------------------------
ann-methods: lsh hypercube ivfflat ivfpq neural

# -------------------------------
# 10. Clean up temporary files
# -------------------------------
clean:
	rm -f blast_results.txt
	rm -f results_*.txt
	rm -rf blast/
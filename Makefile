.PHONY: all embeddings blast lsh hypercube ivfflat ivfpq neural clean

# Default target - run everything
all: embeddings blast lsh hypercube ivfflat ivfpq neural

# -------------------------------
# 1. Generate Embeddings
# -------------------------------
embeddings:
	@echo "Generating protein embeddings using ESM-2..."
	python protein_embed.py -i "data-query-sets&pfam-info/swissprot_50k.fasta" -o output.dat
	@echo "Embeddings generated and saved to output.dat"

# -------------------------------
# 2. Create BLAST database
# -------------------------------
blast:
	@echo "Creating BLAST database..."
	mkdir -p blast
	makeblastdb -in "data-query-sets&pfam-info/swissprot_50k.fasta" -dbtype prot -out blast/swissprot_db
	@echo "BLAST database created in blast/swissprot_db"

# -------------------------------
# 3. Run BLAST search
# -------------------------------
blast-search: blast
	@echo "Running BLASTp search..."
	blastp -query "data-query-sets&pfam-info/targets.fasta" -db blast/swissprot_db -out blast_results.txt -outfmt 6 -evalue 0.01
	@echo "BLAST results saved to blast_results.txt"

# -------------------------------
# 4. LSH Search
# -------------------------------
lsh: embeddings blast-search
	@echo "Running LSH search..."
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_lsh.txt -blast blast_results.txt \
		-method lsh --lsh_k 10 --lsh_L 5 --lsh_w 4.0 -N 10 --recall_N 50
	@echo "LSH results saved to results_lsh.txt"

# -------------------------------
# 5. Hypercube Search
# -------------------------------
hypercube: embeddings blast-search
	@echo "Running Hypercube search..."
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_hypercube.txt \
		-blast blast_results.txt -method hypercube --hypercube_k 10 --hypercube_M 1000 --hypercube_probes 5 -N 10 --recall_N 50
	@echo "Hypercube results saved to results_hypercube.txt"

# -------------------------------
# 6. IVFFlat Search
# -------------------------------
ivfflat: embeddings blast-search
	@echo "Running IVF-Flat search..."
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_ivfflat.txt -blast blast_results.txt \
		-method ivfflat --ivfflat_nlist 100 --ivfflat_nprobe 10 -N 10 --recall_N 50
	@echo "IVF-Flat results saved to results_ivfflat.txt"

# -------------------------------
# 7. IVFPQ Search
# -------------------------------
ivfpq: embeddings blast-search
	@echo "Running IVF-PQ search..."
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_ivfpq.txt -blast blast_results.txt \
		-method ivfpq --ivfpq_nlist 100 --ivfpq_nprobe 10 --ivfpq_m 8 -N 10 --recall_N 50
	@echo "IVF-PQ results saved to results_ivfpq.txt"

# -------------------------------
# 8. Neural LSH Search
# -------------------------------
neural: embeddings blast-search
	@echo "Running Neural LSH search..."
	python protein_search.py -d output.dat -q "data-query-sets&pfam-info/targets.fasta" -o results_neural_m300.txt -blast blast_results.txt \
		-method neural --neural_epochs 20 --neural_k 20 --neural_m 300 --neural_T 10 -N 10 --recall_N 50
	@echo "Neural LSH results saved to results_neural_m300.txt"

# -------------------------------
# 9. Run all ANN methods
# -------------------------------
ann-methods: lsh hypercube ivfflat ivfpq neural
	@echo "All ANN methods executed successfully!"

# -------------------------------
# 10. Clean up temporary files
# -------------------------------
clean:
	@echo "Cleaning up generated files..."
	rm -f output.dat
	rm -f blast_results.txt
	rm -f results_*.txt
	rm -rf blast/
	@echo "Cleanup completed"
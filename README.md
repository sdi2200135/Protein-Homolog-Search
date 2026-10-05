# 🧬 Remote Homolog Search with ANN & ESM-2 Embeddings

> Searching for distant protein homologs using approximate nearest neighbor methods and protein language model embeddings.

![Language](https://img.shields.io/badge/Language-Python%203.10%2B-blue)
![Team](https://img.shields.io/badge/Team-2%20members-green)

---

## About This Project

This is the **3rd Programming Assignment** for the course *"Software Development for Algorithmic Problems"*.

It implements a pipeline for **remote protein homolog search** — finding evolutionarily related proteins
that share little sequence similarity. The approach combines:

- **ESM-2** (Meta's protein language model) to generate dense vector embeddings from protein sequences
- **Five ANN algorithms** to search efficiently over those embeddings
- **BLAST** results as ground truth for evaluation (Recall@N metric)

This assignment unifies all methods from Assignments 1 & 2 into a single bioinformatics application.

🌐 **[View the interactive presentation](https://sdi2200135.github.io/Protein-Homolog-Search/)**

[![Website](https://img.shields.io/badge/Website-Live-brightgreen)](https://sdi2200135.github.io/Protein-Homolog-Search/)

---

## Team

| Name | Student ID |
|------|-----------|
| Παπαθανασίου Ελένη | 1115202200135 |
| Τόντου Αλτάνη-Δάφνη | 1115202200288 |

---

## Methods Implemented & Compared

| Method | Description |
|--------|-------------|
| **Euclidean LSH** | Locality Sensitive Hashing with L2 distance + multi-probe |
| **Hypercube** | Random projection onto hypercube with Hamming distance |
| **IVF-Flat** | Inverted File index with exact distances via k-means |
| **IVF-PQ** | Inverted File + Product Quantization for memory efficiency |
| **Neural LSH** | Hybrid MLP + graph partitioning (KaHIP) for learned indexing |

---

## Tech Stack

- **Python 3.10+** — main language
- **ESM-2** (`fair-esm`) — protein sequence embeddings
- **PyTorch 2.0+** — deep learning backbone (GPU-accelerated)
- **BioPython** — FASTA parsing
- **KaHIP** — graph partitioning for Neural LSH
- **scikit-learn / NumPy / SciPy** — data processing
- **Makefile** — unified pipeline runner

---

## Project Structure

```
📄 protein_embed.py          → Convert FASTA sequences to ESM-2 embeddings
📄 protein_search.py         → Run & compare all 5 ANN methods

📁 ANN/
   └── euclidean_lsh.py      → LSH with Euclidean distance + multi-probe

📁 Hypercube/
   └── hypercube.py          → Hypercube projection with Hamming distance

📁 IVFFlat/
   └── ivfflat.py            → IVF with exact distances + k-means

📁 IVFPQ/
   └── ivfpq.py              → IVF with Product Quantization

📁 Neural/
   ├── neural_lsh.py         → Hybrid MLP + partitioning method
   └── graph_tools/          → Graph utilities for Neural LSH

📄 swissprot.fasta           → Protein database
📄 targets.fasta             → Query proteins
📄 blast_results.txt         → BLAST ground truth for evaluation
📄 requirements.txt
📄 Makefile
```

---

## Installation

### With pip (virtual environment)

```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
# or manually:
pip install torch numpy biopython tqdm scikit-learn pandas
pip install fair-esm
pip install kahip
```

### With Conda (recommended for GPU)

```bash
conda create -n protein_search python=3.11
conda activate protein_search
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy scipy pandas scikit-learn biopython tqdm
pip install fair-esm
```

Verify ESM-2 installation:
```bash
python -c "import esm; print('ESM installed successfully')"
```

---

## Pipeline

### Step 1 — Generate embeddings

```bash
python protein_embed.py -i swissprot.fasta -o protein_vectors.dat \
  --model esm2_t6_8M_UR50D --batch-size 16
```

| Parameter | Description |
|-----------|-------------|
| `-i` | Input FASTA file |
| `-o` | Output binary embeddings file |
| `--model` | Pretrained ESM-2 model variant |
| `--batch-size` | Batch size for processing |

### Step 2 — Search & evaluate

```bash
python protein_search.py -d protein_vectors.dat -q targets.fasta \
  -blast blast_results.txt -o results.txt -method all \
  --recall_N 50 --N 10
```

| Parameter | Description |
|-----------|-------------|
| `-d` | Embeddings database file |
| `-q` | Query FASTA file |
| `-blast` | BLAST results for evaluation |
| `-o` | Output results file |
| `-method` | `all`, `lsh`, `hypercube`, `ivfflat`, `ivfpq`, `neural` |
| `--recall_N` | N for Recall@N computation |
| `--N` | Number of neighbors to return |

### Step 3 — Run individual methods

```bash
# Euclidean LSH
python protein_search.py -method lsh --lsh_k 10 --lsh_L 5 --lsh_w 4.0

# Hypercube
python protein_search.py -method hypercube --hypercube_k 10 --hypercube_M 1000

# IVF-Flat
python protein_search.py -method ivfflat --ivfflat_nlist 100 --ivfflat_nprobe 10

# IVF-PQ
python protein_search.py -method ivfpq --ivfpq_nlist 100 --ivfpq_m 8

# Neural LSH
python protein_search.py -method neural --neural_epochs 10 --neural_k 10
```

---

## Build & Run (Makefile)

```bash
make all            # full pipeline
make embeddings     # generate ESM-2 embeddings
make blast          # run BLAST
make blast-search   # BLAST search
make lsh            # Euclidean LSH search
make hypercube      # Hypercube search
make ivfflat        # IVF-Flat search
make ivfpq          # IVF-PQ search
make neural         # Neural LSH search
make ann-methods    # run all 5 ANN methods
make clean          # remove generated files
```

---

## Sample Results

```
Method            | Time/query (s) | QPS     | Recall@N
Euclidean LSH     | 0.020          | 50.0    | 0.92
Hypercube         | 0.030          | 33.3    | 0.88
IVF-Flat          | 0.008          | 125.0   | 0.93
IVF-PQ            | 0.005          | 200.0   | 0.90
Neural LSH        | 0.010          | 100.0   | 0.95
```

---

## System Requirements

| | Minimum | Recommended |
|---|---------|-------------|
| **Python** | 3.10 | 3.11 |
| **RAM** | 8GB | 16GB+ |
| **GPU** | — | CUDA 11.8+ |
| **Storage** | 5GB | SSD/NVMe |
| **CPU** | Any | Multi-core |

---

## Key Concepts Demonstrated

- Protein language models (ESM-2) for biological sequence embedding
- Bridging bioinformatics (BLAST, FASTA) with modern ML vector search
- Unified benchmarking of 5 ANN algorithms on a real-world domain
- Graph construction, partitioning, and MLP-based learned indexing
- Recall@N evaluation against BLAST ground truth
- End-to-end ML pipeline: embedding → indexing → search → evaluation

---

## Related

- [Assignment 1 — C++ ANN Search (LSH, Hypercube, IVFFlat, IVFPQ)](../Project)
- [Assignment 2 — Neural LSH in Python](../neural-lsh-python)

---

*3rd Programming Assignment · Software Development for Algorithmic Problems*

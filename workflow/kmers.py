import gzip
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import random
import os
import psutil
import pandas as pd
from Bio import SeqIO
from constants import ID_COLUMN



K = 6
VEC_SIZE = 40
W2V_EPOCHS = 5
FASTA_DIR = "/groups/ds/Win-KID/BVBRC/vitek_ii_cleaned/VITEK_cleaned"
INCLUDE_POSITION = False
TRAIN_SUBSET_SIZE = 1000
WINDOW = 5
random.seed(42)


def fasta_to_kmers(fasta_path, k=K):
    kmers = []
    open_func = gzip.open if fasta_path.endswith(".gz") else open
    with open_func(fasta_path, "rt") as handle:
        records = list(SeqIO.parse(handle, "fasta"))
        for record in records:
            seq = str(record.seq).upper()
            kmers.extend([
                seq[i:i+k] for i in range(len(seq)-k+1)
                if set(seq[i:i+k]).issubset({'A', 'C', 'G', 'T'})
            ])
    return kmers

def iter_kmer_sequences(fasta_ids, fasta_dir, k=K):
    """
    Generator that yields one k-mer sequence list at a time.
    Does not store all sequences in memory at once to reduce RAM.
    """
    for sid in fasta_ids:
        fpath = os.path.join(fasta_dir, f"{sid}.fna.gz")
        if os.path.exists(fpath):
            yield fasta_to_kmers(fpath, k)


def log_memory(prefix=""):
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    print(f"[MEM] {prefix} {mem_mb:.1f} MB used")

class KmerCorpus:
    """
    Re-iterable corpus over a batch of FASTA IDs.
    Each call to __iter__ yields lists of k-mers (one sequence per file).
    """
    def __init__(self, fasta_ids, fasta_dir, k):
        self.fasta_ids = fasta_ids
        self.fasta_dir = fasta_dir
        self.k = k

    def __iter__(self):
        for sid in self.fasta_ids:
            fpath = os.path.join(self.fasta_dir, f"{sid}.fna.gz")
            if os.path.exists(fpath):
                yield fasta_to_kmers(fpath, self.k)

def train_word2vec_model_streaming(
    all_fasta_ids,
    fasta_dir,
    min_count=2,
):
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
    
    model = Word2Vec(
        vector_size=VEC_SIZE,
        window=WINDOW,
        min_count=min_count,
        workers=workers,
        sg=0,           # CBOW (lower memory). switching to skip-gram?
        sample=1e-4,
        negative=5
    )
    
    print("📦 Building vocabulary...")
    log_memory("Before vocab build:")
    model.build_vocab(KmerCorpus(all_fasta_ids, fasta_dir, k=K))
    log_memory("After vocab build:")
    print(f"✅ Vocab size: {len(model.wv)} k-mers")
    
    batch_size = 300
    for i in range(0, len(all_fasta_ids), batch_size):
        batch_ids = all_fasta_ids[i:i+batch_size]
        print(f"🚀 Training batch {i//batch_size + 1} ({len(batch_ids)} files)")
        log_memory("Before training batch:")
        
        corpus = KmerCorpus(batch_ids, fasta_dir, k=K)
        model.train(
            corpus,
            total_examples=len(batch_ids),
            epochs=W2V_EPOCHS
        )
        log_memory("After training batch:")
    
    return model


def encode_sample(sample_id, fasta_dir, model, k=K):
    fpath = os.path.join(fasta_dir, f"{sample_id}.fna.gz")
    if not os.path.exists(fpath):
        return None
    emb_vectors = []
    open_func = gzip.open if fpath.endswith(".gz") else open
    with open_func(fpath, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq).upper()
            for i in range(len(seq) - k + 1):
                kmer = seq[i:i+k]
                if set(kmer).issubset({'A', 'C', 'G', 'T'}) and kmer in model.wv:
                    vec = model.wv[kmer]
                    if INCLUDE_POSITION:
                        rel_pos = i / len(seq)
                        vec = np.append(vec, rel_pos)
                    emb_vectors.append(vec)
    if not emb_vectors:
        return None
    return sample_id, np.mean(emb_vectors, axis=0)

def encode_all_samples(fasta_ids, fasta_dir, model, k=K):
    all_vecs = []
    all_ids = []
    for sid in fasta_ids:
        result = encode_sample(sid, fasta_dir, model, k)
        if result:
            sample_id, mean_vec = result
            all_ids.append(sample_id)
            all_vecs.append(mean_vec)
    if not all_vecs:
        raise ValueError("No k-mer embeddings generated!")
    cols = [f'kmer_{i}' for i in range(len(all_vecs[0]))]
    df = pd.DataFrame(all_vecs, columns=cols)
    df[ID_COLUMN] = all_ids
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


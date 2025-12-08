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
from constants import W2V
import logging

logger = logging.getLogger(__name__)


def log_memory(prefix=""):
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    logger.debug(f"[MEM] {prefix} {mem_mb:.1f} MB used")


def fasta_to_kmers(fasta_path, k=W2V.K_SIZE):
    open_func = gzip.open if fasta_path.endswith(".gz") else open
    with open_func(fasta_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq).upper()
            valid = {"A", "C", "G", "T"}
            contig_kmers = []
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if set(kmer).issubset(valid):
                    contig_kmers.append(kmer)
            if contig_kmers:
                yield contig_kmers


def save_w2v_model(model, output_path):
    """Save a trained Word2Vec model"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    logger.info(f"💾 Saved Word2Vec model to: {output_path}")


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
            if not os.path.exists(fpath):
                logger.warning(f"❌ FASTA file missing: {fpath} — skipping")
                continue
            yield from fasta_to_kmers(fpath, self.k)


def train_word2vec_model_streaming(
    all_fasta_ids,
    fasta_dir,
    min_count=2,
    save_path=None,
):
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    model = Word2Vec(
        vector_size=W2V.VEC_SIZE,
        window=W2V.WINDOW,
        min_count=min_count,  # all 'words' with frequency < min_count are ignored do we want that?
        workers=workers,
        sg=0,  # CBOW (lower memory). switching to skip-gram (sg=1)?
        sample=1e-4,
        negative=5,
        seed=42,
    )
    # hs=1 to use herachical softmax?
    # negative=int -> If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drown.use?

    logger.info("📦 Building vocabulary...")
    log_memory("Before vocab build:")
    model.build_vocab(
        KmerCorpus(all_fasta_ids, fasta_dir, k=W2V.K_SIZE), progress_per=1000
    )
    log_memory("After vocab build:")
    logger.info(f"✅ Vocab size: {len(model.wv)} k-mers")

    for i in range(0, len(all_fasta_ids), W2V.BATCH_SIZE):
        batch_ids = all_fasta_ids[i : i + W2V.BATCH_SIZE]
        logger.info(f"Training batch {i//W2V.BATCH_SIZE + 1} ({len(batch_ids)} files)")
        log_memory("Before training batch:")

        corpus = KmerCorpus(batch_ids, fasta_dir, k=W2V.K_SIZE)
        num_examples = sum(1 for _ in corpus)  # genaue Anzahl der Sätze
        model.train(corpus, total_examples=num_examples, epochs=W2V.W2V_EPOCHS)
        log_memory("After training batch:")

    if save_path:
        save_w2v_model(model, save_path)

    return model


def encode_sample(sample_id, fasta_dir, model, k=W2V.K_SIZE):
    fpath = os.path.join(fasta_dir, f"{sample_id}.fna.gz")
    if not os.path.exists(fpath):
        logger.warning(f"❌ Missing FASTA during encoding: {fpath}")
        return None
    sample_sum = None
    sample_count = 0
    open_func = gzip.open if fpath.endswith(".gz") else open
    with open_func(fpath, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq).upper()
            contig_sum = None
            contig_count = 0
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if set(kmer).issubset({"A", "C", "G", "T"}) and kmer in model.wv:
                    vec = model.wv[kmer]

                    if W2V.INCLUDE_POSITION:
                        rel_pos = i / len(seq)
                        vec = np.concatenate((vec, [rel_pos]))

                    if contig_sum is None:
                        contig_sum = vec.astype(np.float64)
                    else:
                        contig_sum += vec
                    contig_count += 1

            if contig_count > 0:
                contig_mean = contig_sum / contig_count
                if sample_sum is None:
                    sample_sum = contig_mean
                else:
                    sample_sum += contig_mean
                sample_count += 1
    if sample_count == 0:
        return None
    return sample_id, (sample_sum / sample_count)


def encode_all_samples(fasta_ids, fasta_dir, model, k=W2V.K_SIZE):
    all_vecs = []
    all_ids = []
    skipped = 0
    for sid in fasta_ids:
        result = encode_sample(sid, fasta_dir, model, k)
        if result:
            sample_id, mean_vec = result
            all_ids.append(sample_id)
            all_vecs.append(mean_vec)
        else:
            skipped += 1
    logger.info(
        f" Encoding completed: {len(all_ids)} samples encoded, {skipped} skipped"
    )
    if not all_vecs:
        raise ValueError("No k-mer embeddings generated!")
    cols = [f"kmer_{i}" for i in range(len(all_vecs[0]))]
    df = pd.DataFrame(all_vecs, columns=cols)
    df[ID_COLUMN] = all_ids
    scaler = MinMaxScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df


def load_w2v_model(model_path):
    """Load a Word2Vec model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Word2Vec model not found: {model_path}")

    model = Word2Vec.load(model_path)
    logger.info(f"📥 Loaded Word2Vec model from: {model_path}")
    return model

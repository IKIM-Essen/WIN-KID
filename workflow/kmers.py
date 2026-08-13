import gzip
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
import os
import psutil
import pandas as pd
from Bio import SeqIO
from constants import ID_COLUMN
from constants import W2V_SETTINGS
import logging
import joblib
from config import CONCAT_CONTIGS, MAX_CONTIGS

logger = logging.getLogger(__name__)


def log_memory(prefix=""):
    process = psutil.Process()
    mem_mb = process.memory_info().rss / 1024**2
    logger.debug(f"[MEM] {prefix} {mem_mb:.1f} MB used")


def fasta_to_kmers(fasta_path, k):
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


def load_w2v_model(model_path):
    """Load a Word2Vec model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Word2Vec model not found: {model_path}")

    model = Word2Vec.load(model_path)
    logger.info(f"📥 Loaded Word2Vec model from: {model_path}")
    return model


def save_pca_model(pca, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(pca, output_path)
    logger.info(f"💾 Saved PCA model to: {output_path}")


def load_pca_model(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    pca = joblib.load(path)
    logger.info(f"📥 Loaded PCA model from: {path}")
    return pca


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
    w2v_settings,
    save_path=None,
):
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

    model = Word2Vec(
        vector_size=w2v_settings.vec_size,
        window=w2v_settings.window,
        min_count=w2v_settings.min_count,  # all 'words' with frequency < min_count are ignored do we want that?
        workers=workers,
        sg=w2v_settings.sg,  # CBOW (lower memory). switching to skip-gram (sg=1)?
        sample=w2v_settings.sample,
        negative=w2v_settings.negative,
        seed=w2v_settings.seed,
    )
    # hs=1 to use herachical softmax?
    # negative=int -> If > 0, negative sampling will be used, the int for negative specifies how many "noise words" should be drown.use?

    logger.info("📦 Building vocabulary...")
    log_memory("Before vocab build:")
    model.build_vocab(
        KmerCorpus(all_fasta_ids, fasta_dir, k=w2v_settings.k_size), progress_per=10000
    )
    log_memory("After vocab build:")
    logger.info(f"✅ Vocab size: {len(model.wv)} k-mers")

    for i in range(0, len(all_fasta_ids), w2v_settings.batch_size):
        batch_ids = all_fasta_ids[i : i + w2v_settings.batch_size]
        logger.info(
            f"Training batch {i//w2v_settings.batch_size + 1} ({len(batch_ids)} files)"
        )
        log_memory("Before training batch:")

        corpus = KmerCorpus(batch_ids, fasta_dir, k=w2v_settings.k_size)
        num_examples = sum(1 for _ in corpus)
        model.train(corpus, total_examples=num_examples, epochs=w2v_settings.w2v_epochs)
        log_memory("After training batch:")

    if save_path:
        save_w2v_model(model, save_path)

    return model


def encode_sample(sample_id, fasta_dir, model, w2v_settings):
    k = w2v_settings.k_size
    fpath = os.path.join(fasta_dir, f"{sample_id}.fna.gz")
    if not os.path.exists(fpath):
        logger.warning(f"❌ Missing FASTA during encoding: {fpath}")
        return None
    open_func = gzip.open if fpath.endswith(".gz") else open
    contig_vectors = []
    with open_func(fpath, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq = str(record.seq).upper()
            contig_sum = None
            contig_count = 0
            for i in range(len(seq) - k + 1):
                kmer = seq[i : i + k]
                if set(kmer).issubset({"A", "C", "G", "T"}) and kmer in model.wv:
                    vec = model.wv[kmer]

                    if w2v_settings.include_position:
                        rel_pos = i / len(seq)
                        vec = np.concatenate((vec, [rel_pos]))

                    if contig_sum is None:
                        contig_sum = vec.astype(np.float64)
                    else:
                        contig_sum += vec
                    contig_count += 1

            if contig_count > 0:
                contig_mean = contig_sum / contig_count

                contig_vectors.append(
                    (len(seq), contig_mean)  # contig length  # embedding
                )
    if len(contig_vectors) == 0:
        return None

    ###################################################################
    # OLD METHOD
    ###################################################################

    if not CONCAT_CONTIGS:

        vectors = [vec for _, vec in contig_vectors]

        sample_vec = np.mean(vectors, axis=0)

        return sample_id, sample_vec

    ###################################################################
    # NEW METHOD
    ###################################################################

    contig_vectors.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    vectors = [vec for _, vec in contig_vectors[:MAX_CONTIGS]]

    embedding_size = len(vectors[0])

    while len(vectors) < MAX_CONTIGS:

        vectors.append(
            np.zeros(
                embedding_size,
                dtype=np.float32,
            )
        )

    sample_vec = np.concatenate(vectors)

    if CONCAT_CONTIGS:
        logger.info(
            f"Using concatenated contig embeddings " f"(max contigs={MAX_CONTIGS})"
        )
    else:
        logger.info("Using mean embedding per sample")

    return sample_id, sample_vec


def encode_all_samples(fasta_ids, fasta_dir, model, w2v_settings):
    all_vecs = []
    all_ids = []
    skipped = 0
    for sid in fasta_ids:
        result = encode_sample(sid, fasta_dir, model, w2v_settings)
        if result:
            sample_id, sample_vec = result
            all_ids.append(sample_id)
            all_vecs.append(sample_vec)
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
    return df

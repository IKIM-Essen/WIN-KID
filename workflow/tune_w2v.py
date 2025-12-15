
import os
import random
import logging
import numpy as np
from kmers import (
    train_word2vec_model_streaming,
    encode_all_samples
)
from sklearn.model_selection import ParameterGrid, ParameterSampler
from copy import deepcopy

logger = logging.getLogger(__name__)

def evaluate_w2v_model(model, val_ids, fasta_dir):
    """
    Simple evaluation metric:
    - use reconstruction stability / embedding variance
    - OR average L2 norm magnitude
    - OR downstream proxy task later
    """
    try:
        df = encode_all_samples(val_ids, fasta_dir, model)
        return df.drop(columns=["sample_id"], errors="ignore").values.std()
    except Exception as e:
        logger.warning(f"Evaluation failed: {e}")
        return -np.inf


def tune_word2vec(
    all_fasta_ids,
    fasta_dir,
    W2V_SETTINGS,
    n_samples=10,
    use_random_search=True
):
    logger.info("🔍 Starting Word2Vec hyperparameter tuning...")

    # --- Split into train/val ---
    random.shuffle(all_fasta_ids)
    val_size = max(1, int(0.1 * len(all_fasta_ids)))
    val_ids = all_fasta_ids[:val_size]
    train_ids = all_fasta_ids[val_size:]

    # --- Hyperparameter search space ---
    search_space = {
        "vec_size": [50, 100, 200],
        "window": [5, 10, 20],
        "negative": [5, 10, 20],
        "sg": [0, 1],
        "min_count": [1, 2, 5],
    }

    if use_random_search:
        configs = list(ParameterSampler(search_space, n_iter=n_samples, random_state=42))
    else:
        configs = list(ParameterGrid(search_space))

    logger.info(f"🔎 Tuning over {len(configs)} configurations")

    best_score = -np.inf
    best_model = None
    best_cfg = None

    for idx, cfg in enumerate(configs):
        logger.info(f"⚙️ Training W2V configuration {idx+1}/{len(configs)}: {cfg}")

        # Make a copy of settings
        tuned_settings = deepcopy(W2V_SETTINGS)
        for k, v in cfg.items():
            setattr(tuned_settings, k, v)

        # Train model
        model = train_word2vec_model_streaming(
            train_ids, fasta_dir, tuned_settings, save_path=None
        )

        # Evaluate model on validation set
        score = evaluate_w2v_model(model, val_ids, fasta_dir)
        logger.info(f"   → Score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model
            best_cfg = cfg

    logger.info(f"🏆 Best config: {best_cfg}  (Score={best_score:.4f})")
    return best_model

#Step 2: encode every sentence with a sentence transformer and save the vectors.



import argparse
import json
import os

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from data import load_collection

EMBED_MODEL = "all-MiniLM-L6-v2"  # 384 dimensions, ~5x faster than all-mpnet-base-v2
INDEX_DIR = "index"


def device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build(max_articles):
    paragraphs, sentences = load_collection(max_articles)
    print(f"{len(paragraphs)} paragraphs, {len(sentences)} sentences")

    model = SentenceTransformer(EMBED_MODEL, device=device())
    vecs = model.encode(
        [s["text"] for s in sentences],
        batch_size=256,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float16)
    print(f"Embedding matrix: {vecs.shape} (dimensions = {vecs.shape[1]})")

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(f"{INDEX_DIR}/sentence_vectors.npy", vecs)
    with open(f"{INDEX_DIR}/paragraphs.json", "w") as f:
        json.dump(paragraphs, f)
    with open(f"{INDEX_DIR}/sentences.json", "w") as f:
        json.dump(sentences, f)


def load():
    vecs = np.load(f"{INDEX_DIR}/sentence_vectors.npy")
    with open(f"{INDEX_DIR}/paragraphs.json") as f:
        paragraphs = json.load(f)
    with open(f"{INDEX_DIR}/sentences.json") as f:
        sentences = json.load(f)
    return vecs, paragraphs, sentences


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-articles", type=int, default=20000,
                    help="use a subset while developing; 0 = whole corpus (~170k articles)")
    args = ap.parse_args()
    build(args.max_articles or None)

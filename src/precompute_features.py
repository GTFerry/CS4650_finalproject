import argparse
import json
import random
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from data import load_all, split, MAX_SENTS

OUT = Path("data/features")


def mean_sbert(texts, model_name="all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    from nltk.tokenize import sent_tokenize

    model = SentenceTransformer(model_name)
    embeddings = []
    for text in texts:
        sentences = sent_tokenize(text)[:MAX_SENTS]
        if not sentences:
            sentences = [text[:200]]
        sentence_embeddings = model.encode(sentences, batch_size=128, show_progress_bar=False)
        embeddings.append(sentence_embeddings.mean(axis=0))
    return np.array(embeddings, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--tfidf-vocab", type=int, default=5000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading records...")
    records = load_all()
    random.seed(42)
    train_records = split(records, "train")
    if len(train_records) > args.max_train:
        train_records = random.sample(train_records, args.max_train)
    val_records = split(records, "val")
    test_records = split(records, "test")
    split_records = {"train": train_records, "val": val_records, "test": test_records}
    print(f"  train={len(train_records)} val={len(val_records)} test={len(test_records)}")

    for split_name, records_for_split in split_records.items():
        labels = np.array([record["label"] for record in records_for_split], dtype=np.int64)
        financial_features = np.nan_to_num(
            np.array([record["financial"] for record in records_for_split], dtype=np.float32),
            0.0,
        )
        np.save(OUT / f"labels_{split_name}.npy", labels)
        np.save(OUT / f"financial_{split_name}.npy", financial_features)

    print(f"Fitting TF-IDF (vocab={args.tfidf_vocab})...")
    tfidf = TfidfVectorizer(max_features=args.tfidf_vocab, sublinear_tf=True,
                            stop_words="english", ngram_range=(1, 2))
    X_tr = tfidf.fit_transform([record["text"] for record in train_records])
    sparse.save_npz(OUT / "tfidf_train.npz", X_tr)
    for name in ("val", "test"):
        X = tfidf.transform([record["text"] for record in split_records[name]])
        sparse.save_npz(OUT / f"tfidf_{name}.npz", X)

    print("Computing frozen SBERT mean embeddings...")
    for split_name, records_for_split in split_records.items():
        out_path = OUT / f"sbert_{split_name}.npy"
        if out_path.exists():
            continue
        print(f"  {split_name} ({len(records_for_split)})...")
        np.save(out_path, mean_sbert([record["text"] for record in records_for_split]))

    meta = {"train_size": len(train_records), "val_size": len(val_records), "test_size": len(test_records),
            "tfidf_vocab": args.tfidf_vocab, "sbert_dim": 384}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()

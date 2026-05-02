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
    out = []
    for t in texts:
        sents = sent_tokenize(t)[:MAX_SENTS]
        if not sents:
            sents = [t[:200]]
        e = model.encode(sents, batch_size=128, show_progress_bar=False)
        out.append(e.mean(axis=0))
    return np.array(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train", type=int, default=3000)
    ap.add_argument("--tfidf-vocab", type=int, default=5000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    print("Loading records...")
    records = load_all()
    random.seed(42)
    train = split(records, "train")
    if len(train) > args.max_train:
        train = random.sample(train, args.max_train)
    val  = split(records, "val")
    test = split(records, "test")
    splits = {"train": train, "val": val, "test": test}
    print(f"  train={len(train)} val={len(val)} test={len(test)}")

    for name, recs in splits.items():
        labels = np.array([r["label"] for r in recs], dtype=np.int64)
        fin    = np.nan_to_num(np.array([r["financial"] for r in recs], dtype=np.float32), 0.0)
        np.save(OUT / f"labels_{name}.npy", labels)
        np.save(OUT / f"financial_{name}.npy", fin)

    print(f"Fitting TF-IDF (vocab={args.tfidf_vocab})...")
    tfidf = TfidfVectorizer(max_features=args.tfidf_vocab, sublinear_tf=True,
                            stop_words="english", ngram_range=(1, 2))
    X_tr = tfidf.fit_transform([r["text"] for r in train])
    sparse.save_npz(OUT / "tfidf_train.npz", X_tr)
    for name in ("val", "test"):
        X = tfidf.transform([r["text"] for r in splits[name]])
        sparse.save_npz(OUT / f"tfidf_{name}.npz", X)

    print("Computing frozen SBERT mean embeddings...")
    for name, recs in splits.items():
        out_path = OUT / f"sbert_{name}.npy"
        if out_path.exists():
            continue
        print(f"  {name} ({len(recs)})...")
        np.save(out_path, mean_sbert([r["text"] for r in recs]))

    meta = {"train_size": len(train), "val_size": len(val), "test_size": len(test),
            "tfidf_vocab": args.tfidf_vocab, "sbert_dim": 384}
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()

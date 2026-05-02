import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB

FEAT = Path("data/features")
OUT = Path("checkpoints/baselines")


def load(split):
    labels = np.load(FEAT / f"labels_{split}.npy")
    financial = np.load(FEAT / f"financial_{split}.npy")
    tfidf = sparse.load_npz(FEAT / f"tfidf_{split}.npz")
    sbert = np.load(FEAT / f"sbert_{split}.npy")
    return labels, financial, tfidf, sbert


def evaluate(name, y, pred, t):
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    acc = (y == pred).mean()
    print(f"  {name:<40} f1={f1:.4f}  acc={acc:.4f}  ({t:.1f}s)")
    return {"model": name, "val_macro_f1": round(float(f1), 4),
            "val_acc": round(float(acc), 4), "train_time_s": round(t, 1)}


def fit_eval(name, clf, Xtr, ytr, Xvl, yvl):
    start_time = time.time()
    clf.fit(Xtr, ytr)
    return evaluate(name, yvl, clf.predict(Xvl), time.time() - start_time)


def mlp_sbert_fin(Xtr, ytr, Xvl, yvl, epochs=30, patience=5):
    import torch, torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.utils.class_weight import compute_class_weight

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_dim = Xtr.shape[1]
    num_classes = len(np.unique(ytr))
    model = nn.Sequential(
        nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, num_classes),
    ).to(device)
    class_weights = compute_class_weight("balanced", classes=np.arange(num_classes), y=ytr)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float).to(device))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    loader = DataLoader(TensorDataset(torch.tensor(Xtr, dtype=torch.float),
                                      torch.tensor(ytr, dtype=torch.long)),
                        batch_size=64, shuffle=True)
    validation_features = torch.tensor(Xvl, dtype=torch.float).to(device)

    best_score = -1.0
    best_state = None
    patience_counter = 0
    start_time = time.time()
    for _ in range(epochs):
        model.train()
        for batch_features, batch_labels in loader:
            opt.zero_grad()
            crit(model(batch_features.to(device)), batch_labels.to(device)).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            predictions = model(validation_features).argmax(1).cpu().numpy()
        score = f1_score(yvl, predictions, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = model(validation_features).argmax(1).cpu().numpy()
    return evaluate("MLP (SBERT + financial)", yvl, predictions, time.time() - start_time)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    y_tr, fin_tr, tfidf_tr, sbert_tr = load("train")
    y_vl, fin_vl, tfidf_vl, sbert_vl = load("val")

    results = []
    print("Baselines:")
    results.append(fit_eval("Majority class", DummyClassifier(strategy="most_frequent"), tfidf_tr, y_tr, tfidf_vl, y_vl))
    results.append(fit_eval("LR (financial only)", LogisticRegression(max_iter=1000), fin_tr, y_tr, fin_vl, y_vl))
    results.append(fit_eval("Complement NB (TF-IDF)", ComplementNB(alpha=0.1), tfidf_tr, y_tr, tfidf_vl, y_vl))
    results.append(fit_eval("LR L2 (TF-IDF)", LogisticRegression(max_iter=1000, solver="saga"), tfidf_tr, y_tr, tfidf_vl, y_vl))

    tfidf_fin_tr = sparse.hstack([tfidf_tr, sparse.csr_matrix(fin_tr)])
    tfidf_fin_vl = sparse.hstack([tfidf_vl, sparse.csr_matrix(fin_vl)])
    results.append(fit_eval("LR (TF-IDF + financial)", LogisticRegression(max_iter=1000, solver="saga"), tfidf_fin_tr, y_tr, tfidf_fin_vl, y_vl))

    sbert_fin_tr = np.hstack([sbert_tr, fin_tr])
    sbert_fin_vl = np.hstack([sbert_vl, fin_vl])
    results.append(mlp_sbert_fin(sbert_fin_tr, y_tr, sbert_fin_vl, y_vl))

    print("\nSummary:")
    for r in sorted(results, key=lambda r: -r["val_macro_f1"]):
        print(f"  {r['model']:<40} {r['val_macro_f1']:.4f}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

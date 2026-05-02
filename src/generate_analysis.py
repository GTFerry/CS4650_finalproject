import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EarningsDataset, NUM_CLASSES, load_all, split, split_sentences
from model import TranscriptModel, collate

CLASSES = ["Down", "Flat", "Up"]
FIG_DIR = Path("figures")


def load_model(run, device):
    ckpt = torch.load(Path("checkpoints") / run / "best.pt", map_location=device)
    cfg = ckpt.get("cfg", {})
    model = TranscriptModel(
        num_classes=NUM_CLASSES,
        unfreeze_last=cfg.get("unfreeze_last", 2),
        sbert_chunk=cfg.get("sbert_chunk", 64),
        lstm_hidden=cfg.get("lstm_hidden", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, cfg


def evaluate_test(run, model, cfg, device):
    test_records = split(load_all(), "test")
    print(f"test: {len(test_records)} records")
    loader = DataLoader(
        EarningsDataset(test_records),
        batch_size=cfg.get("batch", 4),
        shuffle=False,
        collate_fn=collate,
    )
    predictions = []
    labels = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval test"):
            logits = model(
                batch["sentences"],
                batch["sent_counts"].to(device),
                batch["financial"].to(device),
            )
            predictions.extend(logits.argmax(1).cpu().tolist())
            labels.extend(batch["labels"].tolist())

    y_true = np.array(labels)
    y_pred = np.array(predictions)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    acc = (y_true == y_pred).mean()
    print(f"\nTest macro F1: {f1:.4f}  Accuracy: {acc:.4f}")
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    out = {"run": run, "split": "test",
           "macro_f1": round(float(f1), 4),
           "accuracy": round(float(acc), 4),
           "n": len(test_records),
           "confusion_matrix": cm.tolist()}
    (Path("checkpoints") / run / "test_results.json").write_text(json.dumps(out, indent=2))


def plot_training_curve(run):
    history = json.loads((Path("checkpoints") / run / "history.json").read_text())
    epochs = [h["epoch"] for h in history]
    tr = [h["tr_f1"] for h in history]
    vl = [h["vl_f1"] for h in history]
    best = max(range(len(vl)), key=lambda i: vl[i])

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(epochs, tr, label="train", linewidth=1.6)
    ax.plot(epochs, vl, label="val", linewidth=1.6)
    ax.scatter([epochs[best]], [vl[best]], color="red", zorder=5,
               label=f"best val F1={vl[best]:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Training curve ({run})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"{run}_training_curve.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}")


def plot_attention(run, model, device):
    validation_records = split(load_all(), "val")
    random.seed(7)
    sample_size = min(100, len(validation_records))
    sampled_records = random.sample(validation_records, sample_size)

    best = None
    with torch.no_grad():
        for record in sampled_records:
            batch = collate([record])
            logits, attention_weights = model(
                batch["sentences"],
                batch["sent_counts"].to(device),
                batch["financial"].to(device),
                return_attention=True,
            )
            pred = logits.argmax(1).item()
            if pred != record["label"]:
                continue
            n = batch["sent_counts"][0].item()
            weights = attention_weights[0, :n].cpu().numpy()
            top_share = float(weights[np.argsort(weights)[-10:]].sum())
            if best is None or abs(record["return"]) > abs(best["return"]):
                best = {
                    "record": record,
                    "weights": weights,
                    "pred": pred,
                    "top10_share": top_share,
                    "return": record["return"],
                }

    if best is None:
        raise RuntimeError("Could not find a correctly predicted validation example for attention plotting.")

    r = best["record"]
    weights = best["weights"]
    sents = split_sentences(r["text"])[:len(weights)]
    n = len(weights)
    uniform = 1.0 / n

    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    ax.bar(range(n), weights, width=1.0, color="#3b78b3", alpha=0.85)
    ax.axhline(uniform, color="red", linestyle="--", linewidth=1.0,
               label=f"uniform = 1/{n} = {uniform:.4f}")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Attention weight")
    ax.set_title(
        f"{r['ticker']} {r['date'].date()}  return={r['return']:+.3f}  "
        f"true={CLASSES[r['label']]}  pred={CLASSES[best['pred']]}",
        fontsize=9,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out = FIG_DIR / f"{run}_attention.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  wrote {out}  (top-10 share = {best['top10_share']:.3f}, "
          f"uniform-10 = {10 * uniform:.3f})")

    # also print the top-5 attended sentences for the report
    top5 = np.argsort(weights)[-5:][::-1]
    print(f"\nTop-5 attended sentences for {r['ticker']} {r['date'].date()}:")
    for i in top5:
        s = sents[i].replace("\n", " ")[:120]
        print(f"  [{weights[i]:.4f}] {s}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--skip-eval", action="store_true")
    args = p.parse_args()

    FIG_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt, cfg = load_model(args.run, device)
    print(f"Loaded {args.run}: epoch={ckpt['epoch']} val_f1={ckpt['val_f1']:.4f}")

    if not args.skip_eval:
        evaluate_test(args.run, model, cfg, device)

    print("\nGenerating training curve...")
    plot_training_curve(args.run)

    print("Generating attention figure...")
    plot_attention(args.run, model, device)


if __name__ == "__main__":
    main()

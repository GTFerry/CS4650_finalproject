import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import EarningsDataset, NUM_CLASSES, load_all, split
from model import TranscriptModel, collate


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_single_epoch(model, loader, criterion, optimizer, scheduler, device, description, use_amp):
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0
    all_predictions = []
    all_labels = []

    if use_amp:
        autocast_context = lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        autocast_context = lambda: torch.cuda.amp.autocast(enabled=False)

    with torch.set_grad_enabled(is_training):
        for batch in tqdm(loader, desc=description):
            with autocast_context():
                logits = model(
                    batch["sentences"],
                    batch["sent_counts"].to(device),
                    batch["financial"].to(device),
                )
                y = batch["labels"].to(device)
                loss = criterion(logits, y)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            all_predictions.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(batch["labels"].tolist())

    accuracy = sum(pred == label for pred, label in zip(all_predictions, all_labels)) / len(all_labels)
    macro_f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    return total_loss / len(loader), accuracy, macro_f1


def build_dataloaders(train_records, val_records, batch_size, num_workers):
    train_loader = DataLoader(
        EarningsDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        EarningsDataset(val_records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=num_workers,
    )
    return train_loader, val_loader


def build_model(args, device):
    model = TranscriptModel(
        num_classes=NUM_CLASSES,
        unfreeze_last=args.unfreeze_last,
        sbert_chunk=args.sbert_chunk,
        lstm_hidden=args.lstm_hidden,
        lstm_layers=args.lstm_layers,
        dropout=args.dropout,
    ).to(device)
    return model


def build_optimizer(model, sbert_learning_rate, head_learning_rate):
    return torch.optim.AdamW([
        {"params": model.sbert_params(), "lr": sbert_learning_rate},
        {"params": model.other_params(), "lr": head_learning_rate},
    ], weight_decay=1e-2)


def build_scheduler(optimizer, train_loader, epochs):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"]],
        steps_per_epoch=len(train_loader),
        epochs=epochs,
    )


def write_json_file(path, data):
    path.write_text(json.dumps(data, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--sbert-lr", type=float, default=3e-6)
    p.add_argument("--unfreeze-last", type=int, default=2)
    p.add_argument("--sbert-chunk", type=int, default=256)
    p.add_argument("--lstm-hidden", type=int, default=256)
    p.add_argument("--lstm-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--max-train", type=int, default=None,
                   help="Cap training sample count (random subsample, seed=42)")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", help="bf16 autocast (H100/A100 only)")
    args = p.parse_args()

    device = get_device()
    out_dir = Path("checkpoints") / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}  Run: {args.run}")

    print("Loading records...")
    records = load_all()
    train_rec = split(records, "train")
    val_rec = split(records, "val")

    if args.max_train and len(train_rec) > args.max_train:
        import random
        random.seed(42)
        train_rec = random.sample(train_rec, args.max_train)
    print(f"  train={len(train_rec)}  val={len(val_rec)}")

    train_loader, val_loader = build_dataloaders(
        train_rec,
        val_rec,
        args.batch,
        args.num_workers,
    )

    y_train = [r["label"] for r in train_rec]
    cw = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=y_train)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))

    model = build_model(args, device)

    n_sb = sum(p.numel() for p in model.sbert_params())
    n_ot = sum(p.numel() for p in model.other_params())
    print(f"Trainable: SBERT={n_sb:,}  head={n_ot:,}  total={n_sb + n_ot:,}")

    optimizer = build_optimizer(model, args.sbert_lr, args.lr)
    scheduler = build_scheduler(optimizer, train_loader, args.epochs)

    use_amp = args.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()
    if args.amp and not use_amp:
        print("Warning: bf16 unavailable on this GPU; running fp32.")

    cfg = vars(args)
    write_json_file(out_dir / "config.json", cfg)

    best_f1, patience, history = -1.0, 0, []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_l, tr_a, tr_f = run_single_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            f"ep{epoch:02d} tr",
            use_amp,
        )
        vl_l, vl_a, vl_f = run_single_epoch(
            model,
            val_loader,
            criterion,
            None,
            None,
            device,
            f"ep{epoch:02d} vl",
            use_amp,
        )
        elapsed = time.time() - t0
        print(f"epoch {epoch:02d}  tr loss={tr_l:.3f} f1={tr_f:.3f}  "
              f"vl loss={vl_l:.3f} f1={vl_f:.3f}  ({elapsed:.0f}s)")
        history.append({
            "epoch": epoch,
            "tr_loss": tr_l,
            "tr_acc": tr_a,
            "tr_f1": tr_f,
            "vl_loss": vl_l,
            "vl_acc": vl_a,
            "vl_f1": vl_f,
        })
        write_json_file(out_dir / "history.json", history)

        if vl_f > best_f1:
            best_f1, patience = vl_f, 0
            torch.save({
                "epoch": epoch,
                "val_f1": vl_f,
                "model_state": model.state_dict(),
                "cfg": cfg,
                "num_classes": NUM_CLASSES,
            }, out_dir / "best.pt")
            print(f"  saved new best (val f1={vl_f:.3f})")
        else:
            patience += 1
            if patience >= args.patience:
                print("Early stopping.")
                break

    print(f"Done. best val f1={best_f1:.3f}")


if __name__ == "__main__":
    main()

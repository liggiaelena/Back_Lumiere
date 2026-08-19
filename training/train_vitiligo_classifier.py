"""Train a face-level healthy/vitiligo classification gate."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loaders(root: Path, batch_size: int, workers: int):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.72, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.1, 0.04),
        transforms.RandomRotation(8),
        transforms.ToTensor(), normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])
    result = {}
    for split, tf, shuffle in (("train", train_tf, True), ("val", eval_tf, False), ("test", eval_tf, False)):
        dataset = datasets.ImageFolder(root / split, transform=tf)
        if dataset.class_to_idx != {"healthy": 0, "vitiligo": 1}:
            raise ValueError(f"Unexpected classes: {dataset.class_to_idx}")
        result[split] = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                   num_workers=workers, pin_memory=True,
                                   persistent_workers=workers > 0)
    return result


@torch.inference_mode()
def predict(model, loader, device):
    model.eval(); labels, probabilities = [], []
    for images, target in loader:
        logits = model(images.to(device, non_blocking=True))
        probabilities.extend(logits.softmax(1)[:, 1].cpu().tolist())
        labels.extend(target.tolist())
    return np.asarray(labels), np.asarray(probabilities)


def metrics(labels, probabilities, threshold):
    pred = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, pred, average="binary", zero_division=0)
    return {
        "threshold": float(threshold), "accuracy": float((pred == labels).mean()),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "false_positive_rate": float(fp / max(tn + fp, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def choose_threshold(labels, probabilities, max_fpr):
    candidates = np.unique(np.r_[0.5, np.linspace(0.01, 0.99, 197), probabilities])
    rows = [metrics(labels, probabilities, value) for value in candidates]
    feasible = [row for row in rows if row["false_positive_rate"] <= max_fpr]
    pool = feasible or rows
    return max(pool, key=lambda row: (row["f1"], row["recall"], row["threshold"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-validation-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    seed_everything(args.seed); args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = loaders(args.dataset, args.batch_size, args.workers)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(nn.Dropout(0.25), nn.Linear(model.fc.in_features, 2))
    model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history, best_auc = [], -1.0
    for epoch in range(1, args.epochs + 1):
        model.train(); loss_sum = 0.0; seen = 0
        for images, target in data["train"]:
            images = images.to(device, non_blocking=True); target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = criterion(model(images), target)
            scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * images.size(0); seen += images.size(0)
        scheduler.step()
        y, p = predict(model, data["val"], device)
        row = {"epoch": epoch, "train_loss": loss_sum / seen, **metrics(y, p, 0.5)}
        history.append(row); print(json.dumps(row), flush=True)
        if row["roc_auc"] > best_auc:
            best_auc = row["roc_auc"]
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "best.pt")
    checkpoint = torch.load(args.output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    val_y, val_p = predict(model, data["val"], device)
    selected = choose_threshold(val_y, val_p, args.max_validation_fpr)
    test_y, test_p = predict(model, data["test"], device)
    report = {"device": str(device), "best_epoch": checkpoint["epoch"],
              "validation": selected, "test": metrics(test_y, test_p, selected["threshold"]),
              "history": history, "classes": {"healthy": 0, "vitiligo": 1}}
    (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.output / "deployment.json").write_text(json.dumps({
        "architecture": "resnet18", "classes": {"healthy": 0, "vitiligo": 1},
        "vitiligo_threshold": selected["threshold"], "input_size": 224,
        "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
    }, indent=2), encoding="utf-8")
    print(json.dumps({"completed": True, **report["test"]}), flush=True)


if __name__ == "__main__":
    main()

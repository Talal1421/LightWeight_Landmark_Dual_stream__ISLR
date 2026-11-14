"""Train the lightweight sign-language recognition model."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import FEATURE_SET_CHOICES, NORMALIZATION_CHOICES, SignLanguageFeatureDataset, collate_fn
    from .model import LightweightSignLanguageModel, count_trainable_parameters, model_size_mb
except ImportError:
    from dataset import FEATURE_SET_CHOICES, NORMALIZATION_CHOICES, SignLanguageFeatureDataset, collate_fn
    from model import LightweightSignLanguageModel, count_trainable_parameters, model_size_mb


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def load_num_classes(config_path: Path) -> int:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return int(config["num_classes"])


def resolve_csv(csv_path: Path | None, splits_dir: Path, split_name: str) -> Path:
    path = csv_path if csv_path is not None else splits_dir / f"{split_name}.csv"
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return path


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "train",
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.set_grad_enabled(is_train):
        for features, lengths, labels in tqdm(loader, desc=desc, leave=False):
            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(features, lengths)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint_safely(
    checkpoint: dict,
    checkpoint_path: Path,
    retries: int = 12,
    delay_seconds: float = 1.0,
) -> Path:
    """Save a checkpoint without crashing on transient Windows file locks."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(f".{checkpoint_path.name}.{uuid.uuid4().hex}.tmp")
    torch.save(checkpoint, temp_path)

    last_error: OSError | None = None
    for attempt in range(retries):
        try:
            os.replace(temp_path, checkpoint_path)
            return checkpoint_path
        except OSError as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(delay_seconds)

    epoch = int(checkpoint.get("epoch", 0))
    val_acc = float(checkpoint.get("val_acc", 0.0))
    fallback_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_epoch{epoch:03d}_acc{val_acc:.4f}_{uuid.uuid4().hex[:8]}{checkpoint_path.suffix}"
    )
    os.replace(temp_path, fallback_path)
    print(
        f"Warning: could not replace {checkpoint_path} after {retries} attempts "
        f"({last_error}). Kept checkpoint at {fallback_path}."
    )
    return fallback_path


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    val_acc: float,
    num_classes: int,
    config_path: Path,
    normalization: str,
    feature_set: str,
    input_landmarks: int,
    input_channels: int,
    add_missing_mask: bool,
) -> dict:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "num_classes": num_classes,
        "config_path": str(config_path),
        "normalization": normalization,
        "feature_set": feature_set,
        "input_landmarks": input_landmarks,
        "input_channels": input_channels,
        "add_missing_mask": add_missing_mask,
    }


def checkpoint_name_for_epoch(best_name: str, epoch: int) -> str:
    path = Path(best_name)
    return f"{path.stem}_epoch{epoch:03d}{path.suffix or '.pt'}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the GRU-attention sign-language model.")
    parser.add_argument("--features_dir", type=Path, default=Path("features"))
    parser.add_argument("--splits_dir", type=Path, default=Path("splits"))
    parser.add_argument("--config_path", type=Path, default=Path("configs") / "labels.json")
    parser.add_argument("--checkpoint_dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--train_csv", type=Path, default=None)
    parser.add_argument("--val_csv", type=Path, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="validation")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--best_name", type=str, default="best_model.pt")
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=100,
        help="Save an additional epoch checkpoint every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="none",
        choices=sorted(NORMALIZATION_CHOICES),
        help="Feature normalization applied during loading.",
    )
    parser.add_argument(
        "--feature_set",
        type=str,
        default="full",
        choices=sorted(FEATURE_SET_CHOICES),
        help="Feature subset to train on. no_face drops pose face landmarks and hand-to-face features.",
    )
    parser.add_argument(
        "--add_missing_mask",
        action="store_true",
        help="Append a 4th channel indicating whether each landmark block was detected by MediaPipe.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    features_dir = args.features_dir.resolve()
    splits_dir = args.splits_dir.resolve()
    config_path = args.config_path.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_csv = resolve_csv(args.train_csv, splits_dir, args.train_split)
    val_csv = resolve_csv(args.val_csv, splits_dir, args.val_split)

    train_dataset = SignLanguageFeatureDataset(
        train_csv,
        features_dir=features_dir,
        normalization=args.normalization,
        feature_set=args.feature_set,
        add_missing_mask=args.add_missing_mask,
    )
    val_dataset = SignLanguageFeatureDataset(
        val_csv,
        features_dir=features_dir,
        normalization=args.normalization,
        feature_set=args.feature_set,
        add_missing_mask=args.add_missing_mask,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    num_classes = load_num_classes(config_path)
    device = resolve_device(args.device)
    input_landmarks = int(train_dataset.feature_shape[0])
    input_channels = int(train_dataset.feature_shape[1])
    model = LightweightSignLanguageModel(
        num_classes=num_classes,
        input_landmarks=input_landmarks,
        input_channels=input_channels,
    ).to(device)

    print(f"Model parameters: {count_trainable_parameters(model):,}")
    print(f"Model size: {model_size_mb(model):.2f} MB")
    print(f"Device: {device}")
    print(f"Normalization: {args.normalization}")
    print(f"Feature set: {args.feature_set} ({input_landmarks}x{input_channels} features per frame)")
    print(f"Missing mask channel: {args.add_missing_mask}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    best_val_acc = -1.0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, desc=f"epoch {epoch} train")
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None, desc=f"epoch {epoch} val")

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = checkpoint_dir / args.best_name
            checkpoint = build_checkpoint(
                epoch,
                model,
                optimizer,
                val_acc,
                num_classes,
                config_path,
                args.normalization,
                args.feature_set,
                input_landmarks,
                input_channels,
                args.add_missing_mask,
            )
            saved_path = save_checkpoint_safely(
                checkpoint,
                checkpoint_path,
            )
            print(f"Saved best checkpoint to {saved_path} (val_acc={val_acc:.4f})")

        if args.checkpoint_interval > 0 and epoch % args.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / checkpoint_name_for_epoch(args.best_name, epoch)
            checkpoint = build_checkpoint(
                epoch,
                model,
                optimizer,
                val_acc,
                num_classes,
                config_path,
                args.normalization,
                args.feature_set,
                input_landmarks,
                input_channels,
                args.add_missing_mask,
            )
            saved_path = save_checkpoint_safely(checkpoint, checkpoint_path)
            print(f"Saved epoch checkpoint to {saved_path} (epoch={epoch:03d}, val_acc={val_acc:.4f})")

    write_history(checkpoint_dir / "training_history.csv", history)
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()

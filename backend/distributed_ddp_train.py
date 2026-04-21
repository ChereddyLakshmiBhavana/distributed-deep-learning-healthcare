"""
Distributed DDP CNN Trainer
Launch with torchrun to train the pneumonia CNN using torch.distributed + DDP.

Example:
    torchrun --standalone --nproc_per_node=2 backend/distributed_ddp_train.py
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


class PneumoniaCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return self.fc(x)


class SyntheticXrayDataset(Dataset):
    def __init__(self, num_samples=1000, image_size=224):
        self.num_samples = num_samples
        self.image_size = image_size
        self.labels = torch.randint(0, 2, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.labels[idx] == 1:
            image = torch.randn(1, self.image_size, self.image_size) * 0.3 + 0.7
        else:
            image = torch.randn(1, self.image_size, self.image_size) * 0.2 + 0.5
        image = torch.clamp(image, 0, 1)
        return image, self.labels[idx]


def is_distributed():
    return all(key in os.environ for key in ("RANK", "WORLD_SIZE", "LOCAL_RANK")) and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed():
    distributed = is_distributed()
    rank = 0
    world_size = 1
    local_rank = 0
    device = torch.device("cpu")

    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        backend = "nccl" if torch.cuda.is_available() else "gloo"

        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cpu")

    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return distributed, rank, world_size, local_rank, device


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return running_loss / max(1, len(loader)), correct / max(1, total)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return running_loss / max(1, len(loader)), correct / max(1, total)


def main():
    distributed, rank, world_size, local_rank, device = setup_distributed()
    is_main_process = rank == 0

    torch.manual_seed(42)
    np.random.seed(42)

    model = PneumoniaCNN(num_classes=2).to(device)
    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs = {"device_ids": [local_rank], "output_device": local_rank}
        model = DDP(model, **ddp_kwargs)

    train_dataset = SyntheticXrayDataset(num_samples=800, image_size=224)
    test_dataset = SyntheticXrayDataset(num_samples=200, image_size=224)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True) if distributed else None
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False) if distributed else None

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, sampler=test_sampler)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    num_epochs = 10
    if is_main_process:
        mode_label = "DDP" if distributed else "single-process"
        print(f"Starting CNN training in {mode_label} mode on {device}")

    for epoch in range(num_epochs):
        if distributed and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if is_main_process:
            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
            )

    if is_main_process:
        print("Training complete, evaluating model...")

    all_preds = []
    all_labels = []
    model_to_eval = unwrap_model(model)
    model_to_eval.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    if is_main_process:
        print("\nDDP CNN Evaluation Results:")
        print("=" * 60)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        print(classification_report(all_labels, all_preds, target_names=["NORMAL", "PNEUMONIA"]))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = MODELS_DIR / "ddp_cnn_model.pth"
        torch.save(
            {
                "model_state_dict": model_to_eval.state_dict(),
                "history": history,
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "distributed_training": distributed,
                "world_size": world_size,
                "created_at": timestamp,
            },
            checkpoint_path,
        )

        info_path = MODELS_DIR / "ddp_cnn_model_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "model_type": "CNN-DDP",
                    "accuracy": float(accuracy),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1_score": float(f1),
                    "distributed_training": distributed,
                    "world_size": world_size,
                    "launch_command": "torchrun --standalone --nproc_per_node=<gpus> backend/distributed_ddp_train.py",
                },
                f,
                indent=2,
            )

        print(f"\n✓ Saved DDP checkpoint to: {checkpoint_path}")
        print(f"✓ Saved metadata to: {info_path}")

    cleanup_distributed()


if __name__ == "__main__":
    main()

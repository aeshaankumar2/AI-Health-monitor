#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import json
import math
import copy
import random
import argparse
import warnings
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models, utils as tvutils

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    SKLEARN_OK = True
except Exception as e:
    SKLEARN_OK = False
    warnings.warn("scikit-learn not available. Some reports will be simplified.")


def set_seed(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

@dataclass
class TrainConfig:
    data_dir: str
    train_subdir: str = "train"
    val_subdir: str = "val"
    img_size: int = 224
    batch_size: int = 32
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5               
    lr_patience: int = 2             
    lr_factor: float = 0.5
    num_workers: int = 4
    seed: int = 42
    amp: bool = True
    model_name: str = "resnet18"     
    freeze_backbone: bool = False
    output_dir: str = "outputs"
    grad_cam_samples: int = 8
    benchmark_warmup: int = 5
    benchmark_iters: int = 50
    prune_amount: float = 0.3        
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"

def makedirs(path: str):
    os.makedirs(path, exist_ok=True)

def count_params(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def human_bytes(n: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path)
    print(f"Saved model to {path} ({human_bytes(size)})")
    return size


def build_transforms(cfg: TrainConfig):
    train_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)], p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(cfg.img_size, scale=(0.85, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf

def build_dataloaders(cfg: TrainConfig):
    train_tf, val_tf = build_transforms(cfg)
    train_path = os.path.join(cfg.data_dir, cfg.train_subdir)
    val_path = os.path.join(cfg.data_dir, cfg.val_subdir)

    train_ds = datasets.ImageFolder(train_path, transform=train_tf)
    val_ds = datasets.ImageFolder(val_path, transform=val_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader, class_names, num_classes


def create_backbone(cfg: TrainConfig, num_classes: int) -> nn.Module:
    if cfg.model_name.lower() == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif cfg.model_name.lower() == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {cfg.model_name}")

    if cfg.freeze_backbone:
        for name, p in model.named_parameters():
            if "fc" not in name and "classifier" not in name:
                p.requires_grad = False
    return model


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()

def run_one_epoch(
    model: nn.Module, loader: DataLoader, criterion, optimizer, device: str, scaler: Optional[GradScaler], train: bool, use_amp: bool
) -> Tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    running_correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if use_amp and scaler is not None:
                with autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                if train:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (outputs.argmax(1) == labels).sum().item()
        total += batch_size

    epoch_loss = running_loss / max(1, total)
    epoch_acc = running_correct / max(1, total)
    return epoch_loss, epoch_acc

def train_model(cfg: TrainConfig, model: nn.Module, train_loader, val_loader, class_names: List[str]):
    device = cfg.device
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.lr_factor, patience=cfg.lr_patience
    )
    scaler = GradScaler(enabled=cfg.amp)

    best_weights = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0
    best_val_loss = math.inf
    patience_counter = 0

    history: List[EpochStats] = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True, use_amp=cfg.amp
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer, device, scaler=None, train=False, use_amp=False
        )
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        history.append(EpochStats(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc,
            val_loss=val_loss, val_acc=val_acc, lr=current_lr
        ))

        print(f"[Epoch {epoch:02d}/{cfg.epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}% | "
              f"LR: {current_lr:.3e} | Time: {dt:.1f}s")

        if val_loss < best_val_loss - 1e-4 or val_acc > best_val_acc + 1e-4:
            best_val_loss = min(best_val_loss, val_loss)
            best_val_acc = max(best_val_acc, val_acc)
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            print("  ↳ New best model found. Saving checkpoint...")
            save_model(model, os.path.join(cfg.output_dir, "checkpoint_best.pth"))
        else:
            patience_counter += 1
            print(f"  ↳ No improvement. EarlyStopping patience {patience_counter}/{cfg.patience}")
            if patience_counter >= cfg.patience:
                print("  ↳ Early stopping triggered.")
                break

    model.load_state_dict(best_weights)
    save_model(model, os.path.join(cfg.output_dir, "model_final.pth"))

    with open(os.path.join(cfg.output_dir, "history.json"), "w") as f:
        json.dump([asdict(h) for h in history], f, indent=2)

    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}%")
    return model, history


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str, class_names: List[str]) -> Dict:
    model.eval()
    all_logits, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0)
    preds = logits.argmax(1)
    acc = (preds == labels).float().mean().item()

    results = {"accuracy": acc}
    if SKLEARN_OK:
        y_true = labels.numpy()
        y_pred = preds.numpy()
        report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
        results["classification_report"] = report

        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
    else:
        num_classes = len(class_names)
        prec_list, rec_list, f1_list = [], [], []
        for c in range(num_classes):
            tp = int(((preds == c) & (labels == c)).sum())
            fp = int(((preds == c) & (labels != c)).sum())
            fn = int(((preds != c) & (labels == c)).sum())
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            prec_list.append(precision); rec_list.append(recall); f1_list.append(f1)
        results["macro_precision"] = float(np.mean(prec_list))
        results["macro_recall"] = float(np.mean(rec_list))
        results["macro_f1"] = float(np.mean(f1_list))

    return results

def plot_confusion_matrices(cm: np.ndarray, class_names: List[str], outdir: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           title='Confusion Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    path_raw = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(path_raw, dpi=160)
    plt.close(fig)

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(cm_norm, interpolation='nearest', cmap='Greens')
    ax2.figure.colorbar(im2, ax2)
    ax2.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           ylabel='True label',
           title='Confusion Matrix (Normalized)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax2.text(j, i, f"{cm_norm[i, j]*100:.1f}%",
                     ha="center", va="center",
                     color="white" if cm_norm[i, j] > 0.5 else "black")
    fig2.tight_layout()
    path_norm = os.path.join(outdir, "confusion_matrix_normalized.png")
    plt.savefig(path_norm, dpi=160)
    plt.close(fig2)

    print(f"Saved confusion matrices:\n  - {path_raw}\n  - {path_norm}")

class GradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer_name]
        self.activations = None
        self.gradients = None
        self.hook_handles = [
            self.target_layer.register_forward_hook(self._forward_hook),
            self.target_layer.register_full_backward_hook(self._backward_hook)
        ]

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, class_idx: Optional[int], logits: torch.Tensor):
        grads = self.gradients 
        acts = self.activations  
        weights = grads.mean(dim=(2, 3), keepdim=True)  
        cam = (weights * acts).sum(dim=1, keepdim=True)  
        cam = F.relu(cam)
        cam_min = cam.view(cam.size(0), -1).min(dim=1)[0].view(-1,1,1,1)
        cam_max = cam.view(cam.size(0), -1).max(dim=1)[0].view(-1,1,1,1)
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-9)
        return cam 

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()

def make_gradcam_samples(model: nn.Module, loader: DataLoader, device: str,
                         class_names: List[str], outdir: str,
                         target_layer_name: str = "layer4"):

    model.eval()
    cam_maker = GradCAM(model, target_layer_name)
    saved = 0
    max_save = min(16, len(loader.dataset))

    torch.set_grad_enabled(True)

    for imgs, labels in loader:
        imgs = imgs.to(device)
        imgs.requires_grad_(True)         

        logits = model(imgs)
        preds = logits.argmax(1)

        loss = logits.gather(1, preds.view(-1, 1)).sum()
        model.zero_grad(set_to_none=True)
        loss.backward(retain_graph=True)

        cam = cam_maker(class_idx=None, logits=logits).cpu()
        imgs_cpu = imgs.detach().cpu()

        for i in range(imgs_cpu.size(0)):
            if saved >= max_save:
                cam_maker.remove_hooks()
                print(f"Grad-CAM: saved {saved} samples to {outdir}")
                torch.set_grad_enabled(False)
                return

            img = denormalize_image(imgs_cpu[i])
            heat = cam[i, 0].numpy()
            overlay = overlay_heatmap(img, heat)

            gt = class_names[labels[i].item()]
            pd = class_names[preds[i].item()]
            path = os.path.join(outdir, f"gradcam_{saved:02d}_gt-{gt}_pred-{pd}.png")

            plt.imsave(path, overlay)
            saved += 1

    cam_maker.remove_hooks()
    torch.set_grad_enabled(False)
    print(f"Grad-CAM: saved {saved} samples to {outdir}")


def denormalize_image(t: torch.Tensor) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406]).reshape(3,1,1)
    std = np.array([0.229, 0.224, 0.225]).reshape(3,1,1)
    x = t.cpu().numpy()
    x = (x * std + mean).clip(0,1)
    x = np.transpose(x, (1,2,0))
    return x

def overlay_heatmap(img: np.ndarray, heat: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_color = plt.cm.jet(heat)[..., :3]
    overlay = (1 - alpha) * img + alpha * heat_color
    overlay = overlay.clip(0,1)
    return overlay


def apply_global_pruning(model: nn.Module, amount: float = 0.3):
    import torch.nn.utils.prune as prune
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            parameters_to_prune.append((module, 'weight'))
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    for module, _ in parameters_to_prune:
        try:
            prune.remove(module, 'weight')
        except Exception:
            pass
    return model

def apply_dynamic_quantization(model: nn.Module):
    qmodel = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )
    return qmodel


@torch.no_grad()
@torch.no_grad()
def benchmark_inference(model: nn.Module, loader: DataLoader, device: str, warmup: int, iters: int) -> Dict[str, float]:
    model.eval()
    data_iter = iter(loader)
    imgs, _ = next(data_iter)
    imgs = imgs.to(device)

    for _ in range(warmup):
        _ = model(imgs)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    t0 = time.time()
    n_images = 0

    for _ in range(iters):
        try:
            imgs, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            imgs, _ = next(data_iter)
        imgs = imgs.to(device)
        _ = model(imgs)
        n_images += imgs.size(0)

    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    dt = time.time() - t0
    throughput = n_images / dt
    latency_ms = (dt / iters) * 1000.0

    return {
        "throughput_img_s": throughput,
        "latency_ms_per_iter": latency_ms,
        "images_processed": n_images,
        "time_s": dt
    }


def parse_args():
    p = argparse.ArgumentParser(description="Wound Classifier Training")
    p.add_argument("--data_dir", type=str, default="/Users/aeshaankumar/Desktop/dataset_processed",
               help="Root folder with train/ and val/ subfolders")
    p.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--lr_patience", type=int, default=2)
    p.add_argument("--lr_factor", type=float, default=0.5)
    p.add_argument("--prune_amount", type=float, default=0.3)
    p.add_argument("--benchmark_iters", type=int, default=50)
    p.add_argument("--benchmark_warmup", type=int, default=5)
    return p.parse_args()

def main():
    args = parse_args()
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    cfg = TrainConfig(
        data_dir=args.data_dir,
        train_subdir="train",
        val_subdir="val",
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        num_workers=args.num_workers,
        seed=args.seed,
        amp=not args.no_amp,
        model_name=args.model,
        freeze_backbone=args.freeze_backbone,
        output_dir=args.output_dir,
        benchmark_iters=args.benchmark_iters,
        benchmark_warmup=args.benchmark_warmup,
        prune_amount=args.prune_amount,
        device=device
    )
    print("======== CONFIG ========")
    print(json.dumps(asdict(cfg), indent=2))
    print("========================")

    train_loader, val_loader, class_names, num_classes = build_dataloaders(cfg)
    print(f"Classes ({num_classes}): {class_names}")
    model = create_backbone(cfg, num_classes=num_classes)
    total_params, trainable_params = count_params(model)
    print(f"Model: {cfg.model_name} | Total params: {total_params:,} | Trainable: {trainable_params:,}")

    print("\n[Benchmark] Baseline (untrained) inference speed:")
    base_bench = benchmark_inference(model.to(cfg.device), val_loader, cfg.device, cfg.benchmark_warmup, cfg.benchmark_iters)
    print(json.dumps(base_bench, indent=2))

    model, history = train_model(cfg, model, train_loader, val_loader, class_names)

    print("\n[Evaluate] Final model on validation set:")
    eval_res = evaluate(model, val_loader, cfg.device, class_names)
    print(f"Validation Accuracy: {eval_res['accuracy']*100:.2f}%")
    if SKLEARN_OK:
        print("Classification Report (macro/weighted):")
        cr = eval_res["classification_report"]
        print(json.dumps({
            "macro_avg": cr.get("macro avg", {}),
            "weighted_avg": cr.get("weighted avg", {}),
            "per_class": {k: v for k, v in cr.items() if k in class_names}
        }, indent=2))
        cm = np.array(eval_res["confusion_matrix"])
        plot_confusion_matrices(cm, class_names, cfg.output_dir)

    print("\n[Explainability] Generating Grad-CAM overlays...")
    try:
        target_layer = "layer4"
        if cfg.model_name.startswith("mobile"):
            target_layer = "features"  
        make_gradcam_samples(model, val_loader, cfg.device, class_names, cfg.output_dir, target_layer_name=target_layer)
    except Exception as e:
        print(f"Grad-CAM failed gracefully: {e}")

    print("\n[Benchmark] Trained FP32 model inference speed:")
    trained_bench = benchmark_inference(model, val_loader, cfg.device, cfg.benchmark_warmup, cfg.benchmark_iters)
    print(json.dumps(trained_bench, indent=2))

    fp32_path = os.path.join(cfg.output_dir, "model_fp32.pth")
    fp32_size = save_model(model, fp32_path)

    print("\n[Optimize] Applying global magnitude pruning...")
    pruned_model = copy.deepcopy(model).cpu()
    pruned_model = apply_global_pruning(pruned_model, amount=cfg.prune_amount)
    pruned_model = pruned_model.to(cfg.device)

    print("[Benchmark] Pruned model inference speed:")
    pruned_bench = benchmark_inference(pruned_model, val_loader, cfg.device, cfg.benchmark_warmup, cfg.benchmark_iters)
    print(json.dumps(pruned_bench, indent=2))
    pruned_path = os.path.join(cfg.output_dir, "model_pruned.pth")
    pruned_size = save_model(pruned_model, pruned_path)
    if cfg.device == "mps":
        print("⚠️  Quantization not supported on MPS. Skipping...")
    else:
        print("\n[Optimize] Applying dynamic quantization (Linear layers)...")
        qmodel = apply_dynamic_quantization(copy.deepcopy(pruned_model).cpu())
        qmodel = qmodel.to(cfg.device)


    print("\n[Optimize] Applying dynamic quantization (Linear layers)...")
    qmodel = apply_dynamic_quantization(copy.deepcopy(pruned_model).cpu())
    qmodel = qmodel.to(cfg.device)

    print("[Benchmark] Pruned + Quantized model inference speed:")
    pq_bench = benchmark_inference(qmodel, val_loader, cfg.device, cfg.benchmark_warmup, cfg.benchmark_iters)
    print(json.dumps(pq_bench, indent=2))
    pq_path = os.path.join(cfg.output_dir, "model_pruned_quantized.pth")
    pq_size = save_model(qmodel.cpu(), pq_path)

    print("\n[Evaluate] Pruned + Quantized model on validation set:")
    pq_eval = evaluate(qmodel.to(cfg.device), val_loader, cfg.device, class_names)
    print(f"Validation Accuracy (PQ): {pq_eval['accuracy']*100:.2f}%")

    print("\n========== SUMMARY (Copy-Paste for Resume) ==========")
    baseline_t = base_bench["latency_ms_per_iter"]
    trained_t = trained_bench["latency_ms_per_iter"]
    pruned_t = pruned_bench["latency_ms_per_iter"]
    pq_t = pq_bench["latency_ms_per_iter"]

    speedup_trained_vs_base = 100.0 * (baseline_t - trained_t) / baseline_t
    speedup_pruned_vs_trained = 100.0 * (trained_t - pruned_t) / trained_t
    speedup_pq_vs_trained = 100.0 * (trained_t - pq_t) / trained_t

    fp32_acc = eval_res["accuracy"] * 100.0
    pq_acc  = pq_eval["accuracy"] * 100.0
    acc_delta_pq = pq_acc - fp32_acc

    print(f"Final FP32 Val Accuracy: {fp32_acc:.2f}%")
    print(f"Final PQ  Val Accuracy: {pq_acc:.2f}% (Δ {acc_delta_pq:+.2f} pp)")
    print(f"Model Size FP32: {human_bytes(fp32_size)}")
    print(f"Model Size Pruned: {human_bytes(pruned_size)}")
    print(f"Model Size Pruned+Quantized: {human_bytes(pq_size)}")
    print(f"Latency (ms) — Base(untrained): {baseline_t:.2f} | Trained: {trained_t:.2f} | Pruned: {pruned_t:.2f} | Pruned+Quantized: {pq_t:.2f}")
    print(f"Throughput (img/s) — Trained: {trained_bench['throughput_img_s']:.2f} | Pruned: {pruned_bench['throughput_img_s']:.2f} | P+Q: {pq_bench['throughput_img_s']:.2f}")

    print("\nResume-ready bullets (tune numbers from your actual runs):")
    print(f"- Improved wound image classification accuracy to **{fp32_acc:.1f}%** using transfer learning, "
        f"augmentation, and early stopping.")
    print(f"- Reduced model latency by **{speedup_pruned_vs_trained:.1f}%** via global magnitude pruning "
        f"and by **{speedup_pq_vs_trained:.1f}%** with dynamic quantization (cumulative speedup from trained FP32).")
    print(f"- Compressed model size from **{human_bytes(fp32_size)}** to **{human_bytes(pq_size)}** "
        f"(**{100.0 * (fp32_size - pq_size) / fp32_size:.1f}%** smaller) with minimal accuracy change "
        f"({acc_delta_pq:+.2f} pp).")
    print(f"- Added **Grad-CAM** explainability and full evaluation (confusion matrix, macro/weighted F1).")
    print("=====================================================\n")

if __name__ == "__main__":
    print("healthproj.py is starting main()...")
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ Uncaught exception during main():")
        traceback.print_exc()

    
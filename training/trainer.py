"""
Trainer — Full training loop with mixed precision, scheduling, and logging.

Features:
  - Automatic Mixed Precision (AMP) for faster training on RTX GPUs
  - Differential learning rates for encoders vs. fusion/classifier
  - Cosine annealing with warmup
  - Early stopping on validation F1
  - TensorBoard + optional W&B logging
  - Checkpoint saving (best + periodic)
  - Gradient accumulation for larger effective batch sizes
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.losses import FocalLoss, LabelSmoothingCE
from training.metrics import EmotionMetrics


class Trainer:
    """
    Training loop for multimodal emotion recognition.

    Handles:
      - Mixed precision training
      - Multi-group optimizer with differential LRs
      - Cosine warmup scheduling
      - Early stopping
      - Comprehensive logging
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        dataset_cfg = config.get("dataset", {})
        dataset_name = dataset_cfg.get("name", "meld")

        # Number of classes
        self.num_classes = dataset_cfg.get(dataset_name, {}).get("num_classes", 7)
        self.class_names = dataset_cfg.get(dataset_name, {}).get(
            "class_names", [str(i) for i in range(self.num_classes)]
        )

        # Training hyperparameters
        self.epochs = train_cfg.get("epochs", 50)
        self.accumulation_steps = train_cfg.get("accumulation_steps", 1)
        self.use_amp = train_cfg.get("amp", True) and device == "cuda"
        self.log_every = train_cfg.get("log_every_n_steps", 50)
        self.val_every = train_cfg.get("val_every_n_epochs", 1)
        self.save_top_k = train_cfg.get("save_top_k", 3)

        # Directories
        project_cfg = config.get("project", {})
        self.output_dir = Path(project_cfg.get("output_dir", "./results"))
        self.log_dir = Path(project_cfg.get("log_dir", "./results/logs"))
        self.ckpt_dir = Path(project_cfg.get("checkpoint_dir", "./results/checkpoints"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Loss function
        self.criterion = self._build_loss(train_cfg.get("loss", {}))

        # Optimizer with differential learning rates
        self.optimizer = self._build_optimizer(train_cfg.get("optimizer", {}))

        # Learning rate scheduler
        self.scheduler = self._build_scheduler(train_cfg.get("scheduler", {}))

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Metrics tracker
        self.metrics = EmotionMetrics(
            num_classes=self.num_classes, class_names=self.class_names
        )

        # Early stopping
        es_cfg = train_cfg.get("early_stopping", {})
        self.patience = es_cfg.get("patience", 10)
        self.es_metric = es_cfg.get("metric", "val_f1_weighted")
        self.es_mode = es_cfg.get("mode", "max")
        self.best_metric = float("-inf") if self.es_mode == "max" else float("inf")
        self.patience_counter = 0

        # TensorBoard
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_checkpoints = []  # (metric_value, path) sorted by metric

        # Metadata stamped into every checkpoint for downstream scripts
        self.model_mode = config.get("model", {}).get("mode", "multimodal")
        self.fusion_strategy = config.get("model", {}).get("fusion", {}).get(
            "strategy", "cross_attention"
        )
        self.train_class_distribution = None  # set by train.py after construction

    def _build_loss(self, loss_cfg: dict) -> nn.Module:
        """Build loss function from config."""
        name = loss_cfg.get("name", "focal")

        if name == "focal":
            alpha = loss_cfg.get("focal_alpha", None)
            if alpha and isinstance(alpha, list):
                alpha = torch.FloatTensor(alpha).to(self.device)
            return FocalLoss(
                gamma=loss_cfg.get("focal_gamma", 2.0),
                alpha=alpha,
            )
        elif name == "label_smoothing":
            return LabelSmoothingCE(
                smoothing=loss_cfg.get("label_smoothing", 0.1),
                num_classes=self.num_classes,
            )
        else:
            return nn.CrossEntropyLoss()

    def _build_optimizer(self, opt_cfg: dict) -> torch.optim.Optimizer:
        """
        Build optimizer with differential learning rates.

        Three parameter groups:
          1. Vision encoder — lowest LR (pretrained backbone, fine-tune gently)
          2. Text encoder — low LR (pretrained DeBERTa, fine-tune gently)
          3. Fusion + classifier — highest LR (training from scratch)
        """
        base_lr = opt_cfg.get("lr", 2e-4)
        vision_lr = opt_cfg.get("vision_lr", 1e-5)
        text_lr = opt_cfg.get("text_lr", 2e-5)
        weight_decay = opt_cfg.get("weight_decay", 0.01)

        param_groups = []

        # Vision encoder params
        if hasattr(self.model, "vision_encoder") and self.model.vision_encoder is not None:
            param_groups.append(
                {
                    "params": self.model.vision_encoder.parameters(),
                    "lr": vision_lr,
                    "name": "vision_encoder",
                }
            )

        # Text encoder params
        if hasattr(self.model, "text_encoder") and self.model.text_encoder is not None:
            param_groups.append(
                {
                    "params": self.model.text_encoder.parameters(),
                    "lr": text_lr,
                    "name": "text_encoder",
                }
            )

        # Fusion + classifier params
        other_params = []
        for name_param, param in self.model.named_parameters():
            if not any(
                name_param.startswith(prefix)
                for prefix in ["vision_encoder.", "text_encoder."]
            ):
                other_params.append(param)

        if other_params:
            param_groups.append(
                {
                    "params": other_params,
                    "lr": base_lr,
                    "name": "fusion_classifier",
                }
            )

        # Print LR groups
        for group in param_groups:
            n_params = sum(p.numel() for p in group["params"])
            print(
                f"  [{group['name']}] LR={group['lr']:.2e}, "
                f"params={n_params:,}"
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )
        return optimizer

    def _build_scheduler(self, sched_cfg: dict):
        """Build learning rate scheduler."""
        name = sched_cfg.get("name", "cosine_warmup")
        warmup_epochs = sched_cfg.get("warmup_epochs", 3)
        total_steps = len(self.train_loader) * self.epochs

        if name == "cosine_warmup":
            warmup_steps = len(self.train_loader) * warmup_epochs

            def lr_lambda(step):
                if step < warmup_steps:
                    return step / max(warmup_steps, 1)
                progress = (step - warmup_steps) / max(
                    total_steps - warmup_steps, 1
                )
                return 0.5 * (1 + np.cos(np.pi * progress))

            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer, lr_lambda
            )
        elif name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        elif name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", patience=5, factor=0.5
            )
        else:
            return None

    def train(self) -> Dict[str, float]:
        """
        Run the full training loop.

        Returns:
            dict with best validation metrics.
        """
        print(f"\n{'=' * 60}")
        print(f"Starting training for {self.epochs} epochs")
        print(f"Device: {self.device} | AMP: {self.use_amp}")
        print(f"{'=' * 60}\n")

        best_metrics = {}

        for epoch in range(self.epochs):
            self.current_epoch = epoch

            # ---- Training ----
            train_metrics = self._train_epoch()

            # ---- Validation ----
            if (epoch + 1) % self.val_every == 0:
                val_metrics = self._validate()

                # Log to TensorBoard
                self._log_epoch(train_metrics, val_metrics, epoch)

                # Print summary
                print(
                    f"Epoch {epoch + 1}/{self.epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val F1: {val_metrics.get('f1_weighted', 0):.4f} | "
                    f"Val Acc: {val_metrics.get('accuracy', 0):.4f}"
                )

                # Early stopping check
                current_metric = val_metrics.get(
                    self.es_metric.replace("val_", ""), 0
                )
                improved = self._check_improvement(current_metric)

                if improved:
                    best_metrics = val_metrics.copy()
                    self._save_checkpoint(val_metrics, is_best=True)
                    self.patience_counter = 0
                    print(f"  ★ New best {self.es_metric}: {current_metric:.4f}")
                else:
                    # Don't count patience during LR warmup — the model hasn't
                    # reached full LR yet and epoch-0 can look artificially "best"
                    warmup_epochs = self.config.get("training", {}).get(
                        "scheduler", {}
                    ).get("warmup_epochs", 3)
                    min_epochs = max(
                        warmup_epochs + 2,
                        self.config.get("training", {}).get(
                            "early_stopping", {}
                        ).get("min_epochs", 0),
                    )
                    if epoch < min_epochs:
                        print(
                            f"  [early_stop] Warmup guard active "
                            f"(epoch {epoch + 1} <= {min_epochs}) — patience not counted"
                        )
                    else:
                        self.patience_counter += 1
                        if self.patience_counter >= self.patience:
                            print(
                                f"\nEarly stopping triggered after {epoch + 1} epochs"
                            )
                            break

            # Step scheduler
            if self.scheduler and not isinstance(
                self.scheduler,
                torch.optim.lr_scheduler.ReduceLROnPlateau,
            ):
                pass  # LambdaLR steps per batch, handled in _train_epoch

        self.writer.close()
        print(f"\nTraining complete. Best metrics: {best_metrics}")
        return best_metrics

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch + 1}",
            leave=False,
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)

            forward_kwargs = {"image": image}

            # Add text inputs if available (multimodal mode)
            if "input_ids" in batch:
                forward_kwargs["input_ids"] = batch["input_ids"].to(self.device)
                forward_kwargs["attention_mask"] = batch["attention_mask"].to(
                    self.device
                )

            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                output = self.model(**forward_kwargs)
                logits = output["logits"]
                loss = self.criterion(logits, label)
                loss = loss / self.accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Step scheduler per batch (for cosine warmup)
                if self.scheduler and not isinstance(
                    self.scheduler,
                    torch.optim.lr_scheduler.ReduceLROnPlateau,
                ):
                    self.scheduler.step()

            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = label.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

            # Update progress bar
            pbar.set_postfix(
                loss=f"{loss.item() * self.accumulation_steps:.4f}",
                lr=f"{self.optimizer.param_groups[-1]['lr']:.2e}",
            )

            self.global_step += 1

            # Log to TensorBoard
            if self.global_step % self.log_every == 0:
                self.writer.add_scalar(
                    "train/loss_step",
                    loss.item() * self.accumulation_steps,
                    self.global_step,
                )

        avg_loss = total_loss / len(self.train_loader)
        metrics = self.metrics.compute(
            np.array(all_preds), np.array(all_labels)
        )
        metrics["loss"] = avg_loss

        return metrics

    @torch.no_grad()
    def _validate(self) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            image = batch["image"].to(self.device)
            label = batch["label"].to(self.device)

            forward_kwargs = {"image": image}

            if "input_ids" in batch:
                forward_kwargs["input_ids"] = batch["input_ids"].to(self.device)
                forward_kwargs["attention_mask"] = batch["attention_mask"].to(
                    self.device
                )

            # Forward pass with AMP
            with torch.amp.autocast(device_type=self.device, enabled=self.use_amp):
                output = self.model(**forward_kwargs)
                logits = output["logits"]
                loss = self.criterion(logits, label)

            total_loss += loss.item()
            preds = logits.argmax(dim=1).cpu().numpy()
            labels = label.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

        avg_loss = total_loss / len(self.val_loader)
        metrics = self.metrics.compute(
            np.array(all_preds), np.array(all_labels)
        )
        metrics["loss"] = avg_loss

        return metrics

    def _check_improvement(self, current_metric: float) -> bool:
        """Check if the current metric improved."""
        if self.es_mode == "max":
            improved = current_metric > self.best_metric
        else:
            improved = current_metric < self.best_metric

        if improved:
            self.best_metric = current_metric
        return improved

    def _save_checkpoint(self, metrics: dict, is_best: bool = False):
        """Save model checkpoint."""
        metric_value = metrics.get(
            self.es_metric.replace("val_", ""), 0
        )

        ckpt_name = (
            f"epoch_{self.current_epoch + 1}_f1_{metric_value:.4f}.pt"
        )
        ckpt_path = self.ckpt_dir / ckpt_name

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "global_step": self.global_step,
            # Metadata for downstream scripts (evaluate, analyze_ambiguity)
            "model_mode": self.model_mode,
            "fusion_strategy": self.fusion_strategy,
            "class_names": self.class_names,
            "num_classes": self.num_classes,
            "dataset_name": self.config.get("dataset", {}).get("name", "meld"),
        }
        if self.train_class_distribution:
            checkpoint["train_class_distribution"] = self.train_class_distribution

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, ckpt_path)

        if is_best:
            best_path = self.ckpt_dir / "best_model.pt"
            torch.save(checkpoint, best_path)

        # Manage top-k checkpoints
        self.best_checkpoints.append((metric_value, str(ckpt_path)))
        self.best_checkpoints.sort(
            key=lambda x: x[0], reverse=(self.es_mode == "max")
        )

        while len(self.best_checkpoints) > self.save_top_k:
            _, old_path = self.best_checkpoints.pop()
            if os.path.exists(old_path) and "best_model" not in old_path:
                os.remove(old_path)

    def _log_epoch(
        self,
        train_metrics: dict,
        val_metrics: dict,
        epoch: int,
    ):
        """Log metrics to TensorBoard."""
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"train/{key}", value, epoch)

        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"val/{key}", value, epoch)

        # Log learning rates
        for i, group in enumerate(self.optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            self.writer.add_scalar(f"lr/{name}", group["lr"], epoch)

    def load_checkpoint(self, checkpoint_path: str):
        """Load a checkpoint to resume training or for evaluation."""
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0) + 1
        self.global_step = checkpoint.get("global_step", 0)

        print(
            f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}"
        )
        return checkpoint.get("metrics", {})

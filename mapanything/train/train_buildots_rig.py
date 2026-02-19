#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""
Training script for MapAnything 4-camera equirectangular rig pose estimation.

Loads a pretrained MapAnything checkpoint, adds rig-specific modules
(rig_rot_encoder, timestamp_encoder, Cam360PoseHead), and fine-tunes
the full model with differential learning rates.

Usage:
    python -m mapanything.train.train_buildots_rig \
        --pretrained_ckpt /path/to/mapanything_checkpoint.pth \
        --data_root /bd-resources/jenia/dataset/buildots_da3 \
        --output_dir /path/to/output
"""

import argparse
import datetime
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from mapanything.datasets.buildots_rig import (
    BuildotsRigAdapter,
    buildots_rig_collate_fn,
)
from mapanything.train.rig_losses import Rig360PoseLoss
from mapanything.utils.train_tools import (
    NativeScalerWithGradNormCount as NativeScaler,
)

# Enable TF32
if hasattr(torch.backends.cuda, "matmul") and hasattr(
    torch.backends.cuda.matmul, "allow_tf32"
):
    torch.backends.cuda.matmul.allow_tf32 = True


def get_args():
    parser = argparse.ArgumentParser("MapAnything Rig Pose360 Training")

    # Data
    parser.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root directory for Buildots dataset segments.",
    )
    parser.add_argument("--seq_len", type=int, default=10)
    parser.add_argument("--num_timestamps", type=int, default=3)
    parser.add_argument("--image_size", type=int, nargs=2, default=[350, 350])
    parser.add_argument("--fov", type=int, default=90)

    # Model
    parser.add_argument(
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="Path to pretrained MapAnything checkpoint.",
    )

    # Training
    parser.add_argument("--output_dir", type=str, default="output/rig_pose360")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_new", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_freq", type=int, default=5)
    parser.add_argument("--eval_freq", type=int, default=5)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--no_amp", dest="use_amp", action="store_false")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument(
        "--accum_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps. Effective batch size = batch_size * accum_steps.",
    )

    # Loss weights
    parser.add_argument("--trans_weight", type=float, default=1.0)
    parser.add_argument("--yaw_weight", type=float, default=1.0)
    parser.add_argument("--rig_trans_var_weight", type=float, default=0.1)
    parser.add_argument("--rig_yaw_consistency_weight", type=float, default=0.1)

    return parser.parse_args()


def build_param_groups(model, lr_pretrained, lr_new, weight_decay):
    """Build optimizer param groups with differential LR for new vs pretrained modules."""
    new_module_names = {"rig_rot_encoder", "timestamp_encoder", "pose360_head"}

    pretrained_params = []
    pretrained_params_no_wd = []
    new_params = []
    new_params_no_wd = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_new = any(m in name for m in new_module_names)
        no_wd = param.dim() < 2 or "bias" in name or "norm" in name

        if is_new:
            if no_wd:
                new_params_no_wd.append(param)
            else:
                new_params.append(param)
        else:
            if no_wd:
                pretrained_params_no_wd.append(param)
            else:
                pretrained_params.append(param)

    param_groups = [
        {"params": pretrained_params, "lr": lr_pretrained, "weight_decay": weight_decay},
        {"params": pretrained_params_no_wd, "lr": lr_pretrained, "weight_decay": 0.0},
        {"params": new_params, "lr": lr_new, "weight_decay": weight_decay * 0.2},
        {"params": new_params_no_wd, "lr": lr_new, "weight_decay": 0.0},
    ]
    return [g for g in param_groups if len(g["params"]) > 0]


def cosine_lr_schedule(optimizer, epoch, total_epochs, warmup_epochs, base_lrs, min_lr):
    """Cosine LR schedule with linear warmup."""
    if epoch < warmup_epochs:
        scale = epoch / max(warmup_epochs, 1)
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        scale = 0.5 * (1.0 + np.cos(np.pi * progress))

    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = max(min_lr, base_lrs[i] * scale)


def train_one_epoch(
    model, criterion, data_loader, optimizer, scaler, device, epoch, args, log_writer=None
):
    model.train()
    metric_logger = defaultdict(lambda: 0.0)
    num_batches = 0

    amp_dtype = torch.bfloat16 if args.use_amp else torch.float32

    accum_steps = getattr(args, "accum_steps", 1)
    num_loader = len(data_loader)

    for batch_idx, batch in enumerate(data_loader):
        # batch: list of V view-dicts, each value has shape (B, ...)
        # Move to device
        for view in batch:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device, non_blocking=True)

        if batch_idx % accum_steps == 0:
            optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            preds = model(batch)
            loss, details = criterion(preds, batch)

        if not torch.isfinite(loss):
            print(f"WARNING: non-finite loss {loss.item()} at batch {batch_idx}, skipping")
            continue

        loss_scaled = loss / accum_steps
        do_step = (batch_idx + 1) % accum_steps == 0 or batch_idx == num_loader - 1
        scaler(
            loss_scaled,
            optimizer,
            clip_grad=args.grad_clip,
            parameters=model.parameters(),
            update_grad=do_step,
        )

        # Logging
        for k, v in details.items():
            metric_logger[k] += v
        num_batches += 1

        if batch_idx % 50 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"  [Epoch {epoch}][{batch_idx}/{len(data_loader)}] "
                f"loss={details['total_loss']:.4f} "
                f"trans={details['trans_loss']:.4f} "
                f"yaw={details['yaw_loss']:.4f} "
                f"rig_tv={details['rig_trans_var']:.5f} "
                f"rig_yc={details['rig_yaw_consistency']:.5f} "
                f"lr={lr:.2e}"
            )

    # Average metrics
    if num_batches > 0:
        for k in metric_logger:
            metric_logger[k] /= num_batches

    if log_writer is not None:
        for k, v in metric_logger.items():
            log_writer.add_scalar(f"train/{k}", v, epoch)

    return dict(metric_logger)


@torch.no_grad()
def eval_one_epoch(model, criterion, data_loader, device, epoch, args):
    model.eval()
    metric_logger = defaultdict(lambda: 0.0)
    num_batches = 0
    amp_dtype = torch.bfloat16 if args.use_amp else torch.float32

    for batch in data_loader:
        for view in batch:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.use_amp, dtype=amp_dtype):
            preds = model(batch)
            loss, details = criterion(preds, batch)

        for k, v in details.items():
            metric_logger[k] += v
        num_batches += 1

    if num_batches > 0:
        for k in metric_logger:
            metric_logger[k] /= num_batches

    return dict(metric_logger)


def main():
    args = get_args()

    # Setup
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    # ---- Dataset ----
    # Import BuildotsDataset from the user's Buildots codebase
    import sys
    buildots_path = (
        args.buildots_code_path
        or os.environ.get("BUILDOTS_CODE_PATH", "/Users/jenia/projects/buildots/pycode")
    )
    if buildots_path not in sys.path:
        sys.path.insert(0, buildots_path)
    from research.positioning_net.buildots_dataset_generator import BuildotsDataset

    train_ds = BuildotsDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        image_size=tuple(args.image_size),
        fov=args.fov,
        debug=False,
        training=True,
    )
    val_ds = BuildotsDataset(
        root_dir=args.data_root,
        seq_len=args.seq_len,
        image_size=tuple(args.image_size),
        fov=args.fov,
        debug=False,
        training=False,
    )

    train_adapter = BuildotsRigAdapter(train_ds, num_timestamps=args.num_timestamps)
    val_adapter = BuildotsRigAdapter(val_ds, num_timestamps=args.num_timestamps)

    train_loader = DataLoader(
        train_adapter,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=buildots_rig_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_adapter,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=buildots_rig_collate_fn,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model ----
    from mapanything.models import init_model_from_config

    # Load pretrained MapAnything (the config must already exist as a Hydra model config)
    # For now, we load from checkpoint with strict=False
    if args.pretrained_ckpt is not None:
        print(f"Loading pretrained checkpoint: {args.pretrained_ckpt}")
        ckpt = torch.load(args.pretrained_ckpt, map_location="cpu", weights_only=False)

        # Extract model config from checkpoint if available
        if "model_config" in ckpt:
            model_config = ckpt["model_config"]
        else:
            print(
                "WARNING: No model_config in checkpoint. "
                "You need to provide a model instance with the correct config."
            )
            raise ValueError(
                "Checkpoint does not contain model_config. "
                "Please provide a model with the rig config already set up."
            )

        from mapanything.models import model_factory

        model = model_factory("mapanything", **model_config)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        print(f"Loaded pretrained weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing keys (expected for new modules): {missing[:20]}...")
        del ckpt
    else:
        print("No pretrained checkpoint provided. Initializing from config.")
        try:
            model = init_model_from_config("mapanything", device="cpu")
        except Exception as e:
            print(f"Could not load via Hydra config: {e}")
            print("Please provide --pretrained_ckpt.")
            return

    model = model.to(device)
    print(f"Model on {device}, parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Loss ----
    criterion = Rig360PoseLoss(
        trans_weight=args.trans_weight,
        yaw_weight=args.yaw_weight,
        rig_trans_var_weight=args.rig_trans_var_weight,
        rig_yaw_consistency_weight=args.rig_yaw_consistency_weight,
    ).to(device)

    # ---- Optimizer ----
    param_groups = build_param_groups(model, args.lr, args.lr_new, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95))
    base_lrs = [g["lr"] for g in optimizer.param_groups]
    scaler = NativeScaler()

    print(f"Optimizer param groups:")
    for i, g in enumerate(optimizer.param_groups):
        print(f"  Group {i}: lr={g['lr']:.2e}, wd={g['weight_decay']:.4f}, params={len(g['params'])}")

    # ---- TensorBoard ----
    log_writer = SummaryWriter(log_dir=args.output_dir)

    # ---- Training loop ----
    best_loss = float("inf")
    print(f"\nStarting training for {args.epochs} epochs")
    print(f"  Train samples: {len(train_adapter)}, Val samples: {len(val_adapter)}")
    print(f"  Views per sample: {args.num_timestamps * 4}")
    accum_steps = getattr(args, "accum_steps", 1)
    eff_bs = args.batch_size * accum_steps
    if accum_steps > 1:
        print(f"  Batch size: {args.batch_size} (effective: {eff_bs}, accum_steps: {accum_steps})")
    else:
        print(f"  Batch size: {args.batch_size}")

    for epoch in range(1, args.epochs + 1):
        cosine_lr_schedule(
            optimizer, epoch, args.epochs, args.warmup_epochs, base_lrs, args.min_lr
        )

        train_stats = train_one_epoch(
            model, criterion, train_loader, optimizer, scaler, device, epoch, args, log_writer
        )

        # Evaluation
        if args.eval_freq > 0 and epoch % args.eval_freq == 0:
            val_stats = eval_one_epoch(model, criterion, val_loader, device, epoch, args)
            val_loss = val_stats.get("total_loss", float("inf"))
            print(
                f"  [Val Epoch {epoch}] "
                f"loss={val_loss:.4f} "
                f"trans={val_stats.get('trans_loss', 0):.4f} "
                f"yaw={val_stats.get('yaw_loss', 0):.4f}"
            )

            for k, v in val_stats.items():
                log_writer.add_scalar(f"val/{k}", v, epoch)

            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.output_dir, "checkpoint-best.pth")
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "best_loss": best_loss,
                    },
                    save_path,
                )
                print(f"  New best! Saved to {save_path}")

        # Periodic save
        if args.save_freq > 0 and epoch % args.save_freq == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint-{epoch:04d}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                save_path,
            )

        # Log
        log_stats = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()}}
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")

    # Final save
    save_path = os.path.join(args.output_dir, "checkpoint-last.pth")
    torch.save({"epoch": args.epochs, "model": model.state_dict()}, save_path)
    print(f"\nTraining complete. Last checkpoint: {save_path}")
    print(f"Best validation loss: {best_loss:.4f}")

    total_time = time.time() - time.time()
    log_writer.close()


if __name__ == "__main__":
    main()

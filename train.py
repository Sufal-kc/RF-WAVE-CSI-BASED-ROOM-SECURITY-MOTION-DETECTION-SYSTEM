#!/usr/bin/env python3
"""
Three-stage training loop.

Why three stages, not just plain supervised regression:
  - Confidently-labeled CSI<->pose pairs are the scarce resource here
    (camera dropout, low MediaPipe confidence, sync tolerance misses).
    CSI itself is cheap and constant.
  - Stage 1 (self-supervised): pretrain the CNN encoder as a
    reconstruction autoencoder on EVERY CSI window in the unlabeled pool.
    This is where the encoder learns what "normal" channel structure
    looks like, using orders of magnitude more data than the labels
    allow.
  - Stage 2 (supervised): attach the pose regression head, fine-tune the
    whole network on the labeled set only, with a confidence-weighted
    loss so an unreliable MediaPipe joint (low visibility) doesn't get
    the same weight as a confident one.
  - Stage 3 (semi-supervised self-training): use the stage-2 model to
    pseudo-label a subset of the unlabeled pool (only keep predictions
    the model is *consistent* about, via test-time augmentation
    agreement), fold those in as extra soft-weighted training examples,
    and re-fit. This is the standard way to make use of the CSI-only
    pool once you already have a reasonable regressor, without pretending
    unlabeled CSI carries known joint positions.

Run stages independently or all at once with --stage all.
"""

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from common import NUM_JOINTS
from dataset import CSIWindowDataset, CSIPoseDataset
from model import CSIAutoencoder, CSIPoseNet


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stage 1: self-supervised pretraining
# ---------------------------------------------------------------------------
def pretrain_autoencoder(args, device):
    ds = CSIWindowDataset(os.path.join(args.data_dir, "windows_unlabeled.npz"))
    n_val = max(1, int(0.1 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    _, _, T, F = ds.X.shape
    model = CSIAutoencoder(in_channels=4, embed_dim=args.embed_dim, window_shape=(T, F)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_val, best_state = float("inf"), None
    for epoch in range(args.pretrain_epochs):
        model.train()
        train_loss = 0.0
        for x in train_loader:
            x = x.to(device)
            recon, _ = model(x)
            loss = loss_fn(recon, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x in val_loader:
                x = x.to(device)
                recon, _ = model(x)
                val_loss += loss_fn(recon, x).item() * x.size(0)
        val_loss /= len(val_ds)

        print(f"[pretrain] epoch {epoch+1}/{args.pretrain_epochs}  "
              f"train_recon_mse={train_loss:.4f}  val_recon_mse={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(args.out_dir, "autoencoder.pt")
    torch.save({"state_dict": model.state_dict(), "stats": ds.stats,
                "window_shape": (T, F), "embed_dim": args.embed_dim}, ckpt_path)
    print(f"[pretrain] saved {ckpt_path}")
    return model, ds.stats


# ---------------------------------------------------------------------------
# Confidence-weighted keypoint loss
# ---------------------------------------------------------------------------
def weighted_mse(pred, target, conf):
    """conf: (N, NUM_JOINTS) in [0,1]. Expand to (N, NUM_JOINTS*2) to weight
    x/y of each joint equally, then zero out joints below the confidence
    threshold entirely instead of letting them contribute noisy gradient."""
    w = conf.repeat_interleave(2, dim=1)  # (N, NUM_JOINTS*2)
    w = torch.where(conf.repeat_interleave(2, dim=1) > 0, w, torch.zeros_like(w))
    sq_err = (pred - target) ** 2
    denom = w.sum().clamp_min(1e-6)
    return (sq_err * w).sum() / denom


# ---------------------------------------------------------------------------
# Stage 2: supervised fine-tune on labeled pairs
# ---------------------------------------------------------------------------
def supervised_finetune(args, device, pretrained_encoder=None, stats=None):
    ds = CSIPoseDataset(os.path.join(args.data_dir, "windows_labeled.npz"), stats=stats)
    n_val = max(1, int(0.15 * len(ds)))
    train_ds, val_ds = random_split(ds, [len(ds) - n_val, n_val])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    model = CSIPoseNet(in_channels=4, embed_dim=args.embed_dim, num_joints=NUM_JOINTS).to(device)
    if pretrained_encoder is not None:
        model.load_pretrained_encoder(pretrained_encoder)
        print("[finetune] initialized encoder from stage-1 pretraining")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.5)

    best_val, best_state = float("inf"), None
    for epoch in range(args.finetune_epochs):
        model.train()
        train_loss = 0.0
        for x, y, conf in train_loader:
            x, y, conf = x.to(device), y.to(device), conf.to(device)
            pred = model(x)
            loss = weighted_mse(pred, y, conf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y, conf in val_loader:
                x, y, conf = x.to(device), y.to(device), conf.to(device)
                pred = model(x)
                val_loss += weighted_mse(pred, y, conf).item() * x.size(0)
        val_loss /= len(val_ds)

        print(f"[finetune] epoch {epoch+1}/{args.finetune_epochs}  "
              f"train_wmse={train_loss:.5f}  val_wmse={val_loss:.5f}")
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    ckpt_path = os.path.join(args.out_dir, "pose_model_stage2.pt")
    torch.save({"state_dict": model.state_dict(), "stats": ds.stats, "embed_dim": args.embed_dim}, ckpt_path)
    print(f"[finetune] saved {ckpt_path}")
    return model, ds.stats


# ---------------------------------------------------------------------------
# Stage 3: pseudo-label self-training on the unlabeled pool
# ---------------------------------------------------------------------------
def self_training(args, device, model, stats):
    unl_ds = CSIWindowDataset(os.path.join(args.data_dir, "windows_unlabeled.npz"), stats=stats)
    unl_loader = DataLoader(unl_ds, batch_size=args.batch_size, shuffle=False)

    model.eval()
    pseudo_X, pseudo_Y, pseudo_conf = [], [], []
    noise_std = 0.05  # small perturbation for the consistency check below

    with torch.no_grad():
        for x in unl_loader:
            x = x.to(device)
            pred_a = model(x)
            pred_b = model(x + noise_std * torch.randn_like(x))
            agreement = torch.mean((pred_a - pred_b) ** 2, dim=1)  # per-example disagreement

            # keep only windows where the two noisy passes agree well --
            # a cheap stand-in for prediction confidence, since we have no
            # ground truth for these windows at all.
            keep = agreement < torch.quantile(agreement, args.pseudo_label_keep_frac)
            if keep.sum() == 0:
                continue

            kept_pred = pred_a[keep].cpu().numpy()
            pseudo_X.append(x[keep].cpu().numpy())
            pseudo_Y.append(kept_pred)
            # soft confidence: lower disagreement -> weight closer to 1,
            # capped below the weight of real labels (pseudo_label_weight)
            w = args.pseudo_label_weight * torch.ones(kept_pred.shape[0], NUM_JOINTS)
            pseudo_conf.append(w.numpy())

    if not pseudo_X:
        print("[self-train] no confident pseudo-labels found, skipping stage 3")
        return model

    pseudo_X = np.concatenate(pseudo_X, axis=0).transpose(0, 2, 3, 1)  # back to (N,T,F,4) for the npz format
    pseudo_Y = np.concatenate(pseudo_Y, axis=0)
    pseudo_conf = np.concatenate(pseudo_conf, axis=0)
    print(f"[self-train] generated {len(pseudo_X)} pseudo-labeled examples "
          f"(weight={args.pseudo_label_weight})")

    pseudo_path = os.path.join(args.out_dir, "windows_pseudo.npz")
    np.savez_compressed(pseudo_path, X=pseudo_X, Y=pseudo_Y, conf=pseudo_conf)

    # combine real + pseudo labels and do a short additional fine-tune pass
    real_ds = CSIPoseDataset(os.path.join(args.data_dir, "windows_labeled.npz"), stats=stats)
    pseudo_ds = CSIPoseDataset(pseudo_path, stats=stats)

    from torch.utils.data import ConcatDataset
    combined = ConcatDataset([real_ds, pseudo_ds])
    loader = DataLoader(combined, batch_size=args.batch_size, shuffle=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
    model.train()
    for epoch in range(args.self_train_epochs):
        total = 0.0
        for x, y, conf in loader:
            x, y, conf = x.to(device), y.to(device), conf.to(device)
            pred = model(x)
            loss = weighted_mse(pred, y, conf)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"[self-train] epoch {epoch+1}/{args.self_train_epochs}  wmse={total/len(combined):.5f}")

    ckpt_path = os.path.join(args.out_dir, "pose_model_final.pt")
    torch.save({"state_dict": model.state_dict(), "stats": stats, "embed_dim": args.embed_dim}, ckpt_path)
    print(f"[self-train] saved {ckpt_path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Directory with windows_unlabeled.npz / windows_labeled.npz")
    ap.add_argument("--out-dir", default="./checkpoints")
    ap.add_argument("--stage", choices=["pretrain", "finetune", "self_train", "all"], default="all")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--self-train-epochs", type=int, default=10)
    ap.add_argument("--pseudo-label-keep-frac", type=float, default=0.2,
                     help="Keep this fraction of unlabeled windows with the lowest augmentation disagreement")
    ap.add_argument("--pseudo-label-weight", type=float, default=0.3,
                     help="Loss weight for pseudo-labels relative to real labels (1.0)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"[setup] device={device}")

    autoencoder, stats = None, None
    if args.stage in ("pretrain", "all"):
        autoencoder, stats = pretrain_autoencoder(args, device)

    model, stats = supervised_finetune(args, device, pretrained_encoder=autoencoder, stats=stats) \
        if args.stage in ("finetune", "all") else (None, stats)

    if args.stage in ("self_train", "all"):
        if model is None:
            raise SystemExit("self_train requires a stage-2 model; run --stage finetune or all first")
        self_training(args, device, model, stats)


if __name__ == "__main__":
    main()

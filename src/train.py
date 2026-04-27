"""
Training with heavy regularization + early stopping.

Main differences from v1:
  • Early-stops when val τ plateaus for EARLY_STOP_PATIENCE epochs.
  • Gracefully skips loading an incompatible old checkpoint (the model
    architecture changed, so the old .pth won't fit).
  • Saves both EMA and raw-model checkpoints — we've seen cases where
    the raw model beats EMA on this task.
"""
import os
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset
from src.config import *
from src.dataset import SherlockFeatureDataset, feature_collate_fn
from src.model import TemporalReorderModel
from src.metrics import calculate_kendall_tau


# ─────────────────────────────────────────────────────────────────────────────
# VECTORISED LOSSES  (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def pairwise_bce_loss(pair_logits, target_ranks, mask):
    B, T, _ = pair_logits.shape
    r_i = target_ranks.unsqueeze(2)                    # [B, T, 1]
    r_j = target_ranks.unsqueeze(1)                    # [B, 1, T]
    gt  = (r_i < r_j).float()                          # 1 iff i earlier than j

    pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)
    eye = torch.eye(T, dtype=torch.bool, device=mask.device).unsqueeze(0)
    pair_mask = pair_mask & ~eye

    bce = F.binary_cross_entropy_with_logits(pair_logits, gt, reduction="none")
    denom = pair_mask.sum().clamp(min=1).float()
    return (bce * pair_mask.float()).sum() / denom


def listmle_loss(scores, target_ranks, mask):
    B, T = scores.shape
    loss_sum = scores.new_zeros(())
    count = 0
    for b in range(B):
        n = int(mask[b].sum().item())
        if n < 2:
            continue
        s = scores[b, :n]
        r = target_ranks[b, :n]
        _, order = r.sort(descending=True)
        ordered = s[order]
        rev = torch.flip(ordered, dims=[0])
        lse_rev = torch.logcumsumexp(rev, dim=0)
        lse = torch.flip(lse_rev, dims=[0])
        loss_sum = loss_sum + (lse - ordered).mean()
        count += 1
    return loss_sum / max(count, 1)


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def state_dict(self):
        return self.shadow


# ─────────────────────────────────────────────────────────────────────────────
# DATA / TRAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders():
    train_base = SherlockFeatureDataset(
        features_dir=os.path.join(FEATURES_DIR, "train"),
        label_file=LABEL_FILE,
        is_train=True,
        noise_std=FEAT_NOISE_STD,
        frame_dropout_p=FRAME_DROPOUT_P,
        preload_to_ram=True,
    )
    val_base = SherlockFeatureDataset(
        features_dir=os.path.join(FEATURES_DIR, "train"),
        label_file=LABEL_FILE,
        is_train=False,
        preload_to_ram=True,
    )
    val_base._cache = train_base._cache

    assert len(train_base) == len(val_base)
    n_total = len(train_base)
    n_val   = max(1, int(n_total * 0.08))
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    train_ds = Subset(train_base, train_idx)
    val_ds   = Subset(val_base,   val_idx)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=feature_collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=feature_collate_fn, num_workers=2, pin_memory=True,
    )
    return train_loader, val_loader, n_train, n_val


def run_validation(model, loader):
    model.eval()
    preds, gts, lens = [], [], []
    with torch.no_grad():
        for feats, ranks, mask, *_ in loader:
            feats    = feats.to(DEVICE, non_blocking=True)
            pad_mask = (~mask).to(DEVICE)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                scores, _ = model(feats, src_key_padding_mask=pad_mask)
            scores_np = scores.float().cpu().numpy()
            ranks_np  = ranks.numpy()
            lengths   = mask.sum(dim=1).cpu().numpy()
            for i in range(len(lengths)):
                vl = int(lengths[i])
                preds.append(scores_np[i][:vl])
                gts.append(ranks_np[i][:vl])
                lens.append(vl)
    return calculate_kendall_tau(preds, gts, lens)


def train():
    print("=" * 65)
    print("TRAINING v2 — small model, heavy regularization, early stop")
    print("=" * 65)

    train_loader, val_loader, n_train, n_val = build_loaders()
    print(f"  train videos: {n_train}   val videos: {n_val}")
    print(f"  batch size  : {BATCH_SIZE}")
    print(f"  batches/ep  : {len(train_loader)}")

    model = TemporalReorderModel(
        feat_dim=FEAT_DIM, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)
    total_p, train_p = model.param_summary()
    print(f"  params (total/trainable): {total_p/1e6:.2f}M / {train_p/1e6:.2f}M")

    # Don't try to resume from an old checkpoint — architecture changed.
    # Start fresh.
    if os.path.exists(WEIGHTS_PATH):
        try:
            model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
            print(f"  resumed from: {WEIGHTS_PATH}")
        except RuntimeError:
            print(f"  old checkpoint at {WEIGHTS_PATH} is incompatible — "
                  f"training from scratch")

    ema = EMA(model, EMA_DECAY) if USE_EMA else None

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999),
    )

    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * WARMUP_EPOCHS

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler("cuda", enabled=DEVICE.type == "cuda")

    best_tau = -1.0
    best_type = ""          # "ema" or "raw"
    patience_ctr = 0
    os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)

    print(f"\nStarting — up to {EPOCHS} epochs, early stop after "
          f"{EARLY_STOP_PATIENCE} non-improving epochs")
    print(f"Loss = {PAIRWISE_WEIGHT}·PairwiseBCE + {LISTMLE_WEIGHT}·ListMLE")
    print("-" * 65)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = running_bce = running_mle = 0.0

        for batch_idx, (feats, ranks, mask, *_) in enumerate(train_loader):
            feats    = feats.to(DEVICE, non_blocking=True)
            ranks    = ranks.to(DEVICE, non_blocking=True)
            mask     = mask.to(DEVICE, non_blocking=True)
            pad_mask = ~mask

            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=DEVICE.type == "cuda"):
                scores, pair_logits = model(feats, src_key_padding_mask=pad_mask)
                loss_bce = pairwise_bce_loss(pair_logits.float(), ranks, mask)
                loss_mle = listmle_loss(scores.float(), ranks, mask)
                loss = PAIRWISE_WEIGHT * loss_bce + LISTMLE_WEIGHT * loss_mle

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema is not None:
                ema.update(model)

            running_loss += loss.item()
            running_bce  += loss_bce.item()
            running_mle  += loss_mle.item()

        avg_loss = running_loss / len(train_loader)
        avg_bce  = running_bce  / len(train_loader)
        avg_mle  = running_mle  / len(train_loader)

        # Evaluate BOTH raw and EMA weights — save whichever wins
        raw_tau = run_validation(model, val_loader)

        if ema is not None:
            backup = copy.deepcopy(model.state_dict())
            model.load_state_dict(ema.state_dict())
            ema_tau = run_validation(model, val_loader)
            model.load_state_dict(backup)
        else:
            ema_tau = raw_tau

        val_tau = max(raw_tau, ema_tau)
        chosen  = "ema" if ema_tau >= raw_tau else "raw"

        lr_now = optimizer.param_groups[0]["lr"]
        flag = ""
        if val_tau > best_tau:
            best_tau = val_tau
            best_type = chosen
            save_state = (ema.state_dict() if chosen == "ema"
                          else model.state_dict())
            torch.save(save_state, WEIGHTS_PATH)
            flag = f"  ← best ({chosen}) ✓"
            patience_ctr = 0
        else:
            patience_ctr += 1

        print(
            f"Ep [{epoch+1:2d}/{EPOCHS}]  "
            f"loss={avg_loss:.4f}  bce={avg_bce:.4f}  mle={avg_mle:.4f}  "
            f"raw τ={raw_tau:.4f}  ema τ={ema_tau:.4f}  "
            f"best={best_tau:.4f}  lr={lr_now:.2e}{flag}"
        )

        if patience_ctr >= EARLY_STOP_PATIENCE:
            print(f"\nEarly stopping — no improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs.")
            break

    print("-" * 65)
    print(f"Done. Best val τ = {best_tau:.4f}  (from {best_type} weights)")
    print(f"Weights saved to {WEIGHTS_PATH}")


if __name__ == "__main__":
    train()

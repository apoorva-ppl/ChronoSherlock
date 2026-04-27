"""
Cached-feature dataset.

Instead of decoding videos and running a 22M-param ViT every epoch,
we load the pre-extracted feature tensors saved by extract_features.py.

One epoch is now bottlenecked on disk I/O + transformer math, not
on cv2.VideoCapture and DINOv2 forward passes. Expect ~20–40s per epoch
on a T4 for 5000 training videos.
"""
import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class SherlockFeatureDataset(Dataset):
    """
    Each item returns:
        features          : FloatTensor [N, FEAT_DIM]
        target_ranks      : FloatTensor [N]      (only when labels given)
        vid_id            : str
        sampled_indices   : LongTensor [N]       (positions in corrupted vid)
        total_frames      : int

    If `is_train`, light augmentation (Gaussian noise + frame dropout) is
    applied to the features on-the-fly.
    """

    def __init__(self, features_dir, label_file=None,
                 is_train=False, noise_std=0.0, frame_dropout_p=0.0,
                 preload_to_ram=True):
        self.features_dir     = Path(features_dir)
        self.is_train         = is_train
        self.noise_std        = noise_std
        self.frame_dropout_p  = frame_dropout_p
        self.preload_to_ram   = preload_to_ram
        self.labels           = {}

        if label_file:
            with open(label_file, "r") as f:
                self.labels = json.load(f)

        # List of (vid_id, path_to_feature_file)
        pairs = []
        for p in sorted(self.features_dir.glob("*.pt")):
            vid_id = p.stem
            if (not self.labels) or (vid_id in self.labels):
                pairs.append((vid_id, p))
        self.pairs = pairs
        self.video_ids = [p[0] for p in self.pairs]

        # Optionally load everything to RAM up front. 5100 videos × ~75 frames
        # × 384 dim × 2 B ≈ 290 MB — tiny.
        self._cache = {}
        if self.preload_to_ram:
            for vid_id, path in self.pairs:
                self._cache[vid_id] = torch.load(path, map_location="cpu")

    def __len__(self):
        return len(self.pairs)

    def _load(self, vid_id, path):
        if vid_id in self._cache:
            return self._cache[vid_id]
        return torch.load(path, map_location="cpu")

    def __getitem__(self, idx):
        vid_id, path = self.pairs[idx]
        payload = self._load(vid_id, path)

        feats  = payload["features"].float()               # [N, FEAT_DIM]
        sampled = payload["sampled_indices"].long()        # [N]
        total   = int(payload["total_frames"])

        # ── Augmentation (train only) ─────────────────────────────────
        if self.is_train:
            # Random frame dropout: at least 4 frames must survive
            if self.frame_dropout_p > 0 and feats.size(0) > 8:
                keep_mask = torch.rand(feats.size(0)) > self.frame_dropout_p
                if keep_mask.sum() < 4:
                    keep_mask[:4] = True
                feats   = feats[keep_mask]
                sampled = sampled[keep_mask]

            # Gaussian noise on features
            if self.noise_std > 0:
                feats = feats + torch.randn_like(feats) * self.noise_std

        # ── Compute target ranks from labels ──────────────────────────
        if not self.labels:
            return feats, vid_id, sampled, total

        full_order = self.labels[vid_id]          # len = n_total
        n_total    = len(full_order)

        # rank_of_position[p] = normalised chronological rank of corrupted-
        # video position p. 0.0 = earliest, 1.0 = latest.
        rank_of_position = np.zeros(n_total, dtype=np.float32)
        denom = max(n_total - 1, 1)
        for rank, pos in enumerate(full_order):
            if 0 <= pos < n_total:
                rank_of_position[pos] = rank / denom

        sampled_np = sampled.numpy()
        target_ranks = np.array([
            rank_of_position[i] if 0 <= i < n_total else 0.0
            for i in sampled_np
        ], dtype=np.float32)

        return feats, torch.from_numpy(target_ranks), vid_id, sampled, total


def feature_collate_fn(batch):
    """
    Collate variable-length feature sequences.

    Returns, for training:
        feats_padded   : [B, T_max, FEAT_DIM]
        ranks_padded   : [B, T_max]     (padding value -1.0)
        mask           : [B, T_max]     (True for valid frames)
        vid_ids        : tuple[str]
        sampled_list   : list[LongTensor] (per-video original positions)
        totals         : list[int]

    For inference (no labels): skips ranks_padded.
    """
    has_labels = len(batch[0]) == 5    # (feats, ranks, vid_id, sampled, total)

    if has_labels:
        feats_list, ranks_list, vid_ids, sampled_list, totals = zip(*batch)
    else:
        feats_list, vid_ids, sampled_list, totals = zip(*batch)
        ranks_list = None

    feats_padded = pad_sequence(feats_list, batch_first=True)     # [B, T, D]
    lengths = torch.tensor([f.size(0) for f in feats_list])
    T_max   = feats_padded.size(1)
    mask    = torch.arange(T_max)[None, :] < lengths[:, None]      # [B, T]

    if has_labels:
        ranks_padded = pad_sequence(
            ranks_list, batch_first=True, padding_value=-1.0
        )
        return feats_padded, ranks_padded, mask, vid_ids, list(sampled_list), list(totals)
    else:
        return feats_padded, mask, vid_ids, list(sampled_list), list(totals)

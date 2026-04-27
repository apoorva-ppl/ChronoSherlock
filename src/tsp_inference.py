"""
TSP chain-ordering with guaranteed-correct output format.

Reads sample_submission.csv to get the exact expected frame count for every
test video, then ensures the submission has precisely that many elements as
a valid 1..N permutation.

Two regimes per video:
  • n_sampled == expected_frames  →  TSP path maps directly to frame indices
  • n_sampled <  expected_frames  →  assign temporal scores from TSP path,
    interpolate to all frames, argsort → full 1..N permutation

Run:   !python -m src.tsp_inference
"""
import os
import sys
import ast
import csv
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from src.config import *
from src.dataset import SherlockFeatureDataset, feature_collate_fn


# ─────────────────────────────────────────────────────────────────────────────
# Orientation model (loads either v1 or v2 weights)
# ─────────────────────────────────────────────────────────────────────────────

class _V1OrientationModel(nn.Module):
    """Mirrors v1 architecture so old weights load cleanly."""
    def __init__(self, feat_dim=384, embed_dim=384, num_heads=8,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim), nn.LayerNorm(embed_dim),
            nn.GELU(), nn.Dropout(dropout),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm  = nn.LayerNorm(embed_dim)
        self.score_head  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(embed_dim // 2, 1),
        )
        self.pair_q = nn.Linear(embed_dim, embed_dim // 2)
        self.pair_k = nn.Linear(embed_dim, embed_dim // 2)

    def forward(self, feats, src_key_padding_mask=None):
        x = self.input_proj(feats)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        return self.score_head(x).squeeze(-1)


def load_orientation_model(weights_path):
    if not os.path.exists(weights_path):
        return None
    state = torch.load(weights_path, map_location="cpu")
    is_v1 = any(k.startswith("pair_q.") for k in state.keys())
    embed_dim = state["input_proj.0.weight"].shape[0]
    num_layers = 1 + max(
        int(k.split(".")[2])
        for k in state.keys() if k.startswith("transformer.layers.")
    )
    if is_v1:
        print(f"  detected v1 weights  (embed={embed_dim}, layers={num_layers})")
        model = _V1OrientationModel(
            feat_dim=FEAT_DIM, embed_dim=embed_dim,
            num_heads=8, num_layers=num_layers, dropout=0.1,
        )
    else:
        from src.model import TemporalReorderModel
        print(f"  detected v2 weights  (embed={embed_dim}, layers={num_layers})")
        model = TemporalReorderModel(
            feat_dim=FEAT_DIM, embed_dim=embed_dim,
            num_heads=8, num_layers=num_layers, dropout=DROPOUT,
        )
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def model_scores(model, feats, mask):
    feats_   = feats.to(DEVICE)
    pad_mask = (~mask).to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=DEVICE.type == "cuda"):
        out = model(feats_, src_key_padding_mask=pad_mask)
    if isinstance(out, tuple):
        out = out[0]
    n = int(mask.sum().item())
    return out[0, :n].float().cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# TSP helpers
# ─────────────────────────────────────────────────────────────────────────────

def cosine_distance_matrix(feats):
    norm   = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
    feats_ = feats / norm
    sim    = feats_ @ feats_.T
    return 1.0 - sim


def nearest_neighbor_path(dist):
    N = dist.shape[0]
    d = dist.copy()
    np.fill_diagonal(d, np.inf)
    start = d.min(axis=1).argmax()
    path    = [int(start)]
    visited = np.zeros(N, dtype=bool)
    visited[start] = True
    current = start
    for _ in range(N - 1):
        row = dist[current].copy()
        row[visited] = np.inf
        nxt = int(row.argmin())
        path.append(nxt)
        visited[nxt] = True
        current = nxt
    return path


def two_opt(path, dist, max_sweeps=30):
    N = len(path)
    if N < 4:
        return path
    path = list(path)
    for _ in range(max_sweeps):
        improved = False
        for i in range(1, N - 1):
            for j in range(i + 1, N):
                if j == N - 1:
                    old = dist[path[i - 1], path[i]]
                    new = dist[path[i - 1], path[j]]
                else:
                    old = (dist[path[i - 1], path[i]]
                           + dist[path[j], path[j + 1]])
                    new = (dist[path[i - 1], path[j]]
                           + dist[path[i], path[j + 1]])
                if new + 1e-12 < old:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    improved = True
        if not improved:
            break
    return path


def orient_by_scores(path, scores):
    if len(path) < 3:
        return path
    s = np.asarray(scores, dtype=np.float32)[path]
    xs = np.arange(len(path), dtype=np.float32)
    xs_rank = xs.argsort().argsort().astype(np.float32)
    s_rank  = s.argsort().argsort().astype(np.float32)
    corr    = np.corrcoef(xs_rank, s_rank)[0, 1]
    if np.isnan(corr):
        return path
    return list(reversed(path)) if corr < 0 else path


# ─────────────────────────────────────────────────────────────────────────────
# Read expected frame counts from sample_submission.csv
# ─────────────────────────────────────────────────────────────────────────────

def load_expected_frame_counts(sample_sub_path):
    """
    Returns dict { "video_XXXX": int } with the number of frames
    expected in the submission for each test video.
    """
    counts = {}
    with open(sample_sub_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order = ast.literal_eval(row["order"])
            counts[row["ID"]] = len(order)
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# Build full 1..N permutation from TSP path on sampled frames
# ─────────────────────────────────────────────────────────────────────────────

def tsp_path_to_full_order(path, sampled_indices, n_valid, expected_n):
    """
    path             : list of indices into sampled array (len = n_valid)
    sampled_indices  : np.array of original frame indices that were sampled
    n_valid          : number of valid sampled frames
    expected_n       : total frames the grading expects

    Returns a list of length expected_n: a permutation of 1..expected_n
    giving the chronological ordering of all frames (1-indexed).
    """
    if n_valid >= expected_n:
        # We have a feature for every frame → direct mapping
        # path[k] is the k-th frame in chronological order (as sampled index)
        # sampled_indices[path[k]] is the original frame position
        return [int(sampled_indices[path[k]]) + 1
                for k in range(expected_n)]

    # We sampled fewer frames than exist. Assign temporal scores to sampled
    # frames based on their TSP position, then interpolate to all frames.
    # Score convention: higher = later in time.
    tsp_scores = np.zeros(n_valid, dtype=np.float64)
    for position_in_path, sampled_idx in enumerate(path):
        tsp_scores[sampled_idx] = position_in_path / max(n_valid - 1, 1)

    # Interpolate scores to every frame index 0..expected_n-1
    sampled_orig = sampled_indices[:n_valid].astype(np.float64)
    full_scores  = np.interp(
        np.arange(expected_n, dtype=np.float64),
        sampled_orig,
        tsp_scores,
    )
    # argsort ascending → chronological order (lowest score = earliest)
    full_order = np.argsort(full_scores)
    return (full_order + 1).tolist()     # 1-indexed


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate_tsp_submission(submission_name="submission_tsp.csv"):
    print("=" * 65)
    print("TSP CHAIN-ORDERING BASELINE")
    print("=" * 65)

    # Load expected frame counts
    sample_sub = SAMPLE_SUB_PATH
    if not os.path.exists(sample_sub):
        # Try a few fallback locations
        candidates = [
            "/content/dataset/sample_submission.csv",
            os.path.join(DRIVE_ROOT, "sample_submission.csv"),
            os.path.join(DRIVE_ROOT, "data", "sample_submission.csv"),
        ]
        for c in candidates:
            if os.path.exists(c):
                sample_sub = c
                break
    if not os.path.exists(sample_sub):
        print(f"ERROR: cannot find sample_submission.csv")
        print(f"  tried: {SAMPLE_SUB_PATH}")
        print("  Please set SAMPLE_SUB_PATH in config.py to the correct path.")
        sys.exit(1)

    expected_counts = load_expected_frame_counts(sample_sub)
    print(f"  expected frame counts loaded for {len(expected_counts)} videos")
    print(f"  frame range: {min(expected_counts.values())} – "
          f"{max(expected_counts.values())}")

    # Load features
    test_feat_dir = os.path.join(FEATURES_DIR, "test")
    if not os.path.isdir(test_feat_dir):
        print(f"\nERROR: {test_feat_dir} does not exist.")
        print("Run feature extraction first:  !python -m src.extract_features")
        sys.exit(1)

    test_dataset = SherlockFeatureDataset(
        features_dir=test_feat_dir, label_file=None,
        is_train=False, preload_to_ram=True,
    )
    if len(test_dataset) == 0:
        print(f"\nERROR: no .pt files in {test_feat_dir}")
        print("Run feature extraction:  !python -m src.extract_features")
        sys.exit(1)
    print(f"  cached test features: {len(test_dataset)}")

    # Load orientation model
    model = None
    if os.path.exists(WEIGHTS_PATH):
        try:
            model = load_orientation_model(WEIGHTS_PATH)
            print(f"  using model for direction: {WEIGHTS_PATH}")
        except Exception as e:
            print(f"  [WARN] could not load model ({e}); using heuristic direction")

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=feature_collate_fn, num_workers=0,
    )

    results = []
    n_interpolated = 0

    for i, (feats, mask, vid_ids, sampled_list, totals) in enumerate(test_loader):
        vid_id  = vid_ids[0]
        sampled = sampled_list[0].numpy()
        n_valid = int(mask.sum().item())

        # Expected frame count from sample_submission
        expected_n = expected_counts.get(vid_id)
        if expected_n is None:
            # Fallback: use what cv2 reported during extraction
            expected_n = int(totals[0])
            print(f"  [WARN] {vid_id} not in sample_submission; "
                  f"using cv2 count = {expected_n}")

        feats_np = feats[0, :n_valid].float().numpy()

        # ── TSP on sampled frames ───────────────────────────────────
        dist = cosine_distance_matrix(feats_np)
        path = nearest_neighbor_path(dist)
        path = two_opt(path, dist)

        # ── Orient direction ────────────────────────────────────────
        if model is not None:
            scores = model_scores(model, feats, mask)
            path   = orient_by_scores(path, scores)
        else:
            # Heuristic: prefer direction where feature-change accelerates
            deltas = np.diff(feats_np[path], axis=0)
            mags   = np.linalg.norm(deltas, axis=1)
            if len(mags) >= 4:
                first_half  = mags[:len(mags) // 2].mean()
                second_half = mags[len(mags) // 2:].mean()
                if first_half > second_half:
                    path = path[::-1]

        # ── Build full 1..N permutation ─────────────────────────────
        predicted_order = tsp_path_to_full_order(
            path, sampled, n_valid, expected_n
        )
        if n_valid < expected_n:
            n_interpolated += 1

        # Sanity: must be a permutation of 1..expected_n
        assert len(predicted_order) == expected_n, \
            f"{vid_id}: got {len(predicted_order)} elements, expected {expected_n}"
        assert sorted(predicted_order) == list(range(1, expected_n + 1)), \
            f"{vid_id}: not a valid 1..{expected_n} permutation"

        order_str = "[" + ", ".join(map(str, predicted_order)) + "]"
        results.append({"ID": vid_id, "order": order_str})

        if (i + 1) % 25 == 0 or (i + 1) == len(test_dataset):
            print(f"  processed {i + 1}/{len(test_dataset)}")

    if n_interpolated > 0:
        print(f"\n  NOTE: {n_interpolated} videos needed interpolation "
              f"(had more frames than MAX_FRAMES={MAX_FRAMES})")
        print(f"  For best TSP quality, re-extract with MAX_FRAMES >= "
              f"{max(expected_counts.values())}")

    df = pd.DataFrame(results)
    df["_n"] = df["ID"].apply(lambda x: int(x.split("_")[-1]))
    df = df.sort_values("_n").drop(columns=["_n"])[["ID", "order"]]

    # Verify all expected videos are present
    missing = set(expected_counts.keys()) - set(df["ID"])
    if missing:
        print(f"\n  WARNING: {len(missing)} videos missing from submission!")
        print(f"  Missing: {sorted(missing)[:10]}...")

    out_path = os.path.join(os.path.dirname(SUBMISSION_PATH), submission_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSubmission written to: {out_path}")
    print(f"  {len(df)} rows, format validated ✓")
    print(df.head(3).to_string(index=False))


if __name__ == "__main__":
    generate_tsp_submission()

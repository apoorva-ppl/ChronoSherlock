"""
Model-based inference with correct output format.

Reads sample_submission.csv to determine the exact expected frame count
per video, ensuring the output is always a valid 1..N permutation.
"""
import os
import sys
import ast
import csv
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from src.config import *
from src.dataset import SherlockFeatureDataset, feature_collate_fn
from src.model import TemporalReorderModel


def load_expected_frame_counts(sample_sub_path):
    counts = {}
    with open(sample_sub_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            order = ast.literal_eval(row["order"])
            counts[row["ID"]] = len(order)
    return counts


def find_sample_submission():
    candidates = [
        SAMPLE_SUB_PATH,
        "/content/dataset/sample_submission.csv",
        os.path.join(DRIVE_ROOT, "sample_submission.csv"),
        os.path.join(DRIVE_ROOT, "data", "sample_submission.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def borda_order_from_pairwise(pair_logits, n_valid):
    sub   = pair_logits[:n_valid, :n_valid]
    probs = 1.0 / (1.0 + np.exp(-sub))
    np.fill_diagonal(probs, 0.0)
    borda = probs.sum(axis=1)
    return np.argsort(-borda).tolist()


@torch.no_grad()
def tta_forward(model, feats, pad_mask, n_passes=TTA_PASSES, noise_std=0.02):
    scores_sum = None
    pair_sum   = None
    for _ in range(n_passes):
        noisy = feats + torch.randn_like(feats) * noise_std
        with torch.autocast(device_type="cuda", dtype=torch.float16,
                            enabled=DEVICE.type == "cuda"):
            s, p = model(noisy, src_key_padding_mask=pad_mask)
        s = s.float().cpu().numpy()
        p = p.float().cpu().numpy()
        scores_sum = s if scores_sum is None else scores_sum + s
        pair_sum   = p if pair_sum   is None else pair_sum + p
    return scores_sum / n_passes, pair_sum / n_passes


def generate_submission():
    print("=" * 65)
    print("INFERENCE (model-based)")
    print("=" * 65)

    # Expected frame counts
    sample_path = find_sample_submission()
    expected_counts = {}
    if sample_path:
        expected_counts = load_expected_frame_counts(sample_path)
        print(f"  expected counts for {len(expected_counts)} videos")
    else:
        print("  [WARN] sample_submission.csv not found — using cv2 frame counts")

    test_feat_dir = os.path.join(FEATURES_DIR, "test")
    test_dataset = SherlockFeatureDataset(
        features_dir=test_feat_dir, label_file=None,
        is_train=False, preload_to_ram=True,
    )
    print(f"  test videos: {len(test_dataset)}")

    model = TemporalReorderModel(
        feat_dim=FEAT_DIM, embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS, dropout=DROPOUT,
    ).to(DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()
    print(f"  loaded weights: {WEIGHTS_PATH}")

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        collate_fn=feature_collate_fn, num_workers=0,
    )

    results = []
    for i, batch in enumerate(test_loader):
        feats, mask, vid_ids, sampled_list, totals = batch
        vid_id   = vid_ids[0]
        sampled  = sampled_list[0].numpy()
        n_valid  = int(mask.sum().item())

        expected_n = expected_counts.get(vid_id, int(totals[0]))

        feats    = feats.to(DEVICE, non_blocking=True)
        pad_mask = (~mask).to(DEVICE)

        avg_scores, avg_pair = tta_forward(model, feats, pad_mask)
        avg_scores = avg_scores[0]
        avg_pair   = avg_pair[0]

        if n_valid >= expected_n:
            sampled_order = borda_order_from_pairwise(avg_pair, expected_n)
            predicted_order = [int(sampled[k]) + 1 for k in sampled_order]
        else:
            # Assign scores from Borda, interpolate to all frames
            sampled_order = borda_order_from_pairwise(avg_pair, n_valid)
            borda_scores = np.zeros(n_valid, dtype=np.float64)
            for pos, si in enumerate(sampled_order):
                borda_scores[si] = pos / max(n_valid - 1, 1)
            full_scores = np.interp(
                np.arange(expected_n, dtype=np.float64),
                sampled[:n_valid].astype(np.float64),
                borda_scores,
            )
            full_order = np.argsort(full_scores)
            predicted_order = (full_order + 1).tolist()

        assert len(predicted_order) == expected_n
        assert sorted(predicted_order) == list(range(1, expected_n + 1))

        order_str = "[" + ", ".join(map(str, predicted_order)) + "]"
        results.append({"ID": vid_id, "order": order_str})

        if (i + 1) % 25 == 0 or (i + 1) == len(test_dataset):
            print(f"  processed {i+1}/{len(test_dataset)}")

    df = pd.DataFrame(results)
    df["_n"] = df["ID"].apply(lambda x: int(x.split("_")[-1]))
    df = df.sort_values("_n").drop(columns=["_n"])[["ID", "order"]]

    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSubmission written to: {SUBMISSION_PATH}")
    print(f"  {len(df)} rows ✓")


if __name__ == "__main__":
    generate_submission()

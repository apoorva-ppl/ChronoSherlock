"""
One-time feature extraction.

Walks every video in TRAIN_DIR and TEST_DIR, extracts up to MAX_FRAMES
frames uniformly, runs DINOv2-Small over them in FP16, and caches the
resulting feature tensor to disk as:

    FEATURES_DIR/train/<video_id>.pt
    FEATURES_DIR/test/<video_id>.pt

Each .pt file stores a dict:
    {
        "features": FloatTensor [N, 384],   # fp16 on disk, fp32 at load
        "sampled_indices": LongTensor [N],   # corrupted-video positions
        "total_frames": int,
    }

Run once before training. Safe to re-run: videos whose features already
exist on disk are skipped, so you can resume if Colab disconnects.
"""
import os
import sys
import time
import torch
import cv2
import numpy as np
from pathlib import Path
from torchvision import transforms
from src.config import (
    TRAIN_DIR, TEST_DIR, FEATURES_DIR, MAX_FRAMES, IMG_SIZE,
    EXTRACT_BATCH, EXTRACT_FP16, BACKBONE_HUB, FEAT_DIM, DEVICE,
)

_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def load_backbone():
    """Load DINOv2-Small from torch.hub, eval mode, on GPU."""
    print(f"Loading backbone: {BACKBONE_HUB} ...")
    model = torch.hub.load(
        "facebookresearch/dinov2", BACKBONE_HUB,
        pretrained=True, verbose=False,
    )
    model.eval().to(DEVICE)
    for p in model.parameters():
        p.requires_grad = False
    if EXTRACT_FP16:
        model = model.half()
    return model


def sample_frames(video_path, max_frames, img_size):
    """Uniformly sample up to `max_frames` frames as a normalised tensor."""
    cap = cv2.VideoCapture(str(video_path))
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
    n_sample = min(max_frames, total)
    indices = np.linspace(0, total - 1, n_sample, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frames.append(torch.zeros(3, img_size, img_size))
            continue
        frame = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # HWC uint8 → CHW float in [0,1]
        t = torch.from_numpy(frame).permute(2, 0, 1).float().div_(255.0)
        frames.append(_NORM(t))
    cap.release()

    if not frames:
        return torch.zeros((1, 3, img_size, img_size)), np.array([0]), total
    return torch.stack(frames), indices, total


@torch.no_grad()
def extract_one(model, video_path, out_path):
    """Extract features for a single video and save to out_path."""
    frames, sampled_indices, total = sample_frames(video_path, MAX_FRAMES, IMG_SIZE)

    # Batch through the backbone
    feats_all = []
    for i in range(0, frames.size(0), EXTRACT_BATCH):
        batch = frames[i:i + EXTRACT_BATCH].to(DEVICE, non_blocking=True)
        if EXTRACT_FP16:
            batch = batch.half()
        out = model(batch)           # [B, FEAT_DIM] (DINOv2 returns CLS token)
        feats_all.append(out.float().cpu())
    feats = torch.cat(feats_all, dim=0)  # [N, FEAT_DIM]

    # Store as fp16 on disk to halve storage — we cast back to fp32 at load time
    torch.save({
        "features": feats.half(),
        "sampled_indices": torch.from_numpy(sampled_indices).long(),
        "total_frames": int(total),
    }, out_path)


def extract_split(model, video_dir, feat_dir):
    """Extract features for every .mp4 in video_dir, saving to feat_dir."""
    video_dir = Path(video_dir)
    feat_dir  = Path(feat_dir)
    feat_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(video_dir.glob("*.mp4"))
    print(f"  {len(videos)} videos in {video_dir}")

    done, skipped = 0, 0
    t0 = time.time()
    for idx, vp in enumerate(videos):
        vid_id = vp.stem.split(" ")[0]     # strip trailing " (1)" etc.
        out_path = feat_dir / f"{vid_id}.pt"
        if out_path.exists():
            skipped += 1
            continue
        try:
            extract_one(model, vp, out_path)
            done += 1
        except Exception as e:
            print(f"  [WARN] failed on {vp.name}: {e}")

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            eta  = (len(videos) - idx - 1) / max(rate, 1e-6)
            print(f"    [{idx+1:5d}/{len(videos)}]  "
                  f"done={done} skipped={skipped}  "
                  f"rate={rate:.1f} vids/s  ETA={eta/60:.1f} min")

    print(f"  Finished: new={done}, already-cached={skipped}")


def main():
    print("=" * 60)
    print("FEATURE EXTRACTION")
    print("=" * 60)
    print(f"Device      : {DEVICE}")
    print(f"Backbone    : {BACKBONE_HUB} ({FEAT_DIM}-dim)")
    print(f"FP16        : {EXTRACT_FP16}")
    print(f"Image size  : {IMG_SIZE}")
    print(f"Max frames  : {MAX_FRAMES}")
    print(f"Batch       : {EXTRACT_BATCH}")
    print(f"Output dir  : {FEATURES_DIR}")
    print("-" * 60)

    model = load_backbone()

    print("\n[TRAIN split]")
    extract_split(model, TRAIN_DIR, os.path.join(FEATURES_DIR, "train"))

    print("\n[TEST split]")
    extract_split(model, TEST_DIR,  os.path.join(FEATURES_DIR, "test"))

    print("\nAll features cached.")


if __name__ == "__main__":
    main()

import os, sys, json, csv, ast, time, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import kendalltau
from scipy.spatial.distance import cdist
import pandas as pd

from src.config import (
    FEATURES_DIR, LABEL_FILE, WEIGHTS_PATH, DEVICE, DRIVE_ROOT,
    FEAT_DIM, SAMPLE_SUB_PATH, MAX_FRAMES, EMBED_DIM, NUM_HEADS,
    NUM_LAYERS, DROPOUT,
)

#location of raw files
_VIDEO_DIRS = {
    "train": ["/content/dataset/train",
              os.path.join(DRIVE_ROOT, "data", "train")],
    "test":  ["/content/dataset/test",
              os.path.join(DRIVE_ROOT, "data", "test")],
}

PIXEL_SIZE = 48        
USE_GRAYSCALE = False    #RGB performed better than grayscale because color provides additional visual information.

#locate the vdo
def find_video_dir(split):
    for d in _VIDEO_DIRS.get(split, []):
        if os.path.isdir(d):
            return d
    return None


def find_video_path(video_dir, vid_id):
    video_dir = Path(video_dir)
    for p in video_dir.glob(f"{vid_id}*.mp4"):
        return str(p)
    return None

#part1
#converts raw vdo into numerical features vectors
def extract_pixel_features(video_path, max_frames=MAX_FRAMES):
    cap = cv2.VideoCapture(video_path) #open the vdo such that openCV can read its frames
    total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1) #Find how many frames are present in the video
    n_sample = min(max_frames, total) #decides the number of frames to process at once
    indices = np.linspace(0, total - 1, n_sample, dtype=int) #Select frames evenly throughout the video.

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            sz = PIXEL_SIZE
            frames.append(np.zeros(sz * sz * (1 if USE_GRAYSCALE else 3),
                                   dtype=np.float32))
            continue
        frame = cv2.resize(frame, (PIXEL_SIZE, PIXEL_SIZE),
                           interpolation=cv2.INTER_AREA)
        if USE_GRAYSCALE:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype(np.float32).ravel() / 255.0
        frames.append(frame)
    cap.release()

    feats = np.stack(frames)  # [N, D]
    return feats, indices, total

#part-2
#nn heuristic provides a fast approximation by repeatedly visiting the closest unvisited frame, making it suitable for hundreds of frames.
#since, tsp exactly is computationally expensive 
def nn_tsp(dist, max_starts=None):
    N = dist.shape[0]
    if N <= 2:
        return list(range(N))
    starts = range(N) if (max_starts is None or max_starts >= N) \
             else np.random.choice(N, max_starts, replace=False)
    best_path, best_cost = None, np.inf
    for start in starts:
        path = [int(start)]
        vis = np.zeros(N, dtype=bool); vis[start] = True
        cur, cost = start, 0.0
        for _ in range(N - 1):
            row = dist[cur].copy(); row[vis] = np.inf
            nxt = int(row.argmin()); cost += row[nxt]
            path.append(nxt); vis[nxt] = True; cur = nxt
        if cost < best_cost:
            best_cost, best_path = cost, path
    return best_path

#part-3
#improves that path by removing inefficient connections and reducing the overall path cost.
def two_opt(path, dist, sweeps=15):
    N = len(path)
    if N < 4: return path
    for _ in range(sweeps):
        improved = False
        for i in range(1, N - 1):
            di = dist[path[i - 1], path[i]]
            for j in range(i + 1, N): #reverse the same edge instead of reconnecting same edge , connect em differently
                if j == N - 1:
                    old, new = di, dist[path[i - 1], path[j]] #compare costs after reversal
                else:
                    old = di + dist[path[j], path[j + 1]]
                    new = dist[path[i - 1], path[j]] + dist[path[i], path[j + 1]]
                if new + 1e-12 < old:
                    path[i:j + 1] = path[i:j + 1][::-1]
                    di = dist[path[i - 1], path[i]]
                    improved = True
        if not improved:
            break
    return path


def solve_tsp(feats, max_starts=80, max_2opt=15):
    """Build distance matrix from pixel features and solve TSP."""
    dist = cdist(feats, feats, metric="sqeuclidean")
    path = nn_tsp(dist, max_starts=max_starts)
    return two_opt(path, dist, sweeps=max_2opt)

# (A) Trained transformer model
class _V1Model(nn.Module):
    def __init__(self, fd=384, ed=384, nh=8, nl=4, dr=0.1):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(fd, ed), nn.LayerNorm(ed), nn.GELU(), nn.Dropout(dr))
        enc = nn.TransformerEncoderLayer(
            d_model=ed, nhead=nh, dim_feedforward=ed * 4, dropout=dr,
            batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc, num_layers=nl)
        self.final_norm = nn.LayerNorm(ed)
        self.score_head = nn.Sequential(
            nn.Linear(ed, ed // 2), nn.GELU(), nn.Dropout(dr),
            nn.Linear(ed // 2, 1))
        self.pair_q = nn.Linear(ed, ed // 2)
        self.pair_k = nn.Linear(ed, ed // 2)

    def forward(self, x, **kw):
        x = self.input_proj(x)
        x = self.transformer(x, **kw)
        return self.score_head(self.final_norm(x)).squeeze(-1)


def load_model():
    if not os.path.exists(WEIGHTS_PATH):
        return None
    st = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    v1 = any(k.startswith("pair_q.") for k in st)
    ed = st["input_proj.0.weight"].shape[0]
    nl = 1 + max(int(k.split(".")[2])
                 for k in st if k.startswith("transformer.layers."))
    if v1:
        m = _V1Model(fd=FEAT_DIM, ed=ed, nh=8, nl=nl)
    else:
        from src.model import TemporalReorderModel
        m = TemporalReorderModel(feat_dim=FEAT_DIM, embed_dim=ed,
                                 num_heads=8, num_layers=nl, dropout=DROPOUT)
    m.load_state_dict(st)
    m.to(DEVICE).eval()
    return m


def load_dino_features(vid_id, split="train"):
    """Load cached DINOv2 features for a single video (for model direction)."""
    p = Path(FEATURES_DIR) / split / f"{vid_id}.pt"
    if not p.exists():
        return None
    pay = torch.load(p, map_location="cpu", weights_only=False)
    return pay["features"].float().numpy()

#transformer model vote
#uses the trained Transformer to determine whether the TSP path is forward or reversed
@torch.no_grad()
def model_direction_score(model, dino_feats, path):

    N = len(path)
    if dino_feats is None or dino_feats.shape[0] < N:
        return 0.0
    ft = torch.from_numpy(dino_feats[:N]).unsqueeze(0).to(DEVICE)
    with torch.autocast(device_type="cuda", dtype=torch.float16,
                        enabled=DEVICE.type == "cuda"):
        out = model(ft)
    if isinstance(out, tuple):
        out = out[0]
    scores = out[0, :N].float().cpu().numpy()
    pos = np.zeros(N)
    for i, idx in enumerate(path):
        pos[idx] = i / max(N - 1, 1)
    c = np.corrcoef(scores, pos)[0, 1]
    return c if not np.isnan(c) else 0.0


# (B) Physics-based pixel heuristics(to solve direction ambiguity)
def pixel_direction_score(pixel_feats, path):
    N = len(path)
    if N < 6:
        return 0.0

    ordered = pixel_feats[path]  # [N, D]

    scores = []


    diffs = np.diff(ordered, axis=0)
    step_norms = np.linalg.norm(diffs, axis=1)
    if len(step_norms) >= 6:
        third = len(step_norms) // 3
        first_third = step_norms[:third].mean()
        last_third = step_norms[-third:].mean()
        # Positive if steps get larger → consistent with acceleration
        scores.append((last_third - first_third) /
                       (first_third + last_third + 1e-9))


    if len(step_norms) >= 10:
        half = len(step_norms) // 2
        # Second derivative: changes in step size
        accel = np.abs(np.diff(step_norms)) 
        first_half_smooth = accel[:half].mean()
        second_half_smooth = accel[half:].mean()
        # Prefer direction where early part is smoother
        scores.append((second_half_smooth - first_half_smooth) /
                       (first_half_smooth + second_half_smooth + 1e-9))


    D = pixel_feats.shape[1]
    sz = PIXEL_SIZE
    ch = 1 if USE_GRAYSCALE else 3
    if D == sz * sz * ch:
        # Reshape to images
        imgs = ordered.reshape(N, sz, sz, ch) if ch > 0 else ordered.reshape(N, sz, sz)
        if ch == 3:
            lum = 0.299 * imgs[:, :, :, 2] + 0.587 * imgs[:, :, :, 1] + 0.114 * imgs[:, :, :, 0]
        else:
            lum = imgs.squeeze(-1) if ch == 1 else imgs
        # Compute vertical center of mass per frame
        row_weights = np.arange(sz).reshape(1, sz, 1)
        vcm = (lum * row_weights).sum(axis=(1, 2)) / (lum.sum(axis=(1, 2)) + 1e-9)
        # Fit linear trend
        xs = np.arange(N, dtype=np.float64)
        if vcm.std() > 1e-6:
            slope = np.polyfit(xs, vcm, 1)[0]
            # Positive slope = objects moving down = gravity = forward
            scores.append(np.clip(slope * 50, -1, 1))

    if not scores:
        return 0.0
    return float(np.mean(scores))


# (C) Ensemble
def predict_direction(model, dino_feats, pixel_feats, path):
    """Returns 0=forward, 1=reverse."""
    vote = 0.0

    # Model signal (strongest proven signal: gave τ=0.0945 on its own)
    if model is not None:
        mc = model_direction_score(model, dino_feats, path)
        vote += mc * 2.0

    # Physics pixel heuristics
    pc = pixel_direction_score(pixel_feats, path)
    vote += pc * 1.0

    return 1 if vote < 0 else 0


def evaluate(max_videos=None):
    print("=" * 65)
    print("EVALUATION ON TRAINING DATA (pixel-feature TSP)")
    print("=" * 65)

    video_dir = find_video_dir("train")
    if not video_dir:
        print("  ERROR: training videos not found!")
        print("  Run the data copy step first.")
        sys.exit(1)
    print(f"  Videos: {video_dir}")

    with open(LABEL_FILE) as f:
        labels = json.load(f)
    print(f"  Labels: {len(labels)} videos")

    model = load_model()
    if model:
        print(f"  Model loaded: {WEIGHTS_PATH}")
    else:
        print("  No model — using pixel heuristics only for direction")

    has_dino = os.path.isdir(os.path.join(FEATURES_DIR, "train"))
    if has_dino:
        print(f"  DINOv2 features available for direction")

    # Collect video IDs that exist on disk AND have labels
    all_vids = []
    for p in sorted(Path(video_dir).glob("*.mp4")):
        vid_id = p.stem.split(" ")[0]
        if vid_id in labels:
            all_vids.append((vid_id, str(p)))

    if max_videos and len(all_vids) > max_videos:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_vids), max_videos, replace=False)
        all_vids = [all_vids[i] for i in sorted(idx)]

    print(f"  Evaluating {len(all_vids)} videos...\n")

    undirected, directed_m, directed_p, directed_e = [], [], [], []
    t0 = time.time()

    for i, (vid_id, vpath) in enumerate(all_vids):
        gt = labels[vid_id]
        N = len(gt)

        # Extract pixel features
        pf, si, total = extract_pixel_features(vpath)
        if pf.shape[0] < N:
            # cv2 got fewer frames than the label expects — skip for eval
            continue
        pf_used = pf[:N]

        # TSP on pixel distance
        path = solve_tsp(pf_used, max_starts=80, max_2opt=15)

        # Undirected tau
        tau_f, _ = kendalltau(path, gt)
        tau_r, _ = kendalltau(path[::-1], gt)
        tau_f = 0 if np.isnan(tau_f) else tau_f
        tau_r = 0 if np.isnan(tau_r) else tau_r
        undirected.append(max(abs(tau_f), abs(tau_r)))

        # Direction: model only
        dino = load_dino_features(vid_id, "train") if (model and has_dino) else None
        if model and dino is not None:
            mc = model_direction_score(model, dino, path)
            mp = 1 if mc < 0 else 0
            mpath = path[::-1] if mp == 1 else path
            t_m, _ = kendalltau(mpath, gt)
            directed_m.append(0 if np.isnan(t_m) else t_m)

        # Direction: pixel heuristic only
        pc = pixel_direction_score(pf_used, path)
        pp = 1 if pc < 0 else 0
        ppath = path[::-1] if pp == 1 else path
        t_p, _ = kendalltau(ppath, gt)
        directed_p.append(0 if np.isnan(t_p) else t_p)

        # Direction: ensemble
        ep = predict_direction(model, dino, pf_used, path)
        epath = path[::-1] if ep == 1 else path
        t_e, _ = kendalltau(epath, gt)
        directed_e.append(0 if np.isnan(t_e) else t_e)

        if (i + 1) % 100 == 0:
            el = time.time() - t0
            r = (i + 1) / el
            print(f"  [{i+1:5d}/{len(all_vids)}]  "
                  f"uτ={np.mean(undirected):.4f}  "
                  f"{'mτ=' + f'{np.mean(directed_m):.4f}  ' if directed_m else ''}"
                  f"pτ={np.mean(directed_p):.4f}  "
                  f"eτ={np.mean(directed_e):.4f}  "
                  f"{r:.1f}v/s  ETA={int((len(all_vids)-i-1)/r)}s")

    print(f"\n  {'─' * 55}")
    print(f"  RESULTS ({len(undirected)} training videos):")
    print(f"    Undirected τ (path quality):    {np.mean(undirected):.4f}")
    if directed_m:
        print(f"    Model-directed τ:               {np.mean(directed_m):.4f}")
    print(f"    Pixel-heuristic-directed τ:     {np.mean(directed_p):.4f}")
    print(f"    Ensemble-directed τ:            {np.mean(directed_e):.4f}")
    print(f"  {'─' * 55}")
    return model

def load_expected():
    for p in [SAMPLE_SUB_PATH, os.path.join(DRIVE_ROOT, "sample_submission.csv"),
              os.path.join(DRIVE_ROOT, "data", "sample_submission.csv"),
              "/content/dataset/sample_submission.csv"]:
        if os.path.exists(p):
            c = {}
            with open(p) as f:
                for r in csv.DictReader(f):
                    c[r["ID"]] = len(ast.literal_eval(r["order"]))
            print(f"  Counts from {p}")
            return c
    return {}


def submit(model, name="submission_tsp.csv"):
    print("\n" + "=" * 65)
    print("GENERATING TEST SUBMISSION (pixel-feature TSP)")
    print("=" * 65)

    exp = load_expected()
    if exp:
        print(f"  {len(exp)} videos expected, "
              f"frames {min(exp.values())}–{max(exp.values())}")

    video_dir = find_video_dir("test")
    if not video_dir:
        print("  ERROR: test videos not found!")
        sys.exit(1)
    print(f"  Videos: {video_dir}")

    has_dino = os.path.isdir(os.path.join(FEATURES_DIR, "test"))

    results = []
    t0 = time.time()

    for vid_id in sorted(exp.keys()):
        vpath = find_video_path(video_dir, vid_id)
        if vpath is None:
            print(f"  WARNING: {vid_id} not found in {video_dir}")
            continue

        expected_n = exp[vid_id]

        # Extract pixel features
        pf, si, total = extract_pixel_features(vpath)
        N_feats = pf.shape[0]

        # TSP
        path = solve_tsp(pf, max_starts=None, max_2opt=20)

        # Direction
        dino = load_dino_features(vid_id, "test") if (model and has_dino) else None
        dp = predict_direction(model, dino, pf, path)
        if dp == 1:
            path = path[::-1]

        # Map to 1-indexed submission format
        if N_feats >= expected_n:
            order = [int(si[path[k]]) + 1 for k in range(expected_n)]
        else:
            ts = np.zeros(N_feats, dtype=np.float64)
            for pos, idx in enumerate(path):
                ts[idx] = pos / max(N_feats - 1, 1)
            fs = np.interp(np.arange(expected_n, dtype=np.float64),
                           si[:N_feats].astype(np.float64), ts)
            order = (np.argsort(fs) + 1).tolist()

        assert len(order) == expected_n, f"{vid_id}: {len(order)} != {expected_n}"
        assert sorted(order) == list(range(1, expected_n + 1)), f"{vid_id}: bad perm"

        order_str = "[" + ", ".join(map(str, order)) + "]"
        results.append({"ID": vid_id, "order": order_str})

        n_done = len(results)
        if n_done % 50 == 0 or n_done == len(exp):
            print(f"    [{n_done}/{len(exp)}]  {time.time() - t0:.0f}s")

    df = pd.DataFrame(results)
    df["_n"] = df["ID"].apply(lambda x: int(x.split("_")[-1]))
    df = df.sort_values("_n").drop(columns=["_n"])[["ID", "order"]]

    missing = set(exp.keys()) - set(df["ID"])
    if missing:
        print(f"  WARNING: {len(missing)} missing videos!")

    out = os.path.join(DRIVE_ROOT, name)
    df.to_csv(out, index=False)
    print(f"\n  Saved: {out} ({len(df)} rows ✓)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--max-train", type=int, default=None)
    a = ap.parse_args()

    if not a.eval and not a.submit:
        print("Usage: python -m src.solve --eval --submit")
        sys.exit(0)

    model = load_model()
    if model:
        print(f"Loaded model: {WEIGHTS_PATH}")

    if a.eval:
        model = evaluate(a.max_train)

    if a.submit:
        submit(model)

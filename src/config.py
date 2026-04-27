import torch

# ── Paths ─────────────────────────────────────────────────────────────
TRAIN_DIR    = "/content/dataset/train"
TEST_DIR     = "/content/dataset/test"
LABEL_FILE   = "/content/dataset/train_labels.json"

DRIVE_ROOT       = "/content/drive/MyDrive/MLWare26-Sherlock Files"
WEIGHTS_PATH     = f"{DRIVE_ROOT}/weights/transformer_best_model.pth"
SUBMISSION_PATH  = f"{DRIVE_ROOT}/submission.csv"
SAMPLE_SUB_PATH  = f"{DRIVE_ROOT}/data/sample_submission.csv"

# Features on Drive so they survive VM recycling.
FEATURES_DIR = f"{DRIVE_ROOT}/features"

# ── Feature extractor ─────────────────────────────────────────────────
BACKBONE_HUB   = "dinov2_vits14"
FEAT_DIM       = 384
IMG_SIZE       = 224
EXTRACT_BATCH  = 64
EXTRACT_FP16   = True

# CRITICAL: the problem statement says "at most 150 frames" but the actual
# test set has videos with up to 288 frames. We set MAX_FRAMES = 300 to
# be safe and capture every single frame in every video.
MAX_FRAMES     = 300

# ── Transformer head ──────────────────────────────────────────────────
EMBED_DIM  = 256
NUM_HEADS  = 8
NUM_LAYERS = 2
DROPOUT    = 0.30

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE    = 32
EPOCHS        = 30
LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 5e-4
WARMUP_EPOCHS = 2
GRAD_CLIP     = 1.0

FEAT_NOISE_STD     = 0.10
FRAME_DROPOUT_P    = 0.30

EARLY_STOP_PATIENCE = 7

USE_EMA    = True
EMA_DECAY  = 0.995

PAIRWISE_WEIGHT = 1.0
LISTMLE_WEIGHT  = 0.10

# ── Inference ─────────────────────────────────────────────────────────
TTA_PASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

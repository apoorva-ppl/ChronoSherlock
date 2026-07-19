#all important settings (paths, model parameters, training hyperparameters, etc.) are stored here.
import torch
# paths(where dataset is stored)
TRAIN_DIR    = "/content/dataset/train"
TEST_DIR     = "/content/dataset/test"
LABEL_FILE   = "/content/dataset/train_labels.json"

DRIVE_ROOT       = "/content/drive/MyDrive/MLWare26-Sherlock Files"
WEIGHTS_PATH     = f"{DRIVE_ROOT}/weights/transformer_best_model.pth" #Stores the trained Transformer model.
SUBMISSION_PATH  = f"{DRIVE_ROOT}/submission.csv"
SAMPLE_SUB_PATH  = f"{DRIVE_ROOT}/data/sample_submission.csv"

# Stores extracted DINOv2 features
#instead of every epoch we extract once reuse cached file (faster)
FEATURES_DIR = f"{DRIVE_ROOT}/features"


BACKBONE_HUB   = "dinov2_vits14" #which pretrained model to use
FEAT_DIM       = 384 #each frame has 384 img
IMG_SIZE       = 224 
EXTRACT_BATCH  = 64 #no.of images processed together(large batch more gpu)
EXTRACT_FP16   = True #stores features in float16 not float32 to cut the storage to almost half (while training they convert back to float32)
MAX_FRAMES     = 300 #to cover every video (since 288 frames in dataset)

#transformer settings
EMBED_DIM  = 256 #smaller emebedding -> less computations -> lower memory -> less overfitting
NUM_HEADS  = 8 #uses 8 attention heads(multiple heads-> model focuses more on relationships btw diff frames)
NUM_LAYERS = 2 #a deep network would overfit
DROPOUT    = 0.30 #randomly disable 30% neurons during training (prevents overfitting)

#training params
BATCH_SIZE    = 32 #32 vdo/training step 
EPOCHS        = 30
LEARNING_RATE = 2e-4 #controls how much weight changes per updates
WEIGHT_DECAY  = 5e-4 #regularisation(prevents v. large weights)
WARMUP_EPOCHS = 2 #start with small learning rate + increase gradually
GRAD_CLIP     = 1.0 #limits extremely large gradients(exploding gradients)

FEAT_NOISE_STD     = 0.10 #add gaussian noise for data augmentation + model becomes robust
FRAME_DROPOUT_P    = 0.30 #randomly remove frames during training (data augmentation)

EARLY_STOP_PATIENCE = 7 #prevents overfitting

USE_EMA    = True #uses Average of many previous weights not curr_weights
EMA_DECAY  = 0.995

#loss=pairwiseBCE+ListMLE
PAIRWISE_WEIGHT = 1.0 
LISTMLE_WEIGHT  = 0.10

# inference
TTA_PASSES = 4 #run predictions 4 times n combine the results, reduces randomness

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

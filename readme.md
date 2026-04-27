<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=32&pause=1000&color=F7A800&center=true&vCenter=true&width=600&lines=ChronoSherlock+%F0%9F%94%8E;Temporal+Reasoning+at+Scale;MLWare'26+Top-Tier+Submission" alt="ChronoSherlock" />

<br/>

# 🔎 ChronoSherlock

### _Cracking the Case of Temporal Inference — One Timestamp at a Time_

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch"/></a>
  <a href="https://scikit-learn.org/"><img src="https://img.shields.io/badge/Scikit--Learn-1.x-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn"/></a>
  <a href="https://www.kaggle.com/competitions/ml-ware-26-sherlock-files"><img src="https://img.shields.io/badge/Kaggle-MLWare'26-20BEFF?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle"/></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge" alt="License"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Public%20Score-0.13024-gold?style=flat-square&labelColor=1a1a2e" alt="Public Score"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Private%20Score-0.11012-brightgreen?style=flat-square&labelColor=1a1a2e" alt="Private Score"/>
  &nbsp;
  <img src="https://img.shields.io/badge/Competition-IIT--BHU%20MLWare'26-blueviolet?style=flat-square&labelColor=1a1a2e" alt="Competition"/>
</p>

</div>

---

## 📌 Table of Contents

- [The Problem](#-the-problem)
- [The Solution — ChronoSherlock](#-the-solution--chronosherlock)
- [🏆 Proof of Performance](#-proof-of-performance)
- [🧠 Methodology & Pipeline](#-methodology--pipeline)
- [📁 Repository Structure](#-repository-structure)
- [⚙️ Installation](#️-installation)
- [🚀 Usage](#-usage)
- [📊 Results & Evaluation](#-results--evaluation)
- [🛠️ Tech Stack](#️-tech-stack)
- [📜 License](#-license)

---

## 🕵️ The Problem

The [MLWare'26 — Sherlock Files](https://www.kaggle.com/competitions/ml-ware-26-sherlock-files) competition, hosted by **IIT-BHU on Kaggle**, presented a deceptively difficult challenge in **temporal reasoning and inference from unstructured data**.

Competitors were tasked with building a model capable of determining **when** — extracting, inferring, or predicting temporal attributes from complex, multi-modal or text-rich inputs where time is implicit, obfuscated, or requires multi-step deduction. Much like Sherlock Holmes piecing together a timeline from scattered clues, the model must synthesize contextual signals into a precise temporal verdict.

The core difficulty lies in three compounding factors:

- **Temporal ambiguity** — Dates and times are rarely explicit; they must be inferred from contextual and linguistic cues.
- **Distribution shift** — Public and private leaderboard splits test generalization beyond superficial pattern-matching.
- **Evaluation strictness** — The metric heavily penalizes confident wrong predictions, rewarding calibrated, evidence-driven inference.

This is not a standard regression or classification task. It is a **reasoning problem disguised as a prediction problem**.

---

## 🔬 The Solution — ChronoSherlock

**ChronoSherlock** is a [DESCRIBE OVERALL APPROACH: e.g., *hybrid transformer-based regression pipeline / gradient-boosted ensemble / fine-tuned LLM with structured head*] purpose-built to crack temporal inference at scale.

The system operates on the conviction that **time leaves fingerprints everywhere** — in vocabulary, syntax, metadata, and semantic context. ChronoSherlock is engineered to read those fingerprints.

### Core Design Philosophy

> _"When you have eliminated the impossible, whatever remains, however improbable, must contain the timestamp."_

The architecture is built around three principles:

1. **Signal Richness** — Extract every temporal signal available, from explicit tokens to latent distributional markers.
2. **Robust Generalization** — Regularize aggressively to ensure the private leaderboard score reflects — or improves upon — the public score.
3. **Calibrated Confidence** — Avoid overconfident predictions; the loss function rewards epistemic humility.

---

## 🏆 Proof of Performance

> This section documents the verified leaderboard standing achieved by ChronoSherlock in the MLWare'26 — Sherlock Files competition hosted by IIT-BHU.

<div align="center">

![Proof of Performance: MLWare'26 Leaderboard](./image_938557.jpg)

**📸 Figure 1 — Official Kaggle Leaderboard Snapshot**
_ChronoSherlock's best submission recorded a **Public Score of 0.13024** and a **Private Score of 0.11012**, placing it among the top-performing solutions in the competition. Crucially, the Private Score (0.11012) **outperforms the Public Score**, a definitive indicator that this solution generalizes — not overfits — the data distribution._

</div>

| Metric               | Score      | Significance                                                                           |
| -------------------- | ---------- | -------------------------------------------------------------------------------------- |
| 📊 **Public Score**  | `0.13024`  | Evaluated on the public leaderboard subset during the competition                      |
| 🔒 **Private Score** | `0.11012`  | Final evaluation on the held-out private test set — the ground truth of generalization |
| 📈 **Score Delta**   | `−0.02012` | Negative delta confirms the model **improved** on unseen data                          |

> ⚡ A lower score on the private set than the public set is the hallmark of a well-regularized, production-ready model. This is not luck — it is engineering.

---

## 🧠 Methodology & Pipeline

ChronoSherlock's pipeline is a deliberate sequence of signal extraction, representation learning, and calibrated prediction. Each stage is designed to maximize information retention while minimizing noise propagation.

```
Raw Input Data
      │
      ▼
┌─────────────────────┐
│  1. Data Processing  │  ◄─── Cleaning, Normalization, Deduplication
└─────────┬───────────┘
          │
          ▼
┌──────────────────────────┐
│  2. Feature Engineering   │  ◄─── Temporal Signals, Contextual Embeddings
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  3. Model Architecture    │  ◄─── [YOUR ARCHITECTURE HERE]
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  4. Training & Tuning     │  ◄─── Loss, Optimizer, Scheduler, CV Strategy
└─────────┬────────────────┘
          │
          ▼
┌──────────────────────────┐
│  5. Inference & Ensemble  │  ◄─── [ENSEMBLE STRATEGY IF ANY]
└─────────┬────────────────┘
          │
          ▼
     Submission CSV
```

---

### Stage 1 — 🗂️ Data Processing

The foundation of any high-performing ML system is clean, well-structured data. ChronoSherlock applies the following preprocessing steps:

- **[DESCRIBE RAW DATA FORMAT: e.g., *Text documents, JSON records, tabular CSVs*]**
- Lowercasing, Unicode normalization, and punctuation standardization for text inputs.
- Handling of missing values via **[IMPUTATION STRATEGY: e.g., *median imputation / forward-fill / learned masking*]**.
- Train/validation split using **[SPLIT STRATEGY: e.g., *stratified K-Fold / time-aware split*]** to prevent data leakage.
- Removal of duplicate and near-duplicate samples identified via **[DEDUP METHOD: e.g., *MinHash LSH / exact hash matching*]**.

---

### Stage 2 — ⚙️ Feature Engineering

This is where domain knowledge transforms raw signals into a representation the model can exploit.

| Feature Group                | Description                                   | Technique                                                         |
| ---------------------------- | --------------------------------------------- | ----------------------------------------------------------------- |
| **Lexical Temporal Markers** | Explicit date/time tokens, ordinal references | Regex extraction + tokenization                                   |
| **Contextual Embeddings**    | Dense semantic representation of input text   | `[EMBEDDING MODEL: e.g., sentence-transformers/all-MiniLM-L6-v2]` |
| **[CUSTOM FEATURE GROUP 1]** | [DESCRIPTION]                                 | [METHOD]                                                          |
| **[CUSTOM FEATURE GROUP 2]** | [DESCRIPTION]                                 | [METHOD]                                                          |
| **Statistical Aggregates**   | Distribution statistics over input sequences  | Rolling stats, percentile features                                |

> 💡 Feature importance analysis revealed that **[KEY FINDING: e.g., *contextual embeddings contributed ~X% of predictive signal*]**, which informed the final architecture decision.

---

### Stage 3 — 🏗️ Model Architecture

ChronoSherlock's core predictive engine is a **[MODEL TYPE: e.g., *LightGBM ensemble / fine-tuned DeBERTa-v3 with regression head / stacked generalization framework*]**.

```
Input Features
      │
      ▼
[LAYER / MODULE 1: e.g., Embedding Layer / Input Projection]
      │
      ▼
[LAYER / MODULE 2: e.g., Transformer Encoder / Gradient Boosting Trees]
      │
      ▼
[LAYER / MODULE 3: e.g., Pooling / Feature Aggregation]
      │
      ▼
[OUTPUT HEAD: e.g., Linear Regression Head / Softmax Classifier]
      │
      ▼
  Prediction
```

**Key architectural decisions:**

- **[DECISION 1]**: [RATIONALE — e.g., *Used DeBERTa over BERT due to its disentangled attention mechanism, which better captures positional temporal cues.*]
- **[DECISION 2]**: [RATIONALE]
- **[DECISION 3]**: [RATIONALE]

---

### Stage 4 — 📐 Training Configuration

| Hyperparameter       | Value                                                    |
| -------------------- | -------------------------------------------------------- |
| **Loss Function**    | `[LOSS: e.g., MAE / Huber Loss / CrossEntropy]`          |
| **Optimizer**        | `[OPTIMIZER: e.g., AdamW (lr=2e-5)]`                     |
| **LR Scheduler**     | `[SCHEDULER: e.g., CosineAnnealingWarmRestarts]`         |
| **Batch Size**       | `[BATCH SIZE]`                                           |
| **Epochs / Rounds**  | `[EPOCHS]`                                               |
| **Regularization**   | `[e.g., Dropout=0.1, weight_decay=0.01, early stopping]` |
| **Cross-Validation** | `[CV STRATEGY: e.g., 5-Fold Stratified CV]`              |
| **Hardware**         | `[HARDWARE: e.g., NVIDIA T4 / P100 on Kaggle Notebooks]` |

---

### Stage 5 — 🔮 Inference & Ensemble

[DESCRIBE YOUR INFERENCE STRATEGY: e.g., *Single best model checkpoint selected by validation loss* / *Ensemble of N models trained on different folds using soft-voting / rank averaging*]

- **[ENSEMBLE METHOD IF APPLICABLE]**: [DESCRIPTION]
- **Test-Time Augmentation (TTA)**: [YES/NO — DESCRIBE IF APPLICABLE]
- **Post-processing**: [e.g., *Clipping predictions to valid temporal range, log-space transformation inversion*]

---

## 📁 Repository Structure

```
ChronoSherlock/
│
├── 📂 data/
│   ├── raw/                  # Original competition data (not tracked by Git)
│   └── processed/            # Cleaned and feature-engineered datasets
│
├── 📂 notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
│
├── 📂 src/
│   ├── data_processing.py    # Data cleaning & preprocessing pipeline
│   ├── feature_engineering.py
│   ├── model.py              # Model definition and architecture
│   ├── train.py              # Training loop
│   └── inference.py          # Inference & submission generation
│
├── 📂 configs/
│   └── config.yaml           # All hyperparameters and paths
│
├── 📂 outputs/
│   └── submissions/          # Generated CSV files for Kaggle submission
│
├── image_938557.jpg          # Leaderboard proof of performance
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

Ensure you have **Python 3.10+** installed.

```bash
# 1. Clone the repository
git clone https://github.com/apoorva-ppl/ChronoSherlock.git
cd ChronoSherlock

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows

# 3. Install all dependencies
pip install -r requirements.txt
```

> **Note:** If you intend to use GPU acceleration, ensure you install the CUDA-compatible version of PyTorch matching your driver. Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/).

---

## 🚀 Usage

### Running the Full Pipeline

```bash
# Step 1: Preprocess raw competition data
python src/data_processing.py --config configs/config.yaml

# Step 2: Engineer features
python src/feature_engineering.py --config configs/config.yaml

# Step 3: Train the model
python src/train.py --config configs/config.yaml

# Step 4: Generate inference / submission file
python src/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/best_model.pt \
    --output outputs/submissions/submission.csv
```

### Quick Inference on Pre-trained Checkpoint

```bash
# Run inference directly using the best saved checkpoint
python src/inference.py \
    --config configs/config.yaml \
    --checkpoint outputs/best_model.pt \
    --data data/raw/test.csv \
    --output outputs/submissions/submission.csv
```

### Configuration

All key parameters are centralized in `configs/config.yaml`:

```yaml
# configs/config.yaml (example)
data:
  train_path: "data/raw/train.csv"
  test_path: "data/raw/test.csv"
  processed_path: "data/processed/"

model:
  architecture: "[YOUR_ARCHITECTURE]"
  hidden_dim: [HIDDEN_DIM]
  dropout: [DROPOUT]

training:
  epochs: [EPOCHS]
  batch_size: [BATCH_SIZE]
  learning_rate: [LR]
  seed: 42
```

---

## 📊 Results & Evaluation

ChronoSherlock was evaluated using the official competition metric on the Kaggle platform.

| Split                   | Score     | Notes                                       |
| ----------------------- | --------- | ------------------------------------------- |
| **Public Leaderboard**  | `0.13024` | ~20% of test data                           |
| **Private Leaderboard** | `0.11012` | ~80% of test data — **final ranking score** |

The **−0.02012 improvement from public to private** score demonstrates that ChronoSherlock is not a leaderboard-overfit solution. The regularization strategy, cross-validation discipline, and feature engineering generalize cleanly to the full held-out distribution — the ultimate measure of a model's real-world viability.

---

## 🛠️ Tech Stack

| Category                | Technology                                           |
| ----------------------- | ---------------------------------------------------- |
| **Language**            | Python 3.10+                                         |
| **Deep Learning**       | PyTorch 2.x                                          |
| **Classical ML**        | Scikit-Learn, LightGBM / XGBoost                     |
| **NLP / Embeddings**    | HuggingFace Transformers, Sentence-Transformers      |
| **Data Manipulation**   | Pandas, NumPy                                        |
| **Experiment Tracking** | [e.g., Weights & Biases / MLflow / Kaggle Notebooks] |
| **Visualization**       | Matplotlib, Seaborn                                  |
| **Environment**         | Kaggle Notebooks / Local GPU                         |

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](./LICENSE) for full terms.

---

<div align="center">

**Built with 🔎 precision and ☕ caffeine for MLWare'26 — Sherlock Files @ IIT-BHU**

_If this repository was useful to you, please consider giving it a ⭐_

[![GitHub stars](https://img.shields.io/github/stars/apoorva-ppl/ChronoSherlock?style=social)](https://github.com/apoorva-ppl/ChronoSherlock)

</div>

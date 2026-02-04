# Trustworthy Visual Inference (TVI)

This project implements a **Trustworthy Visual Inference** framework that synergizes **Parametric** (ResNet18) and **Non-Parametric** (Optimal Transport) representations via Evidential Fusion.

## 🚀 Features

- **Training-Free / Post-Hoc Robustness**: Can be applied to trained models.
- **Dual-Stream Evidence**:
    - **Parametric**: Standard Softmax/Evidence from Logits.
    - **Non-Parametric**: Sinkhorn Optimal Transport distance to a Support Set (Training examples).
- **Evidential Fusion**: Dempster-Shafer theory combines evidences to quantify **Uncertainty**.
- **OOD Detection**: High performance in detecting Out-of-Distribution samples (e.g., Noise, SVHN).

## 📂 Project Structure

```
├── conf/               # Configuration files (cifar10.json, cifar100.json)
├── src/                # Source code
│   ├── config.py       # Dynamic configuration loader
│   ├── dataset.py      # Data loaders (CIFAR-10/100)
│   ├── model.py        # ResNet18 Backbone
│   ├── ot_module.py    # Optimal Transport (Sinkhorn) Calculation
│   ├── evidence_module.py # Evidence Extraction
│   ├── fusion_module.py   # Dempster-Shafer Fusion
│   ├── train_backbone.py  # Training Script
│   └── inference.py       # Inference & Verification Script
├── results/            # Results (Logs, Metrics, Plots) - organized by dataset
├── checkpoints/        # Model Weights
├── run.sh              # Startup Script
└── requirements.txt    # Python Dependencies
```

## 🛠️ Installation

```bash
# If using conda
conda create -n tvi python=3.10
conda activate tvi

# Install dependencies
pip install -r requirements.txt
```

## 🏃 Usage

You can use the provided `run.sh` script to launch experiments.

### 1. Training

Train the backbone model (ResNet18):

```bash
# Train on CIFAR-10
./run.sh train conf/cifar10.json

# Train on CIFAR-100
./run.sh train conf/cifar100.json
```

### 2. Inference & Evaluation

Run the full TVI pipeline (Parametric + OT + Fusion) to generate metrics and plots:

```bash
# Inference on CIFAR-10
./run.sh inference conf/cifar10.json

# Inference on CIFAR-100
./run.sh inference conf/cifar100.json
```

## 📊 Results

Results are saved in `results/<dataset_name>/`:
- `metrics.json`: Accuracy (Parametric, Non-Parametric, Fusion), ECE, AUROC.
- `experiment.log`: Detailed logs.
- `uncertainty_distribution.png`: Visualization of uncertainty for ID vs OOD.

## 🔬 Methodology

1.  **Backbone**: ResNet18 extracts features `(B, 512, 4, 4)`.
2.  **Optimal Transport**: We compute the **Sinkhorn Distance** between test image features and support set features, respecting spatial structure.
3.  **Fusion**: Evidence from the network (Softmax) and Memory (OT) is fused using Dempster-Shafer rules to handle conflicting information.

## 📝 License

MIT

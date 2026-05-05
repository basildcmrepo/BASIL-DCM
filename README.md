<h1 align="center">
  <img src="images/logo.png" alt="BASIL Logo" width="40" align="middle"/> 
  BASIL-DCM: Biophysical Amortized Scalable Inference for Latent Dynamic Causal Modeling
</h1>

This repository contains the official PyTorch implementation of **BASIL** (Physics-Informed Amortized Inference Model), as introduced in our NeurIPS paper: *Estimating the directed, weighted, and signed network of influences among brain regions from fMRI*.

BASIL addresses the computational bottleneck of classical Dynamic Causal Modeling (DCM) by using amortized inference. It combines a Mamba-based temporal encoder with an ROI-wise Spatial Transformer to estimate subject-specific directed connectivity and biophysical DCM parameters in a single forward pass, regularized by a differentiable cross-spectral density (CSD) objective.

![Overview of BASIL . BASIL amortizes DCM inversion by mapping resting-state fMRI to
subject-specific effective connectivity and biophysical parameters. ROI time series are first encoded
by a Mamba-based temporal module with phase-aware timing features. A spatial Transformer then
models adaptive inter-regional interactions. A subject-level FiLM module conditions connectivity
prediction on global brain-state context. Estimated DCM parameters are passed through a differen-
tiable CSD module, enforcing consistency with the DCM forward model in the spectral domain.](images/Fig1.jpg)

## 📂 Repository Structure

```text
BASIL-DCM/
├── data/                   
├── checkpoints/            # Directory for saved model weights
├── src/
│   ├── model/
│   │   ├── basil_core.py   # Main BASIL model architecture
│   │   └── components.py   # Mamba/GRU encoders, Phase CNN, and MLP heads
│   ├── physics/
│   │   ├── test_CSD_torch.py # Differentiable Analytic CSD module
│   │   └── __init__.py
│   ├── utils/
│   │   ├── data_loader.py  # Dataset and TargetScaler
│   │   ├── losses.py       # Composite, contrastive, and CSD loss functions
│   │   └── metrics.py      # Edge, sign, and network validation metrics
│   └── trainer.py          # PyTorch Lightning training module
├── scripts/
│   └── train_basil.py      # Main execution script for training
├── requirements.txt        # Python dependencies
└── README.md
```

## ⚙️ Installation

We recommend using a virtual environment (e.g., Conda) to manage dependencies.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/basildcmrepo/BASIL-DCM.git
   cd BASIL-DCM
   ```

2. **Create a Conda environment:**
   ```bash
   conda create -n basil_env python=3.10
   conda activate basil_env
   ```

3. **Install PyTorch:**
   Please install PyTorch matching your CUDA version. For example (CUDA 11.8):
   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: `mamba-ssm` requires a CUDA compiler (like `nvcc`) available in your system path during installation.*

## 📊 Dataset Preparation

The model expects empirical fMRI data derived from the Human Connectome Project (HCP). 

Place your `.npz` parameter and time-series files inside the `data/DCM_params/` directory (or update the `DATA_DIR` path in the data loader). Ensure the following keys/files are present:
* `time_series.npz` (Y)
* `A.npz` (Effective Connectivity means)
* `A_Vp.npz` (Effective Connectivity variances)
* `transit.npz`, `aa.npz`, `b.npz`, `c.npz` (Biophysical parameters)
* `CSD.npz` and `Hz.npz` (Cross-spectral density and frequency bins)

## 🚀 Training

To train BASIL from scratch, simply execute the training script. This script utilizes PyTorch Lightning and logs metrics to Weights & Biases (`wandb`).

```bash
python scripts/train_basil.py
```

You can configure hyperparameters (e.g., Mamba layers, Transformer heads, CSD loss weight) directly within the `cfg` dictionary inside `scripts/train_basil.py`.

## 📜 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@inproceedings{basil2026,
  title={Estimating the directed, weighted, and signed network of influences among brain regions from fMRI},
  author={Author Names},
  booktitle={Placeholder},
  year={2026}
}
```

## License
MIT License
Copyright (c) 2026 [Annonymous]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
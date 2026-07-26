<h1 align="center">
  <img src="images/logo.png" alt="BASIL Logo" width="60" align="middle"/> 
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


## 🧭 Quick-Start Tutorial for New Users

This section provides a minimal end-to-end workflow for preparing a dataset, configuring BASIL, training the model, and adapting the code to a new resting-state fMRI cohort. The main training entry point is `scripts/train_basil.py`, while the dataset and target-scaling logic are implemented in `src/utils/data_loader.py`.

### 1. Understand the expected inputs

Each sample consists of an ROI-by-time resting-state fMRI recording together with the corresponding DCM parameters used as supervised simulation targets.

| Input or target | Description | Typical shape |
| --- | --- | --- |
| `Y` | Resting-state fMRI ROI time series | `(subjects, R, T)` |
| `A` | Directed effective-connectivity mean matrix | `(subjects, R, R)` |
| `A_Vp` | Edge-wise effective-connectivity variance | `(subjects, R, R)` |
| `transit` | ROI-specific hemodynamic transit-time parameters | `(subjects, R)` |
| `aa` | Endogenous neuronal-fluctuation parameters | Subject-level parameter vector |
| `b` | Observation-noise parameters | Subject-level parameter vector |
| `c` | ROI-specific observation-noise parameters | `(subjects, R)` |
| `CSD` | Empirical or simulated cross-spectral density | See `src/utils/data_loader.py` |
| `Hz` | Frequency bins corresponding to the CSD | `(frequencies,)` |

The exact storage conventions used by the repository are defined in `src/utils/data_loader.py`. Before training on a new dataset, verify that all arrays use a consistent subject order and that every target corresponds to the same subject as its time series.

### 2. Prepare the dataset

Place the required `.npz` files inside `data/DCM_params/`, or update the data path used by the loader.

A recommended preparation workflow is:

1. Parcellate each fMRI recording into the same ordered set of `R` ROIs.
2. Ensure that all recordings contain the same number of time points `T`, or apply a consistent cropping/padding strategy before loading.
3. Generate or load the corresponding DCM parameters and CSD targets.
4. Split the data by subject before constructing training and validation loaders.
5. Keep all simulations derived from the same empirical subject within the same split to prevent subject-level leakage.

For a new dataset, inspect the arrays before training:

```python
import numpy as np

Y = np.load("data/DCM_params/time_series.npz")
A = np.load("data/DCM_params/A.npz")

print(Y.files)
print(A.files)
```

Then confirm the selected arrays have compatible subject dimensions and the expected ROI/time dimensions.

### 3. Configure the model

The model configuration is defined in the `cfg` dictionary in `scripts/train_basil.py`.

```python
cfg = dict(
    name="BASIL_custom_dataset",
    temporal_type="mamba",
    d_time=96,
    mamba_d_model=64,
    mamba_layers=4,
    d_node=128,
    n_spatial_layers=4,
    n_heads=8,
    dropout=0.1,
    lambda_csd=0.1,
    lambda_A_contrast=0.5,
)
```

The most important settings are:

| Setting | Meaning |
| --- | --- |
| `d_time` | Temporal embedding dimension produced by the temporal encoder |
| `mamba_d_model` | Internal Mamba state dimension |
| `mamba_layers` | Number of temporal Mamba layers |
| `d_node` | ROI/node embedding dimension used by the spatial Transformer |
| `n_spatial_layers` | Number of spatial self-attention layers |
| `n_heads` | Number of spatial attention heads |
| `dropout` | Dropout probability |
| `lambda_csd` | Weight of the differentiable CSD consistency loss |
| `lambda_A_contrast` | Weight of the connectivity contrastive objective |

Set `R` to the number of ROIs and `T` to the number of time points in your dataset. These values must agree with the arrays returned by the data loader.

### 4. Construct the data loaders

The training script expects:

- `train_loader`
- `val_loader`
- `Hz_t`
- `scaler`

These objects should be created using the dataset and scaling utilities in `src/utils/data_loader.py`. The train/validation split should be performed at the empirical-subject level rather than at the individual simulation level.

Before launching training, inspect one batch:

```python
batch = next(iter(train_loader))

if isinstance(batch, dict):
    for key, value in batch.items():
        if hasattr(value, "shape"):
            print(f"{key}: {tuple(value.shape)}")
else:
    print(type(batch))
```

This is a useful check for mismatched ROI counts, time-series lengths, missing targets, or incorrect batch dimensions.

### 5. Launch training

Once the loaders, frequency grid, scaler, and data dimensions are defined, run:

```python
from scripts.train_basil import run_experiment

best_checkpoint = run_experiment(
    cfg=cfg,
    train_loader=train_loader,
    val_loader=val_loader,
    Hz_t=Hz_t,
    scaler=scaler,
    R=100,
    T=1200,
    project="basil-dcm",
    max_epochs=200,
    device_ids=[0],
)

print(f"Best checkpoint: {best_checkpoint}")
```

The model is trained with PyTorch Lightning. The checkpoint with the lowest `val/loss_A_total` is saved under:

```text
checkpoints/<experiment_name>/
```

Training metrics and learning-rate information are logged to Weights & Biases.

### 6. Adapting BASIL to a new dataset

For a new cohort or acquisition protocol:

1. Use the same ROI ordering for every subject.
2. Update `R` and `T` to match the new data.
3. Recompute the frequency grid and CSD targets when the repetition time or time-series length changes.
4. Fit the target scaler using the training subjects only.
5. Use a subject-level train/validation/test split.
6. Start from the released checkpoint for cross-dataset fine-tuning, or initialise the model from scratch for dataset-specific training.
7. Report the exact preprocessing, ROI definition, number of time points, frequency range, and data split used.

When the new cohort is small, repeated subject-level cross-validation is preferable to relying on a single train/test split. All samples derived from the same empirical subject must remain in the same fold.

### 7. Common troubleshooting checks

- **CUDA or `mamba-ssm` installation errors:** confirm that the installed PyTorch and CUDA versions are compatible and that `nvcc` is available.
- **Shape mismatch in the temporal encoder:** verify that the loaded time series have shape `(batch, R, T)` and that `R` and `T` match the model constructor.
- **Shape mismatch in the connectivity loss:** confirm that `A` and `A_Vp` use the same ROI ordering and have shape `(batch, R, R)`.
- **Unexpected CSD loss values:** confirm that `CSD` and `Hz` were generated using the same sampling interval and frequency convention expected by the physics module.
- **Poor validation performance:** check subject-level splitting, target scaling, parameter ranges, and whether the new dataset differs substantially from the simulation distribution.
- **Out-of-memory errors:** reduce the batch size first; for larger parcellations, also consider reducing `d_node`, the number of attention heads, or the number of spatial layers.

### 8. Reproducibility checklist

For a reproducible experiment, record:

- dataset and preprocessing version;
- ROI parcellation and ROI ordering;
- number of subjects, ROIs, and time points;
- train/validation/test subject identifiers;
- simulation parameter ranges;
- model configuration;
- random seed;
- batch size, learning rate, and number of epochs;
- selected checkpoint and validation criterion.

The provided training script fixes the PyTorch Lightning seed and saves the best checkpoint according to `val/loss_A_total`.

## 📜 Citation

If you use this code or model in your research, please cite our paper:

```bibtex
@inproceedings{basil2026,
  title={BASIL-DCM: Biophysical Amortized Scalable Inference for Latent Dynamic Causal Modeling},
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
"""
Training entry point for BASIL-DCM experiments.

This script provides:
1. A configuration-driven constructor for the BASIL-DCM architecture.
2. A PyTorch Lightning training routine with Weights & Biases logging.
3. Best-checkpoint selection based on validation effective-connectivity loss.

The data loaders, frequency vector (``Hz_t``), parameter scaler, and data
dimensions must be created by the calling notebook or data-preparation script.
"""

import os

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.model.basil_core import BasilDCM
from src.trainer import BasilDCMLit


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model_from_cfg(R, T, cfg):
    """Construct a BASIL-DCM model from an experiment configuration.

    Parameters
    ----------
    R : int
        Number of brain regions (ROIs).
    T : int
        Number of fMRI time points per sample.
    cfg : dict
        Experiment configuration containing the temporal and spatial encoder
        dimensions, layer counts, attention heads, and dropout probability.

    Returns
    -------
    BasilDCM
        Configured BASIL-DCM model.
    """
    return BasilDCM(
        R=R,
        T=T,
        d_time=cfg["d_time"],
        temporal_type=cfg.get("temporal_type", "mamba"),
        mamba_d_model=cfg.get("mamba_d_model", 64),
        mamba_layers=cfg.get("mamba_layers", 4),
        d_node=cfg.get("d_node", 128),
        n_spatial_layers=cfg.get("n_spatial_layers", 4),
        n_heads=cfg.get("n_heads", 8),
        dropout=cfg.get("dropout", 0.1),
    )


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def run_experiment(
    cfg,
    train_loader,
    val_loader,
    Hz_t,
    scaler,
    R,
    T,
    project="basil-dcm",
    max_epochs=200,
    device_ids=[0],
):
    """Train one BASIL-DCM configuration and return its best checkpoint.

    Parameters
    ----------
    cfg : dict
        Model and loss configuration. The dictionary must contain ``name``,
        ``lambda_csd``, and ``lambda_A_contrast`` in addition to the model
        hyperparameters consumed by :func:`build_model_from_cfg`.
    train_loader, val_loader
        PyTorch data loaders for model fitting and validation.
    Hz_t : torch.Tensor
        Frequency grid used by the spectral-DCM/CSD objective.
    scaler
        Parameter scaler used to transform target biophysical parameters.
    R : int
        Number of ROIs.
    T : int
        Number of fMRI time points.
    project : str, optional
        Weights & Biases project name.
    max_epochs : int, optional
        Maximum number of training epochs.
    device_ids : list[int], optional
        GPU indices passed to the PyTorch Lightning trainer.

    Returns
    -------
    str
        Path to the checkpoint with the lowest ``val/loss_A_total``.
    """
    import wandb

    # Fix the random seed across Python workers and accelerator processes to
    # improve reproducibility across repeated experiment runs.
    pl.seed_everything(0, workers=True)

    # Create one W&B run per configuration. Model artifact logging is disabled
    # because checkpoint management is handled explicitly below.
    wandb_logger = WandbLogger(
        project=project,
        name=cfg["name"],
        log_model=False,
    )

    # Build the neural architecture and wrap it in the Lightning module that
    # implements the training, validation, optimisation, and loss logic.
    model = build_model_from_cfg(R, T, cfg)

    lit = BasilDCMLit(
        model=model,
        Hz=Hz_t,
        scaler=scaler,
        lambda_csd=cfg["lambda_csd"],
        lambda_A_contrast=cfg["lambda_A_contrast"],
    )

    # Save only the checkpoint with the lowest validation connectivity loss.
    # Each experiment receives a dedicated directory to avoid overwriting
    # checkpoints from other configurations.
    ckpt_dir = os.path.join("checkpoints", cfg["name"])

    ckpt = ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="val/loss_A_total",
        mode="min",
        save_top_k=1,
        filename=f"{cfg['name']}-{{epoch:02d}}",
    )

    # Record the learning-rate schedule once per epoch in W&B.
    lrmon = LearningRateMonitor(logging_interval="epoch")

    # Full-precision training is used here. ``device_ids`` may contain one or
    # more GPU indices, depending on the intended training configuration.
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=device_ids,
        precision="32",
        logger=wandb_logger,
        callbacks=[ckpt, lrmon],
        log_every_n_steps=1,
    )

    # Fit the model using the supplied subject-level training and validation
    # partitions. Any subject-level leakage prevention must be handled when the
    # data loaders are constructed.
    trainer.fit(
        lit,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # Close the current W&B run cleanly before returning the selected model.
    wandb.finish()

    return ckpt.best_model_path


# ---------------------------------------------------------------------------
# Example experiment configuration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Architecture and objective settings used for the HCP-100 experiment.
    # The configuration name encodes the principal architectural choices to
    # make experiment tracking and checkpoint identification easier.
    cfg = dict(
        name=(
            "BASIL_HCP100_dtim96_dmamba64_mambalayers4_"
            "dnode128_spatial4_heads8_Transit2p5"
        ),
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

    # The following objects must be defined before launching the experiment:
    #   - train_loader and val_loader: subject-separated data loaders
    #   - Hz_t: frequency grid for the spectral objective
    #   - scaler: target-parameter scaler
    #
    # Example:
    # best_checkpoint = run_experiment(
    #     cfg=cfg,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     Hz_t=Hz_t,
    #     scaler=scaler,
    #     R=100,
    #     T=1200,
    #     device_ids=[0],
    # )
    # print(f"Best checkpoint: {best_checkpoint}")

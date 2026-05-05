import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src.model.basil_core import BasilDCM
from src.trainer import BasilDCMLit
# Assumes you load your DataLoader and Hz_t in this file as per notebook logic

def build_model_from_cfg(R, T, cfg):
    return BasilDCM(
        R=R, T=T, d_time=cfg["d_time"], temporal_type=cfg.get("temporal_type", "mamba"),
        mamba_d_model=cfg.get("mamba_d_model", 64), mamba_layers=cfg.get("mamba_layers", 4),
        d_node=cfg.get("d_node", 128), n_spatial_layers=cfg.get("n_spatial_layers", 4),
        n_heads=cfg.get("n_heads", 8), dropout=cfg.get("dropout", 0.1)
    )

def run_experiment(cfg, train_loader, val_loader, Hz_t, scaler, R, T, project="basil-dcm", max_epochs=200, device_ids=[0]):
    import wandb
    pl.seed_everything(0, workers=True)
    
    wandb_logger = WandbLogger(project=project, name=cfg["name"], log_model=False)
    model = build_model_from_cfg(R, T, cfg)
    
    lit = BasilDCMLit(
        model=model, Hz=Hz_t, scaler=scaler, lambda_csd=cfg["lambda_csd"],
        lambda_A_contrast=cfg["lambda_A_contrast"]
    )

    ckpt_dir = os.path.join("checkpoints", cfg["name"])
    ckpt = ModelCheckpoint(dirpath=ckpt_dir, monitor="val/loss_A_total", mode="min", save_top_k=1, filename=f"{cfg['name']}-{{epoch:02d}}")
    lrmon = LearningRateMonitor(logging_interval="epoch")

    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="gpu", devices=device_ids, precision="32",
        logger=wandb_logger, callbacks=[ckpt, lrmon], log_every_n_steps=1
    )

    trainer.fit(lit, train_loader, val_loader)
    wandb.finish()
    return ckpt.best_model_path

if __name__ == "__main__":
    cfg = dict(
        name="BASIL_HCP100_dtim96_dmamba64_mambalayers4_dnode128_spatial4_heads8_Transit2p5",
        temporal_type="mamba", d_time=96, mamba_d_model=64, mamba_layers=4,
        d_node=128, n_spatial_layers=4, n_heads=8, dropout=0.1, lambda_csd=0.1, lambda_A_contrast=0.5
    )
    # run_experiment(cfg, train_loader, val_loader, Hz_t, scaler, R=100, T=1200, device_ids=[0])
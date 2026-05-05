import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict
import numpy as np

from src.utils.losses import (
    weighted_mse_A, cosine_loss_A, pearson_loss_per_subject, 
    sign_consistency_loss, kl_gaussian_elementwise, 
    subject_contrastive_loss, csd_loss_logmag
)
from src.physics.test_CSD_torch import spm_csd_analytic_torch # Assumes you moved your physics script here

class BasilDCMLit(pl.LightningModule):
    def __init__(self, model, Hz: torch.Tensor, scaler=None, lr: float = 3e-4, weight_decay: float = 1e-4,
                 lambda_csd: float = 0.0, w_A_mse: float = 1.0, w_A_cos: float = 0.5, w_A_corr: float = 1.0,
                 w_A_sign: float = 1.0, lambda_A_kl: float = 1e-3, lambda_A_resid: float = 1.0, lambda_A_contrast: float = 2.0,
                 w_transit: float = 3.0, csd_warmup_epochs: int = 30, tau_A_strong: float = 0.02, w_A_strong: float = 5.0,
                 vp_kl_cap: float = 0.25, vp_mse_cap: float = 10.0, log_every_n_epochs: int = 1):
        super().__init__()
        self.save_hyperparameters(ignore=["model", "scaler"])
        self.model = model
        self.register_buffer("Hz", Hz.view(-1).float())
        self.scaler = scaler
        
        # Hyperparams
        self.lr, self.weight_decay, self.lambda_csd = lr, weight_decay, lambda_csd
        self.w_A_mse, self.w_A_cos, self.w_A_corr, self.w_A_sign = w_A_mse, w_A_cos, w_A_corr, w_A_sign
        self.lambda_A_kl, self.lambda_A_resid, self.lambda_A_contrast = lambda_A_kl, lambda_A_resid, lambda_A_contrast
        self.w_transit, self.csd_warmup_epochs = w_transit, csd_warmup_epochs
        self.tau_A_strong, self.w_A_strong, self.vp_kl_cap, self.vp_mse_cap = tau_A_strong, w_A_strong, vp_kl_cap, vp_mse_cap
        self.log_every_n_epochs = log_every_n_epochs
        self._val_cache = []

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def _has_scaled(self, batch, name): return (self.scaler is not None) and (f"{name}_z" in batch)

    def _mse_param(self, batch, name, pred):
        tgt = batch[name].to(self.device)
        if self._has_scaled(batch, name):
            return F.mse_loss(self.scaler.transform(name, pred), batch[f"{name}_z"].to(self.device))
        return F.mse_loss(pred, tgt)

    def _compute_losses(self, batch, out):
        A_true, A_vp = batch["A_mean"].to(self.device), batch["A_vp"].to(self.device)
        A_true_reg = batch["A_mean_z"].to(self.device) if self._has_scaled(batch, "A_mean") else A_true
        A_pred_reg = self.scaler.transform("A_mean", out["A_mu"]) if self._has_scaled(batch, "A_mean") else out["A_mu"]

        loss_A_mse_w = weighted_mse_A(A_pred_reg, A_true_reg, A_vp, self.tau_A_strong, self.w_A_strong, vp_cap=self.vp_mse_cap)
        loss_A_cos = cosine_loss_A(A_pred_reg, A_true_reg)
        loss_A_corr = pearson_loss_per_subject(A_pred_reg, A_true_reg)
        loss_A_sign = sign_consistency_loss(out["A_mu"], A_true, self.tau_A_strong)
        
        loss_A_kl = kl_gaussian_elementwise(out["A_mu"], out["A_var"], A_true, A_vp.clamp_min(1e-6).clamp_max(self.vp_kl_cap)) if self.lambda_A_kl > 0 else torch.tensor(0.0, device=self.device)
        
        A_true_c, A_pred_c = A_true - A_true.mean(dim=0, keepdim=True), out["A_mu"] - out["A_mu"].mean(dim=0, keepdim=True)
        loss_A_resid = ((A_pred_c - A_true_c) ** 2).mean() / (A_true_c ** 2).mean().clamp_min(1e-8)
        loss_A_contrast = subject_contrastive_loss(out["A_mu"], A_true, 0.1)

        loss_A_total = (self.w_A_mse * loss_A_mse_w + self.w_A_corr * loss_A_corr + self.w_A_sign * loss_A_sign +
                        self.lambda_A_kl * loss_A_kl + self.lambda_A_resid * loss_A_resid + self.lambda_A_contrast * loss_A_contrast)

        loss_neural = self._mse_param(batch, "a", out["a"]) + self._mse_param(batch, "b", out["b"]) + self._mse_param(batch, "c", out["c"])
        loss_hemo = self.w_transit * self._mse_param(batch, "transit", out["transit"])

        loss_csd, lambda_csd_eff = torch.tensor(0.0, device=self.device), 0.0
        if self.lambda_csd > 0:
            lambda_csd_eff = self.lambda_csd * min(1.0, float(self.current_epoch + 1) / float(self.csd_warmup_epochs))
            CSD_pred = spm_csd_analytic_torch(out["A_mu"], torch.nan_to_num(out["a"]), torch.nan_to_num(out["b"]), torch.nan_to_num(out["c"]), torch.nan_to_num(out["transit"]), torch.nan_to_num(batch["decay"]), torch.nan_to_num(batch["epsilon"]), self.Hz)
            loss_csd = csd_loss_logmag(CSD_pred, batch["CSD_data"].to(self.device))

        loss_total = loss_A_total + loss_neural + loss_hemo + lambda_csd_eff * loss_csd
        return loss_total, {"loss_total": loss_total, "loss_A_total": loss_A_total, "loss_neural": loss_neural, "loss_hemo": loss_hemo, "loss_csd": loss_csd}

    def training_step(self, batch, batch_idx):
        loss, comps = self._compute_losses(batch, self.model(batch))
        self.log_dict({f"train/{k}": v for k, v in comps.items()}, on_epoch=True, prog_bar=True, batch_size=batch["Y"].shape[0])
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss, comps = self._compute_losses(batch, self.model(batch))
        self.log_dict({f"val/{k}": v for k, v in comps.items()}, on_epoch=True, prog_bar=True, batch_size=batch["Y"].shape[0])
        return loss
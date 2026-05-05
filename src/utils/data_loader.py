import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.signal import hilbert
from typing import Dict, Optional, Tuple

class TargetScaler:
    """
    Stores mean/std for each target tensor; provides transform/inverse_transform.
    Global mean/std per target name (not per-dimension).
    """
    def __init__(self, eps: float = 1e-6):
        self.eps = float(eps)
        self.stats = {}  # name -> (mean, std)

    @torch.no_grad()
    def fit(self, name: str, x: torch.Tensor):
        x = x.detach().cpu()
        mean = x.mean()
        std = x.std().clamp_min(self.eps)
        self.stats[name] = (mean, std)

    def _check(self, name: str):
        if name not in self.stats:
            raise KeyError(f"TargetScaler: '{name}' not fitted. Available: {list(self.stats.keys())}")

    def transform(self, name: str, x: torch.Tensor) -> torch.Tensor:
        self._check(name)
        mean, std = self.stats[name]
        mean = mean.to(x.device, x.dtype)
        std  = std.to(x.device, x.dtype)
        return (x - mean) / std

    def inverse(self, name: str, x: torch.Tensor) -> torch.Tensor:
        self._check(name)
        mean, std = self.stats[name]
        mean = mean.to(x.device, x.dtype)
        std  = std.to(x.device, x.dtype)
        return x * std + mean


class DCMParamDataset(Dataset):
    def __init__(
        self,
        Y: torch.Tensor,                # (S,R,T)
        A_mean: torch.Tensor,           # (S,R,R)
        A_vp: torch.Tensor,             # (S,R,R) variance
        a: torch.Tensor,                # (S,2)
        b: torch.Tensor,                # (S,2)
        c: torch.Tensor,                # (S,R)
        transit: torch.Tensor,          # (S,R)
        decay: torch.Tensor,            # (S,1)
        epsilon: torch.Tensor,          # (S,1)
        CSD_data: torch.Tensor,         # (S,F,R,R) complex
        Hz: torch.Tensor,               # (F,)
        vp_floor: float = 1e-6,
        vp_cap: float = 10.0,           
        normalize_Y: bool = True,
    ):
        self.Y = Y.float()
        self.A_mean = A_mean.float()
        self.A_vp = A_vp.float()
        self.a = a.float()
        self.b = b.float()
        self.c = c.float()
        self.transit = transit.float()
        self.decay = decay.float()
        self.epsilon = epsilon.float()
        self.CSD_data = CSD_data.to(torch.complex64)
        self.Hz = Hz.view(-1).float()

        self.normalize_Y = bool(normalize_Y)
        self.scaler: Optional[TargetScaler] = None

        S, R, T = self.Y.shape

        # filter subjects with any NaN/Inf
        def finite_mask(x: torch.Tensor) -> torch.Tensor:
            return torch.isfinite(x.reshape(S, -1)).all(dim=1)

        mask = (
            finite_mask(self.Y) & finite_mask(self.A_mean) & finite_mask(self.A_vp) &
            finite_mask(self.a) & finite_mask(self.b) & finite_mask(self.c) &
            finite_mask(self.transit) & finite_mask(self.decay) & finite_mask(self.epsilon) &
            finite_mask(self.CSD_data.real.float()) & finite_mask(self.CSD_data.imag.float())
        )

        self.kept_indices = torch.where(mask)[0].cpu().numpy()
        self.dropped_indices = torch.where(~mask)[0].cpu().numpy()

        n_bad = int((~mask).sum().item())
        if n_bad > 0:
            print(f"[DCMParamDataset] Filtering out {n_bad}/{S} subjects with NaN/Inf.")
            self.Y = self.Y[mask]
            self.A_mean = self.A_mean[mask]
            self.A_vp = self.A_vp[mask]
            self.a = self.a[mask]
            self.b = self.b[mask]
            self.c = self.c[mask]
            self.transit = self.transit[mask]
            self.decay = self.decay[mask]
            self.epsilon = self.epsilon[mask]
            self.CSD_data = self.CSD_data[mask]

        self.S = self.Y.shape[0]

        # variance safety
        self.A_vp = torch.nan_to_num(self.A_vp, nan=vp_floor, posinf=vp_cap, neginf=vp_floor)
        self.A_vp = self.A_vp.clamp_min(vp_floor).clamp_max(vp_cap)

    def set_scaler(self, scaler: TargetScaler):
        self.scaler = scaler

    def __len__(self):
        return self.S

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        Y = self.Y[idx]  # (R,T)
        if self.normalize_Y:
            Y = (Y - Y.mean(dim=-1, keepdim=True)) / (Y.std(dim=-1, keepdim=True) + 1e-6)

        analytic_signal = hilbert(Y.numpy())
        phase_data = torch.from_numpy(np.angle(analytic_signal)).float()

        item = {
            "Y": Y,                         
            "Y_phase": phase_data,
            "A_mean": self.A_mean[idx],     
            "A_vp": self.A_vp[idx],         
            "a": self.a[idx],               
            "b": self.b[idx],               
            "c": self.c[idx],               
            "transit": self.transit[idx],   
            "decay": self.decay[idx],       
            "epsilon": self.epsilon[idx],   
            "CSD_data": self.CSD_data[idx], 
        }

        # Standardized targets
        if self.scaler is not None:
            item["A_mean_z"] = self.scaler.transform("A_mean", item["A_mean"])
            item["a_z"]       = self.scaler.transform("a", item["a"])
            item["b_z"]       = self.scaler.transform("b", item["b"])
            item["c_z"]       = self.scaler.transform("c", item["c"])
            item["transit_z"] = self.scaler.transform("transit", item["transit"])
            item["decay_z"]   = self.scaler.transform("decay", item["decay"])
            item["epsilon_z"] = self.scaler.transform("epsilon", item["epsilon"])

        return item


def make_splits(S: int, val_frac: float = 0.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(S)
    rng.shuffle(idx)
    n_val = int(round(S * val_frac))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    return train_idx, val_idx
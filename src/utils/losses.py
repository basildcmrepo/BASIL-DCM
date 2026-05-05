import torch
import torch.nn.functional as F
from typing import Optional

def pearson_loss_per_subject(A_pred: torch.Tensor, A_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    Ap = A_pred.reshape(A_pred.size(0), -1).float()
    At = A_true.reshape(A_true.size(0), -1).float()
    Ap = Ap - Ap.mean(dim=-1, keepdim=True)
    At = At - At.mean(dim=-1, keepdim=True)
    corr = (Ap * At).sum(dim=-1) / (torch.sqrt((Ap**2).sum(dim=-1).clamp_min(eps) * (At**2).sum(dim=-1).clamp_min(eps)) + eps)
    return (1.0 - corr).mean()

def cosine_loss_A(A_pred: torch.Tensor, A_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    B = A_true.shape[0]
    Ap = A_pred.reshape(B, -1)
    At = A_true.reshape(B, -1)
    cos = (Ap * At).sum(dim=1) / (Ap.norm(dim=1).clamp_min(eps) * At.norm(dim=1).clamp_min(eps))
    return (1.0 - cos).mean()

def sign_consistency_loss(A_pred: torch.Tensor, A_true: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    mask = A_true.abs() > tau
    if not mask.any(): return torch.tensor(0.0, device=A_pred.device)
    return F.mse_loss(torch.tanh(A_pred[mask] * 20.0), torch.sign(A_true[mask]))

def weighted_mse_A(A_pred: torch.Tensor, A_true: torch.Tensor, A_vp: Optional[torch.Tensor] = None, tau: float = 0.02, w_strong: float = 5.0, eps: float = 1e-6, vp_cap: float = 10.0) -> torch.Tensor:
    w = torch.ones_like(A_true)
    if tau > 0: w = w * (1.0 + (A_true.abs() > tau).float() * (w_strong - 1.0))
    if A_vp is not None:
        conf = 1.0 / A_vp.clamp_min(eps)
        conf = torch.clamp(conf / (conf.mean(dim=(-1, -2), keepdim=True) + eps), max=vp_cap)
        w = w * conf
    return (w * (A_pred - A_true) ** 2).sum() / w.sum().clamp_min(1.0)

def kl_gaussian_elementwise(mu_q, var_q, mu_p, var_p, eps=1e-8) -> torch.Tensor:
    var_q, var_p = var_q.clamp_min(eps), var_p.clamp_min(eps)
    return 0.5 * (torch.log(var_q) - torch.log(var_p) + (var_p + (mu_p - mu_q) ** 2) / var_q - 1.0).mean()

def subject_contrastive_loss(A_pred: torch.Tensor, A_true: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B = A_true.shape[0]
    zp, zt = F.normalize(A_pred.reshape(B, -1).float(), dim=1), F.normalize(A_true.reshape(B, -1).float(), dim=1)
    return F.cross_entropy(zp @ zt.T / temperature, torch.arange(B, device=A_true.device))

def csd_loss_logmag(CSD_pred: torch.Tensor, CSD_target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return ((torch.log(torch.abs(CSD_pred) + eps) - torch.log(torch.abs(CSD_target) + eps)) ** 2).mean()
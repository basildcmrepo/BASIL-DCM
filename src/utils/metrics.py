import torch

def _flatten(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1)

def corrcoef_flat(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pearson correlation over ALL elements (scalar)."""
    x = _flatten(x).float()
    y = _flatten(y).float()
    x = x - x.mean()
    y = y - y.mean()
    return (x * y).mean() / (x.std() + eps) / (y.std() + eps)

def r2_score_flat(y_true: torch.Tensor, y_pred: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    y_true = _flatten(y_true).float()
    y_pred = _flatten(y_pred).float()
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().clamp_min(eps)
    return 1.0 - ss_res / ss_tot

def mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(x - y))

def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - y) ** 2))

def safe_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()

def safe_complex_to_cpu(x: torch.Tensor) -> torch.Tensor:
    return x.detach().cpu().to(torch.complex64)

def sign_accuracy_strong(A_pred: torch.Tensor, A_true: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    B, R, _ = A_true.shape
    diag = torch.eye(R, dtype=torch.bool, device=A_true.device).unsqueeze(0).expand(B, -1, -1)
    mask = (A_true.abs() > tau) & (~diag)

    if not mask.any():
        return torch.tensor(float("nan"), device=A_true.device)

    acc = (torch.sign(A_pred[mask]) == torch.sign(A_true[mask])).float().mean()
    return acc

def sign_recall_pos_neg(A_pred: torch.Tensor, A_true: torch.Tensor, tau: float = 0.02):
    B, R, _ = A_true.shape
    diag = torch.eye(R, dtype=torch.bool, device=A_true.device).unsqueeze(0).expand(B, -1, -1)

    pos_mask = (A_true > tau) & (~diag)
    neg_mask = (A_true < -tau) & (~diag)

    pos_recall = torch.tensor(float("nan"), device=A_true.device)
    neg_recall = torch.tensor(float("nan"), device=A_true.device)

    if pos_mask.any():
        pos_recall = (A_pred[pos_mask] > 0).float().mean()
    if neg_mask.any():
        neg_recall = (A_pred[neg_mask] < 0).float().mean()

    return pos_recall, neg_recall

def signed_confusion_counts(A_pred: torch.Tensor, A_true: torch.Tensor, tau: float = 0.02):
    B, R, _ = A_true.shape
    diag = torch.eye(R, dtype=torch.bool, device=A_true.device).unsqueeze(0).expand(B, -1, -1)
    mask = (A_true.abs() > tau) & (~diag)

    if not mask.any():
        return None

    true_sign = (A_true[mask] > 0).long()   # 1 = positive, 0 = negative
    pred_sign = (A_pred[mask] > 0).long()

    tp = ((true_sign == 1) & (pred_sign == 1)).sum().item()
    tn = ((true_sign == 0) & (pred_sign == 0)).sum().item()
    fp = ((true_sign == 0) & (pred_sign == 1)).sum().item()
    fn = ((true_sign == 1) & (pred_sign == 0)).sum().item()

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
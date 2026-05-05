import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm.modules.mamba_simple import Mamba
except ImportError:
    from mamba_ssm import Mamba

class TemporalEncoderMambaPooling(nn.Module):
    def __init__(self, d_time: int, d_model: int = 32, n_layers: int = 2, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.in_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([Mamba(d_model=d_model) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.att_w = nn.Parameter(torch.randn(d_model) * 0.02)

        out_in = 2 * d_model if bidirectional else d_model
        self.out_proj_global = nn.Sequential(nn.Linear(out_in, out_in), nn.GELU(), nn.Linear(out_in, d_time))
        self.out_proj_timing = nn.Sequential(nn.Linear(out_in, out_in), nn.GELU(), nn.Linear(out_in, d_time))

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("btd,d->bt", h, self.att_w)
        a = torch.softmax(logits, dim=1)
        return torch.einsum("btd,bt->bd", h, a)

    def forward(self, y: torch.Tensor):
        x = self.in_proj(y)
        hf = x
        for blk in self.blocks: hf = blk(hf)
        hf = self.drop(self.ln(hf))
        z_pooled = self._pool(hf)
        z_last = hf[:, -1, :]

        if self.bidirectional:
            xr = torch.flip(x, dims=[1])
            hb = xr
            for blk in self.blocks: hb = blk(hb)
            hb = self.drop(self.ln(hb))
            z_pooled = torch.cat([z_pooled, self._pool(hb)], dim=-1)
            z_last = torch.cat([z_last, hb[:, -1, :]], dim=-1)

        return self.out_proj_global(z_pooled), self.out_proj_timing(z_last)

class TemporalEncoderGRUPooling(nn.Module):
    def __init__(self, d_time: int, hidden_size: int = 64, n_layers: int = 2, dropout: float = 0.0, bidirectional: bool = True):
        super().__init__()
        self.gru = nn.GRU(1, hidden_size, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0.0, bidirectional=bidirectional)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.ln = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.att_w = nn.Parameter(torch.randn(out_dim) * 0.02)
        self.out_proj_global = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, d_time))
        self.out_proj_timing = nn.Sequential(nn.Linear(out_dim, out_dim), nn.GELU(), nn.Linear(out_dim, d_time))

    def _pool(self, h: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("btd,d->bt", h, self.att_w)
        a = torch.softmax(logits, dim=1)
        return torch.einsum("btd,bt->bd", h, a)

    def forward(self, y: torch.Tensor):
        h, _ = self.gru(y)
        h = self.drop(self.ln(h))
        return self.out_proj_global(self._pool(h)), self.out_proj_timing(h[:, -1, :])

class SpatialTransformerROI(nn.Module):
    def __init__(self, R: int, d_node: int, n_layers: int, n_heads: int, dropout: float):
        super().__init__()
        self.R = R
        self.roi_embed = nn.Embedding(R, d_node)
        self.roi_scale = nn.Parameter(torch.tensor(0.0001))
        enc_layer = nn.TransformerEncoderLayer(d_node, n_heads, 4 * d_node, dropout, "gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        B, R, _ = Z.shape
        pos = self.roi_embed(torch.arange(R, device=Z.device)).unsqueeze(0).expand(B, -1, -1)
        return self.encoder(Z + self.roi_scale * pos)

class MultiScalePhaseCNN(nn.Module):
    def __init__(self, out_dim: int = 24):
        super().__init__()
        c1, c2 = out_dim // 3, out_dim // 3
        c3 = out_dim - c1 - c2
        self.conv3 = nn.Conv1d(1, c1, 3, padding=1)
        self.conv9 = nn.Conv1d(1, c2, 9, padding=4)
        self.conv21 = nn.Conv1d(1, c3, 21, padding=10)
        self.fuse = nn.Sequential(nn.GELU(), nn.Conv1d(out_dim, out_dim, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = torch.cat([self.conv3(x), self.conv9(x), self.conv21(x)], dim=1)
        return self.fuse(z).squeeze(-1)

class SubjectFiLM(nn.Module):
    def __init__(self, d_feat: int, d_cond: int, hidden: int = None, mod_scale: float = 0.3):
        super().__init__()
        hidden = d_feat if hidden is None else hidden
        self.mod_scale = mod_scale
        self.norm = nn.LayerNorm(d_feat)
        self.to_gamma = nn.Sequential(nn.LayerNorm(d_cond), nn.Linear(d_cond, hidden), nn.GELU(), nn.Linear(hidden, d_feat))
        self.to_beta = nn.Sequential(nn.LayerNorm(d_cond), nn.Linear(d_cond, hidden), nn.GELU(), nn.Linear(hidden, d_feat))

    def forward(self, Z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.to_gamma(cond).unsqueeze(1)
        beta  = self.to_beta(cond).unsqueeze(1)
        return self.norm(Z) * (1.0 + self.mod_scale * torch.tanh(gamma)) + self.mod_scale * beta

class EdgeHeadAMLP(nn.Module):
    def __init__(self, d_node: int, d_cond: int, hidden: int = 128, min_var: float = 1e-6):
        super().__init__()
        self.min_var = min_var
        in_dim = 4 * d_node + d_cond
        self.mu_net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.var_net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.out_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, Zhat: torch.Tensor, g_subject: torch.Tensor):
        B, R, D = Zhat.shape
        zi = Zhat.unsqueeze(2).expand(B, R, R, D)
        zj = Zhat.unsqueeze(1).expand(B, R, R, D)
        gs = g_subject.unsqueeze(1).unsqueeze(1).expand(B, R, R, g_subject.shape[-1])
        feat = torch.cat([zi, zj, zi - zj, zi * zj, gs], dim=-1)
        
        A_mu = torch.tanh(self.mu_net(feat).squeeze(-1)) * self.out_scale * 5.0
        A_var = F.softplus(self.var_net(feat).squeeze(-1)) + self.min_var
        return A_mu, A_var

class StrongGlobalHead(nn.Module):
    def __init__(self, d_in: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 2 * d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_in, d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_in, out_dim))
    def forward(self, g: torch.Tensor) -> torch.Tensor: return self.net(g)

class StrongROIWiseHead(nn.Module):
    def __init__(self, d_in: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, 2 * d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_in, d_in), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_in, out_dim))
    def forward(self, Zhat: torch.Tensor) -> torch.Tensor: return self.net(Zhat)
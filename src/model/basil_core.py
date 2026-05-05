import torch
import torch.nn as nn
from typing import Dict
from .components import (
    TemporalEncoderMambaPooling, TemporalEncoderGRUPooling,
    MultiScalePhaseCNN, SpatialTransformerROI, SubjectFiLM,
    EdgeHeadAMLP, StrongGlobalHead, StrongROIWiseHead
)

class BasilDCM(nn.Module):
    """BASIL: Physics-Informed Amortized Inference for Effective Connectivity."""
    def __init__(
        self, R: int, T: int, d_time: int = 64, temporal_type: str = "mamba",
        mamba_d_model: int = 64, mamba_layers: int = 4, gru_hidden: int = 64, gru_layers: int = 2,
        d_node: int = 128, n_spatial_layers: int = 4, n_heads: int = 4,
        dropout: float = 0.1, bidirectional: bool = True
    ):
        super().__init__()
        self.R, self.T = R, T
        
        if temporal_type.lower() == "mamba":
            self.temporal = TemporalEncoderMambaPooling(d_time, mamba_d_model, mamba_layers, 0.0, bidirectional)
        elif temporal_type.lower() == "gru":
            self.temporal = TemporalEncoderGRUPooling(d_time, gru_hidden, gru_layers, 0.0, bidirectional)
        else:
            raise ValueError("Unknown temporal_type. Use 'mamba' or 'gru'.")

        self.phase_cnn = MultiScalePhaseCNN(out_dim=24)
        self.node_proj = nn.Sequential(nn.LayerNorm(d_time), nn.Linear(d_time, d_node), nn.GELU())
        self.spatial = SpatialTransformerROI(R, d_node, n_spatial_layers, n_heads, dropout)

        self.subject_summary = nn.Sequential(nn.LayerNorm(d_time), nn.Linear(d_time, d_node), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_node, d_node), nn.GELU())
        self.subject_film_A = SubjectFiLM(d_node, d_node, d_node, 0.3)
        self.edge_A = EdgeHeadAMLP(d_node, d_node, 128)

        self.pool_A = nn.Sequential(nn.LayerNorm(d_node), nn.Linear(d_node, 1))
        self.pool_neural = nn.Sequential(nn.LayerNorm(d_node), nn.Linear(d_node, 1))
        self.pool_hemo = nn.Sequential(nn.LayerNorm(d_node), nn.Linear(d_node, 1))

        self.a_head = StrongGlobalHead(d_node, 2, dropout)
        self.b_head = StrongGlobalHead(d_node, 2, dropout)
        self.c_head = StrongROIWiseHead(d_node, 1, dropout)

        phase_dim = 24
        self.transit_pre_head = nn.Sequential(
            nn.LayerNorm(d_time + phase_dim), nn.Linear(d_time + phase_dim, 4 * d_time), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * d_time, 2 * d_time), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d_time, d_time), nn.GELU(), nn.Linear(d_time, 1)
        )
        self.transit_refine_head = nn.Sequential(
            nn.LayerNorm(d_node), nn.Linear(d_node, 2 * d_node), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2 * d_node, d_node), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_node, 1)
        )

    def _pool_graph(self, Zhat: torch.Tensor, pool_net: nn.Module) -> torch.Tensor:
        logits = pool_net(Zhat).squeeze(-1)
        w = torch.softmax(logits, dim=1)
        return torch.einsum("brd,br->bd", Zhat, w)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        Y, Y_phase = batch["Y"], batch["Y_phase"]
        B, R, T = Y.shape

        y_seq, y_phase = Y.reshape(B * R, T, 1), Y_phase.reshape(B * R, 1, T)
        z_global, z_timing = self.temporal(y_seq)
        z_phase = self.phase_cnn(y_phase)

        z_transit_input = torch.cat([z_global, z_phase], dim=-1)
        transit_pre_raw = self.transit_pre_head(z_transit_input).reshape(B, R)

        z_nodes = self.node_proj(z_global).reshape(B, R, -1)
        Zhat_base = self.spatial(z_nodes)

        g_subject_time = z_global.reshape(B, R, -1).mean(dim=1)
        g_subject = self.subject_summary(g_subject_time)

        Zhat_A = self.subject_film_A(Zhat_base, g_subject)
        g_neural = self._pool_graph(Zhat_base, self.pool_neural)

        A_mu, A_var = self.edge_A(Zhat_A, g_subject)
        transit_raw = transit_pre_raw + 0.25 * self.transit_refine_head(Zhat_base).squeeze(-1)

        return {
            "A_mu": A_mu, "A_var": A_var,
            "a": self.a_head(g_neural), "b": self.b_head(g_neural),
            "c": self.c_head(Zhat_base).squeeze(-1),
            "transit": transit_raw, "transit_pre": transit_pre_raw,
        }
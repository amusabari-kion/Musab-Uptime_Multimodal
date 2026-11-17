"""
model.py: Multimodal Anomaly Detection Architectures
Author: A. Musab Ari
Project: YZM5110 Research Project - Multimodal Early Warning and Anomaly Detection in Uptime Applications

This module implements:
1. Baseline MultiMon: 1D-CNN Telemetry Encoder + Transformer Log Encoder + Cross-Modal Attention.
2. GatedMultiMon (Our Improvement): Dynamic Gated Fusion Layer with Modality Reliability
   Estimation to handle missing/corrupted logging inputs (e.g. syslog daemon crashes).
"""

import math
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:
    class TimeSeriesEncoder(nn.Module):
        """
        1D-CNN Encoder for multivariate time-series telemetry (CPU, RAM, Disk, Network).
        Input shape: (batch_size, num_features, seq_len)
        Output shape: (batch_size, seq_len_reduced, hidden_dim) or (batch_size, hidden_dim)
        """
        def __init__(self, in_channels: int = 8, hidden_dim: int = 128, kernel_size: int = 3):
            super().__init__()
            self.conv1 = nn.Conv1d(in_channels, hidden_dim // 2, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn1 = nn.BatchNorm1d(hidden_dim // 2)
            self.conv2 = nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.relu = nn.ReLU()
            self.pool = nn.AdaptiveAvgPool1d(1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (B, C, L)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            pooled = self.pool(out).squeeze(-1)  # (B, hidden_dim)
            return pooled


    class LogTextEncoder(nn.Module):
        """
        Transformer-based / Embedding Log Encoder for unstructured log tokens.
        Can be initialized from scratch or act as a wrapper for pre-trained BERT embeddings.
        Input shape: (batch_size, seq_len) token IDs or (batch_size, seq_len, embed_dim)
        Output shape: (batch_size, hidden_dim)
        """
        def __init__(self, vocab_size: int = 5000, embed_dim: int = 128, hidden_dim: int = 128, num_layers: int = 2):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 2,
                dropout=0.1,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.proj = nn.Linear(embed_dim, hidden_dim)
            self.relu = nn.ReLU()

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
            # input_ids: (B, L)
            emb = self.embedding(input_ids)  # (B, L, D)
            src_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
            out = self.transformer_encoder(emb, src_key_padding_mask=src_key_padding_mask)
            # Global mean pooling over sequence length
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).expand_as(out).float()
                sum_out = torch.sum(out * mask_expanded, dim=1)
                lens = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                pooled = sum_out / lens
            else:
                pooled = out.mean(dim=1)
            return self.relu(self.proj(pooled))


    class CrossModalAttention(nn.Module):
        """
        Cross-Modal Attention layer.
        Queries come from Time-Series telemetry (H_ts), Keys & Values from Logs (H_text).
        H_cross = softmax( (H_ts W_q) (H_text W_k)^T / sqrt(d) ) (H_text W_v)
        """
        def __init__(self, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
            # q: (B, D) -> unsqueeze to (B, 1, D)
            # kv: (B, D) -> unsqueeze to (B, 1, D)
            q_seq = q.unsqueeze(1)
            kv_seq = kv.unsqueeze(1)
            attn_out, _ = self.mha(q_seq, kv_seq, kv_seq)
            attn_out = attn_out.squeeze(1)
            return self.norm(q + self.dropout(attn_out))


    class DynamicGatedFusion(nn.Module):
        """
        Proposed Improvement: Dynamic Gated Fusion (DGF) Layer.
        
        Rationale:
        In industrial cloud deployments, logging daemons (e.g. syslog, fluentd) frequently crash,
        suffer network partition buffer drops, or experience disk starvation during severe outages.
        Standard cross-modal attention forces a normalized softmax distribution over missing/corrupted
        log tokens, severely polluting metric representations.
        
        The Dynamic Gated Fusion computes a reliability gate:
            g = sigmoid(W_g [h_ts || h_text] + b_g)
        When logs are missing or degraded, g dynamically throttles down to 0, shielding the
        telemetry anomaly detector from catastrophic noise.
        """
        def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
            super().__init__()
            self.gate_fc = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Sigmoid()
            )
            self.proj_ts = nn.Linear(hidden_dim, hidden_dim)
            self.proj_cross = nn.Linear(hidden_dim, hidden_dim)
            self.norm = nn.LayerNorm(hidden_dim)

        def forward(self, h_ts: torch.Tensor, h_cross: torch.Tensor, h_text: torch.Tensor) -> torch.Tensor:
            # h_ts: (B, D), h_cross: (B, D), h_text: (B, D)
            concat_repr = torch.cat([h_ts, h_text], dim=-1)  # (B, 2*D)
            gate = self.gate_fc(concat_repr)  # (B, D) in range [0, 1]
            
            # Dynamic weighting: telemetry provides constant backbone, cross-attention is gated
            fused = self.proj_ts(h_ts) + gate * self.proj_cross(h_cross)
            return self.norm(fused), gate


    class MultiMon(nn.Module):
        """
        Baseline MultiMon Architecture:
        Telemetry (1D-CNN) + Log (Transformer) + Cross-Modal Attention -> Classifier
        """
        def __init__(self, num_metrics: int = 8, vocab_size: int = 5000, hidden_dim: int = 128, num_classes: int = 2):
            super().__init__()
            self.metric_encoder = TimeSeriesEncoder(in_channels=num_metrics, hidden_dim=hidden_dim)
            self.log_encoder = LogTextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
            self.cross_attention = CrossModalAttention(hidden_dim=hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

        def forward(self, metrics: torch.Tensor, log_ids: torch.Tensor, log_mask: torch.Tensor = None):
            # metrics: (B, C, L), log_ids: (B, T)
            h_ts = self.metric_encoder(metrics)
            h_text = self.log_encoder(log_ids, log_mask)
            h_cross = self.cross_attention(h_ts, h_text)
            logits = self.classifier(h_cross)
            return logits, None  # None for gate weights


    class GatedMultiMon(nn.Module):
        """
        Proposed GatedMultiMon Architecture:
        Telemetry (1D-CNN) + Log (Transformer) + Cross-Modal Attention + Dynamic Gated Fusion -> Classifier
        """
        def __init__(self, num_metrics: int = 8, vocab_size: int = 5000, hidden_dim: int = 128, num_classes: int = 2):
            super().__init__()
            self.metric_encoder = TimeSeriesEncoder(in_channels=num_metrics, hidden_dim=hidden_dim)
            self.log_encoder = LogTextEncoder(vocab_size=vocab_size, hidden_dim=hidden_dim)
            self.cross_attention = CrossModalAttention(hidden_dim=hidden_dim)
            self.gated_fusion = DynamicGatedFusion(hidden_dim=hidden_dim)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, num_classes)
            )

        def forward(self, metrics: torch.Tensor, log_ids: torch.Tensor, log_mask: torch.Tensor = None):
            h_ts = self.metric_encoder(metrics)
            h_text = self.log_encoder(log_ids, log_mask)
            h_cross = self.cross_attention(h_ts, h_text)
            fused, gate = self.gated_fusion(h_ts, h_cross, h_text)
            logits = self.classifier(fused)
            return logits, gate

else:
    # Fallback placeholder classes if torch is not installed
    class MultiMon:
        def __init__(self, *args, **kwargs):
            self.name = "MultiMon (Baseline)"

    class GatedMultiMon:
        def __init__(self, *args, **kwargs):
            self.name = "GatedMultiMon (Proposed Robust)"


def get_model(model_name: str, num_metrics: int = 8, vocab_size: int = 5000, hidden_dim: int = 128):
    """Factory function to instantiate models."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required to instantiate PyTorch models. Run 'pip install torch'.")
    name = model_name.lower()
    if name in ["multimon", "baseline"]:
        return MultiMon(num_metrics=num_metrics, vocab_size=vocab_size, hidden_dim=hidden_dim)
    elif name in ["gated_multimon", "gatedmultimon", "proposed"]:
        return GatedMultiMon(num_metrics=num_metrics, vocab_size=vocab_size, hidden_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}. Choose 'multimon' or 'gated_multimon'.")


if __name__ == "__main__":
    if TORCH_AVAILABLE:
        print("[model.py] Testing PyTorch architectures...")
        B, C, L, T = 4, 8, 30, 20
        metrics = torch.randn(B, C, L)
        log_ids = torch.randint(0, 500, (B, T))
        mask = torch.ones(B, T)

        baseline = get_model("multimon")
        out_base, _ = baseline(metrics, log_ids, mask)
        print(f"MultiMon Baseline Output Shape: {out_base.shape} (Expected: [{B}, 2])")

        gated = get_model("gated_multimon")
        out_gated, gate = gated(metrics, log_ids, mask)
        print(f"GatedMultiMon Output Shape: {out_gated.shape}, Gate Shape: {gate.shape}")
        print("[model.py] Model architectures validated successfully.")
    else:
        print("[model.py] PyTorch is not yet installed in this environment. Models defined as interfaces.")

"""
Transformer head with RankNet-style (antisymmetric) pairwise logits.

Previously the pairwise head was Q·K^T — unconstrained, so the model could
learn non-transitive "A precedes B precedes C precedes A" relationships
that overfit the training set. Kendall tau requires a total order, so
giving the model freedom to violate transitivity is actively harmful.

Now: pair_logits[i,j] = score_j − score_i  (antisymmetric by construction).
 - Enforces a total order
 - Halves the ranking-specific parameter count
 - Ties the two heads together so they can't contradict each other

Scalar-score convention (unchanged): higher score = LATER in time.
Therefore pair_logits[i,j] = score_j − score_i is HIGH when j is later
than i, i.e. when i precedes j — which is exactly the label we train
against.
"""
import torch
import torch.nn as nn


class TemporalReorderModel(nn.Module):
    def __init__(self, feat_dim=384, embed_dim=256, num_heads=8,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.feat_dim  = feat_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.final_norm  = nn.LayerNorm(embed_dim)

        # Only a scalar-score head. No separate pairwise projection.
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, feats, src_key_padding_mask=None):
        """
        feats                : [B, T, FEAT_DIM]
        src_key_padding_mask : [B, T], True at PADDING positions

        Returns:
            scores      : [B, T]      higher = later in time
            pair_logits : [B, T, T]   score_j − score_i
        """
        x = self.input_proj(feats)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        scores = self.score_head(x).squeeze(-1)               # [B, T]

        # Antisymmetric pairwise logits (RankNet).
        # pair_logits[b, i, j] = scores[b, j] - scores[b, i]
        #                      > 0 when j is later than i, i.e. i precedes j
        pair_logits = scores.unsqueeze(1) - scores.unsqueeze(2)

        return scores, pair_logits

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

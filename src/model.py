import torch
import torch.nn as nn


class TemporalReorderModel(nn.Module):
    def __init__(self, feat_dim=384, embed_dim=256, num_heads=8,
                 num_layers=2, dropout=0.3):
        super().__init__()
        self.feat_dim  = feat_dim
        self.embed_dim = embed_dim

        self.input_proj = nn.Sequential( #contains(linear,layernorm,GELU,dropout)
            nn.Linear(feat_dim, embed_dim), #learns task specific representation instead of directly using generic DINOV2 features
            nn.LayerNorm(embed_dim), 
            nn.GELU(), #GELU provides smoother nonlinear activation than ReLU
            nn.Dropout(dropout), #stabilizes the transformer output before passing it to the prediction head.
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
        #ensures consistency + transitive frame rankings
        #pairwise seperate predictions have v large params + inconsistent result
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, feats, src_key_padding_mask=None):
        x = self.input_proj(feats)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.final_norm(x)
        scores = self.score_head(x).squeeze(-1)        
        pair_logits = scores.unsqueeze(1) - scores.unsqueeze(2)

        return scores, pair_logits

    def param_summary(self):
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

import torch
import torch.nn as nn

class MultiHeadLatentAttention(nn.Module):
    """Attend from learned latent queries to modality tokens.
    Returns a fused latent vector by averaging the attended queries.
    """
    def __init__(self, d_model=256, n_heads=4, n_latent=8):
        super().__init__()
        self.n_latent = n_latent
        self.latent = nn.Parameter(torch.randn(n_latent, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, modality_tokens: torch.Tensor):
        # modality_tokens: [B, T, D]
        B = modality_tokens.size(0)
        Q = self.latent.unsqueeze(0).repeat(B, 1, 1)  # [B, L, D]
        attn_out, _ = self.attn(Q, modality_tokens, modality_tokens, need_weights=False)  # [B, L, D]
        out = self.norm(attn_out + Q)
        fused = out.mean(dim=1)  # [B, D]
        return fused

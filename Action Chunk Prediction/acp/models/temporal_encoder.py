# acp/models/temporal_encoder.py
import torch, math
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1,L,D]

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L]

class TransformerTemporalEncoder(nn.Module):
    def __init__(self, d_model=256, depth=4, heads=4, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model*4,
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.pe = PositionalEncoding(d_model)
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, x):  # x: [B,T,D]
        x = self.pe(x)
        return self.enc(x)

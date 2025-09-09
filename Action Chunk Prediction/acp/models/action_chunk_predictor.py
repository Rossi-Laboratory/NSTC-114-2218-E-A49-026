# acp/models/action_chunk_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .temporal_encoder import TransformerTemporalEncoder
from .boundary_detector import BoundaryDetector
from .vq.vector_quantizer import VectorQuantizer
from .fast.aggregator import segment_mean

class ActionChunkPredictor(nn.Module):
    """Encode sequence -> boundary logits -> aggregate -> VQ (optional)."""
    def __init__(self, d_model=256, use_vq=True, codebook_size=128, depth=4, heads=4, dropout=0.1):
        super().__init__()
        self.enc = TransformerTemporalEncoder(d_model, depth, heads, dropout)
        self.boundary = BoundaryDetector(d_model, hidden=d_model, dropout=dropout)
        self.use_vq = use_vq
        self.vq = VectorQuantizer(codebook_size, d_model) if use_vq else None

    @staticmethod
    def logits_to_bool(logits, thresh=0.5):
        return (torch.sigmoid(logits) > thresh)

    def forward(self, seq):  # seq: [B,T,D]
        h = self.enc(seq)                    # [B,T,D]
        logits = self.boundary(h)            # [B,T]
        ends = self.logits_to_bool(logits)   # [B,T] bool
        chunks = segment_mean(h, ends)       # [B,N,D]
        codes = None; vq_loss = torch.tensor(0.0, device=seq.device)
        if self.use_vq:
            chunks, vq_loss, codes = self.vq(chunks)  # [B,N,D], scalar, [B,N]
        return {"boundary_logits": logits, "emb": chunks, "codes": codes, "vq_loss": vq_loss}

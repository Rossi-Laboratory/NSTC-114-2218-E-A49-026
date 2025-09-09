# acp/__init__.py
"""
Action Chunk Prediction (ACP)
=============================

ACP is a research framework for learning **multi-step robot behaviors**
using **action chunks** instead of step-by-step prediction.

Key Features
------------
- FAST-style boundary detection & chunk aggregation
- VQ-based chunk discretization (codebook, quantizer)
- Integration with Vision-Language-Action (VLA-MoE-Manipulation)
- Modular training pipeline (optim, scheduler, losses)
- Inference utilities (decode, export, visualize)

Subpackages
-----------
- acp.data          : datasets & collate functions
- acp.models        : model components (temporal encoder, boundary, VQ, FAST)
- acp.training      : training loop, losses, optim, scheduler
- acp.inference     : inference, decode, export, visualization
- acp.integration   : bridge to external VLA-MoE-Manipulation repo
- acp.utils         : logging, metrics, distributed, checkpoint helpers
"""

__version__ = "0.1.0"

# convenience imports
from acp.models.action_chunk_predictor import ActionChunkPredictor

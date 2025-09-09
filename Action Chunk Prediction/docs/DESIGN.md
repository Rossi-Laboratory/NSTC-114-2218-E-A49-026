# DESIGN
- Temporal encoder (Transformer) -> boundary detector -> segment pooling (FAST-like) -> VQ.
- Optional CL3 hierarchical module can be added later.
- Integration with VLA-MoE happens via `acp/integration/vla_moe_bridge.py` only.

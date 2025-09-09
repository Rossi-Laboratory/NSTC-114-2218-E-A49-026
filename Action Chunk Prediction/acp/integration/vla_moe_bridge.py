# acp/integration/vla_moe_bridge.py
import os, sys

def _add_vla_path(repo_root: str):
    vla_path = os.path.join(repo_root, "third_party", "VLA-MoE-Manipulation")
    if os.path.isdir(vla_path) and vla_path not in sys.path:
        sys.path.insert(0, vla_path)
    return vla_path

def load_vla_inference(repo_root: str):
    vla_path = _add_vla_path(repo_root)
    try:
        from inference import InferenceSession  # from VLA-MoE-Manipulation
        return InferenceSession
    except Exception as e:
        raise ImportError(
            f"Cannot import VLA InferenceSession from {vla_path}. "
            "Make sure you cloned your existing VLA-MoE-Manipulation into third_party/."
        ) from e

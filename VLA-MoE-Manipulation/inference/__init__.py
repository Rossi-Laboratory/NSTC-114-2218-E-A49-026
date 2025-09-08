# inference/__init__.py
"""
Utilities for running VLA-MoE inference programmatically.

Exports:
- load_yaml(path): read a YAML file
- load_model(infer_cfg, model_cfg, device): build VLAMoE and (optionally) load checkpoint
- prepare_inputs(command, image=None, infer_cfg=None, device="cpu"): pack tensors
- run_model(model, inputs): forward pass -> (action, weights)
- pretty_print(action, weights, expert_names): stringify results
- InferenceSession: small helper for one-liner inference
"""

from typing import Optional, Tuple, Dict, Any, Union
import os
import yaml
import torch
import numpy as np
from PIL import Image

from models.vla_moe import VLAMoE
from utils.checkpoint import load_checkpoint
from data.dataset import tokenize

__all__ = [
    "load_yaml",
    "load_model",
    "prepare_inputs",
    "run_model",
    "pretty_print",
    "InferenceSession",
]


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(
    infer_cfg: Dict[str, Any],
    model_cfg: Dict[str, Any],
    device: Union[str, torch.device] = "cpu",
) -> VLAMoE:
    device = torch.device(device)
    model = VLAMoE(model_cfg).to(device)
    ckpt = infer_cfg.get("checkpoint_path")
    if ckpt and os.path.isfile(ckpt):
        load_checkpoint(ckpt, model)
        print(f"[INFO] Loaded checkpoint: {ckpt}")
    else:
        if ckpt:
            print(f"[WARN] Checkpoint not found at {ckpt}; using random weights.")
    return model


def _to_tensor(x, device, dtype=torch.float32):
    t = torch.as_tensor(x, dtype=dtype)
    if t.ndim == 1:
        t = t.unsqueeze(0)
    return t.to(device)


def prepare_inputs(
    command: str,
    image: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
    infer_cfg: Optional[Dict[str, Any]] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, torch.Tensor]:
    ds_cfg = (infer_cfg or {}).get("dataset", {})
    H = int(ds_cfg.get("image_size", 128))
    W = H
    vocab_size = int(ds_cfg.get("vocab_size", 512))
    max_text_len = int(ds_cfg.get("max_text_len", 16))
    proprio_dim = int(ds_cfg.get("proprio_dim", 7))
    tactile_dim = int(ds_cfg.get("tactile_dim", 4))

    # image handling
    if image is None:
        img_t = torch.rand(3, H, W)
    elif isinstance(image, str):
        img = Image.open(image).convert("RGB").resize((W, H))
        img_t = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            img_t = torch.from_numpy(arr).float().permute(2, 0, 1) / 255.0
        else:
            raise ValueError("np.ndarray image must be HxWx3 or HxWx4.")
    elif torch.is_tensor(image):
        img_t = image.float()
    else:
        raise ValueError("Unsupported image type for 'image'.")

    text_ids = tokenize(command, vocab_size, max_text_len)
    proprio = torch.zeros(proprio_dim, dtype=torch.float32)
    tactile = torch.zeros(tactile_dim, dtype=torch.float32)

    return {
        "image": _to_tensor(img_t, device),
        "text_ids": _to_tensor(text_ids, device, dtype=torch.long),
        "proprio": _to_tensor(proprio, device),
        "tactile": _to_tensor(tactile, device),
        "command": command,
    }


@torch.no_grad()
def run_model(model: VLAMoE, inputs: Dict[str, torch.Tensor]):
    model.eval()
    action, weights = model(
        inputs["image"], inputs["text_ids"], inputs["proprio"], inputs["tactile"]
    )
    return action, weights


def pretty_print(
    action: torch.Tensor,
    weights: torch.Tensor,
    expert_names=("grasping", "placement", "tooluse", "tactile"),
) -> str:
    a = action.squeeze(0).detach().cpu().numpy()
    w = weights.squeeze(0).detach().cpu().numpy()
    lines = [
        "Predicted action [dx, dy, dz, droll, dpitch, dyaw, gripper]:",
        np.array2string(a, precision=4),
        "Expert weights:",
        ", ".join(f"{n}: {w[i]:.3f}" for i, n in enumerate(expert_names)),
    ]
    return "\n".join(lines)


class InferenceSession:
    """
    Small helper wrapper:

    >>> sess = InferenceSession("configs/inference.yaml", "configs/model_moe.yaml", device="cpu")
    >>> action, weights = sess.predict("pick up the red block", image_path="some.jpg")
    """

    def __init__(
        self,
        inference_cfg_path: str = "configs/inference.yaml",
        model_cfg_path: str = "configs/model_moe.yaml",
        device: Union[str, torch.device] = "cpu",
    ):
        self.infer_cfg = load_yaml(inference_cfg_path)
        self.model_cfg = load_yaml(model_cfg_path)
        self.device = torch.device(
            device if device is not None else self.infer_cfg.get("device", "cpu")
        )
        self.model = load_model(self.infer_cfg, self.model_cfg, self.device)

    def predict(
        self,
        command: str,
        image_path: Optional[str] = None,
        image_array: Optional[np.ndarray] = None,
        image_tensor: Optional[torch.Tensor] = None,
    ):
        # choose one image source by priority: tensor > array > path > None
        image = (
            image_tensor
            if image_tensor is not None
            else (image_array if image_array is not None else image_path)
        )
        inputs = prepare_inputs(
            command, image=image, infer_cfg=self.infer_cfg, device=self.device
        )
        return run_model(self.model, inputs)

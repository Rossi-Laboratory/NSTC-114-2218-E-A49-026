# inference/visualization.py
"""
Visualization utilities for VLA-MoE manipulation.

Features
--------
1) visualize_action(...)          : overlay a 2D arrow for the predicted (dx, dy) on the image,
                                    plus optional expert-weight bar chart and 7D action bar.
2) visualize_batch_grid(...)      : show a grid of multiple samples (images + predictions).
3) draw_action_arrow(...)         : low-level helper to draw the 2D arrow.
4) bar_expert_weights(...)        : bar chart for expert mixture weights.
5) bar_action_vector(...)         : bar chart for the 7D action vector.

Notes
-----
- Accepts torch.Tensor / numpy.ndarray / PIL.Image inputs for images.
- Keeps dependencies minimal: numpy + matplotlib (+ optional torch/PIL if available).
- The 7D action is interpreted as:
    [dx, dy, dz, droll, dpitch, dyaw, gripper]
  Only (dx, dy) are drawn as a 2D arrow for quick visual intuition.
"""

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt

try:
    import torch
except Exception:  # torch is optional here
    torch = None

try:
    from PIL import Image
except Exception:  # PIL is optional here
    Image = None


ActionVec = Union[np.ndarray, "torch.Tensor", Sequence[float]]  # 7D
Weights = Union[np.ndarray, "torch.Tensor", Sequence[float]]    # E
ImageLike = Union[np.ndarray, "torch.Tensor", "Image.Image"]


DIM_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


# -----------------------------
# utilities
# -----------------------------
def _to_numpy_image(img: ImageLike) -> np.ndarray:
    """Convert image to float32 numpy array in [0, 1], shape (H, W, 3)."""
    if torch is not None and isinstance(img, torch.Tensor):
        # (C,H,W) or (H,W,C)
        arr = img.detach().cpu().float().numpy()
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            # assume CHW
            arr = np.transpose(arr, (1, 2, 0))
        # drop alpha if present
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = np.clip(arr, 0.0, 1.0)
        return arr

    if isinstance(img, np.ndarray):
        arr = img
        if arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
            # HWC
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            arr = arr.astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return np.clip(arr, 0.0, 1.0)
        elif arr.ndim == 2:
            # grayscale -> stack to 3-ch
            arr = np.stack([arr, arr, arr], axis=-1).astype(np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            return np.clip(arr, 0.0, 1.0)

    if Image is not None and isinstance(img, Image.Image):
        arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
        return arr

    raise ValueError("Unsupported image type. Expect torch.Tensor, np.ndarray, or PIL.Image.")


def _to_numpy_vec(x: Union[np.ndarray, "torch.Tensor", Sequence[float]]) -> np.ndarray:
    """Convert vector (action or weights) to 1D float numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy().reshape(-1)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32).reshape(-1)
    return np.asarray(list(x), dtype=np.float32).reshape(-1)


# -----------------------------
# low-level drawing helpers
# -----------------------------
def draw_action_arrow(
    ax: plt.Axes,
    action: ActionVec,
    center: Optional[Tuple[float, float]] = None,
    scale: Optional[float] = None,
    annotate: bool = True,
) -> None:
    """
    Draw a 2D arrow using (dx, dy) from the 7D action vector.
    - center defaults to image center in pixel coords
    - scale defaults to min(H, W) / 6
    """
    a = _to_numpy_vec(action)
    assert a.shape[0] >= 2, "Action must be at least 2D for (dx, dy)."
    dx, dy = float(a[0]), float(a[1])

    H, W = ax.images[0].get_array().shape[:2] if ax.images else (480, 640)
    if center is None:
        center = (W / 2.0, H / 2.0)
    if scale is None:
        scale = min(H, W) / 6.0

    x0, y0 = center
    x1 = x0 + dx * scale
    y1 = y0 - dy * scale  # y axis points down in image coordinates

    ax.arrow(x0, y0, x1 - x0, y1 - y0, width=0.0, head_width=min(H, W) * 0.02, length_includes_head=True)
    if annotate:
        ax.scatter([x0], [y0], s=20)
        ax.text(x0, y0, " start", va="bottom", ha="left")
        ax.text(x1, y1, " end", va="top", ha="left")


def bar_expert_weights(
    ax: plt.Axes,
    weights: Weights,
    expert_names: Optional[Sequence[str]] = None,
    title: str = "Expert Mixture Weights",
) -> None:
    w = _to_numpy_vec(weights)
    if expert_names is None or len(expert_names) != len(w):
        expert_names = [f"E{i}" for i in range(len(w))]
    ax.bar(range(len(w)), w)
    ax.set_xticks(range(len(w)))
    ax.set_xticklabels(expert_names, rotation=30, ha="right")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


def bar_action_vector(
    ax: plt.Axes,
    action: ActionVec,
    dim_names: Sequence[str] = DIM_NAMES,
    title: str = "Action (7D)",
) -> None:
    a = _to_numpy_vec(action)
    n = len(a)
    labels = list(dim_names[:n]) if dim_names else [f"d{i}" for i in range(n)]
    ax.bar(range(n), a)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=0)
    ax.set_title(title)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)


# -----------------------------
# main entrypoints
# -----------------------------
def visualize_action(
    image: ImageLike,
    action: ActionVec,
    weights: Optional[Weights] = None,
    expert_names: Optional[Sequence[str]] = None,
    command: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Show a 2-panel figure:
      Left  : RGB image with 2D arrow for (dx, dy)
      Right : Top: expert weights bar (if provided); Bottom: 7D action bar
    """
    img = _to_numpy_image(image)
    a = _to_numpy_vec(action)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0])

    # Left: image + arrow
    ax_img = fig.add_subplot(gs[:, 0])
    ax_img.imshow(img)
    ax_img.set_axis_off()
    draw_action_arrow(ax_img, a)
    if command:
        ax_img.set_title(f"Command: {command}")

    # Right-top: expert weights
    ax_top = fig.add_subplot(gs[0, 1])
    if weights is not None:
        bar_expert_weights(ax_top, weights, expert_names=expert_names)
    else:
        ax_top.set_axis_off()
        ax_top.text(0.5, 0.5, "No expert weights provided.", ha="center", va="center")

    # Right-bottom: action bar
    ax_bot = fig.add_subplot(gs[1, 1])
    bar_action_vector(ax_bot, a)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def visualize_batch_grid(
    images: Sequence[ImageLike],
    actions: Sequence[ActionVec],
    weights_list: Optional[Sequence[Weights]] = None,
    commands: Optional[Sequence[str]] = None,
    expert_names: Optional[Sequence[str]] = None,
    ncols: int = 3,
    figsize_per_cell: Tuple[float, float] = (4.0, 3.5),
    show: bool = True,
    save_path: Optional[str] = None,
):
    """
    Visualize multiple samples in a grid.
    Each cell shows the image with (dx, dy) arrow and a small bar of expert weights below.

    Args:
        images: list of images
        actions: list of 7D action vectors
        weights_list: list of expert weight vectors (optional)
        commands: list of strings (optional)
        ncols: columns in the grid
        figsize_per_cell: (width, height) per cell
    """
    N = len(images)
    ncols = max(1, ncols)
    nrows = (N + ncols - 1) // ncols

    fig = plt.figure(figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows * 1.4))
    gs = fig.add_gridspec(nrows, ncols)

    for i in range(N):
        r, c = divmod(i, ncols)
        sub_gs = gs[r, c].subgridspec(2, 1, height_ratios=[3, 2])

        # Image + arrow
        ax_img = fig.add_subplot(sub_gs[0, 0])
        img = _to_numpy_image(images[i])
        ax_img.imshow(img)
        ax_img.set_axis_off()
        draw_action_arrow(ax_img, actions[i])
        if commands and i < len(commands) and commands[i]:
            ax_img.set_title(commands[i], fontsize=10)

        # Expert weights mini bar (if provided)
        ax_bar = fig.add_subplot(sub_gs[1, 0])
        if weights_list is not None and i < len(weights_list) and weights_list[i] is not None:
            bar_expert_weights(ax_bar, weights_list[i], expert_names=expert_names, title="Experts")
        else:
            ax_bar.set_axis_off()

    # hide any empty cells
    for j in range(N, nrows * ncols):
        r, c = divmod(j, ncols)
        ax = fig.add_subplot(gs[r, c])
        ax.set_axis_off()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig

"""
Type conversion utilities between ComfyUI tensors, PIL Images, and base64 strings.

All image paths in the node pack flow through these converters so
format changes only need to be made once.
"""

import numpy as np
import torch
from PIL import Image as PILImage


# ---------------------------------------------------------------------------
# ComfyUI tensor ↔ PIL Image
# ---------------------------------------------------------------------------

def comfy_tensor_to_pil(tensor: torch.Tensor, max_size: int = 3072) -> PILImage.Image:
    """Convert the first image in a ComfyUI IMAGE batch to a PIL Image.

    Resizes the image if the longest edge exceeds `max_size` (default 2048) 
    to prevent API timeout/payload size errors.

    Args:
        tensor: Tensor of shape [B, H, W, C], float32 in [0, 1].
        max_size: Maximum allowed size for the longest edge. Set to None to disable.

    Returns:
        An RGB PIL Image.
    """
    img_np = tensor[0].cpu().numpy()
    img_np = (img_np * 255.0).clip(0, 255).astype(np.uint8)
    img = PILImage.fromarray(img_np, mode="RGB")
    
    if max_size is not None:
        w, h = img.size
        longest_edge = max(w, h)
        if longest_edge > max_size:
            scale = max_size / float(longest_edge)
            new_w = int(w * scale)
            new_h = int(h * scale)
            img = img.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
            
    return img


def pil_to_comfy_tensor(pil_image: PILImage.Image) -> torch.Tensor:
    """Convert a PIL Image to a single-image ComfyUI tensor.

    Returns:
        Tensor of shape [1, H, W, 3], float32 in [0, 1].
    """
    rgb = pil_image.convert("RGB")
    np_array = np.array(rgb).astype(np.float32) / 255.0
    return torch.from_numpy(np_array).unsqueeze(0)



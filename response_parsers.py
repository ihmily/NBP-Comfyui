"""
Unified response parsers for Gemini API responses.

Each parser converts an SDK response object into a tuple of
(ComfyUI IMAGE tensor, metadata string).
"""

import torch
from io import BytesIO
from PIL import Image as PILImage
from .type_converters import pil_to_comfy_tensor




# ---------------------------------------------------------------------------
# Gemini generate_content response  (NanoBanana / GeminiChatImage)
# ---------------------------------------------------------------------------

def parse_gemini_response(response) -> tuple[torch.Tensor, str]:
    """Parse a `GenerateContentResponse` from `client.models.generate_content()`.

    Handles mixed text + image responses from Gemini models.

    Args:
        response: SDK response with `.candidates[0].content.parts`.

    Returns:
        (image_tensor [N,H,W,C], concatenated text response).

    Raises:
        RuntimeError: If no image was generated.
    """
    text_parts: list[str] = []
    image_tensors: list[torch.Tensor] = []

    if not response.candidates or not response.candidates[0].content:
        raise RuntimeError(
            "No valid response from the model. The response was empty or malformed."
        )

    parts = response.candidates[0].content.parts or []

    for part in parts:
        if part.text is not None:
            text_parts.append(part.text)
        elif part.inline_data is not None:
            # The SDK Part exposes .as_image() → PIL Image
            try:
                pil_img = part.as_image()
                image_tensors.append(pil_to_comfy_tensor(pil_img))
            except Exception:
                # Fallback: raw bytes
                pil_img = PILImage.open(BytesIO(part.inline_data.data)).convert("RGB")
                image_tensors.append(pil_to_comfy_tensor(pil_img))

    if not image_tensors:
        text_preview = " ".join(text_parts)[:200] if text_parts else "No content"
        raise RuntimeError(
            f"No image was generated. The model returned text only: {text_preview}"
        )

    batch = torch.cat(image_tensors, dim=0)
    text = "\n".join(text_parts)
    return batch, text

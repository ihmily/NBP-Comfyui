"""
ComfyUI-Google-Imagen: Custom node pack for Google Imagen and Nano Banana image generation.
"""

from .nodes import (
    GoogleImagenAPIKeyNode, 
    NanoBananaGenerate, 
    NanoBananaChat
)

NODE_CLASS_MAPPINGS = {
    "GoogleImagenAPIKey": GoogleImagenAPIKeyNode,
    "NanoBananaGenerate": NanoBananaGenerate,
    "NanoBananaChat": NanoBananaChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleImagenAPIKey": "Google API Key",
    "NanoBananaGenerate": "🍌 Nano Banana Generate",
    "NanoBananaChat": "🍌 Nano Banana Chat Edit",
}

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

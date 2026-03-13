# Comprehensive PRD: ComfyUI Custom Node for Google Nano Banana Image Generation

**This PRD defines a ComfyUI custom node package that integrates Google's Nano Banana Pro and Nano Banana 2 image generation models, plus the Imagen 4 family, through Google AI Studio API keys.** The document maps every API parameter, reference image option, and batch capability to specific ComfyUI node inputs and outputs. It draws on official Google documentation, the ComfyUI node development API, and architectural patterns from 14+ existing community implementations.

The two "Nano Banana" models use Google's Gemini `generateContent` endpoint — **not** the standalone Imagen `:predict` endpoint — which means they support conversational multi-turn editing, up to **14 reference images**, and resolutions up to **4K**. A separate Imagen 4 node leverages the `:predict` endpoint for its deterministic seed control and explicit mask-based editing. Together, these nodes cover the full spectrum of Google's image generation capabilities accessible via a single AI Studio API key.

---

## 1. Models and their capabilities at a glance

The package targets five models across two distinct API systems. Understanding this split is essential — it determines endpoint, parameter structure, and feature availability.

### Nano Banana models (Gemini `generateContent` endpoint)

| Model Name | Model ID | Key Capabilities |
|---|---|---|
| **Nano Banana 2** | `gemini-3.1-flash-image-preview` | Speed-optimized, 512px–4K output, **14 aspect ratios** including extreme (1:8, 8:1), thinking levels, Google Search grounding, up to 14 reference images |
| **Nano Banana Pro** | `gemini-3-pro-image-preview` | Studio-quality creative control, 1K–4K output, 10 aspect ratios, thinking with "High" mode, up to 11 reference images (6 object + 5 character) |
| Nano Banana (original) | `gemini-2.5-flash-image` | Legacy stable model, 1K output only, 5 aspect ratios, free tier eligible |

### Imagen 4 models (`:predict` endpoint)

| Model Name | Model ID | Key Capabilities |
|---|---|---|
| **Imagen 4 Standard** | `imagen-4.0-generate-001` | 1K–2K output, 1–4 batch, seed control, negative prompt (via editing model), watermark toggle |
| **Imagen 4 Ultra** | `imagen-4.0-ultra-generate-001` | Highest quality, 1K–2K, same params as Standard |
| **Imagen 4 Fast** | `imagen-4.0-fast-generate-001` | Fastest generation, 1K only, `enhancePrompt` should be `false` for complex prompts |

The fundamental architectural difference: **Nano Banana models generate images as part of a conversational response** (interleaved text + image parts), while **Imagen models use a dedicated image generation endpoint** returning an array of base64 images. This means the node designs diverge at the API call and response-parsing layers.

---

## 2. Node architecture and package structure

Based on analysis of the cleanest existing implementations (NakanoSanku/ComfyUI-Gemini for separation of concerns, ru4ls/ComfyUI_Nano_Banana for feature completeness), the package should contain **four primary nodes** plus utility helpers.

### Package folder layout

```
ComfyUI/custom_nodes/comfyui-google-imagen/
├── __init__.py                    # Node registration (NODE_CLASS_MAPPINGS)
├── nodes/
│   ├── __init__.py
│   ├── nano_banana_generate.py    # Nano Banana text-to-image + image-to-image
│   ├── nano_banana_chat.py        # Multi-turn conversational editing node
│   ├── imagen_generate.py         # Imagen 4 text-to-image
│   └── imagen_edit.py             # Imagen 4 editing (inpaint, outpaint, style/subject ref)
├── helpers/
│   ├── __init__.py
│   ├── client_manager.py          # API client creation and key management
│   ├── type_converters.py         # ComfyUI tensor ↔ base64/PIL conversions
│   ├── response_parsers.py        # Extract images/text from API responses
│   └── error_handlers.py          # User-friendly error messages
├── requirements.txt               # google-genai>=1.0.0
├── .env.template                  # Template for API key config
└── README.md
```

### `__init__.py` registration

```python
from .nodes.nano_banana_generate import NanoBananaGenerate
from .nodes.nano_banana_chat import NanoBananaChat
from .nodes.imagen_generate import ImagenGenerate
from .nodes.imagen_edit import ImagenEdit

NODE_CLASS_MAPPINGS = {
    "NanoBananaGenerate": NanoBananaGenerate,
    "NanoBananaChat":     NanoBananaChat,
    "ImagenGenerate":     ImagenGenerate,
    "ImagenEdit":         ImagenEdit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NanoBananaGenerate": "🍌 Nano Banana Generate",
    "NanoBananaChat":     "🍌 Nano Banana Chat Edit",
    "ImagenGenerate":     "🖼️ Imagen 4 Generate",
    "ImagenEdit":         "🖼️ Imagen 4 Edit",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
```

### `requirements.txt`

```
google-genai>=1.0.0
Pillow>=9.0.0
numpy
torch>=2.0.0
```

The `google-genai` SDK (not the older `google-generativeai`) is the current recommended package. It provides both `client.models.generate_content()` for Nano Banana and `client.models.generate_images()` for Imagen.

---

## 3. Node 1 — Nano Banana Generate (text-to-image and image-to-image)

This is the primary workhorse node. It covers both pure text-to-image generation and reference-image-guided generation using Nano Banana Pro or Nano Banana 2.

### INPUT_TYPES specification

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": "Describe the image to generate..."
            }),
            "model": ([
                "gemini-3.1-flash-image-preview",   # Nano Banana 2
                "gemini-3-pro-image-preview",        # Nano Banana Pro
                "gemini-2.5-flash-image",            # Nano Banana (original)
            ], {"default": "gemini-3.1-flash-image-preview"}),
            "aspect_ratio": ([
                "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                "9:16", "16:9", "21:9",
                "1:4", "4:1", "1:8", "8:1",  # NB2 only
            ], {"default": "1:1"}),
            "image_size": ([
                "512px", "1K", "2K", "4K"
            ], {"default": "1K"}),
            "response_modality": ([
                "IMAGE", "TEXT_AND_IMAGE"
            ], {"default": "IMAGE"}),
        },
        "optional": {
            "api_key": ("STRING", {
                "default": "",
                "multiline": False,
                "placeholder": "Leave blank to use GEMINI_API_KEY env var"
            }),
            "reference_image_1": ("IMAGE",),
            "reference_image_2": ("IMAGE",),
            "reference_image_3": ("IMAGE",),
            "reference_image_4": ("IMAGE",),
            "reference_image_5": ("IMAGE",),
            "reference_image_6": ("IMAGE",),
            "reference_image_7": ("IMAGE",),
            "reference_image_8": ("IMAGE",),
            "reference_image_9": ("IMAGE",),
            "reference_image_10": ("IMAGE",),
            "reference_image_11": ("IMAGE",),
            "reference_image_12": ("IMAGE",),
            "reference_image_13": ("IMAGE",),
            "reference_image_14": ("IMAGE",),
            "thinking_level": ([
                "minimal", "High"
            ], {"default": "minimal"}),
            "temperature": ("FLOAT", {
                "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05
            }),
            "batch_count": ("INT", {
                "default": 1, "min": 1, "max": 4, "step": 1,
                "display": "number"
            }),
            "enable_search_grounding": ("BOOLEAN", {
                "default": False,
                "label_on": "Enabled",
                "label_off": "Disabled"
            }),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 0xffffffffffffffff
            }),
        },
        "hidden": {
            "unique_id": "UNIQUE_ID",
        }
    }
```

### RETURN_TYPES and execution

```python
RETURN_TYPES = ("IMAGE", "STRING")
RETURN_NAMES = ("images", "response_text")
CATEGORY = "Google AI/Image Generation"
FUNCTION = "generate"
OUTPUT_NODE = False

@classmethod
def IS_CHANGED(cls, **kwargs):
    return float("NaN")  # Always re-execute (API call)
```

### Parameter mapping — Nano Banana models

Every ComfyUI input maps to the API as follows:

| ComfyUI Input | API Location | Notes |
|---|---|---|
| `prompt` | `contents[0].parts[0].text` | Natural language prompt. Max ~480 tokens recommended. Supports multilingual. |
| `model` | URL path: `models/{model}:generateContent` | Model ID string directly in endpoint URL |
| `aspect_ratio` | `generationConfig.imageConfig.aspectRatio` | NB2 supports 14 ratios; Pro supports 10. Extreme ratios (1:8, 8:1) are NB2-only. Node should validate and warn. |
| `image_size` | `generationConfig.imageConfig.imageSize` | Values: `"512px"`, `"1K"`, `"2K"`, `"4K"`. Must be uppercase K. `512px` is NB2-only. |
| `response_modality` | `generationConfig.responseModalities` | `["IMAGE"]` or `["TEXT", "IMAGE"]`. Use IMAGE-only for cleaner output; TEXT_AND_IMAGE for captioned results. |
| `reference_image_N` | `contents[0].parts[N].inline_data` | Each reference image encoded as `{"mime_type": "image/png", "data": "<BASE64>"}`. Up to 14 images for NB2, 11 for Pro. |
| `thinking_level` | `generationConfig.thinkingConfig.thinkingLevel` | `"minimal"` (default for NB2) or `"High"`. Higher = better quality but more tokens billed. |
| `temperature` | `generationConfig.temperature` | 0.0–2.0. Controls randomness. |
| `batch_count` | Loop: make N separate API calls | Nano Banana generates **1 image per call** (unlike Imagen). Batch requires N sequential calls. |
| `enable_search_grounding` | `tools` array | If true: `[{"google_search": {}}]`. For NB2: `{"google_search": {"searchTypes": {"webSearch": {}, "imageSearch": {}}}}` |
| `seed` | Used to set `random.seed()` before call | **No native seed parameter** in Gemini API. Use for local reproducibility signal only. |
| `api_key` | Client initialization | Falls back to `os.environ.get("GEMINI_API_KEY")` if empty. |

### Reference image handling — critical details

Nano Banana models accept reference images as inline base64 data within the `contents.parts` array. There is **no explicit reference type parameter** (unlike Imagen's `REFERENCE_TYPE_STYLE` / `REFERENCE_TYPE_SUBJECT`). Instead, reference behavior is controlled entirely through the text prompt.

- **Style transfer**: Include reference image + prompt like "Generate an image in the style of this reference"
- **Subject consistency**: Include reference + prompt like "Create a new scene with this character"
- **Image editing**: Include reference + prompt like "Remove the background from this image"
- **Compositing**: Include multiple references + prompt describing how to combine them

**Conversion pipeline** for each reference image input:

```python
def comfyui_tensor_to_genai_part(tensor):
    """Convert ComfyUI IMAGE tensor to google-genai inline_data part."""
    # tensor shape: [B, H, W, C], take first image in batch
    image_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(image_np, "RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    b64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return {"inline_data": {"mime_type": "image/png", "data": b64_data}}
```

**Supported input MIME types**: `image/png`, `image/jpeg`, `image/webp` (up to 5MB per image inline).

### API call construction — Python SDK

```python
from google import genai
from google.genai import types

client = genai.Client(api_key=api_key)

# Build content parts
parts = [{"text": prompt}]
for ref_image in reference_images:  # list of tensors
    parts.append(comfyui_tensor_to_genai_part(ref_image))

response = client.models.generate_content(
    model=model,
    contents=[{"parts": parts}],
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"] if text_and_image else ["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        ),
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level,
        ),
        temperature=temperature,
    ),
)
```

### Response parsing

The response contains interleaved `text` and `inline_data` parts:

```python
images = []
texts = []
for part in response.parts:
    if part.text is not None:
        texts.append(part.text)
    elif part.inline_data is not None:
        # Decode base64 PNG to ComfyUI tensor
        img_bytes = base64.b64decode(part.inline_data.data)
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        np_img = np.array(pil_img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(np_img)  # [H, W, C]
        images.append(tensor)

# Stack into batch tensor [B, H, W, C]
image_batch = torch.stack(images) if images else empty_image_tensor()
response_text = "\n".join(texts)
return (image_batch, response_text)
```

---

## 4. Node 2 — Nano Banana Chat Edit (multi-turn conversational editing)

This node enables **iterative image refinement** through multi-turn conversation — a capability unique to the Gemini-based models. Users can chain multiple Chat Edit nodes together, each sending a new instruction that builds on the previous conversation context.

### INPUT_TYPES specification

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "instruction": ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": "Edit instruction (e.g., 'Change the sky to sunset')"
            }),
            "model": ([
                "gemini-3.1-flash-image-preview",
                "gemini-3-pro-image-preview",
                "gemini-2.5-flash-image",
            ], {"default": "gemini-3.1-flash-image-preview"}),
        },
        "optional": {
            "api_key": ("STRING", {"default": ""}),
            "input_image": ("IMAGE",),
            "chat_history": ("CHAT_HISTORY",),  # Custom type for multi-turn
            "aspect_ratio": ([
                "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                "9:16", "16:9", "21:9"
            ], {"default": "1:1"}),
            "image_size": (["1K", "2K", "4K"], {"default": "1K"}),
        },
    }

RETURN_TYPES = ("IMAGE", "STRING", "CHAT_HISTORY")
RETURN_NAMES = ("edited_image", "response_text", "chat_history")
```

The `CHAT_HISTORY` custom type carries the conversation state between chained nodes. Under the hood, it stores the list of content turns needed to reconstruct context for the next API call:

```python
# Chat session via SDK
chat = client.chats.create(
    model=model,
    config=types.GenerateContentConfig(
        response_modalities=["TEXT", "IMAGE"],
        image_config=types.ImageConfig(aspect_ratio=aspect_ratio, image_size=image_size),
    ),
)
# First turn: send image + instruction
response = chat.send_message([input_image_part, instruction])
# Subsequent turns: send just instruction
response = chat.send_message(instruction)
```

---

## 5. Node 3 — Imagen 4 Generate (text-to-image with seed control)

This node uses the Imagen `:predict` endpoint. Its advantages over Nano Banana are **deterministic seed control**, **1–4 images per call**, explicit **negative prompts**, and configurable **safety filters**.

### INPUT_TYPES specification

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": "Describe the image..."
            }),
            "model": ([
                "imagen-4.0-generate-001",       # Standard
                "imagen-4.0-ultra-generate-001",  # Ultra
                "imagen-4.0-fast-generate-001",   # Fast
                "imagen-3.0-generate-002",        # Imagen 3 (legacy)
            ], {"default": "imagen-4.0-generate-001"}),
            "number_of_images": ("INT", {
                "default": 1, "min": 1, "max": 4, "step": 1
            }),
            "aspect_ratio": ([
                "1:1", "3:4", "4:3", "9:16", "16:9"
            ], {"default": "1:1"}),
        },
        "optional": {
            "api_key": ("STRING", {"default": ""}),
            "negative_prompt": ("STRING", {
                "default": "",
                "multiline": True,
                "placeholder": "What to avoid..."
            }),
            "image_size": (["1K", "2K"], {"default": "1K"}),
            "seed": ("INT", {
                "default": 0, "min": 0, "max": 4294967295
            }),
            "add_watermark": ("BOOLEAN", {
                "default": True,
                "label_on": "SynthID On",
                "label_off": "SynthID Off"
            }),
            "enhance_prompt": ("BOOLEAN", {
                "default": True,
                "label_on": "Enhanced",
                "label_off": "Raw"
            }),
            "person_generation": ([
                "dont_allow", "allow_adult", "allow_all"
            ], {"default": "allow_adult"}),
            "safety_filter_level": ([
                "block_low_and_above",
                "block_medium_and_above",
                "block_only_high"
            ], {"default": "block_medium_and_above"}),
            "output_format": (["image/png", "image/jpeg"], {"default": "image/png"}),
            "jpeg_quality": ("INT", {
                "default": 80, "min": 0, "max": 100, "step": 5
            }),
        },
    }

RETURN_TYPES = ("IMAGE",)
RETURN_NAMES = ("images",)
```

### Parameter mapping — Imagen 4 models

| ComfyUI Input | API Parameter Path | Constraints |
|---|---|---|
| `prompt` | `instances[0].prompt` | Max 480 tokens. English only for Imagen. |
| `model` | URL path: `models/{model}:predict` | — |
| `number_of_images` | `parameters.sampleCount` | 1–4. Returns array of that size. |
| `aspect_ratio` | `parameters.aspectRatio` | 5 options only. |
| `image_size` | `parameters.imageSize` | `"1K"` or `"2K"`. Fast model: 1K only. |
| `seed` | `parameters.seed` | **Only works when `addWatermark` is `false`**. Max 4294967295. Node should auto-disable watermark when seed > 0. |
| `add_watermark` | `parameters.addWatermark` | SynthID invisible watermark. |
| `enhance_prompt` | `parameters.enhancePrompt` | LLM-based prompt rewriting. Set `false` for Fast model with complex prompts. |
| `negative_prompt` | `parameters.negativePrompt` | Supported on editing model; may have limited effect on generate models. |
| `person_generation` | `parameters.personGeneration` | `allow_all` blocked in EU/UK/CH/MENA regions. |
| `safety_filter_level` | `parameters.safetySetting` | — |
| `output_format` | `parameters.outputOptions.mimeType` | — |
| `jpeg_quality` | `parameters.outputOptions.compressionQuality` | 0–100, JPEG only. |

### API call and response

```python
response = client.models.generate_images(
    model=model,
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images=number_of_images,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        seed=seed if not add_watermark else None,
        add_watermark=add_watermark,
        enhance_prompt=enhance_prompt,
        person_generation=person_generation,
        safety_filter_level=safety_filter_level,
        include_rai_reason=True,
    ),
)

# Parse response
images = []
for gen_image in response.generated_images:
    pil_img = gen_image.image._pil_image  # SDK provides PIL access
    np_img = np.array(pil_img.convert("RGB")).astype(np.float32) / 255.0
    images.append(torch.from_numpy(np_img))
image_batch = torch.stack(images)  # [B, H, W, C]
return (image_batch,)
```

---

## 6. Node 4 — Imagen 4 Edit (inpainting, outpainting, style/subject reference)

This node uses the `imagen-3.0-capability-001` model (editing/customization model) with explicit reference types and mask support.

### INPUT_TYPES specification

```python
@classmethod
def INPUT_TYPES(cls):
    return {
        "required": {
            "prompt": ("STRING", {"default": "", "multiline": True}),
            "edit_mode": ([
                "EDIT_MODE_INPAINT_INSERTION",
                "EDIT_MODE_INPAINT_REMOVAL",
                "EDIT_MODE_OUTPAINT",
            ], {"default": "EDIT_MODE_INPAINT_INSERTION"}),
        },
        "optional": {
            "api_key": ("STRING", {"default": ""}),
            "base_image": ("IMAGE",),
            "mask_image": ("MASK",),
            "style_reference": ("IMAGE",),
            "subject_reference": ("IMAGE",),
            "subject_description": ("STRING", {
                "default": "",
                "placeholder": "Description of the subject"
            }),
            "subject_type": ([
                "SUBJECT_TYPE_PERSON",
                "SUBJECT_TYPE_ANIMAL",
                "SUBJECT_TYPE_PRODUCT",
                "SUBJECT_TYPE_DEFAULT",
            ], {"default": "SUBJECT_TYPE_DEFAULT"}),
            "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            "guidance_scale": ("INT", {
                "default": 60, "min": 0, "max": 500, "step": 5
            }),
            "number_of_images": ("INT", {
                "default": 1, "min": 1, "max": 4, "step": 1
            }),
            "aspect_ratio": ([
                "1:1", "3:4", "4:3", "9:16", "16:9"
            ], {"default": "1:1"}),
            "person_generation": ([
                "dont_allow", "allow_adult", "allow_all"
            ], {"default": "allow_adult"}),
            "safety_filter_level": ([
                "block_low_and_above",
                "block_medium_and_above",
                "block_only_high"
            ], {"default": "block_medium_and_above"}),
        },
    }
```

### Reference image API format (Imagen editing model)

```python
# Reference images structure for Imagen capability model
reference_images = []

if base_image is not None:
    reference_images.append({
        "referenceImage": {
            "rawReferenceImage": {
                "image": {"bytesBase64Encoded": tensor_to_base64(base_image)}
            }
        },
        "referenceType": "REFERENCE_TYPE_RAW"
    })

if mask_image is not None:
    reference_images.append({
        "referenceImage": {
            "maskReferenceImage": {
                "maskImage": {"bytesBase64Encoded": mask_to_base64(mask_image)},
                "maskImageConfig": {"maskMode": "MASK_MODE_USER_PROVIDED"}
            }
        },
        "referenceType": "REFERENCE_TYPE_MASK"
    })

if style_reference is not None:
    reference_images.append({
        "referenceImage": {
            "styleReferenceImage": {
                "image": {"bytesBase64Encoded": tensor_to_base64(style_reference)}
            }
        },
        "referenceType": "REFERENCE_TYPE_STYLE"
    })

if subject_reference is not None:
    reference_images.append({
        "referenceImage": {
            "subjectReferenceImage": {
                "image": {"bytesBase64Encoded": tensor_to_base64(subject_reference)},
                "subjectDescription": subject_description,
                "subjectType": subject_type
            }
        },
        "referenceType": "REFERENCE_TYPE_SUBJECT"
    })
```

---

## 7. Image tensor handling and conversion utilities

All image data flowing through ComfyUI uses the tensor format **`[B, H, W, C]`** with `torch.float32` dtype and values in `[0.0, 1.0]`. The conversion helpers in `helpers/type_converters.py` form the critical bridge between ComfyUI and the Google API.

### Complete conversion functions

```python
import torch, numpy as np, base64, io
from PIL import Image, ImageOps

def tensor_to_base64(tensor: torch.Tensor, format: str = "PNG") -> str:
    """ComfyUI IMAGE tensor [B,H,W,C] → base64 string (first image in batch)."""
    img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np, "RGB")
    buf = io.BytesIO()
    pil_img.save(buf, format=format)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def mask_to_base64(mask: torch.Tensor) -> str:
    """ComfyUI MASK tensor [B,H,W] → base64 grayscale PNG."""
    mask_np = (mask[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(mask_np, "L")
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_tensor(b64_string: str) -> torch.Tensor:
    """Base64 image string → ComfyUI IMAGE tensor [1,H,W,C]."""
    img_bytes = base64.b64decode(b64_string)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    pil_img = ImageOps.exif_transpose(pil_img)
    np_img = np.array(pil_img).astype(np.float32) / 255.0
    return torch.from_numpy(np_img).unsqueeze(0)  # [1, H, W, C]

def batch_base64_to_tensor(b64_list: list[str]) -> torch.Tensor:
    """List of base64 images → single batched tensor [B,H,W,C]."""
    tensors = [base64_to_tensor(b64) for b64 in b64_list]
    return torch.cat(tensors, dim=0)
```

**Key constraint for batching**: all images in a batch tensor must share the same height and width. When the API returns images of varying sizes (unlikely but possible), resize to a common dimension before stacking.

---

## 8. API key management strategy

Based on best practices from the most popular community nodes (ShmuelRonen, ru4ls, if-ai), the recommended three-tier fallback:

```python
import os, json

def resolve_api_key(node_input_key: str = "") -> str:
    """Resolve API key: node input → env var → config file."""
    # Tier 1: Direct node input (least secure, but convenient)
    if node_input_key.strip():
        return node_input_key.strip()
    
    # Tier 2: Environment variable (recommended)
    env_key = os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        return env_key
    
    # Tier 3: Config file in node directory
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            key = json.load(f).get("api_key", "")
            if key:
                return key
    
    raise ValueError(
        "No API key found. Set GEMINI_API_KEY environment variable, "
        "provide key in node input, or create config.json."
    )
```

The environment variable approach keeps keys out of workflow JSON files (which are easily shared). The node input serves as a convenient override for testing.

---

## 9. Batch processing across both API systems

### Imagen 4 batch (native)

Imagen natively supports **1–4 images per call** via `sampleCount`. The response returns an array of predictions, each converted to a tensor and stacked:

```python
# Imagen: all images come in one response
image_tensors = []
for gen_img in response.generated_images:
    image_tensors.append(pil_to_tensor(gen_img.image._pil_image))
return (torch.stack(image_tensors),)  # [B, H, W, C]
```

### Nano Banana batch (sequential calls)

Nano Banana generates **1 image per API call**. To produce a batch, make N sequential calls and concatenate:

```python
all_images = []
for i in range(batch_count):
    response = client.models.generate_content(model=model, contents=contents, config=config)
    for part in response.parts:
        if part.inline_data is not None:
            all_images.append(base64_to_tensor(part.inline_data.data))
if all_images:
    return (torch.cat(all_images, dim=0),)
```

**Batch API alternative**: For high-volume workflows, Google offers a **Batch API** that processes requests asynchronously (up to 24hr turnaround) at **50% cost reduction**. This could be exposed as a separate "Batch Queue" node for production pipelines, but is not required for the initial release.

---

## 10. Error handling requirements

Based on common failure modes identified across existing implementations:

| Error Condition | Detection | User-Facing Behavior |
|---|---|---|
| Missing API key | Empty string after fallback chain | Raise `ValueError` with setup instructions |
| Invalid API key | HTTP 401/403 from API | Clear error: "Invalid API key. Verify at ai.google.dev" |
| Rate limit exceeded | HTTP 429 | "Rate limit reached. Wait and retry, or upgrade tier." |
| Content blocked by safety | Response contains `raiFilteredReason` / no images | Return placeholder image + warning text in STRING output |
| Model not available in region | HTTP 400 with region error | "Model unavailable in your region. Try a different model." |
| `allow_all` in restricted region | API rejection for EU/UK/CH/MENA | "Person generation 'allow_all' not available in your region." |
| Seed + watermark conflict | Logical conflict | Auto-disable watermark when seed > 0, log warning |
| Extreme aspect ratio on wrong model | 1:8/8:1 on non-NB2 model | Validate and fall back to closest supported ratio |
| Network timeout | Connection error | Retry once, then raise with timeout advice |

Every node should implement `IS_CHANGED` returning `float("NaN")` to ensure re-execution on each run (API calls should never be cached).

---

## 11. Pricing and rate limits users should know

### Per-image costs (paid tier)

| Model | ~Cost per Image | Notes |
|---|---|---|
| Nano Banana 2 @ 1K | **$0.067** | 1,120 output tokens × $60/1M tokens |
| Nano Banana 2 @ 4K | **$0.151** | 2,520 output tokens |
| Nano Banana Pro @ 1K | **$0.134** | 1,120 output tokens × $120/1M tokens |
| Nano Banana Pro @ 4K | **$0.240** | 2,000 output tokens |
| Nano Banana (original) | **$0.039** | 1,290 tokens × $30/1M tokens. Free tier eligible. |
| Imagen 4 Standard/Ultra/Fast | Varies | Per-image pricing on Google's pricing page |

### Key rate limit facts

- **Free tier**: Extremely limited for image generation. Nano Banana Pro and NB2 require paid API keys.
- **Tier 1** (billing enabled): Moderate limits, model-specific.
- **Tier 2** ($250+ spend, 30+ days): Higher limits.
- **Tier 3** ($1,000+ spend, 30+ days): Highest limits.
- Thinking tokens are billed regardless of `includeThoughts` setting.
- Search grounding: **5,000 free prompts/month**, then $14/1,000 queries.

---

## 12. Validation rules and model-specific constraints

The node implementation must enforce these constraints to prevent API errors:

```python
# Aspect ratio validation per model
ASPECT_RATIOS = {
    "gemini-3.1-flash-image-preview": [
        "1:1","1:4","1:8","2:3","3:2","3:4","4:1","4:3","4:5","5:4","8:1","9:16","16:9","21:9"
    ],
    "gemini-3-pro-image-preview": [
        "1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"
    ],
    "gemini-2.5-flash-image": ["1:1","3:4","4:3","9:16","16:9"],
}

IMAGE_SIZES = {
    "gemini-3.1-flash-image-preview": ["512px", "1K", "2K", "4K"],
    "gemini-3-pro-image-preview": ["1K", "2K", "4K"],
    "gemini-2.5-flash-image": ["1K"],
}

MAX_REFERENCE_IMAGES = {
    "gemini-3.1-flash-image-preview": 14,  # 10 object + 4 character
    "gemini-3-pro-image-preview": 11,       # 6 object + 5 character
    "gemini-2.5-flash-image": 5,
}

# Imagen seed/watermark constraint
# When seed > 0, force add_watermark = False
```

---

## 13. Complete feature matrix

This matrix summarizes every parameter across all four nodes:

| Feature | NB Generate | NB Chat Edit | Imagen Generate | Imagen Edit |
|---|:---:|:---:|:---:|:---:|
| Text prompt | ✅ | ✅ | ✅ | ✅ |
| Negative prompt | ❌ (use prompt) | ❌ | ⚠️ (limited) | ✅ |
| Model selection | 3 models | 3 models | 4 models | 1 model |
| Aspect ratio | 5–14 options | 5–14 options | 5 options | 5 options |
| Image size | 512px–4K | 1K–4K | 1K–2K | — |
| Seed control | ❌ (no native) | ❌ | ✅ (w/o watermark) | ✅ |
| Batch count | 1–4 (sequential) | 1 | 1–4 (native) | 1–4 |
| Reference images | Up to 14 | 1 (input) | ❌ | 4 (typed) |
| Style reference | Via prompt | Via prompt | ❌ | ✅ (explicit) |
| Subject reference | Via prompt | Via prompt | ❌ | ✅ (explicit) |
| Mask/inpainting | ❌ | ❌ | ❌ | ✅ |
| Multi-turn editing | ❌ | ✅ | ❌ | ❌ |
| Thinking config | ✅ | ❌ | ❌ | ❌ |
| Search grounding | ✅ | ✅ | ❌ | ❌ |
| Temperature | ✅ | ❌ | ❌ | ❌ |
| Safety filter | ❌ | ❌ | ✅ | ✅ |
| Person generation | ❌ | ❌ | ✅ | ✅ |
| Watermark control | ❌ | ❌ | ✅ | ✅ |
| Enhance prompt | ❌ | ❌ | ✅ | ❌ |
| Guidance scale | ❌ | ❌ | ❌ | ✅ |
| Output format | PNG (fixed) | PNG (fixed) | PNG/JPEG | PNG/JPEG |
| Text output | ✅ | ✅ | ❌ | ❌ |
| Chat history I/O | ❌ | ✅ | ❌ | ❌ |

---

## Conclusion

This PRD specifies a **four-node architecture** that cleanly separates the two API systems (Gemini `generateContent` for Nano Banana, Imagen `:predict` for Imagen 4) while presenting a consistent ComfyUI interface. Three design decisions deserve emphasis.

First, **reference images in Nano Banana are prompt-driven, not parameter-driven**. Unlike Imagen's explicit `REFERENCE_TYPE_STYLE` / `REFERENCE_TYPE_SUBJECT` flags, Nano Banana relies on natural language instructions to determine how reference images are used. This is both a strength (flexibility) and a challenge (less deterministic). The node should include placeholder text in the prompt field guiding users on how to describe reference image intent.

Second, **batch generation differs fundamentally between the two systems**. Imagen returns 1–4 images per call natively; Nano Banana requires sequential calls. The `batch_count` parameter on the Nano Banana node should be clearly documented as triggering multiple API calls with proportional cost and latency.

Third, **seed determinism is only available through Imagen 4**, and only when SynthID watermarking is disabled. For reproducible workflows, Imagen 4 is the correct choice. Nano Banana should expose a seed input for local randomness signaling, but the node tooltip must clarify that true server-side determinism is not guaranteed.

The helper module pattern (client_manager, type_converters, response_parsers, error_handlers) mirrors the cleanest community implementation and ensures the codebase remains maintainable as Google adds models or changes API parameters. The `google-genai` SDK should be the sole dependency for API communication — avoid the deprecated `google-generativeai` package.
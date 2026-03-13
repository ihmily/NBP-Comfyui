# NBP-ComfyUI PRD Implementation Plan

Migrate from raw `requests`-based API calls to the `google-genai` SDK. Replace the current 3 nodes with 4 new nodes per the PRD.

## User Review Required

> [!IMPORTANT]
> **Breaking change**: The old [GoogleImagenAPIKeyNode](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#24-57), [ImagenGenerateNode](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#59-292), and [NanoBananaNode](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#294-426) will be **replaced** by 4 new nodes. Any existing ComfyUI workflows using the old nodes will need to be rewired.

> [!IMPORTANT]
> **Dependency**: Requires `google-genai>=1.0.0` (the new unified SDK). The deprecated `google-generativeai` package is NOT used. Please confirm this is acceptable.

---

## Proposed Changes

### Helper Modules

#### [NEW] [client_manager.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/client_manager.py)

Singleton wrapper around `google.genai.Client`. Resolves API key via 3-tier priority (env var → config file → node input). Exposes `get_client(api_key_override="")` returning a configured `Client` instance.

#### [NEW] [type_converters.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/type_converters.py)

- `comfy_tensor_to_pil(tensor)` — ComfyUI IMAGE tensor (B,H,W,C float32) → PIL Image
- `pil_to_comfy_tensor(image)` — PIL Image → ComfyUI IMAGE tensor
- `batch_pils_to_tensor(images)` — list of PIL Images → batched tensor
- [comfy_tensor_to_base64(tensor, fmt)](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/image_utils.py#27-42) — tensor → base64 string

#### [NEW] [response_parsers.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/response_parsers.py)

- `parse_imagen_response(response)` — extract PIL images and metadata from Imagen SDK response
- `parse_gemini_response(response)` — extract images and text from Gemini SDK response
- `parse_imagen_edit_response(response)` — extract edited images from Imagen edit response

#### [NEW] [error_handlers.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/error_handlers.py)

- Custom exception classes: `APIKeyError`, `GenerationBlockedError`, `QuotaExceededError`
- `handle_api_error(exception)` — maps SDK exceptions to user-friendly ComfyUI messages

---

### Node Implementations

#### [NEW] [nodes.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py) (full rewrite)

The current [nodes.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py) (426 lines) will be **completely rewritten** with 4 node classes:

**1. `NanoBananaGenerate`** — Text-to-image via Gemini SDK
- Models: `gemini-2.0-flash-preview-image-generation`, `gemini-2.5-flash-preview-04-17`
- Inputs: prompt, model, response_modalities (`IMAGE` / `TEXT_AND_IMAGE`), optional input_image
- Uses `client.models.generate_content()` with `responseModalities` config
- Returns: [(IMAGE, STRING)](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#54-57) — generated image tensor + text response

**2. `NanoBananaChat`** — Multi-turn conversational image editing
- Same models as NanoBananaGenerate
- Uses `client.chats.create()` for persistent chat sessions
- Maintains conversation history for iterative editing
- Inputs: prompt, model, response_modalities, optional input_image, optional chat_history
- Returns: [(IMAGE, STRING, NANO_BANANA_CHAT_HISTORY)](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#54-57) — image + text + chat state

**3. [ImagenGenerate](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#59-292)** — Text-to-image via Imagen SDK
- Models: `imagen-4.0-generate-001`, `imagen-4.0-fast-generate-001`, `imagen-4.0-ultra-generate-001`, `imagen-3.0-generate-002`, `imagen-3.0-fast-generate-001`
- Inputs: prompt, model, aspect_ratio, num_images, image_size, seed, enhance_prompt, add_watermark, safety_setting, person_generation, output_format, language
- Uses `client.models.generate_images()`
- Seed constraint logic: seed > 0 auto-disables watermark + enhance_prompt
- Returns: [(IMAGE, STRING)](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#54-57) — batched image tensor + generation info

**4. `ImagenEdit`** — Image editing/inpainting via Imagen SDK
- Models: Same as ImagenGenerate (subset supporting editing)
- Inputs: prompt, input_image, optional mask_image, model, edit parameters
- Uses `client.models.edit_image()`
- Returns: [(IMAGE, STRING)](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#54-57) — edited image tensor + info

---

### Registration

#### [MODIFY] [__init__.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/__init__.py)

Update `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` with new 4-node set. Remove old class references.

---

### Cleanup

#### [DELETE] [api_utils.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/api_utils.py)

Replaced by `client_manager.py`. Raw `requests` calls no longer needed.

#### [DELETE] [image_utils.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/image_utils.py)

Replaced by `type_converters.py` and `response_parsers.py`.

---

## Verification Plan

### Manual Verification

Since this is a ComfyUI custom node pack with no existing test infrastructure, verification is manual:

1. **Load test**: Start ComfyUI and confirm no import errors in the console. All 4 nodes should appear under the `Nano Banana Pack` category in the node browser.
2. **Node wiring**: Drag each node onto the canvas and confirm all inputs/outputs render correctly.
3. **API key resolution**: Test that passing an API key via the node input field works (requires valid Google API key).
4. **Generation test**: Connect a `NanoBananaGenerate` node → `Preview Image` node and run with a simple prompt to confirm image output.

> [!NOTE]
> Full end-to-end generation testing requires a valid Google API key and network access. The user should confirm this works in their environment after implementation.

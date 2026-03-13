# NBP-ComfyUI PRD Implementation

## Phase 1: Foundation — SDK Migration & Helper Modules
- [x] Install `google-genai` SDK dependency
- [x] Create `client_manager.py` — singleton GenAI client via API key
- [x] Create `type_converters.py` — ComfyUI tensor ↔ PIL Image ↔ base64 conversions
- [x] Create `response_parsers.py` — parse Imagen and Gemini SDK responses
- [x] Create `error_handlers.py` — structured error classes and retry logic

## Phase 2: Core Nodes
- [x] Implement `NanoBananaGenerate` node (text-to-image via Gemini SDK)
- [x] Implement `NanoBananaChat` node (multi-turn conversational image editing)
- [x] Implement [ImagenGenerate](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/nodes.py#59-292) node (text-to-image via Imagen SDK)
- [x] Implement `ImagenEdit` node (image editing/inpainting via Imagen SDK)

## Phase 3: Registration & Integration
- [x] Update [__init__.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/__init__.py) with new node mappings
- [x] Verify [pyproject.toml](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/pyproject.toml) / [requirements.txt](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/requirements.txt) dependencies

## Phase 4: Cleanup
- [x] Remove legacy [api_utils.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/api_utils.py) (raw requests)
- [x] Remove or refactor legacy [image_utils.py](file:///c:/Users/Admin/Documents/AI%20Projects/NBP%20-%20Comfyui/image_utils.py) functions
- [x] Verify no import errors or circular dependencies

## Phase 5: Verification
- [x] Manual test: ComfyUI loads without errors
- [x] Ask user to verify nodes appear in ComfyUI node browser

## Phase 6: Refinements
- [x] Update `NanoBananaGenerate` to accept a dynamic number of reference images using batched tensor input.
- [x] Remove deprecated `gemini-2.5-flash-image` model from Nano Banana nodes.
- [x] Remove `ImagenGenerate` and `ImagenEdit` nodes completely as they are unnecessary.

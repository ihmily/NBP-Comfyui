# ComfyUI-Google-Imagen

Custom node pack for **Google Imagen 4** and **Nano Banana** (Gemini image generation) in ComfyUI.

## Nodes

| Node | Description |
|------|-------------|
| **Google API Key** | Provides API key via env var, config file, or manual input |
| **Imagen Generate** | Text-to-image with Imagen 4 models (`:predict` endpoint) |
| **Nano Banana Generate** | Text-to-image and image editing via Gemini models (`:generateContent` endpoint) |

## Installation

1. Clone into `ComfyUI/custom_nodes/`:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/your-username/ComfyUI-Google-Imagen.git
   ```
2. Install dependencies:
   ```bash
   pip install -r ComfyUI-Google-Imagen/requirements.txt
   ```
3. Restart ComfyUI.

## API Key Setup

Provide your Google API key via one of these methods (checked in order):

1. **Environment variable**: Set `GOOGLE_API_KEY` or `GEMINI_API_KEY`
2. **Config file**: Create `google_api_key.txt` in the extension directory with your key
3. **Node input**: Wire a `Google API Key` node and type the key directly

## Imagen Generate

Generates images using Google's Imagen 4 models via the `:predict` endpoint.

### Models

| Model | Quality | Speed |
|-------|---------|-------|
| `imagen-4.0-generate-001` | High | Standard |
| `imagen-4.0-fast-generate-001` | Good | Fast |
| `imagen-4.0-ultra-generate-001` | Highest | Slow |
| `imagen-3.0-generate-002` | Standard | Standard |
| `imagen-3.0-fast-generate-001` | Standard | Fast |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | — | Text prompt (max 480 tokens) |
| `model` | `imagen-4.0-generate-001` | Model selection |
| `aspect_ratio` | `1:1` | `1:1`, `3:4`, `4:3`, `9:16`, `16:9` |
| `num_images` | `1` | 1–4 images per request |
| `image_size` | `1K` | `1K` or `2K` (2K: Standard/Ultra models only) |
| `seed` | `0` | Set >0 for determinism (auto-disables watermark + enhance) |
| `enhance_prompt` | `true` | LLM-based prompt rewriting |
| `add_watermark` | `true` | SynthID watermark |
| `safety_setting` | `block_medium_and_above` | Safety filter threshold |
| `person_generation` | `allow_all` | People in images control |
| `output_format` | `image/png` | PNG, JPEG, or WebP |

### Outputs

- **images**: ComfyUI IMAGE tensor (batch of generated images)
- **generation_info**: Model name, image count, filtered images, enhanced prompt

## Nano Banana Generate

Generates images using Google's Gemini vision-language models via the `:generateContent` endpoint. Also supports image editing when an input image is provided.

### Models

| Model | Speed | Quality |
|-------|-------|---------|
| `gemini-3.1-flash-image-preview` | Fastest | Good |
| `gemini-3.1-pro-image-preview` | Standard | Highest |
| `gemini-2.5-flash-image` | Fast | Good |

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `prompt` | — | Text prompt for generation or editing |
| `model` | `gemini-3.1-flash-image-preview` | Model selection |
| `response_modalities` | `IMAGE` | `IMAGE` or `TEXT_AND_IMAGE` |
| `input_image` | Optional | Input image for editing/transformation |

### Outputs

- **image**: ComfyUI IMAGE tensor
- **text_response**: Text returned by the model (empty if `IMAGE` mode)

## License

MIT

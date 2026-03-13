# Upgrade Instructions: v1 → v2 — Missing Parameters & Model-Specific Guards

**Scope**: You already have working Nano Banana Generate and Nano Banana Chat Edit nodes for `gemini-3-pro-image-preview` (Pro) and `gemini-3.1-flash-image-preview` (NB2/Flash). This document tells you exactly what to add, where, and how to guard for model-specific incompatibilities.

---

## Critical Model Differences to Guard Against

Before adding any parameter, hardcode this reference table. **Every new optional input must be validated against it at runtime.**

```python
MODEL_CONSTRAINTS = {
    "gemini-3-pro-image-preview": {
        "display_name": "Nano Banana Pro",
        "supports_thinking": False,          # ⚠️ thinking_config causes API error
        "supports_512px": False,             # ⚠️ 512px causes API error
        "thinking_levels": [],               # N/A
        "aspect_ratios": [
            "1:1", "2:3", "3:2", "3:4", "4:3",
            "4:5", "5:4", "9:16", "16:9", "21:9"
        ],                                   # 10 ratios — NO extreme ratios
        "image_sizes": ["1K", "2K", "4K"],
        "max_reference_images": 14,
        "supports_search_grounding": True,
        "supports_image_search": False,      # Web search only
    },
    "gemini-3.1-flash-image-preview": {
        "display_name": "Nano Banana 2 (Flash)",
        "supports_thinking": True,
        "supports_512px": True,
        "thinking_levels": ["minimal", "low", "medium", "high"],  # 4 levels
        "aspect_ratios": [
            "1:1", "1:4", "1:8", "2:3", "3:2", "3:4",
            "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"
        ],                                   # 14 ratios including extreme
        "image_sizes": ["512px", "1K", "2K", "4K"],
        "supports_search_grounding": True,
        "supports_image_search": True,       # Web + Image search
    },
}
```

---

## Parameters to Add — Nano Banana Generate Node

### 1. `system_instruction` — STRING (multiline)

**What it does**: A persistent instruction that shapes the model's behavior across the entire generation. Think of it as a "persona" for the model — it sits outside the user prompt and is never overridden by it.

**Where it goes in the API**: Top-level field in `GenerateContentConfig`, NOT inside `generationConfig`.

**ComfyUI input definition**:
```python
"system_instruction": ("STRING", {
    "default": "",
    "multiline": True,
    "placeholder": "Optional system prompt (e.g., 'You are a professional product photographer. Always produce studio-lit compositions.')",
    "tooltip": "Persistent instruction that guides model behavior. Sits above the user prompt. Use for consistent style direction, persona, or constraints across generations."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    system_instruction=system_instruction.strip() if system_instruction.strip() else None,
    # ... other params
)
```

**Model guard**: Works on **both** Pro and Flash. No guard needed.

---

### 2. `top_p` — FLOAT

**What it does**: Nucleus sampling. The model considers only the smallest set of tokens whose cumulative probability is at least `top_p`. Lower values = more focused/deterministic output. Higher = more creative/varied.

**ComfyUI input definition**:
```python
"top_p": ("FLOAT", {
    "default": 0.95,
    "min": 0.0,
    "max": 1.0,
    "step": 0.01,
    "tooltip": "Nucleus sampling threshold (0.0–1.0). Model considers the smallest token set whose cumulative probability ≥ this value. Lower = more focused, higher = more diverse. Tip: adjust either temperature OR top_p, not both simultaneously."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    top_p=top_p,
    # ...
)
```

**Model guard**: Works on **both** Pro and Flash. No guard needed.

---

### 3. `top_k` — INT

**What it does**: Top-k sampling. The model only considers the K most probable next tokens. Lower K = more predictable, higher K = more variety.

**ComfyUI input definition**:
```python
"top_k": ("INT", {
    "default": 40,
    "min": 1,
    "max": 100,
    "step": 1,
    "tooltip": "Top-k sampling (1–100). Model considers only the K most probable tokens at each step. Lower = more predictable, higher = more creative. Default 40 is a good starting point."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    top_k=top_k,
    # ...
)
```

**Model guard**: Works on **both**. No guard needed.

---

### 4. `candidate_count` — INT

**What it does**: Number of response candidates the API generates internally. Each candidate is a complete, independent response (including an image). The API returns all of them. This is NOT the same as batch — it happens in a single API call.

**ComfyUI input definition**:
```python
"candidate_count": ("INT", {
    "default": 1,
    "min": 1,
    "max": 4,
    "step": 1,
    "tooltip": "Number of response candidates generated in a SINGLE API call. Each candidate may contain a different image variation. Cost scales linearly with candidate count. Different from batch_count (which makes separate calls)."
}),
```

**Important implementation note**: When `candidate_count > 1`, you need to iterate over `response.candidates` (plural), not just `response.parts`:

```python
images = []
texts = []

for candidate in response.candidates:
    if candidate.content and candidate.content.parts:
        for part in candidate.content.parts:
            if part.text is not None:
                texts.append(part.text)
            elif part.inline_data is not None:
                images.append(base64_to_tensor(part.inline_data.data))
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    candidate_count=candidate_count,
    # ...
)
```

**Model guard**: Works on **both**. However, keep max at 4 — higher values may silently fail or return fewer candidates than requested. Log a warning if returned candidates < requested.

---

### 5. `max_output_tokens` — INT

**What it does**: Hard ceiling on total output tokens (text + thinking + image tokens combined). Image tokens count toward this limit: approximately 1,120 tokens for a 1K image, ~2,520 for 4K. If the limit is too low, the image may be truncated or not generated at all.

**ComfyUI input definition**:
```python
"max_output_tokens": ("INT", {
    "default": 8192,
    "min": 1024,
    "max": 32768,
    "step": 512,
    "tooltip": "Maximum output tokens (text + thinking + image combined). Image tokens: ~1,120 for 1K, ~1,600 for 2K, ~2,520 for 4K. If set too low, the image may fail to generate. Minimum recommended: 2048 for 1K, 4096 for 2K+, 8192 for 4K."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    max_output_tokens=max_output_tokens,
    # ...
)
```

**Model guard**: Works on **both**. Add a soft validation warning:

```python
IMAGE_TOKEN_ESTIMATES = {"512px": 800, "1K": 1120, "2K": 1600, "4K": 2520}
min_recommended = IMAGE_TOKEN_ESTIMATES.get(image_size, 1120) + 512  # buffer for text/thinking
if max_output_tokens < min_recommended:
    print(f"⚠️ [NanoBanana] max_output_tokens ({max_output_tokens}) may be too low for {image_size} images. Recommended minimum: {min_recommended}")
```

---

### 6. `stop_sequences` — STRING

**What it does**: A list of strings that cause the model to stop generating when encountered. Useful when using `TEXT_AND_IMAGE` mode and you want to prevent the model from generating excessive text after the image.

**ComfyUI input definition**:
```python
"stop_sequences": ("STRING", {
    "default": "",
    "multiline": False,
    "placeholder": "Comma-separated (e.g., STOP,END,---)",
    "tooltip": "Comma-separated list of strings that stop generation when encountered. Useful with TEXT_AND_IMAGE mode to limit text output. Leave empty to disable."
}),
```

**SDK mapping** (parse comma-separated string):
```python
parsed_stops = [s.strip() for s in stop_sequences.split(",") if s.strip()] if stop_sequences.strip() else None

config = types.GenerateContentConfig(
    stop_sequences=parsed_stops,
    # ...
)
```

**Model guard**: Works on **both**. No guard needed.

---

### 7. `seed` — INT

**What it does**: Influences the random sampling process. For text-only Gemini models, seed provides near-deterministic output. **For image generation, seed influences but does NOT guarantee identical images** — this is a known limitation of the Nano Banana models.

**ComfyUI input definition**:
```python
"seed": ("INT", {
    "default": 0,
    "min": 0,
    "max": 2147483647,
    "step": 1,
    "tooltip": "Seed for sampling (0 = random). ⚠️ For image generation, this influences but does NOT guarantee deterministic output. Same seed + same prompt will produce SIMILAR but not identical images. For true reproducibility, use Imagen 4 models instead."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    seed=seed if seed > 0 else None,
    # ...
)
```

**Model guard**: Works on **both**. No guard needed.

---

### 8. `thinking_level` — COMBO (dropdown)

**What it does**: Controls the depth of internal reasoning before the model generates. Higher thinking = better composition, more accurate text rendering, more considered layouts — but costs more tokens and takes longer.

**⚠️ THIS IS THE MOST IMPORTANT MODEL GUARD**: `thinking_config` is ONLY supported by `gemini-3.1-flash-image-preview`. Sending it to `gemini-3-pro-image-preview` **will cause an API error**.

**ComfyUI input definition**:
```python
"thinking_level": ([
    "none", "minimal", "low", "medium", "high"
], {
    "default": "minimal",
    "tooltip": "Thinking depth before generation (Flash/NB2 ONLY). Higher = better quality but more tokens billed. 'none' disables thinking. ⚠️ IGNORED for Nano Banana Pro — Pro does not support thinking_config."
}),
```

**SDK mapping with model guard**:
```python
thinking_config = None
if MODEL_CONSTRAINTS[model]["supports_thinking"] and thinking_level != "none":
    thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
elif not MODEL_CONSTRAINTS[model]["supports_thinking"] and thinking_level != "none":
    print(f"⚠️ [NanoBanana] thinking_level='{thinking_level}' ignored — {MODEL_CONSTRAINTS[model]['display_name']} does not support thinking_config. Using default behavior.")

config = types.GenerateContentConfig(
    thinking_config=thinking_config,  # Will be None for Pro
    # ...
)
```

---

### 9. Safety Settings — 4 separate COMBO dropdowns

**What they do**: Control the threshold at which the API blocks content for each of 4 harm categories. Setting these to `BLOCK_NONE` reduces false-positive blocks on legitimate content like fashion photography, medical illustrations, etc.

**⚠️ Important caveat to surface in tooltips**: Even with `BLOCK_NONE` on all 4 categories, there is an **additional internal image-specific safety filter** that cannot be disabled via API. If the model responds with "I can't help with that", it's this internal filter, not these settings.

**ComfyUI input definitions** (4 separate dropdowns):
```python
"safety_hate_speech": ([
    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
], {
    "default": "OFF",
    "tooltip": "Hate speech filter threshold. BLOCK_NONE = flag but don't block. OFF = disable entirely. Note: an additional internal image safety filter operates independently and cannot be configured."
}),

"safety_harassment": ([
    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
], {
    "default": "OFF",
    "tooltip": "Harassment filter threshold. BLOCK_NONE = flag but don't block. OFF = disable entirely."
}),

"safety_sexually_explicit": ([
    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
], {
    "default": "OFF",
    "tooltip": "Sexually explicit content filter. This is the most common cause of blocked image generations. BLOCK_NONE = most permissive configurable setting."
}),

"safety_dangerous_content": ([
    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
], {
    "default": "OFF",
    "tooltip": "Dangerous content filter threshold. BLOCK_NONE = flag but don't block. OFF = disable entirely."
}),
```

**SDK mapping** (build the array):
```python
SAFETY_CATEGORIES = {
    "safety_hate_speech":        "HARM_CATEGORY_HATE_SPEECH",
    "safety_harassment":         "HARM_CATEGORY_HARASSMENT",
    "safety_sexually_explicit":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "safety_dangerous_content":  "HARM_CATEGORY_DANGEROUS_CONTENT",
}

safety_settings = []
for input_name, category in SAFETY_CATEGORIES.items():
    threshold = locals_or_kwargs[input_name]  # however you access the input value
    if threshold != "OFF":
        safety_settings.append(
            types.SafetySetting(category=category, threshold=threshold)
        )

config = types.GenerateContentConfig(
    safety_settings=safety_settings if safety_settings else None,
    # ...
)
```

**Model guard**: Works on **both** Pro and Flash. No guard needed.

---

### 10. `presence_penalty` — FLOAT

**What it does**: Penalizes tokens that have already appeared in the output. Positive values encourage more diverse vocabulary. Mainly affects the text portion when using `TEXT_AND_IMAGE` mode, but may subtly influence image generation through the model's internal token planning.

**ComfyUI input definition**:
```python
"presence_penalty": ("FLOAT", {
    "default": 0.0,
    "min": -2.0,
    "max": 2.0,
    "step": 0.1,
    "tooltip": "Penalizes already-appeared tokens (-2.0 to 2.0). Positive = more diverse output, negative = more repetitive. Primarily affects text output in TEXT_AND_IMAGE mode. Default 0.0 is neutral."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
    # ...
)
```

**Model guard**: Works on **both**. No guard needed.

---

### 11. `frequency_penalty` — FLOAT

**What it does**: Penalizes tokens proportionally to how often they've appeared. Stronger effect than `presence_penalty` for reducing repetitive patterns.

**ComfyUI input definition**:
```python
"frequency_penalty": ("FLOAT", {
    "default": 0.0,
    "min": -2.0,
    "max": 2.0,
    "step": 0.1,
    "tooltip": "Penalizes tokens by their frequency (-2.0 to 2.0). Positive = less repetition, negative = more repetition. Stronger effect than presence_penalty. Default 0.0 is neutral."
}),
```

**SDK mapping**:
```python
config = types.GenerateContentConfig(
    frequency_penalty=frequency_penalty if frequency_penalty != 0.0 else None,
    # ...
)
```

**Model guard**: Works on **both**. No guard needed.

---

### 12. `enable_search_grounding` — BOOLEAN + search type guard

**What it does**: Allows the model to search Google before generating, grounding the image in real-world visual references. NB2 (Flash) supports both web search AND image search; Pro supports web search only.

**ComfyUI input definition**:
```python
"enable_search_grounding": ("BOOLEAN", {
    "default": False,
    "label_on": "Search ON",
    "label_off": "Search OFF",
    "tooltip": "Enable Google Search grounding before generation. Model will search for visual references to improve accuracy. Flash/NB2 supports web + image search; Pro supports web search only. Note: 5,000 free search prompts/month, then $14/1,000."
}),
```

**SDK mapping with model guard**:
```python
tools = None
if enable_search_grounding:
    if model == "gemini-3.1-flash-image-preview":
        # NB2 supports both web and image search
        tools = [{"google_search": {"search_types": {"web_search": {}, "image_search": {}}}}]
    else:
        # Pro supports web search only
        tools = [{"google_search": {}}]

# Pass tools separately or via config depending on SDK version
```

---

## Parameters to Add — Nano Banana Chat Edit Node

Add these to the Chat Edit node as well (subset of Generate node params):

| Parameter | Add to Chat Edit? | Notes |
|---|---|---|
| `system_instruction` | ✅ Yes | Shapes the editing persona |
| `top_p` | ✅ Yes | Affects edit variations |
| `top_k` | ✅ Yes | Affects edit variations |
| `temperature` | ✅ Yes (if not already there) | Key for edit creativity |
| `max_output_tokens` | ✅ Yes | Prevents truncated edits |
| `thinking_level` | ✅ Yes | With same model guard |
| `safety_*` (4 categories) | ✅ Yes | Same 4 dropdowns |
| `candidate_count` | ❌ No | Chat is single-threaded |
| `stop_sequences` | ❌ No | Not meaningful in chat |
| `seed` | ❌ No | Not meaningful in chat |
| `presence_penalty` | ❌ No | Minimal impact on edits |
| `frequency_penalty` | ❌ No | Minimal impact on edits |
| `enable_search_grounding` | ✅ Yes | Useful for reference-based edits |

---

## Aspect Ratio & Image Size Validation

Add runtime validation that warns (not errors) when the user picks a combination unsupported by their selected model:

```python
def validate_model_params(model, aspect_ratio, image_size, thinking_level, num_refs):
    """Validate params against model constraints. Returns list of warning strings."""
    warnings = []
    c = MODEL_CONSTRAINTS[model]

    if aspect_ratio not in c["aspect_ratios"]:
        warnings.append(
            f"⚠️ Aspect ratio '{aspect_ratio}' is not supported by {c['display_name']}. "
            f"Supported: {', '.join(c['aspect_ratios'])}. The API may reject this or fall back to 1:1."
        )

    if image_size not in c["image_sizes"]:
        warnings.append(
            f"⚠️ Image size '{image_size}' is not supported by {c['display_name']}. "
            f"Supported: {', '.join(c['image_sizes'])}. Falling back to '1K'."
        )

    if thinking_level != "none" and not c["supports_thinking"]:
        warnings.append(
            f"⚠️ thinking_level='{thinking_level}' is not supported by {c['display_name']}. "
            f"This parameter will be silently removed to prevent API errors."
        )

    if num_refs > c["max_reference_images"]:
        warnings.append(
            f"⚠️ {num_refs} reference images provided but {c['display_name']} supports max {c['max_reference_images']}. "
            f"Extra images will be truncated."
        )

    return warnings
```

Call this at the start of your `generate()` function and print each warning.

---

## Error Handling & Visual Logging (Constraint #4)

### Structured log format

Create a helper that formats errors for the ComfyUI console with clear visual hierarchy:

```python
import time

class NanoBananaLogger:
    PREFIX = "🍌 [NanoBanana]"

    @staticmethod
    def info(msg):
        print(f"{NanoBananaLogger.PREFIX} ℹ️  {msg}")

    @staticmethod
    def warn(msg):
        print(f"{NanoBananaLogger.PREFIX} ⚠️  {msg}")

    @staticmethod
    def error(msg):
        print(f"{NanoBananaLogger.PREFIX} ❌ {msg}")

    @staticmethod
    def success(msg):
        print(f"{NanoBananaLogger.PREFIX} ✅ {msg}")

    @staticmethod
    def api_call(model, image_size, aspect_ratio):
        print(f"\n{'='*60}")
        print(f"{NanoBananaLogger.PREFIX} API Call")
        print(f"  Model:        {model}")
        print(f"  Size:         {image_size}")
        print(f"  Aspect Ratio: {aspect_ratio}")
        print(f"  Time:         {time.strftime('%H:%M:%S')}")
        print(f"{'='*60}")

    @staticmethod
    def api_result(success, candidates_returned, images_extracted, tokens_used=None, duration_s=None):
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"\n{'-'*60}")
        print(f"{NanoBananaLogger.PREFIX} Result: {status}")
        print(f"  Candidates:   {candidates_returned}")
        print(f"  Images:       {images_extracted}")
        if tokens_used:
            print(f"  Tokens used:  {tokens_used}")
        if duration_s:
            print(f"  Duration:     {duration_s:.1f}s")
        print(f"{'-'*60}\n")

    @staticmethod
    def safety_block(finish_reason, safety_ratings=None):
        print(f"\n{'!'*60}")
        print(f"{NanoBananaLogger.PREFIX} ❌ GENERATION BLOCKED")
        print(f"  Reason: {finish_reason}")
        if safety_ratings:
            for rating in safety_ratings:
                print(f"  - {rating.category}: {rating.probability}")
        print(f"  Tip: Try adjusting safety settings or rephrasing your prompt.")
        print(f"  Note: Even with BLOCK_NONE, an internal image filter may still block.")
        print(f"{'!'*60}\n")

    @staticmethod
    def batch_progress(current, total):
        print(f"{NanoBananaLogger.PREFIX} 🔄 Batch {current}/{total}")
```

### Integrate into the generate function

```python
def generate(self, **kwargs):
    logger = NanoBananaLogger

    # 1. Validate params
    warnings = validate_model_params(model, aspect_ratio, image_size, thinking_level, num_refs)
    for w in warnings:
        logger.warn(w)

    # 2. Log API call
    logger.api_call(model, image_size, aspect_ratio)
    start_time = time.time()

    try:
        all_images = []
        all_texts = []

        for batch_i in range(batch_count):
            if batch_count > 1:
                logger.batch_progress(batch_i + 1, batch_count)

            response = client.models.generate_content(model=model, contents=contents, config=config)

            # Check for safety blocks
            for candidate in response.candidates:
                if candidate.finish_reason and candidate.finish_reason != "STOP":
                    logger.safety_block(
                        candidate.finish_reason,
                        getattr(candidate, 'safety_ratings', None)
                    )
                    continue

                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.text is not None:
                            all_texts.append(part.text)
                        elif part.inline_data is not None:
                            all_images.append(base64_to_tensor(part.inline_data.data))

        duration = time.time() - start_time
        logger.api_result(
            success=len(all_images) > 0,
            candidates_returned=len(response.candidates) if response.candidates else 0,
            images_extracted=len(all_images),
            tokens_used=getattr(response, 'usage_metadata', None),
            duration_s=duration
        )

        if not all_images:
            logger.error("No images generated. Returning empty 512x512 placeholder.")
            return (empty_placeholder_tensor(), "\n".join(all_texts) or "No output generated.")

        return (torch.cat(all_images, dim=0), "\n".join(all_texts))

    except Exception as e:
        logger.error(f"API call failed: {type(e).__name__}: {str(e)}")
        if "429" in str(e):
            logger.error("Rate limit hit. Wait 60s or upgrade API tier.")
        elif "thinking" in str(e).lower():
            logger.error(f"thinking_config not supported for {model}. Remove thinking_level or switch to Flash.")
        elif "401" in str(e) or "403" in str(e):
            logger.error("Invalid API key. Verify at https://aistudio.google.com/apikey")
        raise
```

---

## Tooltip Strings Reference (Constraint #5)

Every parameter must have a `"tooltip"` key in its ComfyUI input tuple. Here is the complete set for copy-paste — each written to be understood by someone who has never touched an API:

| Parameter | Tooltip Text |
|---|---|
| `prompt` | `"Describe the image you want to generate. Be specific about subject, style, lighting, composition. Supports multiple languages."` |
| `model` | `"Nano Banana Pro = highest quality, best text rendering. Nano Banana 2 (Flash) = faster, cheaper, supports thinking & 512px & extreme aspect ratios."` |
| `aspect_ratio` | `"Output image proportions. Pro supports 10 ratios. Flash supports 14 including extreme (1:8, 8:1). Invalid ratios for the selected model will be flagged."` |
| `image_size` | `"Output resolution. 512px (Flash only), 1K (1024px), 2K (2048px), 4K (4096px). Larger = higher quality but more tokens billed."` |
| `response_modality` | `"IMAGE = only image output (saves ~3% on tokens). TEXT_AND_IMAGE = image + text description/caption."` |
| `api_key` | `"Your Google AI Studio API key. Leave blank to use GEMINI_API_KEY environment variable."` |
| `system_instruction` | `"Persistent instruction that guides model behavior. Sits above the user prompt. Use for consistent style direction, persona, or constraints across generations."` |
| `temperature` | `"Controls randomness (0.0–2.0). Lower = more predictable, higher = more creative. Google recommends keeping at 1.0 for Gemini 3 models — lower values may cause looping."` |
| `top_p` | `"Nucleus sampling (0.0–1.0). Model considers the smallest token set whose cumulative probability ≥ this value. Lower = more focused. Tip: adjust either temperature OR top_p, not both."` |
| `top_k` | `"Top-k sampling (1–100). Model considers only the K most probable tokens. Lower = more predictable, higher = more creative."` |
| `candidate_count` | `"Response candidates per API call (1–4). Each candidate may produce a different image variation. Cost scales linearly. Different from batch_count."` |
| `max_output_tokens` | `"Max output tokens (text + thinking + image). Image tokens: ~1,120 for 1K, ~1,600 for 2K, ~2,520 for 4K. Too low = image may fail. Recommended minimum: 2048 for 1K, 8192 for 4K."` |
| `stop_sequences` | `"Comma-separated stop strings. Model stops generating when encountered. Useful with TEXT_AND_IMAGE to limit text. Leave empty to disable."` |
| `seed` | `"Sampling seed (0 = random). ⚠️ Influences but does NOT guarantee identical images. Same seed + prompt = similar (not identical) results."` |
| `thinking_level` | `"Reasoning depth (Flash/NB2 ONLY). Higher = better quality, more tokens. ⚠️ IGNORED for Pro — causes API error if sent."` |
| `batch_count` | `"Number of separate API calls (1–8). Each call produces 1 image. Total images = batch_count × candidate_count. Calls run concurrently where rate limits allow."` |
| `safety_hate_speech` | `"Hate speech filter. BLOCK_NONE = most permissive. OFF = disabled. Note: an internal image filter may still block independently."` |
| `safety_harassment` | `"Harassment filter. BLOCK_NONE = most permissive. OFF = disabled."` |
| `safety_sexually_explicit` | `"Sexually explicit filter — most common cause of blocked generations. BLOCK_NONE = most permissive configurable setting."` |
| `safety_dangerous_content` | `"Dangerous content filter. BLOCK_NONE = most permissive. OFF = disabled."` |
| `presence_penalty` | `"Penalizes already-appeared tokens (-2.0 to 2.0). Positive = more diverse. Primarily affects text in TEXT_AND_IMAGE mode."` |
| `frequency_penalty` | `"Penalizes tokens by frequency (-2.0 to 2.0). Positive = less repetition. Stronger than presence_penalty."` |
| `enable_search_grounding` | `"Google Search grounding. Model searches web before generating for real-world accuracy. Flash supports web + image search; Pro web only. 5,000 free/month."` |

---

## Batch Processing Implementation (Constraint #3)

Your batch implementation should combine `candidate_count` (per-call variations) with `batch_count` (number of calls) for maximum throughput:

```python
import concurrent.futures
import time

def _execute_batch(self, client, model, contents, config, batch_count, candidate_count, logger):
    """Execute batch generation with concurrency and rate-limit awareness."""
    all_images = []
    all_texts = []
    failed_calls = 0

    def single_call(call_index):
        try:
            if call_index > 0:
                time.sleep(0.5)  # Basic rate-limit buffer between calls
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as e:
            logger.error(f"Batch call {call_index + 1} failed: {e}")
            return None

    max_workers = min(batch_count, 4)  # Don't exceed 4 concurrent calls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(single_call, i): i
            for i in range(batch_count)
        }

        for future in concurrent.futures.as_completed(futures):
            call_idx = futures[future]
            logger.batch_progress(call_idx + 1, batch_count)

            response = future.result()
            if response is None:
                failed_calls += 1
                continue

            for candidate in response.candidates:
                if candidate.finish_reason and candidate.finish_reason not in ("STOP", None):
                    logger.safety_block(candidate.finish_reason)
                    continue
                if candidate.content:
                    for part in candidate.content.parts:
                        if part.text:
                            all_texts.append(part.text)
                        elif part.inline_data:
                            all_images.append(base64_to_tensor(part.inline_data.data))

    if failed_calls > 0:
        logger.warn(f"{failed_calls}/{batch_count} batch calls failed.")

    total_expected = batch_count * candidate_count
    logger.info(f"Batch complete: {len(all_images)}/{total_expected} images generated.")

    return all_images, all_texts
```

---

## Complete Updated `GenerateContentConfig` Assembly

Here's the final, complete config assembly function with all guards:

```python
def build_config(self, model, response_modality, aspect_ratio, image_size,
                 temperature, top_p, top_k, candidate_count, max_output_tokens,
                 stop_sequences, seed, thinking_level, presence_penalty,
                 frequency_penalty, system_instruction, enable_search_grounding,
                 safety_hate_speech, safety_harassment, safety_sexually_explicit,
                 safety_dangerous_content):

    constraints = MODEL_CONSTRAINTS[model]
    logger = NanoBananaLogger

    # --- Guard: thinking_config ---
    thinking_config = None
    if constraints["supports_thinking"] and thinking_level != "none":
        thinking_config = types.ThinkingConfig(thinking_level=thinking_level)
    elif not constraints["supports_thinking"] and thinking_level != "none":
        logger.warn(f"thinking_level='{thinking_level}' removed — not supported by {constraints['display_name']}")

    # --- Guard: image_size ---
    if image_size not in constraints["image_sizes"]:
        logger.warn(f"image_size='{image_size}' not supported by {constraints['display_name']}. Falling back to '1K'.")
        image_size = "1K"

    # --- Guard: aspect_ratio ---
    if aspect_ratio not in constraints["aspect_ratios"]:
        logger.warn(f"aspect_ratio='{aspect_ratio}' not supported by {constraints['display_name']}. Falling back to '1:1'.")
        aspect_ratio = "1:1"

    # --- Safety settings ---
    safety_map = {
        "HARM_CATEGORY_HATE_SPEECH": safety_hate_speech,
        "HARM_CATEGORY_HARASSMENT": safety_harassment,
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": safety_sexually_explicit,
        "HARM_CATEGORY_DANGEROUS_CONTENT": safety_dangerous_content,
    }
    safety_settings = [
        types.SafetySetting(category=cat, threshold=thresh)
        for cat, thresh in safety_map.items()
        if thresh != "OFF"
    ] or None

    # --- Stop sequences ---
    parsed_stops = (
        [s.strip() for s in stop_sequences.split(",") if s.strip()]
        if stop_sequences and stop_sequences.strip()
        else None
    )

    # --- Search tools ---
    tools = None
    if enable_search_grounding:
        if constraints.get("supports_image_search"):
            tools = [{"google_search": {"search_types": {"web_search": {}, "image_search": {}}}}]
        else:
            tools = [{"google_search": {}}]

    # --- Build config ---
    config = types.GenerateContentConfig(
        response_modalities=(
            ["TEXT", "IMAGE"] if response_modality == "TEXT_AND_IMAGE" else ["IMAGE"]
        ),
        image_config=types.ImageConfig(
            aspect_ratio=aspect_ratio,
            image_size=image_size,
        ),
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        candidate_count=candidate_count,
        max_output_tokens=max_output_tokens,
        stop_sequences=parsed_stops,
        seed=seed if seed > 0 else None,
        presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
        frequency_penalty=frequency_penalty if frequency_penalty != 0.0 else None,
        safety_settings=safety_settings,
        system_instruction=system_instruction.strip() if system_instruction and system_instruction.strip() else None,
        thinking_config=thinking_config,
    )

    return config, tools
```

---

## Checklist

- [ ] Add `system_instruction` (STRING, multiline) to Generate + Chat Edit
- [ ] Add `top_p` (FLOAT 0–1, default 0.95) to Generate + Chat Edit
- [ ] Add `top_k` (INT 1–100, default 40) to Generate + Chat Edit
- [ ] Add `candidate_count` (INT 1–4, default 1) to Generate only
- [ ] Add `max_output_tokens` (INT 1024–32768, default 8192) to Generate + Chat Edit
- [ ] Add `stop_sequences` (STRING, comma-separated) to Generate only
- [ ] Add `seed` (INT 0–2147483647, default 0) to Generate only
- [ ] Add `thinking_level` (COMBO, 5 options) to Generate + Chat Edit **with model guard**
- [ ] Add 4× safety dropdowns to Generate + Chat Edit
- [ ] Add `presence_penalty` (FLOAT -2 to 2) to Generate only
- [ ] Add `frequency_penalty` (FLOAT -2 to 2) to Generate only
- [ ] Add `enable_search_grounding` (BOOLEAN) to Generate + Chat Edit **with model-specific search type**
- [ ] Add `MODEL_CONSTRAINTS` dict with all model-specific rules
- [ ] Add `validate_model_params()` function
- [ ] Add `NanoBananaLogger` class for structured console output
- [ ] Add tooltip to every single parameter
- [ ] Update response parsing to iterate `response.candidates` (not just `response.parts`) for `candidate_count > 1`
- [ ] Update batch logic to use `concurrent.futures.ThreadPoolExecutor`
- [ ] Add `max_output_tokens` soft warning based on `image_size`

"""
ComfyUI node class definitions for Google API and Nano Banana image generation.

Nodes:
  - GoogleImagenAPIKeyNode: Resolves Google API key via 3-tier priority.
  - NanoBananaGenerate: Text-to-image via Gemini SDK.
  - NanoBananaChat: Multi-turn conversational image editing.
"""

from .client_manager import get_client, resolve_api_key
from io import BytesIO
from PIL import Image as PILImage
from .type_converters import (
    comfy_tensor_to_pil, 
    pil_to_comfy_tensor
)
from .error_handlers import handle_api_error
from google.genai import types
import torch
import random
import time
import concurrent.futures

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

MODEL_CONSTRAINTS = {
    "gemini-3-pro-image-preview": {
        "display_name": "Nano Banana Pro",
        "supports_thinking": False,
        "supports_512px": False,
        "thinking_levels": [],
        "aspect_ratios": [
            "1:1", "2:3", "3:2", "3:4", "4:3",
            "4:5", "5:4", "9:16", "16:9", "21:9"
        ],
        "image_sizes": ["1K", "2K", "4K"],
        "max_reference_images": 14,
        "supports_search_grounding": True,
        "supports_image_search": False,
    },
    "gemini-3.1-flash-image-preview": {
        "display_name": "Nano Banana 2 (Flash)",
        "supports_thinking": True,
        "supports_512px": True,
        "thinking_levels": ["minimal", "low", "medium", "high"],
        "aspect_ratios": [
            "1:1", "1:4", "1:8", "2:3", "3:2", "3:4",
            "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"
        ],
        "image_sizes": ["512px", "1K", "2K", "4K"],
        "max_reference_images": 14,
        "supports_search_grounding": True,
        "supports_image_search": True,
    },
}

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

def build_config(model, response_modality, aspect_ratio, image_size,
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

class GoogleImagenAPIKeyNode:
    """Provides a Google API key for Nano Banana nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": (
                            "Google API key. Leave empty to use GOOGLE_API_KEY "
                            "env var or google_api_key.txt config file."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("api_key",)
    FUNCTION = "get_key"
    CATEGORY = "Google AI/Image Generation"
    DESCRIPTION = "Provides Google API key for Nano Banana nodes"

    def get_key(self, api_key: str = "") -> tuple[str]:
        resolved = resolve_api_key(api_key)
        return (resolved,)


def empty_image_tensor():
    return torch.zeros((1, 512, 512, 3), dtype=torch.float32)

class NanoBananaGenerate:
    """Text-to-image and image-to-image using Nano Banana (Gemini generateContent)"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Describe the image to generate...",
                    "tooltip": "Describe the image you want to generate. Be specific about subject, style, lighting, composition. Supports multiple languages."
                }),
                "model": ([
                    "gemini-3.1-flash-image-preview",
                    "gemini-3-pro-image-preview",
                ], {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "Nano Banana Pro = highest quality, best text rendering. Nano Banana 2 (Flash) = faster, cheaper, supports thinking & 512px & extreme aspect ratios."
                }),
                "aspect_ratio": ([
                    "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4",
                    "9:16", "16:9", "21:9",
                    "1:4", "4:1", "1:8", "8:1",
                ], {
                    "default": "1:1",
                    "tooltip": "Output image proportions. Pro supports 10 ratios. Flash supports 14 including extreme (1:8, 8:1). Invalid ratios for the selected model will be flagged."
                }),
                "image_size": ([
                    "512px", "1K", "2K", "4K"
                ], {
                    "default": "1K",
                    "tooltip": "Output resolution. 512px (Flash only), 1K (1024px), 2K (2048px), 4K (4096px). Larger = higher quality but more tokens billed."
                }),
                "response_modality": ([
                    "IMAGE", "TEXT_AND_IMAGE"
                ], {
                    "default": "IMAGE",
                    "tooltip": "IMAGE = only image output (saves ~3% on tokens). TEXT_AND_IMAGE = image + text description/caption."
                }),
                "system_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional system prompt (e.g., 'You are a professional product photographer. Always produce studio-lit compositions.')",
                    "tooltip": "Persistent instruction that guides model behavior. Sits above the user prompt. Use for consistent style direction, persona, or constraints across generations."
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Controls randomness (0.0–2.0). Lower = more predictable, higher = more creative. Google recommends keeping at 1.0 for Gemini 3 models — lower values may cause looping."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling (0.0–1.0). Model considers the smallest token set whose cumulative probability ≥ this value. Lower = more focused. Tip: adjust either temperature OR top_p, not both."
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Top-k sampling (1–100). Model considers only the K most probable tokens. Lower = more predictable, higher = more creative."
                }),
                "candidate_count": ("INT", {
                    "default": 1, "min": 1, "max": 4, "step": 1,
                    "tooltip": "Response candidates per API call (1–4). Each candidate may produce a different image variation. Cost scales linearly. Different from batch_count."
                }),
                "max_output_tokens": ("INT", {
                    "default": 8192, "min": 1024, "max": 32768, "step": 512,
                    "tooltip": "Max output tokens (text + thinking + image). Image tokens: ~1,120 for 1K, ~1,600 for 2K, ~2,520 for 4K. Too low = image may fail. Recommended minimum: 2048 for 1K, 8192 for 4K."
                }),
                "thinking_level": ([
                    "none", "minimal", "low", "medium", "high"
                ], {
                    "default": "minimal",
                    "tooltip": "Reasoning depth (Flash/NB2 ONLY). Higher = better quality, more tokens. ⚠️ IGNORED for Pro — causes API error if sent."
                }),
                "batch_count": ("INT", {
                    "default": 1, "min": 1, "max": 8, "step": 1, "display": "number",
                    "tooltip": "Number of separate API calls (1–8). Each call produces 1 image. Total images = batch_count × candidate_count. Calls run concurrently where rate limits allow."
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "", "forceInput": True,
                    "placeholder": "Leave blank to use GEMINI_API_KEY env var",
                    "tooltip": "Your Google AI Studio API key. Leave blank to use GEMINI_API_KEY environment variable."
                }),
                "reference_image_1": ("IMAGE", {"tooltip": "Individual reference image 1. Unaffected by batch cropping."}),
                "stop_sequences": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "Comma-separated (e.g., STOP,END,---)",
                    "tooltip": "Comma-separated stop strings. Model stops generating when encountered. Useful with TEXT_AND_IMAGE to limit text. Leave empty to disable."
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2147483647, "step": 1,
                    "tooltip": "Sampling seed (0 = random). ⚠️ Influences but does NOT guarantee identical images. Same seed + prompt = similar (not identical) results."
                }),
                "safety_hate_speech": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Hate speech filter. BLOCK_NONE = most permissive. OFF = disabled. Note: an internal image filter may still block independently."
                }),
                "safety_harassment": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Harassment filter. BLOCK_NONE = most permissive. OFF = disabled."
                }),
                "safety_sexually_explicit": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Sexually explicit filter — most common cause of blocked generations. BLOCK_NONE = most permissive configurable setting."
                }),
                "safety_dangerous_content": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Dangerous content filter. BLOCK_NONE = most permissive. OFF = disabled."
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Penalizes already-appeared tokens (-2.0 to 2.0). Positive = more diverse. Primarily affects text in TEXT_AND_IMAGE mode."
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Penalizes tokens by frequency (-2.0 to 2.0). Positive = less repetition. Stronger than presence_penalty."
                }),
                "enable_search_grounding": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Search ON",
                    "label_off": "Search OFF",
                    "tooltip": "Google Search grounding. Model searches web before generating for real-world accuracy. Flash supports web + image search; Pro web only. 5,000 free/month."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "response_text")
    FUNCTION = "generate"
    CATEGORY = "Google AI/Image Generation"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")
        
    def _execute_batch(self, client, model, contents, config, batch_count, candidate_count, logger):
        """Execute batch generation with concurrency and rate-limit awareness."""
        all_images = []
        all_texts = []
        failed_calls = 0
        total_prompt_tokens = 0
        total_candidate_tokens = 0

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

                if hasattr(response, 'usage_metadata') and response.usage_metadata:
                    total_prompt_tokens += getattr(response.usage_metadata, 'prompt_token_count', 0) or 0
                    total_candidate_tokens += getattr(response.usage_metadata, 'candidates_token_count', 0) or 0

                for candidate in response.candidates:
                    if candidate.finish_reason and candidate.finish_reason not in ("STOP", None):
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
                                # We try to extract image via as_image or raw bytes decoding
                                try:
                                    pil_img = part.as_image()
                                    all_images.append(pil_to_comfy_tensor(pil_img))
                                except Exception:
                                    pil_img = PILImage.open(BytesIO(part.inline_data.data)).convert("RGB")
                                    all_images.append(pil_to_comfy_tensor(pil_img))

        if failed_calls > 0:
            logger.warn(f"{failed_calls}/{batch_count} batch calls failed.")

        total_expected = batch_count * candidate_count
        logger.info(f"Batch complete: {len(all_images)}/{total_expected} images generated.")

        tokens_used_str = (
            f"Prompt: {total_prompt_tokens} | Candidates: {total_candidate_tokens} | Total: {total_prompt_tokens + total_candidate_tokens}"
            if (total_prompt_tokens or total_candidate_tokens) else None
        )

        return all_images, all_texts, tokens_used_str

    def generate(self, prompt, model, aspect_ratio, image_size, response_modality,
                 system_instruction, temperature, top_p, top_k, candidate_count, 
                 max_output_tokens, thinking_level, batch_count,
                 api_key="", stop_sequences="", seed=0,
                 safety_hate_speech="BLOCK_NONE", safety_harassment="BLOCK_NONE", 
                 safety_sexually_explicit="BLOCK_NONE", safety_dangerous_content="BLOCK_NONE",
                 presence_penalty=0.0, frequency_penalty=0.0, enable_search_grounding=False, **kwargs):
        
        logger = NanoBananaLogger
        
        # Collect all reference images
        all_refs = []
        if "reference_images" in kwargs and kwargs["reference_images"] is not None:
            B = kwargs["reference_images"].shape[0]
            for i in range(B):
                all_refs.append(kwargs["reference_images"][i:i+1])
                
        for i in range(1, 15):
            key = f"reference_image_{i}"
            if key in kwargs and kwargs[key] is not None:
                img_tensor = kwargs[key]
                B = img_tensor.shape[0]
                for j in range(B):
                    all_refs.append(img_tensor[j:j+1])
        
        # 1. Validate params
        num_refs = len(all_refs)
        warnings = validate_model_params(model, aspect_ratio, image_size, thinking_level, num_refs)
        for w in warnings:
            logger.warn(w)
            
        IMAGE_TOKEN_ESTIMATES = {"512px": 800, "1K": 1120, "2K": 1600, "4K": 2520}
        min_recommended = IMAGE_TOKEN_ESTIMATES.get(image_size, 1120) + 512  # buffer for text/thinking
        if max_output_tokens < min_recommended:
            print(f"⚠️ [NanoBanana] max_output_tokens ({max_output_tokens}) may be too low for {image_size} images. Recommended minimum: {min_recommended}")

        # 2. Log API call
        logger.api_call(model, image_size, aspect_ratio)
        start_time = time.time()
        
        if seed > 0:
            random.seed(seed)

        client = get_client(api_key)

        parts = [prompt]
        max_refs = MODEL_CONSTRAINTS[model]["max_reference_images"]
        for i in range(min(len(all_refs), max_refs)):
            parts.append(comfy_tensor_to_pil(all_refs[i]))

        config, tools = build_config(
            model=model, response_modality=response_modality, aspect_ratio=aspect_ratio,
            image_size=image_size, temperature=temperature, top_p=top_p, top_k=top_k,
            candidate_count=candidate_count, max_output_tokens=max_output_tokens,
            stop_sequences=stop_sequences, seed=seed, thinking_level=thinking_level,
            presence_penalty=presence_penalty, frequency_penalty=frequency_penalty,
            system_instruction=system_instruction, enable_search_grounding=enable_search_grounding,
            safety_hate_speech=safety_hate_speech, safety_harassment=safety_harassment,
            safety_sexually_explicit=safety_sexually_explicit, safety_dangerous_content=safety_dangerous_content
        )

        try:
            if tools:
                config.tools = tools

            all_images, all_texts, tokens_used_str = self._execute_batch(
                client=client, model=model, contents=parts, config=config, 
                batch_count=batch_count, candidate_count=candidate_count, logger=logger
            )

            duration = time.time() - start_time
            
            logger.api_result(
                success=len(all_images) > 0,
                candidates_returned=len(all_images), # rough estimate
                images_extracted=len(all_images),
                tokens_used=tokens_used_str,
                duration_s=duration
            )

            if not all_images:
                logger.error("No images generated. Returning empty 512x512 placeholder.")
                return (empty_image_tensor(), "\n".join(all_texts) or "No output generated.")

            return (torch.cat(all_images, dim=0), "\n---\n".join(all_texts))

        except Exception as e:
            logger.error(f"API call failed: {type(e).__name__}: {str(e)}")
            if "429" in str(e):
                logger.error("Rate limit hit. Wait 60s or upgrade API tier.")
            elif "thinking" in str(e).lower():
                logger.error(f"thinking_config not supported for {model}. Remove thinking_level or switch to Flash.")
            elif "401" in str(e) or "403" in str(e):
                logger.error("Invalid API key. Verify at https://aistudio.google.com/apikey")
            raise

class NanoBananaChat:
    """Multi-turn conversational editing node"""
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Edit instruction (e.g., 'Change the sky to sunset')",
                    "tooltip": "Describe the image you want to generate. Be specific about subject, style, lighting, composition. Supports multiple languages."
                }),
                "model": ([
                    "gemini-3.1-flash-image-preview",
                    "gemini-3-pro-image-preview",
                ], {
                    "default": "gemini-3.1-flash-image-preview",
                    "tooltip": "Nano Banana Pro = highest quality, best text rendering. Nano Banana 2 (Flash) = faster, cheaper, supports thinking & 512px & extreme aspect ratios."
                }),
                "aspect_ratio": ([
                    "1:1", "1:4", "1:8", "2:3", "3:2", "3:4", "4:1", "4:3", "4:5", "5:4", "8:1", "9:16", "16:9", "21:9"
                ], {
                    "default": "1:1",
                    "tooltip": "Output image proportions. Pro supports 10 ratios. Flash supports 14 including extreme (1:8, 8:1). Invalid ratios for the selected model will be flagged."
                }),
                "image_size": ([
                    "512px", "1K", "2K", "4K"
                ], {
                    "default": "1K",
                    "tooltip": "Output resolution. 512px (Flash only), 1K (1024px), 2K (2048px), 4K (4096px). Larger = higher quality but more tokens billed."
                }),
                "system_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "placeholder": "Optional system prompt...",
                    "tooltip": "Persistent instruction that guides model behavior. Sits above the user prompt. Use for consistent style direction, persona, or constraints across generations."
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Controls randomness (0.0–2.0). Lower = more predictable, higher = more creative. Google recommends keeping at 1.0 for Gemini 3 models — lower values may cause looping."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling (0.0–1.0). Model considers the smallest token set whose cumulative probability ≥ this value. Lower = more focused. Tip: adjust either temperature OR top_p, not both."
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 1, "max": 100, "step": 1,
                    "tooltip": "Top-k sampling (1–100). Model considers only the K most probable tokens. Lower = more predictable, higher = more creative."
                }),
                "max_output_tokens": ("INT", {
                    "default": 8192, "min": 1024, "max": 32768, "step": 512,
                    "tooltip": "Max output tokens (text + thinking + image). Image tokens: ~1,120 for 1K, ~1,600 for 2K, ~2,520 for 4K. Too low = image may fail. Recommended minimum: 2048 for 1K, 8192 for 4K."
                }),
                "thinking_level": ([
                    "none", "minimal", "low", "medium", "high"
                ], {
                    "default": "minimal",
                    "tooltip": "Reasoning depth (Flash/NB2 ONLY). Higher = better quality, more tokens. ⚠️ IGNORED for Pro — causes API error if sent."
                }),
            },
            "optional": {
                "api_key": ("STRING", {
                    "default": "", "forceInput": True,
                    "placeholder": "Leave blank to use GEMINI_API_KEY env var",
                    "tooltip": "Your Google AI Studio API key. Leave blank to use GEMINI_API_KEY environment variable."
                }),
                "input_image": ("IMAGE",),
                "chat_history": ("NANO_BANANA_CHAT_HISTORY",),
                "safety_hate_speech": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Hate speech filter. BLOCK_NONE = most permissive. OFF = disabled. Note: an internal image filter may still block independently."
                }),
                "safety_harassment": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Harassment filter. BLOCK_NONE = most permissive. OFF = disabled."
                }),
                "safety_sexually_explicit": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Sexually explicit filter — most common cause of blocked generations. BLOCK_NONE = most permissive configurable setting."
                }),
                "safety_dangerous_content": ([
                    "BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH", "OFF"
                ], {
                    "default": "BLOCK_NONE",
                    "tooltip": "Dangerous content filter. BLOCK_NONE = most permissive. OFF = disabled."
                }),
                "enable_search_grounding": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Search ON",
                    "label_off": "Search OFF",
                    "tooltip": "Google Search grounding. Model searches web before generating for real-world accuracy. Flash supports web + image search; Pro web only. 5,000 free/month."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "NANO_BANANA_CHAT_HISTORY")
    RETURN_NAMES = ("edited_image", "response_text", "chat_history")
    FUNCTION = "generate"
    CATEGORY = "Google AI/Image Generation"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("NaN")

    def generate(self, instruction, model, aspect_ratio, image_size,
                 system_instruction, temperature, top_p, top_k, max_output_tokens,
                 thinking_level, api_key="", input_image=None, chat_history=None,
                 safety_hate_speech="BLOCK_NONE", safety_harassment="BLOCK_NONE", 
                 safety_sexually_explicit="BLOCK_NONE", safety_dangerous_content="BLOCK_NONE",
                 enable_search_grounding=False):
                 
        logger = NanoBananaLogger
        
        # Validate params
        warnings = validate_model_params(model, aspect_ratio, image_size, thinking_level, 0)
        for w in warnings:
            logger.warn(w)

        client = get_client(api_key)

        config, tools = build_config(
            model=model, response_modality="TEXT_AND_IMAGE", aspect_ratio=aspect_ratio,
            image_size=image_size, temperature=temperature, top_p=top_p, top_k=top_k,
            candidate_count=1, max_output_tokens=max_output_tokens,
            stop_sequences="", seed=0, thinking_level=thinking_level,
            presence_penalty=0.0, frequency_penalty=0.0, system_instruction=system_instruction,
            enable_search_grounding=enable_search_grounding,
            safety_hate_speech=safety_hate_speech, safety_harassment=safety_harassment,
            safety_sexually_explicit=safety_sexually_explicit, safety_dangerous_content=safety_dangerous_content
        )
        
        if tools:
            config.tools = tools

        try:
            chat = client.chats.create(model=model, config=config, history=chat_history)
            
            message_parts = []
            if input_image is not None and not chat_history:
                message_parts.append(comfy_tensor_to_pil(input_image))
            message_parts.append(instruction)
            
            # Record start time for latency measurement
            start_time = time.time()
            response = chat.send_message(message_parts)
            duration = time.time() - start_time
            
            from .response_parsers import parse_gemini_response
            img_tensor, txt = parse_gemini_response(response)
            
            usage = getattr(response, 'usage_metadata', None)
            tokens_used_str = None
            if usage:
                tokens_used_str = f"Prompt: {getattr(usage, 'prompt_token_count', 0) or 0} | Candidates: {getattr(usage, 'candidates_token_count', 0) or 0} | Total: {getattr(usage, 'total_token_count', 0) or 0}"

            logger.api_result(
                success=True,
                candidates_returned=len(response.candidates),
                images_extracted=1 if img_tensor.shape[0] > 0 else 0,
                tokens_used=tokens_used_str,
                duration_s=duration
            )
            
            return (img_tensor, txt, chat.get_history())
        except Exception as e:
            logger.error(f"API call failed: {type(e).__name__}: {str(e)}")
            handle_api_error(e)

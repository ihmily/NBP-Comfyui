"""
Singleton manager for the Google GenAI SDK client.

Lazily initialises one `google.genai.Client` per resolved API key
and reuses it for every subsequent call.
"""

import os

# Lazy import — the SDK may not be installed when ComfyUI first scans nodes.
_client_cache: dict = {}  # keyed by api_key string


def _resolve_api_key(override: str = "") -> str:
    """Resolve Google API key using three-tier priority.

    Priority: environment variable → config file → node input override.

    Args:
        override: API key string from the node input. Empty string means not set.

    Returns:
        Resolved API key string.

    Raises:
        RuntimeError: If no API key is found via any method.
    """
    # 1. Environment variable
    env_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    # 2. Config file in the extension directory
    config_path = os.path.join(os.path.dirname(__file__), "google_api_key.txt")
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            file_key = f.read().strip()
        if file_key:
            return file_key

    # 3. Node input override
    if override and override.strip():
        return override.strip()

    raise RuntimeError(
        "No Google API key found. Provide one via:\n"
        "  1. GOOGLE_API_KEY or GEMINI_API_KEY environment variable\n"
        "  2. google_api_key.txt file in the extension directory\n"
        "  3. The Google API Key node connected to the api_key input"
    )


def get_client(api_key_override: str = ""):
    """Return a cached `google.genai.Client`, creating one if necessary.

    The client is cached per resolved API key so switching keys at
    runtime works correctly while avoiding redundant initialisations.

    Args:
        api_key_override: Optional API key string from a node input.

    Returns:
        A `google.genai.Client` instance ready for use.
    """
    from google import genai  # deferred so import errors surface at call-time
    from google.genai.types import HttpOptions

    resolved_key = _resolve_api_key(api_key_override)

    if resolved_key not in _client_cache:
        _client_cache[resolved_key] = genai.Client(
            api_key=resolved_key,
            http_options=HttpOptions(timeout=300000)
        )

    return _client_cache[resolved_key]


def resolve_api_key(override: str = "") -> str:
    """Public wrapper kept for backward-compatibility with the API key node."""
    return _resolve_api_key(override)

"""
Error handling utilities for the ComfyUI Google Imagen node pack.

Maps SDK exceptions and HTTP errors to clear, actionable user messages.
"""


def handle_api_error(exc: Exception) -> str:
    """Convert a google-genai SDK exception into a user-friendly message.

    Args:
        exc: The original exception raised by the SDK.

    Returns:
        A formatted error message string.

    Raises:
        RuntimeError: Always — with the user-friendly message attached.
    """
    msg = str(exc).lower()
    exc_str = str(exc)

    # --- Authentication / key problems ---
    if "api key" in msg or "401" in msg or "unauthenticated" in msg:
        raise RuntimeError(
            "[Google Imagen] Invalid API key. "
            "Set GOOGLE_API_KEY environment variable, place key in "
            "google_api_key.txt, or provide via the API Key node."
        ) from exc

    # --- Permission / safety ---
    if "403" in msg or "permission" in msg or "forbidden" in msg:
        raise RuntimeError(
            "[Google Imagen] Access denied. Your API key may lack permission "
            "for this model, or content policy was violated."
        ) from exc

    # --- Rate limiting ---
    if "429" in msg or "rate" in msg or "quota" in msg or "resource_exhausted" in msg:
        raise RuntimeError(
            "[Google Imagen] Rate limit exceeded. "
            "Google Imagen allows ~20 images/minute. Wait and retry."
        ) from exc

    # --- Bad request / validation ---
    if "400" in msg or "invalid" in msg or "bad request" in msg:
        raise RuntimeError(
            f"[Google Imagen] Invalid request: {exc_str}. "
            f"Check your prompt and parameters."
        ) from exc

    # --- Server errors ---
    if "500" in msg or "503" in msg or "internal" in msg:
        raise RuntimeError(
            f"[Google Imagen] Google API server error. "
            f"Try again in a moment. Details: {exc_str}"
        ) from exc

    # --- Network / timeout ---
    if "timeout" in msg or "timed out" in msg:
        raise RuntimeError(
            "[Google Imagen] Request timed out. "
            "The API may be overloaded — try again."
        ) from exc

    if "connection" in msg or "network" in msg:
        raise RuntimeError(
            "[Google Imagen] Could not connect to the Google API. "
            "Check your internet connection."
        ) from exc

    # --- Catch-all ---
    raise RuntimeError(
        f"[Google Imagen] Unexpected error: {exc_str}"
    ) from exc

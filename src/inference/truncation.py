"""Email content truncation utilities."""

import structlog

logger = structlog.get_logger()

# Approximate characters per token (conservative estimate for multilingual)
CHARS_PER_TOKEN = 3


def estimate_tokens(text: str) -> int:
    """
    Estimate token count from text length.

    This is a rough approximation. Actual tokenization depends on the model.
    For multilingual content, we use a conservative estimate.
    """
    return len(text) // CHARS_PER_TOKEN


def truncate_content(
    content: str,
    max_tokens: int,
    reserve_tokens: int = 200,
) -> tuple[str, bool]:
    """
    Truncate content to fit within token limit.

    Args:
        content: Original email content
        max_tokens: Maximum input tokens for the model
        reserve_tokens: Tokens to reserve for prompt template

    Returns:
        Tuple of (truncated_content, was_truncated)
    """
    available_tokens = max_tokens - reserve_tokens
    max_chars = available_tokens * CHARS_PER_TOKEN

    if len(content) <= max_chars:
        return content, False

    # Truncate and add indicator
    truncated = content[:max_chars].rsplit(" ", 1)[0]  # Cut at word boundary
    truncated = truncated.rstrip() + "\n\n[... content truncated ...]"

    logger.info(
        "content_truncated",
        original_chars=len(content),
        truncated_chars=len(truncated),
        estimated_original_tokens=estimate_tokens(content),
        max_tokens=max_tokens,
    )

    return truncated, True

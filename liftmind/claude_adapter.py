"""
LiftMind Claude Adapter — replaces claude_cli.py.

Uses the Anthropic SDK directly (sync + async) instead of subprocess/CLI.
Provides the same interface that brain.py expects:
  ask_claude(user_query, context, lift_model, rag_results_count, model) -> str
  ask_claude_streaming(user_query, context, lift_model, rag_results_count, model) -> generator[str]
"""
import logging
import os
from pathlib import Path
from typing import Optional, Iterator

import anthropic

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt loader
# ---------------------------------------------------------------------------
_system_prompt_cache: Optional[str] = None

def _load_system_prompt() -> str:
    global _system_prompt_cache
    if _system_prompt_cache:
        return _system_prompt_cache

    # Try relative path from app root
    candidates = [
        Path(__file__).parent.parent / "prompts" / "soul.md",
        Path("/app/prompts/soul.md"),
    ]
    for path in candidates:
        if path.exists():
            _system_prompt_cache = path.read_text(encoding="utf-8")
            logger.info("Loaded soul.md from %s (%d chars)", path, len(_system_prompt_cache))
            return _system_prompt_cache

    # Fallback: minimal inline prompt
    logger.warning("soul.md not found — using minimal fallback system prompt")
    _system_prompt_cache = (
        "You are LiftMind — a senior lift technician assistant for Lift Shop Australia. "
        "Answer technical questions about lifts using only the provided documentation. "
        "Cite sources. Be concise. No preamble."
    )
    return _system_prompt_cache


# ---------------------------------------------------------------------------
# Anthropic client (lazy singleton)
# ---------------------------------------------------------------------------
_client: Optional[anthropic.Anthropic] = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


# ---------------------------------------------------------------------------
# Model name resolution
# ---------------------------------------------------------------------------
_MODEL_MAP = {
    "haiku":  "claude-haiku-4-5-20251001",
    "sonnet": "claude-sonnet-4-6",
    "opus":   "claude-opus-4-6",
}

def _resolve_model(model: Optional[str]) -> str:
    """Resolve 'haiku'/'sonnet' shortname or full model string."""
    if not model:
        return _MODEL_MAP["sonnet"]
    return _MODEL_MAP.get(model.lower(), model)


# ---------------------------------------------------------------------------
# Prompt builder (matches brain.py's expected format)
# ---------------------------------------------------------------------------
def _build_prompt(
    user_query: str,
    context: str = "",
    lift_model: Optional[str] = None,
    rag_results_count: int = 0,
) -> str:
    parts = []

    if lift_model:
        parts.append(f"[Lift Model: {lift_model}]")

    if context:
        parts.append(f"[Documentation Context — {rag_results_count} source(s)]\n{context}")
    else:
        parts.append("[Documentation Context]\nNo relevant documentation found for this query.")

    parts.append(f"[Question]\n{user_query}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public API — matches claude_cli.py interface expected by brain.py
# ---------------------------------------------------------------------------

def ask_claude(
    user_query: str,
    context: str = "",
    lift_model: Optional[str] = None,
    rag_results_count: int = 0,
    model: Optional[str] = None,
) -> str:
    """
    Synchronous Claude call. Returns response text.
    Matches: ask_claude(query, context=..., lift_model=..., rag_results_count=..., model=...)
    """
    model_id = _resolve_model(model)
    system = _load_system_prompt()
    prompt = _build_prompt(user_query, context, lift_model, rag_results_count)

    try:
        client = _get_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        return f"❌ I couldn't get a response right now. Error: {exc}"


def ask_claude_with_image(
    user_query: str,
    image_path: str,
    context: str = "",
    lift_model: Optional[str] = None,
    model: Optional[str] = None,
) -> str:
    """
    Claude call with an image attachment.
    Matches: ask_claude_with_image(query, image_path, context=..., lift_model=..., model=...)
    """
    import base64

    model_id = _resolve_model(model)
    system = _load_system_prompt()
    prompt_text = _build_prompt(user_query, context, lift_model)

    # Determine media type
    ext = Path(image_path).suffix.lower()
    media_type_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}
    media_type = media_type_map.get(ext, "image/jpeg")

    try:
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        client = _get_client()
        response = client.messages.create(
            model=model_id,
            max_tokens=2000,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": image_data}},
                    {"type": "text", "text": prompt_text},
                ],
            }],
        )
        return response.content[0].text
    except Exception as exc:
        logger.error("Claude image API error: %s", exc)
        return f"❌ Image analysis failed. Error: {exc}"


def ask_claude_streaming(
    user_query: str,
    context: str = "",
    lift_model: Optional[str] = None,
    rag_results_count: int = 0,
    model: Optional[str] = None,
    on_chunk=None,
) -> Iterator[str]:
    """
    Streaming Claude call. Yields text chunks.
    Matches: ask_claude_streaming(query, context=..., lift_model=..., rag_results_count=..., model=..., on_chunk=...)
    """
    model_id = _resolve_model(model)
    system = _load_system_prompt()
    prompt = _build_prompt(user_query, context, lift_model, rag_results_count)

    try:
        client = _get_client()
        with client.messages.stream(
            model=model_id,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                if on_chunk:
                    on_chunk(text)
                yield text
    except Exception as exc:
        logger.error("Claude streaming error: %s", exc)
        error_msg = f"❌ Streaming failed. Error: {exc}"
        if on_chunk:
            on_chunk(error_msg)
        yield error_msg

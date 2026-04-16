"""
HyDE (Hypothetical Document Embeddings) for Vague Queries.

When users describe vague symptoms like "weird clicking sound", traditional
search often fails because the query terms don't match manual vocabulary.

HyDE solves this by:
1. Generating a hypothetical technical answer using fast LLM
2. Embedding that hypothetical answer (which uses technical vocabulary)
3. Searching for similar actual content

Example:
  Query: "making a weird clicking sound"
  Hypothetical: "A rhythmic clicking sound typically indicates relay chatter,
                 hydraulic pump valve flutter, or worn gear teeth..."
  → Now vector search finds relevant troubleshooting content
"""
import logging
import re
from typing import Optional, List

try:
    import anthropic
    ANTHROPIC_SDK_AVAILABLE = True
except ImportError:
    ANTHROPIC_SDK_AVAILABLE = False

from liftmind.config import settings

logger = logging.getLogger(__name__)

# HyDE activation rate tracking
_total_queries = 0
_hyde_activated = 0

# Vague symptom patterns that trigger HyDE
VAGUE_SYMPTOM_PATTERNS = [
    r'\bweird\b',
    r'\bstrange\b',
    r'\bodd\b',
    r'\bunusual\b',
    r'\bfunny\b',
    r'\bnoise\b',
    r'\bsound\b',
    r'\bclicking\b',
    r'\bbuzzing\b',
    r'\bhumming\b',
    r'\bvibrat',
    r'\bshak',
    r'\bjerk',
    r'\bwon\'?t\s+(work|move|start|stop|open|close)',
    r'\bdoesn\'?t\s+(work|move|start|stop|open|close)',
    r'\bnot\s+(working|moving|starting|stopping|opening|closing)',
    r'\bsometimes\b',
    r'\bintermittent',
    r'\brandom',
    r'\boccasional',
]

# Compile patterns for efficiency
_VAGUE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in VAGUE_SYMPTOM_PATTERNS]


def is_vague_symptom_query(query: str) -> bool:
    """
    Detect if query describes a vague symptom that might benefit from HyDE.

    Args:
        query: User's search query

    Returns:
        True if query appears to be a vague symptom description
    """
    for pattern in _VAGUE_PATTERNS:
        if pattern.search(query):
            return True
    return False


def should_use_hyde(query: str) -> bool:
    """
    Determine whether to use HyDE for a query and track activation rate.

    This is the preferred entry point for HyDE decisions as it tracks
    activation metrics. Requires HYDE_ACTIVATION_THRESHOLD pattern matches
    (default: 2) to activate, preventing overuse on simple queries.

    Args:
        query: User's search query

    Returns:
        True if HyDE should be activated for this query
    """
    global _total_queries, _hyde_activated
    _total_queries += 1

    # Check if HyDE is enabled
    if not settings.HYDE_ENABLED:
        return False

    # Count matching patterns
    match_count = sum(1 for pattern in _VAGUE_PATTERNS if pattern.search(query))

    # Require minimum pattern matches to activate (prevents overuse)
    threshold = settings.HYDE_ACTIVATION_THRESHOLD
    result = match_count >= threshold

    if result:
        _hyde_activated += 1
        logger.debug(f"HyDE activated: {match_count} pattern matches (threshold: {threshold})")

    # Log rate periodically (every 100 queries)
    if _total_queries % 100 == 0:
        rate = (_hyde_activated / _total_queries) * 100
        logger.info(f"HyDE activation rate: {rate:.1f}% ({_hyde_activated}/{_total_queries})")

    return result


def get_hyde_stats() -> dict:
    """Get current HyDE activation statistics."""
    rate = (_hyde_activated / _total_queries * 100) if _total_queries > 0 else 0.0
    return {
        "total_queries": _total_queries,
        "hyde_activated": _hyde_activated,
        "activation_rate_percent": round(rate, 1)
    }


async def generate_hypothetical_answer(query: str, model: str = None) -> Optional[str]:
    """
    Generate a hypothetical technical answer for a vague query.

    Uses Claude Haiku for speed. The generated answer uses technical vocabulary
    that will match actual manual content when embedded.

    Args:
        query: User's vague symptom description
        model: Lift model context (optional)

    Returns:
        Hypothetical technical answer, or None if generation fails
    """
    try:
        if not ANTHROPIC_SDK_AVAILABLE:
            logger.warning("Anthropic SDK not available, HyDE unavailable")
            return None

        if not settings.ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set, HyDE unavailable")
            return None

        client = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)

        model_context = f" for a {model} lift" if model else ""

        prompt = f"""You are a lift technician. Based on this symptom description{model_context}, write a short (2-3 sentence) technical explanation as if from a manual. Use specific technical terms.

Symptom: {query}

Technical explanation:"""

        response = await client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}]
        )

        hypothetical = response.content[0].text.strip()
        logger.info(f"HyDE generated: {hypothetical[:100]}...")
        return hypothetical

    except Exception as e:
        logger.error(f"HyDE generation failed: {e}")
        return None


def generate_hypothetical_answer_sync(query: str, model: str = None) -> Optional[str]:
    """Synchronous wrapper for generate_hypothetical_answer."""
    import asyncio

    try:
        return asyncio.run(generate_hypothetical_answer(query, model))
    except Exception as e:
        logger.warning(f"HyDE sync wrapper failed: {e}")
        return None


def hyde_search(query: str, model: str = None, search_func=None) -> Optional[List[dict]]:
    """
    Perform HyDE-enhanced search for vague symptom queries.

    Args:
        query: User's vague symptom description
        model: Lift model context
        search_func: Function to call for vector search (passed to avoid circular import)

    Returns:
        Search results using hypothetical answer embedding, or None to fall back
    """
    if not is_vague_symptom_query(query):
        return None

    logger.info(f"HyDE activated for vague query: {query[:50]}...")

    hypothetical = generate_hypothetical_answer_sync(query, model)

    if not hypothetical:
        return None

    # Use the hypothetical answer for search
    if search_func:
        return search_func(hypothetical, model)

    return None

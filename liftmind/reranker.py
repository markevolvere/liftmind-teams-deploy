"""
Cross-Encoder Reranker for LiftMind.

Uses a neural cross-encoder model to rerank search results for quality.
Cross-encoders are more accurate than bi-encoders (vector similarity) because
they see query and document together, enabling deeper semantic comparison.

Model: BAAI/bge-reranker-v2-m3 (or falls back to ms-marco-MiniLM if unavailable)
"""
import logging
import os
import time
from typing import List, Optional

logger = logging.getLogger(__name__)

# Global reranker instance (lazy loaded)
_reranker = None
_reranker_available = None


def _load_reranker():
    """Load the cross-encoder reranker model."""
    global _reranker, _reranker_available

    if _reranker_available is not None:
        return _reranker

    # Check if reranker is disabled
    from liftmind.config import settings
    if not settings.USE_RERANKER:
        logger.info("Reranker disabled via USE_RERANKER=false")
        _reranker_available = False
        return None

    # Force CPU mode to avoid GPU compatibility issues
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

    try:
        from sentence_transformers import CrossEncoder

        # Try BGE reranker first (best quality) - force CPU
        try:
            logger.info("Loading reranker: BAAI/bge-reranker-v2-m3 (CPU mode)")
            _reranker = CrossEncoder('BAAI/bge-reranker-v2-m3', max_length=512, device='cpu')
            _reranker_available = True
            logger.info("BGE reranker loaded on CPU")
            return _reranker
        except Exception as e:
            logger.warning(f"BGE reranker failed: {e}, trying fallback...")

        # Fallback to lighter model - force CPU
        try:
            logger.info("Loading fallback reranker: cross-encoder/ms-marco-MiniLM-L-6-v2 (CPU mode)")
            _reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512, device='cpu')
            _reranker_available = True
            logger.info("Fallback reranker loaded on CPU")
            return _reranker
        except Exception as e:
            logger.error(f"Fallback reranker also failed: {e}")
            _reranker_available = False
            return None

    except ImportError:
        logger.warning("sentence-transformers not installed, reranking disabled")
        _reranker_available = False
        return None


def rerank_results(query: str, results: List[dict], top_k: int = 5) -> List[dict]:
    """
    Neural reranking using cross-encoder.

    Takes query + each chunk text, returns relevance score.
    Far more accurate than vector similarity alone.

    Args:
        query: The user's search query
        results: List of search result dicts with 'content' field
        top_k: Number of top results to return

    Returns:
        Reranked results sorted by relevance score (highest first)
    """
    from liftmind.config import settings

    if not results:
        return []

    # Cap candidates to prevent latency explosion (configurable via settings)
    max_candidates = settings.MAX_RERANK_CANDIDATES
    if len(results) > max_candidates:
        logger.debug(f"Capping rerank candidates from {len(results)} to {max_candidates}")
        results = results[:max_candidates]

    reranker = _load_reranker()

    if reranker is None:
        # Reranking unavailable, return original order
        logger.debug("Reranker unavailable, returning original order")
        return results[:top_k]

    try:
        # Prepare query-document pairs
        pairs = [(query, r.get('content', '')) for r in results]

        # Score all pairs with latency tracking
        start_time = time.time()
        scores = reranker.predict(pairs)
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info(f"Neural reranking {len(pairs)} candidates took {elapsed_ms:.0f}ms")

        # Use configurable latency warning threshold
        if elapsed_ms > settings.RERANK_LATENCY_WARNING_MS:
            logger.warning(f"Reranking exceeded {settings.RERANK_LATENCY_WARNING_MS}ms target: {elapsed_ms:.0f}ms")

        # Attach scores and sort
        scored_results = []
        for i, r in enumerate(results):
            result_copy = r.copy()
            result_copy['rerank_score'] = float(scores[i])
            scored_results.append(result_copy)

        # Sort by rerank score (descending)
        scored_results.sort(key=lambda x: x.get('rerank_score', 0), reverse=True)

        logger.debug(f"Reranked {len(results)} results, top score: {scored_results[0].get('rerank_score', 0):.3f}")

        return scored_results[:top_k]

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        return results[:top_k]


def is_reranker_available() -> bool:
    """Check if reranker is available."""
    _load_reranker()
    return _reranker_available or False


def preload_reranker():
    """Preload the reranker model at startup."""
    logger.info("Preloading reranker model...")
    reranker = _load_reranker()
    if reranker:
        # Warm up with dummy query
        try:
            _ = reranker.predict([("warmup query", "warmup document")])
            logger.info("Reranker preloaded successfully")
        except Exception as e:
            logger.warning(f"Reranker warmup failed: {e}")
    else:
        logger.warning("Reranker not available")

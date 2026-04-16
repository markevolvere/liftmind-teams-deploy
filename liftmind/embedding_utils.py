"""
Embedding Utilities - Shared embedding generation for hybrid search.

Uses all-MiniLM-L6-v2 model (384 dimensions) for semantic embeddings.
Model is loaded once at module level to avoid repeated loading.
"""
import logging
import threading
from functools import lru_cache
from typing import List, Optional

logger = logging.getLogger(__name__)

# Global model instance (lazy loaded) with thread-safe lock
_model = None
_model_lock = threading.Lock()


def _get_model():
    """Get or load the sentence transformer model (thread-safe)."""
    global _model

    # Fast path: model already loaded
    if _model is not None:
        return _model

    with _model_lock:
        # Double-check after acquiring lock (another thread may have loaded it)
        if _model is not None:
            return _model

        # Check if embeddings are disabled
        from liftmind.config import settings
        if not settings.USE_EMBEDDINGS:
            logger.info("Embeddings disabled via USE_EMBEDDINGS=false")
            return None

        try:
            from sentence_transformers import SentenceTransformer
            logger.info("Loading embedding model: all-MiniLM-L6-v2 (CPU mode)")
            # Force CPU to avoid CUDA compatibility issues
            _model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            logger.info(f"Embedding model loaded on CPU. Dimension: {_model.get_sentence_embedding_dimension()}")
            return _model
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            return None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return None


@lru_cache(maxsize=200)
def generate_embedding(text: str) -> Optional[List[float]]:
    """
    Generate embedding vector for text.

    Args:
        text: Input text to embed

    Returns:
        List of 384 floats, or None if model unavailable
    """
    model = _get_model()
    if model is None:
        return None

    try:
        # Truncate very long texts with debug warning
        MAX_TEXT_LENGTH = 8000
        if len(text) > MAX_TEXT_LENGTH:
            logger.debug(f"Truncating text from {len(text)} to {MAX_TEXT_LENGTH} chars for embedding")
            text = text[:MAX_TEXT_LENGTH]

        embedding = model.encode(text, show_progress_bar=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def generate_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Generate embeddings for multiple texts efficiently.

    Args:
        texts: List of input texts

    Returns:
        List of embedding vectors (or None for failed items)
    """
    model = _get_model()
    if model is None:
        return [None] * len(texts)

    try:
        # Truncate long texts with debug warning
        MAX_TEXT_LENGTH = 8000
        truncated = []
        for t in texts:
            if len(t) > MAX_TEXT_LENGTH:
                logger.debug(f"Truncating text from {len(t)} to {MAX_TEXT_LENGTH} chars for embedding")
                truncated.append(t[:MAX_TEXT_LENGTH])
            else:
                truncated.append(t)

        embeddings = model.encode(truncated, show_progress_bar=False)
        return [e.tolist() for e in embeddings]
    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        return [None] * len(texts)


def embedding_dimension() -> int:
    """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
    return 384


def preload_model():
    """Preload the model at startup to avoid cold start latency."""
    logger.info("Preloading embedding model...")
    model = _get_model()
    if model:
        # Generate a dummy embedding to fully initialize
        _ = generate_embedding("warmup")
        logger.info("Embedding model preloaded successfully")
    else:
        logger.warning("Failed to preload embedding model")


def is_model_loaded() -> bool:
    """Check if the embedding model is loaded and available."""
    return _model is not None

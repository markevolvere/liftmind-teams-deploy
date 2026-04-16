"""
Context Cache for Deep Dive Mode.

Loads full manual context for comprehensive queries like
"I'm commissioning a Q1, need the full picture".

CRITICAL: NO RAM CACHING! Full manuals are huge.
Caching 10 PDF text blobs in Python RAM will OOM-kill the server.
Postgres is fast enough - the 50ms DB read is negligible vs 5s LLM generation.
"""
import logging
from typing import Optional

from liftmind.rag import get_db_connection

logger = logging.getLogger(__name__)

# Maximum characters for context (Claude's ~200k token limit ≈ 800k chars)
MAX_CONTEXT_CHARS = 400000  # ~100k tokens, leave room for response


def load_manual_context(model: str) -> Optional[str]:
    """
    Load entire manual content for a lift model from database.

    NO RAM CACHING - fetch from Postgres on demand.
    The 50ms DB read is negligible vs 5s LLM generation.

    Args:
        model: Lift model name (e.g., "Pollock (Q1)")

    Returns:
        Formatted manual content string, or None if too large/not found
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Get all chunks for this model, ordered by document and position
        cur.execute("""
            SELECT c.content, d.filename, c.page_start, c.section_path
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE c.lift_models && %s
            ORDER BY d.filename, c.chunk_index
        """, ([model],))

        chunks = cur.fetchall()

        if not chunks:
            # Try facts table as fallback
            cur.execute("""
                SELECT f.content, d.filename, f.page, f.section_path
                FROM facts f
                JOIN documents d ON f.document_id = d.id
                WHERE f.lift_models && %s
                ORDER BY d.filename, f.page
            """, ([model],))
            chunks = cur.fetchall()

        if not chunks:
            logger.warning(f"No content found for model: {model}")
            return None

        # Format as structured context
        context_parts = []
        current_file = None
        total_chars = 0

        for content, filename, page, section in chunks:
            # Track document changes
            if filename != current_file:
                header = f"\n\n{'='*60}\n=== {filename} ===\n{'='*60}\n"
                context_parts.append(header)
                current_file = filename
                total_chars += len(header)

            # Format chunk with source info
            section_info = f" | {section}" if section else ""
            chunk_text = f"\n[Page {page or '?'}{section_info}]\n{content}\n"

            # Check size limit
            if total_chars + len(chunk_text) > MAX_CONTEXT_CHARS:
                context_parts.append("\n\n[... content truncated due to size limit ...]")
                logger.warning(f"Manual context for {model} truncated at {total_chars} chars")
                break

            context_parts.append(chunk_text)
            total_chars += len(chunk_text)

        full_context = "".join(context_parts)
        logger.info(f"Loaded {len(chunks)} chunks for {model} ({total_chars} chars)")

        return full_context

    except Exception as e:
        logger.error(f"Failed to load manual context for {model}: {e}")
        return None

    finally:
        cur.close()
        conn.close()


def deep_dive_query(query: str, model: str, ask_claude_func) -> Optional[str]:
    """
    Answer with full manual context - like NotebookLM.

    Args:
        query: User's question
        model: Lift model for context
        ask_claude_func: Function to call Claude (passed to avoid circular import)

    Returns:
        Claude's response with full manual context, or None to fall back to RAG
    """
    logger.info(f"Deep dive mode for {model}: {query[:50]}...")

    full_context = load_manual_context(model)

    if not full_context:
        logger.warning(f"Deep dive failed - no context for {model}")
        return None

    # Check if context is too short (might indicate indexing issue)
    if len(full_context) < 1000:
        logger.warning(f"Deep dive context too short ({len(full_context)} chars), falling back to RAG")
        return None

    # Add deep dive header to context
    context_with_header = f"""[DEEP DIVE MODE - Full {model} Manual Context]
You have access to the complete {model} documentation below.
Answer comprehensively using this full context.

{full_context}"""

    try:
        response = ask_claude_func(
            user_query=query,
            context=context_with_header,
            lift_model=model,
            rag_results_count=999  # Indicates full context mode
        )
        return response

    except Exception as e:
        logger.error(f"Deep dive Claude call failed: {e}")
        return None


def get_manual_stats(model: str) -> dict:
    """Get statistics about available manual content for a model."""
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        cur.execute("""
            SELECT COUNT(*), SUM(LENGTH(content))
            FROM chunks WHERE lift_models && %s
        """, ([model],))
        chunk_count, chunk_chars = cur.fetchone()

        cur.execute("""
            SELECT COUNT(*), SUM(LENGTH(content))
            FROM facts WHERE lift_models && %s
        """, ([model],))
        fact_count, fact_chars = cur.fetchone()

        cur.execute("""
            SELECT COUNT(DISTINCT filename)
            FROM documents d
            JOIN chunks c ON c.document_id = d.id
            WHERE c.lift_models && %s
        """, ([model],))
        doc_count = cur.fetchone()[0]

        return {
            "model": model,
            "documents": doc_count or 0,
            "chunks": chunk_count or 0,
            "facts": fact_count or 0,
            "total_chars": (chunk_chars or 0) + (fact_chars or 0),
            "can_deep_dive": (chunk_chars or 0) + (fact_chars or 0) < MAX_CONTEXT_CHARS
        }

    finally:
        cur.close()
        conn.close()

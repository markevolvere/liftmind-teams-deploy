"""
LiftMind Self-Learning Module

When direct PDF reading finds an answer the database missed, learn from it:
1. Extract the relevant page text
2. Chunk it, generate embeddings, insert into chunks + facts tables
3. Tag with learned_at metadata
4. Deduplicate via cosine similarity > 0.95

Result: Next identical/similar question hits the database directly.
"""

import logging
import os
from datetime import datetime
from typing import List, Dict, Optional

import psycopg2

from liftmind.config import settings
from liftmind.embedding_utils import generate_embedding, generate_embeddings_batch
from liftmind.knowledge import smart_chunk_text, detect_chunk_type, detect_lift_models, get_db

logger = logging.getLogger(__name__)

# Cosine similarity threshold for deduplication
DEDUP_SIMILARITY_THRESHOLD = 0.95


def _find_or_create_document(cur, filename: str, lift_model: Optional[str]) -> int:
    """Find existing document by filename or create a placeholder for learned content."""
    cur.execute(
        "SELECT id FROM documents WHERE filename = %s LIMIT 1",
        (filename,)
    )
    row = cur.fetchone()
    if row:
        return row[0]

    # Create a placeholder document record
    lift_models = [lift_model] if lift_model else []
    cur.execute("""
        INSERT INTO documents (filename, file_path, file_type, doc_type, lift_models,
                               index_status, created_at, updated_at)
        VALUES (%s, %s, 'pdf', 'manual', %s, 'complete', NOW(), NOW())
        RETURNING id
    """, (filename, f"learned://{filename}", lift_models))
    return cur.fetchone()[0]


def _is_duplicate_chunk(cur, embedding: List[float], threshold: float = DEDUP_SIMILARITY_THRESHOLD) -> bool:
    """Check if a chunk with very similar embedding already exists."""
    if not embedding:
        return False

    embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

    try:
        cur.execute("""
            SELECT 1 FROM chunks
            WHERE embedding IS NOT NULL
            AND 1 - (embedding <=> %s::vector) > %s
            LIMIT 1
        """, (embedding_str, threshold))
        return cur.fetchone() is not None
    except Exception as e:
        logger.debug(f"Dedup check failed (may be missing vector index): {e}")
        return False


def learn_from_direct_read(
    query: str,
    direct_results: List[Dict],
    lift_model: Optional[str] = None
) -> Dict:
    """
    Learn from content found by direct PDF reading that was missing from the DB.

    Takes the pages that the direct PDF reader found, chunks them, generates
    embeddings, and inserts into the database so future queries hit the DB directly.

    Args:
        query: The original user query that triggered direct PDF reading
        direct_results: List of result dicts from manual_reader with 'content',
                       'filename', 'page_number', etc.
        lift_model: The lift model context (e.g., "Tresa")

    Returns:
        Dict with counts of learned chunks and facts
    """
    stats = {"chunks_added": 0, "chunks_skipped": 0, "errors": []}

    if not direct_results:
        return stats

    conn = None
    try:
        conn = get_db()
        cur = conn.cursor()

        for result in direct_results:
            content = result.get("content", "").strip()
            filename = result.get("filename", "unknown")
            page_number = result.get("page_number")

            if not content or len(content) < 50:
                continue

            try:
                # Find or create the document record
                doc_id = _find_or_create_document(cur, filename, lift_model)

                # Chunk the page content
                page_chunks = smart_chunk_text(content, max_tokens=400, overlap_ratio=0.3)

                if not page_chunks:
                    continue

                # Generate embeddings for all chunks in batch
                chunk_texts = [c["content"] for c in page_chunks]
                embeddings = generate_embeddings_batch(chunk_texts)

                for i, chunk in enumerate(page_chunks):
                    embedding = embeddings[i]

                    # Deduplication: skip if very similar chunk exists
                    if embedding and _is_duplicate_chunk(cur, embedding):
                        stats["chunks_skipped"] += 1
                        continue

                    chunk_type = detect_chunk_type(chunk["content"])
                    lift_models = [lift_model] if lift_model else detect_lift_models(chunk["content"], filename)

                    embedding_str = None
                    if embedding:
                        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

                    # Insert chunk with learned_at metadata
                    cur.execute("""
                        INSERT INTO chunks (
                            document_id, chunk_index, content, chunk_type,
                            page_start, page_end, token_count, lift_models,
                            embedding, learned_at, learned_from_query
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector, NOW(), %s)
                    """, (
                        doc_id,
                        i,
                        chunk["content"],
                        chunk_type,
                        page_number,
                        page_number,
                        chunk.get("tokens", int(len(chunk["content"].split()) * 1.3)),
                        lift_models,
                        embedding_str,
                        query
                    ))

                    stats["chunks_added"] += 1

            except Exception as e:
                logger.error(f"Error learning from {filename} p.{page_number}: {e}")
                stats["errors"].append(f"{filename}:{page_number}: {e}")
                conn.rollback()
                cur = conn.cursor()
                continue

        conn.commit()
        cur.close()

        if stats["chunks_added"] > 0:
            logger.info(
                f"Self-learning: added {stats['chunks_added']} chunks, "
                f"skipped {stats['chunks_skipped']} duplicates "
                f"(query: {query[:60]}...)"
            )

    except Exception as e:
        logger.error(f"Self-learning failed: {e}")
        stats["errors"].append(str(e))
        if conn:
            conn.rollback()
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass

    return stats

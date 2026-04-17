"""
User State Management for LiftMind.

Persists user preferences (selected model) across conversations.
"""
import logging
from typing import Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Lazy schema init -- runs once on first successful DB op.
# Never blocks app startup; never retries forever if DB is down.
_SCHEMA_READY = False


def _get_connection():
    """Get database connection."""
    from liftmind.rag import get_db_connection
    return get_db_connection()


def init_user_state_schema():
    """Initialize user_state table if it doesn't exist. Safe to call many times."""
    global _SCHEMA_READY
    if _SCHEMA_READY:
        return
    try:
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_state (
                user_id TEXT PRIMARY KEY,
                current_model TEXT,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                query_count INTEGER DEFAULT 0,
                pending_question TEXT
            )
        """)
        cur.execute("""
            ALTER TABLE user_state
            ADD COLUMN IF NOT EXISTS pending_question TEXT
        """)
        conn.commit()
        cur.close()
        conn.close()
        _SCHEMA_READY = True
        logger.info("User state schema initialized (lazy)")
    except Exception as e:
        logger.error(f"Lazy schema init failed (will retry on next call): {e}")


def _ensure_schema():
    """Ensure schema exists before any DB op. Cheap no-op after first success."""
    if not _SCHEMA_READY:
        init_user_state_schema()


def get_user_model(user_id: str) -> Optional[str]:
    """
    Get the user's currently selected lift model.

    Args:
        user_id: User identifier (Telegram ID or external system ID)

    Returns:
        The model name or None if not set
    """
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT current_model FROM user_state WHERE user_id = %s
        """, (user_id,))

        row = cur.fetchone()
        cur.close()
        conn.close()

        return row[0] if row else None
    except Exception as e:
        logger.error(f"Error getting user model for {user_id}: {e}")
        return None


def get_user_model_fresh(user_id: str, ttl_minutes: int = 30) -> Optional[str]:
    """
    Get the user's currently selected model, but only if last_active is within TTL.

    Uses Postgres-side timestamp arithmetic to avoid any Python-side naive-vs-aware
    datetime mismatches that can silently blow up the comparison.

    Returns None if the model selection has expired (stale session) OR if the user
    has no current_model set.
    """
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()
        # Use Postgres interval arithmetic -- no Python datetime comparison.
        cur.execute(
            """
            SELECT current_model
            FROM user_state
            WHERE user_id = %s
              AND current_model IS NOT NULL
              AND last_active > (CURRENT_TIMESTAMP - (%s || ' minutes')::interval)
            """,
            (user_id, str(ttl_minutes)),
        )
        row = cur.fetchone()
        cur.close()
        conn.close()

        return row[0] if row else None
    except Exception as e:
        logger.error(f"Error getting fresh user model for {user_id}: {e}")
        return None


def clear_user_model(user_id: str) -> None:
    """Clear the user's selected model (forces picker to re-show)."""
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_state (user_id, current_model, last_active)
            VALUES (%s, NULL, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                current_model = NULL,
                last_active = CURRENT_TIMESTAMP
            """,
            (user_id,),
        )
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Cleared model for user {user_id}")
    except Exception as e:
        logger.error(f"Error clearing model for {user_id}: {e}")


def set_pending_question(user_id: str, question: str) -> None:
    """Stash a question to be answered after the user picks a model."""
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO user_state (user_id, pending_question, last_active)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                pending_question = EXCLUDED.pending_question,
                last_active = CURRENT_TIMESTAMP
            """,
            (user_id, question),
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error setting pending question for {user_id}: {e}")


def pop_pending_question(user_id: str) -> Optional[str]:
    """Return & clear any pending question for this user."""
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT pending_question FROM user_state WHERE user_id = %s",
            (user_id,),
        )
        row = cur.fetchone()
        question = row[0] if row and row[0] else None
        if question:
            cur.execute(
                "UPDATE user_state SET pending_question = NULL WHERE user_id = %s",
                (user_id,),
            )
            conn.commit()
        cur.close()
        conn.close()
        return question
    except Exception as e:
        logger.error(f"Error popping pending question for {user_id}: {e}")
        return None


def set_user_model(user_id: str, model: str) -> None:
    """
    Set the user's current lift model.

    Args:
        user_id: User identifier
        model: The lift model name
    """
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO user_state (user_id, current_model, last_active)
            VALUES (%s, %s, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                current_model = EXCLUDED.current_model,
                last_active = CURRENT_TIMESTAMP
        """, (user_id, model))

        conn.commit()
        cur.close()
        conn.close()

        logger.info(f"Set model '{model}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error setting user model for {user_id}: {e}")


def increment_query_count(user_id: str) -> None:
    """
    Increment the user's query count.

    Args:
        user_id: User identifier
    """
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO user_state (user_id, query_count, last_active)
            VALUES (%s, 1, CURRENT_TIMESTAMP)
            ON CONFLICT (user_id) DO UPDATE SET
                query_count = user_state.query_count + 1,
                last_active = CURRENT_TIMESTAMP
        """, (user_id,))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Error incrementing query count for {user_id}: {e}")


def get_user_stats(user_id: str) -> dict:
    """
    Get user statistics.

    Returns:
        {"current_model": str|None, "query_count": int, "last_active": datetime|None}
    """
    try:
        _ensure_schema()
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("""
            SELECT current_model, query_count, last_active
            FROM user_state WHERE user_id = %s
        """, (user_id,))

        row = cur.fetchone()
        cur.close()
        conn.close()

        if row:
            return {
                "current_model": row[0],
                "query_count": row[1] or 0,
                "last_active": row[2]
            }
        return {"current_model": None, "query_count": 0, "last_active": None}
    except Exception as e:
        logger.error(f"Error getting user stats for {user_id}: {e}")
        return {"current_model": None, "query_count": 0, "last_active": None}

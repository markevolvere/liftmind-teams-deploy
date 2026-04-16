"""
Chat logging for LiftMind — Teams-adapted.

Changed: telegram_user_id (int) → user_id (str) throughout, to support
Teams AAD Object IDs and other string identifiers.
"""
import json
import logging
from typing import Optional
from datetime import datetime
from liftmind.database import get_connection
from liftmind.config import settings

logger = logging.getLogger(__name__)


def log_message(
    user_id: str,
    message_type: str,
    user_message: str = None,
    bot_response: str = None,
    lift_model: str = None,
    rag_sources: list = None,
    response_time_ms: int = None,
) -> Optional[int]:
    """
    Log a chat message.

    Args:
        user_id: Teams AAD Object ID (or any unique string user identifier)
        message_type: 'text', 'command', 'model_select', etc.
        user_message: The user's message
        bot_response: The bot's response text
        lift_model: Current lift model context
        rag_sources: List of RAG source references
        response_time_ms: Response time in ms

    Returns:
        Log entry ID or None if logging fails
    """
    rag_json = json.dumps(rag_sources) if rag_sources else None

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO chat_logs
                        (user_id, message_type, user_message, bot_response,
                         lift_model, rag_sources, response_time_ms)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (user_id, message_type, user_message, bot_response,
                      lift_model, rag_json, response_time_ms))
                log_id = cur.fetchone()[0]

        logger.debug("Logged message: user=%s type=%s id=%s", user_id, message_type, log_id)
        return log_id
    except Exception as exc:
        logger.warning("Failed to log message for user=%s: %s", user_id, exc)
        return None


def get_logs(
    user_id: str = None,
    message_type: str = None,
    lift_model: str = None,
    search_query: str = None,
    start_date: datetime = None,
    end_date: datetime = None,
    page: int = 1,
    limit: int = 50,
) -> dict:
    """Get chat logs with filtering and pagination."""
    offset = (page - 1) * limit
    conditions = []
    params = []

    if user_id is not None:
        conditions.append("user_id = %s")
        params.append(user_id)
    if message_type:
        conditions.append("message_type = %s")
        params.append(message_type)
    if lift_model:
        conditions.append("lift_model = %s")
        params.append(lift_model)
    if start_date:
        conditions.append("created_at >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("created_at <= %s")
        params.append(end_date)
    if search_query:
        conditions.append(
            "to_tsvector('english', COALESCE(user_message, '')) @@ plainto_tsquery('english', %s)"
        )
        params.append(search_query)

    where_clause = " AND ".join(conditions) if conditions else "TRUE"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM chat_logs WHERE {where_clause}", params)
            total = cur.fetchone()[0]

            cur.execute(f"""
                SELECT id, user_id, message_type, user_message, bot_response,
                       lift_model, rag_sources, response_time_ms, created_at
                FROM chat_logs
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, params + [limit, offset])
            rows = cur.fetchall()

    logs = [_row_to_dict(row) for row in rows]
    pages = (total + limit - 1) // limit if limit > 0 else 1

    return {"logs": logs, "total": total, "page": page, "limit": limit, "pages": pages}


def get_user_logs(user_id: str, page: int = 1, limit: int = 50) -> dict:
    """Get logs for a specific user."""
    return get_logs(user_id=user_id, page=page, limit=limit)


def get_stats() -> dict:
    """Get overview statistics."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chat_logs")
            total_messages = cur.fetchone()[0]

            cur.execute("SELECT COUNT(*) FROM chat_logs WHERE created_at >= CURRENT_DATE")
            messages_today = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM chat_logs WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'"
            )
            messages_week = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(DISTINCT user_id) FROM chat_logs "
                "WHERE created_at >= CURRENT_DATE - INTERVAL '7 days'"
            )
            active_users_week = cur.fetchone()[0]

            cur.execute("SELECT message_type, COUNT(*) FROM chat_logs GROUP BY message_type")
            message_types = {row[0]: row[1] for row in cur.fetchall()}

            cur.execute(
                "SELECT AVG(response_time_ms) FROM chat_logs WHERE response_time_ms IS NOT NULL"
            )
            avg_response_time = cur.fetchone()[0]

            cur.execute(
                "SELECT lift_model, COUNT(*) as cnt FROM chat_logs "
                "WHERE lift_model IS NOT NULL GROUP BY lift_model ORDER BY cnt DESC LIMIT 5"
            )
            top_models = {row[0]: row[1] for row in cur.fetchall()}

            cur.execute(
                "SELECT DATE(created_at) as day, COUNT(*) FROM chat_logs "
                "WHERE created_at >= CURRENT_DATE - INTERVAL '14 days' "
                "GROUP BY DATE(created_at) ORDER BY day"
            )
            messages_by_day = {row[0].isoformat(): row[1] for row in cur.fetchall()}

    return {
        "total_messages": total_messages,
        "messages_today": messages_today,
        "messages_week": messages_week,
        "active_users_week": active_users_week,
        "message_types": message_types,
        "avg_response_time_ms": round(avg_response_time) if avg_response_time else None,
        "top_models": top_models,
        "messages_by_day": messages_by_day,
    }


def _row_to_dict(row: tuple) -> dict:
    rag_sources = row[6]
    if isinstance(rag_sources, str):
        rag_sources = json.loads(rag_sources)
    return {
        "id": row[0],
        "user_id": row[1],
        "message_type": row[2],
        "user_message": row[3],
        "bot_response": row[4],
        "lift_model": row[5],
        "rag_sources": rag_sources,
        "response_time_ms": row[7],
        "created_at": row[8].isoformat() if row[8] else None,
    }

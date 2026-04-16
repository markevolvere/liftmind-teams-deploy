"""
Analytics and Query Logging for LiftMind.

Tracks all queries for analysis and quality monitoring.
"""
import logging
from typing import Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def _get_connection():
    """Get database connection."""
    from liftmind.rag import get_db_connection
    return get_db_connection()


def init_analytics_schema():
    """Initialize query_log table if it doesn't exist."""
    conn = _get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS query_log (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            query TEXT NOT NULL,
            query_type TEXT,
            lift_model TEXT,
            model_auto_detected BOOLEAN DEFAULT FALSE,
            rag_results_count INTEGER,
            response_time_ms INTEGER,
            detected_fault_code TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Index for common queries
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_created
        ON query_log (created_at)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_query_log_user
        ON query_log (user_id)
    """)

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Analytics schema initialized")


def log_query(
    user_id: str,
    original_query: str,
    result: dict
) -> None:
    """
    Log a query for analytics.

    Fire-and-forget - failures are logged but don't affect the response.

    Args:
        user_id: User identifier
        original_query: The user's original query
        result: The result dict from process_query()
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()

        cur.execute("""
            INSERT INTO query_log (
                user_id, query, query_type, lift_model,
                model_auto_detected, rag_results_count,
                response_time_ms, detected_fault_code
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            user_id,
            original_query[:500],  # Truncate long queries
            result.get("query_type"),
            result.get("model_used"),
            result.get("model_auto_detected", False),
            result.get("rag_results_count", 0),
            result.get("response_time_ms", 0),
            result.get("detected_code")
        ))

        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        # Fire-and-forget - don't let logging failures affect the user
        logger.error(f"Failed to log query: {e}")


def get_system_stats() -> dict:
    """
    Get system statistics for health check/dashboard.

    Returns:
        {
            "queries_today": int,
            "queries_this_week": int,
            "avg_response_time_ms": int,
            "top_models": [("Elfo Traction", 45), ...],
            "top_query_types": [("fault_code", 120), ...],
            "zero_result_queries": int,
            "index_status": dict
        }
    """
    try:
        conn = _get_connection()
        cur = conn.cursor()

        now = datetime.now()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        week_start = today_start - timedelta(days=7)

        # Queries today
        cur.execute("""
            SELECT COUNT(*) FROM query_log WHERE created_at >= %s
        """, (today_start,))
        queries_today = cur.fetchone()[0]

        # Queries this week
        cur.execute("""
            SELECT COUNT(*) FROM query_log WHERE created_at >= %s
        """, (week_start,))
        queries_this_week = cur.fetchone()[0]

        # Average response time (last 7 days)
        cur.execute("""
            SELECT COALESCE(AVG(response_time_ms), 0)
            FROM query_log WHERE created_at >= %s
        """, (week_start,))
        avg_response_time_ms = int(cur.fetchone()[0])

        # Top models (last 7 days)
        cur.execute("""
            SELECT lift_model, COUNT(*) as cnt
            FROM query_log
            WHERE created_at >= %s AND lift_model IS NOT NULL
            GROUP BY lift_model
            ORDER BY cnt DESC
            LIMIT 5
        """, (week_start,))
        top_models = [(row[0], row[1]) for row in cur.fetchall()]

        # Top query types (last 7 days)
        cur.execute("""
            SELECT query_type, COUNT(*) as cnt
            FROM query_log
            WHERE created_at >= %s AND query_type IS NOT NULL
            GROUP BY query_type
            ORDER BY cnt DESC
        """, (week_start,))
        top_query_types = [(row[0], row[1]) for row in cur.fetchall()]

        # Zero result queries (last 7 days) - quality indicator
        cur.execute("""
            SELECT COUNT(*)
            FROM query_log
            WHERE created_at >= %s AND rag_results_count = 0
        """, (week_start,))
        zero_result_queries = cur.fetchone()[0]

        cur.close()
        conn.close()

        # Get index status
        from liftmind.rag import get_index_status
        try:
            index_status = get_index_status()
        except:
            index_status = {}

        return {
            "queries_today": queries_today,
            "queries_this_week": queries_this_week,
            "avg_response_time_ms": avg_response_time_ms,
            "top_models": top_models,
            "top_query_types": top_query_types,
            "zero_result_queries": zero_result_queries,
            "index_status": index_status
        }

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}")
        return {
            "queries_today": 0,
            "queries_this_week": 0,
            "avg_response_time_ms": 0,
            "top_models": [],
            "top_query_types": [],
            "zero_result_queries": 0,
            "index_status": {},
            "error": str(e)
        }

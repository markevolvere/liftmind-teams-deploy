"""
Database connection pool for LiftMind.

Features:
- Connection health checks
- Retry logic with exponential backoff
- Pool reinitialization as last resort
- Configurable pool sizes from settings

Provides a shared ThreadedConnectionPool for all database operations.
"""
import logging
import time
from contextlib import contextmanager
from typing import Optional
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from liftmind.config import settings

logger = logging.getLogger(__name__)

_pool: Optional[ThreadedConnectionPool] = None
_pool_initialized_at: float = 0


def init_pool(minconn: int = None, maxconn: int = None) -> None:
    """Initialize the database connection pool."""
    global _pool, _pool_initialized_at

    if _pool is not None:
        logger.warning("Connection pool already initialized")
        return

    # Use config values if not specified
    minconn = minconn or settings.DB_POOL_MIN_CONN
    maxconn = maxconn or settings.DB_POOL_MAX_CONN

    try:
        _pool = ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            dsn=settings.DATABASE_URL
        )
        _pool_initialized_at = time.time()
        logger.info(f"Database pool initialized (min={minconn}, max={maxconn})")
    except Exception as e:
        logger.error(f"Failed to initialize database pool: {e}")
        raise


def close_pool() -> None:
    """Close the database connection pool."""
    global _pool, _pool_initialized_at
    if _pool is not None:
        _pool.closeall()
        _pool = None
        _pool_initialized_at = 0
        logger.info("Database pool closed")


def _reinitialize_pool() -> None:
    """Reinitialize pool after failure (last resort)."""
    global _pool, _pool_initialized_at
    logger.warning("Reinitializing database pool...")

    if _pool is not None:
        try:
            _pool.closeall()
        except Exception:
            pass
        _pool = None

    init_pool()


def _check_connection_health(conn) -> bool:
    """Check if a connection is still alive."""
    try:
        # Simple ping query
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            return cur.fetchone()[0] == 1
    except Exception:
        return False


@contextmanager
def get_connection():
    """
    Context manager for safe connection handling with retry logic.

    Implements connection health checks and retry logic:
    1. Get connection from pool
    2. Check if connection is alive
    3. If dead, get a new connection (up to RECONNECT_RETRIES times)
    4. If all retries fail, reinitialize pool

    Usage:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM users")
                results = cur.fetchall()
    """
    global _pool

    if _pool is None:
        init_pool()

    retries = settings.DB_POOL_RECONNECT_RETRIES
    last_error = None

    for attempt in range(retries):
        conn = None
        try:
            conn = _pool.getconn()

            # Check connection health
            if not _check_connection_health(conn):
                logger.warning(f"Dead connection detected (attempt {attempt + 1}/{retries})")
                try:
                    _pool.putconn(conn, close=True)  # Close dead connection
                except Exception:
                    pass
                continue

            # Connection is healthy, use it
            try:
                yield conn
                conn.commit()
                return
            except Exception as e:
                conn.rollback()
                raise

        except psycopg2.OperationalError as e:
            last_error = e
            logger.warning(f"Database connection error (attempt {attempt + 1}/{retries}): {e}")
            if conn:
                try:
                    _pool.putconn(conn, close=True)
                except Exception:
                    pass
            conn = None
            time.sleep(0.5 * (attempt + 1))  # Exponential backoff
            continue

        finally:
            if conn is not None:
                try:
                    _pool.putconn(conn)
                except Exception as e:
                    logger.warning(f"Error returning connection to pool: {e}")

    # All retries exhausted - reinitialize pool as last resort
    logger.error(f"All {retries} connection attempts failed, reinitializing pool")
    _reinitialize_pool()

    # Try one more time with fresh pool
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)


def test_connection() -> bool:
    """Test if the database connection is working."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return False


def execute_query(query: str, params: tuple = None, fetch: bool = True):
    """
    Execute a query and return results.

    Args:
        query: SQL query string
        params: Query parameters
        fetch: Whether to fetch results

    Returns:
        Query results if fetch=True, else None
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            if fetch:
                return cur.fetchall()
            return None


def execute_query_one(query: str, params: tuple = None):
    """Execute a query and return a single result."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchone()


def get_pool_stats() -> dict:
    """Get connection pool statistics."""
    global _pool, _pool_initialized_at

    if _pool is None:
        return {"status": "not_initialized"}

    return {
        "status": "active",
        "min_connections": settings.DB_POOL_MIN_CONN,
        "max_connections": settings.DB_POOL_MAX_CONN,
        "initialized_at": _pool_initialized_at,
        "uptime_seconds": time.time() - _pool_initialized_at if _pool_initialized_at else 0
    }

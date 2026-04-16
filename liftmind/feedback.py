"""
LiftMind Feedback System — Teams-adapted.

Changed:
  - telegram_user_id (int) → user_id (str) throughout
  - Telegram send-message scheduling removed; feedback is recorded in DB only
  - Adaptive Cards (Teams) handle the feedback buttons inline with bot response
  - from app.* → from liftmind.*
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Tuple
import random

from psycopg2.extras import RealDictCursor

from liftmind.database import get_connection

logger = logging.getLogger(__name__)


# Keywords that indicate troubleshooting
TROUBLESHOOTING_KEYWORDS = [
    'fault', 'error', 'problem', 'issue', 'not working', "won't", "doesn't",
    'stuck', 'fail', 'broken', 'stopped', 'troubleshoot', 'help',
    "what's wrong", 'why is', 'how to fix'
]


def is_troubleshooting_query(query: str) -> bool:
    """Check if a query is troubleshooting-related."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in TROUBLESHOOTING_KEYWORDS)


def detect_issue_type(query: str) -> str:
    """Detect the type of issue from the query.

    Returns one of: 'wiring', 'troubleshooting', 'spec'
    """
    query_lower = query.lower()

    if any(term in query_lower for term in ['wire', 'wiring', 'terminal', 'connection', 'cable']):
        return 'wiring'

    if any(term in query_lower for term in [
        'error', 'fault', 'e0', 'e1', 'not working', 'stuck', 'door', 'motor', 'safety', 'level'
    ]):
        return 'troubleshooting'

    return 'spec'


def extract_lift_model(text: str) -> Optional[str]:
    """Extract lift model from text."""
    known_models = [
        'elfo', 'elfo 2', 'elfo xl', 'elfo traction', 'elfo hydraulic',
        'bari', 'e3', 'tresa', 'freedom', 'freedom maxi', 'freedom step',
        'p4', 'pollock', 'supermec', 'supermec 2', 'supermec 3',
        'elvoron', 'boutique'
    ]
    text_lower = text.lower()
    for model in known_models:
        if model in text_lower:
            return model.title()
    return None


def should_request_feedback(user_id: str, conversation_topic: str) -> bool:
    """
    Check if we should request feedback for this conversation.

    Don't ask if already asked in the last 24 hours (max 2 requests).
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*) FROM feedback_candidates
                    WHERE user_id = %s
                      AND created_at > NOW() - INTERVAL '24 hours'
                """, (user_id,))
                recent_count = cur.fetchone()[0]
        return recent_count < 2
    except Exception as exc:
        logger.warning("should_request_feedback check failed: %s", exc)
        return False


def create_feedback_candidate(
    user_id: str,
    user_name: str,
    question: str,
    response: str,
    lift_model: Optional[str] = None,
    fact_ids: List[int] = None,
    chunk_ids: List[int] = None,
    qa_pair_ids: List[int] = None,
) -> Optional[int]:
    """
    Create a feedback candidate entry.

    In Teams, feedback is collected inline via Adaptive Card buttons attached to
    the bot response -- no delayed Telegram message needed.  This function just
    records the candidate so the feedback response can be linked back to it.

    Returns the candidate ID, or None if skipped / on error.
    """
    if not should_request_feedback(user_id, question):
        return None

    if not lift_model:
        lift_model = extract_lift_model(question) or extract_lift_model(response)

    issue_type = detect_issue_type(question)

    # Keep the scheduled_at field so the DB schema stays compatible, but we
    # don't rely on a background job to send a follow-up in Teams -- the card
    # is shown immediately with the response.
    scheduled_time = datetime.now() + timedelta(minutes=random.randint(2, 3))

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feedback_candidates (
                        user_id, user_name, conversation_topic,
                        lift_model, issue_type, original_question, bot_response,
                        fact_ids_used, chunk_ids_used, qa_pairs_used,
                        feedback_scheduled_at, status
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'scheduled')
                    RETURNING id
                """, (
                    user_id,
                    user_name,
                    question[:200],
                    lift_model,
                    issue_type,
                    question,
                    response,
                    fact_ids or [],
                    chunk_ids or [],
                    qa_pair_ids or [],
                    scheduled_time,
                ))
                candidate_id = cur.fetchone()[0]

        logger.info("Created feedback candidate %s for user %s", candidate_id, user_id)
        return candidate_id
    except Exception as exc:
        logger.warning("Failed to create feedback candidate for user=%s: %s", user_id, exc)
        return None


def get_pending_feedback_requests() -> List[Dict]:
    """Get feedback requests that are due (for any background processing)."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, user_id, user_name, lift_model, issue_type, original_question
                    FROM feedback_candidates
                    WHERE status = 'scheduled'
                      AND feedback_scheduled_at <= NOW()
                    ORDER BY feedback_scheduled_at
                    LIMIT 10
                """)
                return list(cur.fetchall())
    except Exception as exc:
        logger.warning("get_pending_feedback_requests failed: %s", exc)
        return []


def mark_feedback_sent(candidate_id: int):
    """Mark a feedback request as sent."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE feedback_candidates
                    SET status = 'sent', feedback_sent_at = NOW()
                    WHERE id = %s
                """, (candidate_id,))
    except Exception as exc:
        logger.warning("mark_feedback_sent failed for id=%s: %s", candidate_id, exc)


def generate_feedback_message(candidate: Dict) -> Tuple[str, List[Dict]]:
    """
    Generate the feedback request message and button definitions.

    In Teams the buttons are rendered as an Adaptive Card -- this function
    returns the same logical structure so callers don't need to change.

    Returns: (message_text, button_list)
    """
    user_name = candidate.get('user_name', 'there')
    if user_name:
        user_name = user_name.split()[0]  # First name only

    message = (
        f"Hey {user_name} \u2014 was that answer helpful?\n\n"
        "Your feedback helps LiftMind get smarter for the whole team."
    )

    buttons = [
        {'text': '\u2705 Correct',  'callback_data': f'fb_correct_{candidate["id"]}'},
        {'text': '\U0001f7e1 Somewhat', 'callback_data': f'fb_partial_{candidate["id"]}'},
        {'text': '\u274c Wrong',    'callback_data': f'fb_wrong_{candidate["id"]}'},
        {'text': '\u23ed\ufe0f Skip',    'callback_data': f'fb_skip_{candidate["id"]}'},
    ]
    return message, buttons


def record_feedback_response(
    candidate_id: int,
    response_type: str,   # 'yes', 'no', 'partial', 'still_working', 'skip'
    user_id: str,
) -> int:
    """
    Record initial feedback response.

    Returns the feedback ID for follow-up.
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO feedback (candidate_id, user_id, did_it_work)
                    VALUES (%s, %s, %s)
                    RETURNING id
                """, (candidate_id, user_id, response_type))
                feedback_id = cur.fetchone()[0]

                cur.execute("""
                    UPDATE feedback_candidates SET status = 'completed' WHERE id = %s
                """, (candidate_id,))

                cur.execute(
                    "SELECT fact_ids_used FROM feedback_candidates WHERE id = %s",
                    (candidate_id,),
                )
                row = cur.fetchone()
                fact_ids = row[0] if row and row[0] else []

        if fact_ids and response_type != 'skip':
            was_helpful = response_type in ['yes', 'correct', 'partial']
            update_fact_confidence(fact_ids, was_helpful)

        return feedback_id
    except Exception as exc:
        logger.error("record_feedback_response failed: %s", exc)
        return -1


def generate_followup_message(response_type: str, feedback_id: int) -> Tuple[str, None]:
    """
    Generate follow-up message text based on initial response.

    In Teams this is sent as a plain text reply; no keyboard needed.
    Returns: (message_text, None)
    """
    if response_type in ('yes', 'correct'):
        return "Cheers! Glad it helped. \U0001f91c", None

    if response_type == 'partial':
        return (
            "Thanks \u2014 what was missing or wrong? A quick note helps me improve.\n\n"
            "Just reply with what you'd change.",
            None,
        )

    if response_type in ('no', 'wrong'):
        return (
            "Thanks for letting me know \u2014 this helps me learn.\n\n"
            "What was the correct answer? Even a quick note helps.\n\n"
            "Just reply with what worked.",
            None,
        )

    if response_type in ('still_working', 'skip'):
        return "No worries! Ask anytime. \U0001f44d", None

    return "", None


def record_followup_response(
    feedback_id: int,
    was_modified: Optional[str],   # 'exact', 'modified', 'different_fix'
    explanation: Optional[str],
):
    """Record follow-up details."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE feedback
                    SET was_modified = %s, user_explanation = %s
                    WHERE id = %s
                """, (was_modified, explanation, feedback_id))

        if explanation and len(explanation) > 10:
            create_verified_fix_from_feedback(feedback_id)
    except Exception as exc:
        logger.error("record_followup_response failed: %s", exc)


def create_verified_fix_from_feedback(feedback_id: int):
    """Create a verified fix entry from feedback."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT f.*, fc.lift_model, fc.original_question, fc.bot_response,
                           fc.user_id, fc.user_name
                    FROM feedback f
                    JOIN feedback_candidates fc ON f.candidate_id = fc.id
                    WHERE f.id = %s
                """, (feedback_id,))
                feedback = cur.fetchone()
                if not feedback:
                    return

                was_correct = feedback['did_it_work'] in ('yes', 'correct')

                symptoms = []
                question = feedback['original_question'].lower()
                for pattern in [
                    r'error\s+(\w+)', r'fault\s+(\w+)', r"won't\s+(\w+)",
                    r'not\s+(\w+ing)', r'stuck\s+(\w+)',
                ]:
                    symptoms.extend(re.findall(pattern, question))

                cur.execute("""
                    INSERT INTO verified_fixes (
                        feedback_id, lift_model, issue_description, symptoms,
                        verified_solution, original_bot_suggestion, was_bot_correct,
                        contributed_by, contributor_name, approved
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, false)
                """, (
                    feedback_id,
                    feedback['lift_model'],
                    feedback['original_question'],
                    symptoms,
                    feedback['user_explanation'] or feedback['bot_response'],
                    feedback['bot_response'],
                    was_correct,
                    feedback['user_id'],
                    feedback['user_name'],
                ))

        logger.info("Created verified fix from feedback %s", feedback_id)
    except Exception as exc:
        logger.error("create_verified_fix_from_feedback failed: %s", exc)


def increment_fix_helpful(fix_ids: list):
    """Increment times_confirmed_helpful for verified fixes."""
    if not fix_ids:
        return
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE verified_fixes
                    SET times_confirmed_helpful = times_confirmed_helpful + 1
                    WHERE id = ANY(%s)
                """, (fix_ids,))
        logger.info("Incremented times_confirmed_helpful for fix IDs: %s", fix_ids)
    except Exception as exc:
        logger.error("increment_fix_helpful failed: %s", exc)


def update_fact_confidence(fact_ids: List[int], was_helpful: bool):
    """Update citation/helpfulness counts. Confidence calculated by view."""
    if not fact_ids:
        return
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for fact_id in fact_ids:
                    if was_helpful:
                        cur.execute("""
                            UPDATE facts
                            SET times_cited = times_cited + 1,
                                times_confirmed_helpful = times_confirmed_helpful + 1
                            WHERE id = %s
                        """, (fact_id,))
                    else:
                        cur.execute("""
                            UPDATE facts
                            SET times_cited = times_cited + 1
                            WHERE id = %s
                        """, (fact_id,))
    except Exception as exc:
        logger.error("update_fact_confidence failed: %s", exc)


# ============================================================================
# STATS AND REPORTING
# ============================================================================

def get_feedback_stats() -> Dict:
    """Get feedback system statistics."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        COUNT(*) as total_candidates,
                        COUNT(*) FILTER (WHERE status = 'completed') as completed,
                        COUNT(*) FILTER (WHERE status = 'sent') as awaiting_response,
                        COUNT(*) FILTER (WHERE status = 'scheduled') as scheduled
                    FROM feedback_candidates
                """)
                candidates = cur.fetchone()

                cur.execute("""
                    SELECT did_it_work, COUNT(*) as count
                    FROM feedback
                    GROUP BY did_it_work
                """)
                responses = {row['did_it_work']: row['count'] for row in cur.fetchall()}

                cur.execute("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE approved = true) as approved,
                        COUNT(*) FILTER (WHERE was_bot_correct = true) as bot_was_correct
                    FROM verified_fixes
                """)
                fixes = cur.fetchone()

                cur.execute("""
                    SELECT contributor_name, COUNT(*) as fixes
                    FROM verified_fixes
                    WHERE approved = true AND contributor_name IS NOT NULL
                    GROUP BY contributor_name
                    ORDER BY fixes DESC
                    LIMIT 5
                """)
                top_contributors = [
                    {'name': row['contributor_name'], 'fixes': row['fixes']}
                    for row in cur.fetchall()
                ]

        total_sent = candidates['completed'] + responses.get('still_working', 0)
        response_rate = (candidates['completed'] / total_sent * 100) if total_sent > 0 else 0

        total_responses = sum(responses.values())
        positive = responses.get('yes', 0) + responses.get('partial', 0)
        accuracy = (positive / total_responses * 100) if total_responses > 0 else 0

        return {
            'candidates': dict(candidates),
            'responses': responses,
            'verified_fixes': dict(fixes),
            'top_contributors': top_contributors,
            'response_rate': round(response_rate, 1),
            'bot_accuracy': round(accuracy, 1),
        }
    except Exception as exc:
        logger.error("get_feedback_stats failed: %s", exc)
        return {}


def get_pending_fixes_for_review() -> List[Dict]:
    """Get verified fixes pending admin approval."""
    try:
        with get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT id, lift_model, issue_description, symptoms, verified_solution,
                           original_bot_suggestion, was_bot_correct, contributor_name, created_at
                    FROM verified_fixes
                    WHERE approved = false
                    ORDER BY created_at DESC
                """)
                return list(cur.fetchall())
    except Exception as exc:
        logger.error("get_pending_fixes_for_review failed: %s", exc)
        return []


def approve_verified_fix(fix_id: int, approve: bool = True):
    """Approve or reject a verified fix."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                if approve:
                    cur.execute("UPDATE verified_fixes SET approved = true WHERE id = %s", (fix_id,))
                else:
                    cur.execute("DELETE FROM verified_fixes WHERE id = %s", (fix_id,))
    except Exception as exc:
        logger.error("approve_verified_fix failed for id=%s: %s", fix_id, exc)

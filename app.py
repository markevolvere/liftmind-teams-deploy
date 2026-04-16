"""
LiftMind Teams Bot — app.py

Aiohttp + Bot Framework SDK v4 + LiftMind RAG brain.

Architecture:
  - Bot Framework adapter handles Teams auth + activity routing
  - Whitelist stored in PostgreSQL (same DB as LiftMind)
  - All lift Q&A handled by liftmind.brain.process_query() via asyncio.to_thread()
  - Adaptive Cards used for model selection and feedback collection
"""

import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from aiohttp import web

from botbuilder.core import TurnContext, Bot
from botbuilder.core.teams import TeamsInfo
from botbuilder.integration.aiohttp import (
    CloudAdapter,
    ConfigurationBotFrameworkAuthentication,
)
from botbuilder.schema import (
    Activity, ActivityTypes, Attachment,
    SuggestedActions, CardAction, ActionTypes,
)

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("liftmind-teams")

# ── Config ─────────────────────────────────────────────────────────────────────
# Support both Microsoft* and MICROSOFT_* env var conventions
def _env(*names, default=""):
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return default

APP_ID       = _env("MicrosoftAppId", "MICROSOFT_APP_ID")
APP_PASSWORD = _env("MicrosoftAppPassword", "MICROSOFT_APP_PASSWORD")
APP_TYPE     = _env("MicrosoftAppType", "MICROSOFT_APP_TYPE", default="MultiTenant")
APP_TENANTID = _env("MicrosoftAppTenantId", "MICROSOFT_APP_TENANT_ID")
PORT         = int(os.environ.get("PORT", 3978))

# Admin seed: always has access
_SEED_EMAIL = os.environ.get("ADMIN_SEED_EMAIL", "mjeanes@liftshop.com.au")


# ── Bot Framework CloudAdapter ────────────────────────────────────────────────
class _BotAuthConfig:
    """Config object read by ConfigurationBotFrameworkAuthentication.

    Uses the attribute names the SDK expects (APP_ID, APP_PASSWORD, APP_TYPE,
    APP_TENANTID). CloudAdapter handles SingleTenant and MultiTenant token
    acquisition + validation automatically based on APP_TYPE.
    """
    APP_ID = APP_ID
    APP_PASSWORD = APP_PASSWORD
    APP_TYPE = APP_TYPE
    APP_TENANTID = APP_TENANTID

logger.info(
    "[BOTAUTH] APP_TYPE=%s APP_ID_set=%s APP_TENANTID_set=%s",
    APP_TYPE, bool(APP_ID), bool(APP_TENANTID),
)

_auth = ConfigurationBotFrameworkAuthentication(_BotAuthConfig())
adapter = CloudAdapter(_auth)


# ── PostgreSQL whitelist (uses same DB as LiftMind) ───────────────────────────

def _get_pg_conn():
    """Return a raw psycopg2 connection (caller must close)."""
    from liftmind.database import get_connection
    return get_connection()


def _ensure_whitelist_table():
    """Create bot_authorised_users table if it doesn't exist, seed admin."""
    try:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS bot_authorised_users (
                        id         SERIAL PRIMARY KEY,
                        email      TEXT NOT NULL UNIQUE,
                        added_by   TEXT NOT NULL DEFAULT 'SYSTEM',
                        added_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    )
                """)
                # Seed admin user if table was just created or is empty
                cur.execute(
                    "INSERT INTO bot_authorised_users (email, added_by) "
                    "VALUES (%s, 'SYSTEM') ON CONFLICT (email) DO NOTHING",
                    (_SEED_EMAIL,),
                )
        logger.info("Whitelist table ensured (seed: %s)", _SEED_EMAIL)
    except Exception as exc:
        logger.error("Failed to ensure whitelist table: %s", exc)


def _is_user_authorised(email: str) -> bool:
    """Check if an email is in the authorised users list."""
    try:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM bot_authorised_users WHERE LOWER(email) = LOWER(%s)",
                    (email,),
                )
                row = cur.fetchone()
                return (row[0] > 0) if row else False
    except Exception as exc:
        logger.error("Whitelist check failed: %s", exc)
        return False


def _add_authorised_user(email: str, added_by: str) -> str:
    """Add a user to the whitelist. Returns status message."""
    email = email.strip().lower()
    if not email.endswith("@liftshop.com.au"):
        return f"**Cannot add** `{email}` — only @liftshop.com.au addresses are allowed."
    try:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM bot_authorised_users WHERE LOWER(email) = %s",
                    (email,),
                )
                if cur.fetchone()[0] > 0:
                    return f"`{email}` is already authorised."
                cur.execute(
                    "INSERT INTO bot_authorised_users (email, added_by) VALUES (%s, %s)",
                    (email, added_by.lower()),
                )
        return f"**Added** `{email}` to the authorised users list. ✅"
    except Exception as exc:
        logger.error("Failed to add user %s: %s", email, exc)
        return f"Error adding user: {exc}"


def _remove_authorised_user(email: str) -> str:
    """Remove a user from the whitelist. Returns status message."""
    email = email.strip().lower()
    if email == _SEED_EMAIL.lower():
        return f"Cannot remove the admin account `{email}`."
    try:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM bot_authorised_users WHERE LOWER(email) = %s",
                    (email,),
                )
                removed = cur.rowcount > 0
        if removed:
            return f"**Removed** `{email}` from the authorised users list."
        return f"`{email}` was not found in the authorised users list."
    except Exception as exc:
        logger.error("Failed to remove user %s: %s", email, exc)
        return f"Error removing user: {exc}"


def _list_authorised_users() -> str:
    """Return a formatted list of all authorised users."""
    try:
        with _get_pg_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT email, added_by, added_at FROM bot_authorised_users ORDER BY added_at"
                )
                rows = cur.fetchall()
        if not rows:
            return "No authorised users found."
        lines = ["**Authorised Users:**\n"]
        for email, added_by, added_at in rows:
            dt_str = added_at.strftime("%d/%m/%Y") if added_at else "N/A"
            lines.append(f"- `{email}` — added by `{added_by}` on {dt_str}")
        return "\n".join(lines)
    except Exception as exc:
        logger.error("Failed to list users: %s", exc)
        return f"Error listing users: {exc}"


async def _get_user_email(ctx: TurnContext) -> str:
    """Get the user's email from Teams context."""
    try:
        member = await TeamsInfo.get_member(ctx, ctx.activity.from_property.id)
        return (member.email or "").strip().lower()
    except Exception as exc:
        logger.warning("Could not get Teams member email: %s", exc)
        try:
            channel_data = ctx.activity.channel_data or {}
            upn = channel_data.get("userPrincipalName", "")
            if upn:
                return upn.strip().lower()
        except Exception:
            pass
        return ""


# ── Adaptive Cards ─────────────────────────────────────────────────────────────

def _feedback_card(candidate_id: int) -> Attachment:
    """Build a compact feedback Adaptive Card to attach to bot responses."""
    card = {
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": [
            {
                "type": "TextBlock",
                "text": "Was that helpful?",
                "size": "Small",
                "color": "Accent",
                "isSubtle": True,
            }
        ],
        "actions": [
            {
                "type": "Action.Submit",
                "title": "✅ Yes",
                "data": {"feedback_type": "correct", "candidate_id": candidate_id},
            },
            {
                "type": "Action.Submit",
                "title": "🟡 Partial",
                "data": {"feedback_type": "partial", "candidate_id": candidate_id},
            },
            {
                "type": "Action.Submit",
                "title": "❌ Wrong",
                "data": {"feedback_type": "wrong", "candidate_id": candidate_id},
            },
        ],
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    }
    return Attachment(
        content_type="application/vnd.microsoft.card.adaptive",
        content=card,
    )


# Canonical list of lift models in the knowledge base.
# Grouped here by family for the prompt text; flattened for chip actions.
LIFT_MODELS = [
    # Elfo family
    "Elfo", "Elfo 2", "E3",
    "Elfo Cabin", "Elfo Electronic",
    "Elfo Hydraulic controller", "Elfo Traction",
    # Supermec family
    "Supermec", "Supermec 2", "Supermec 3",
    # Freedom family
    "Freedom", "Freedom MAXI", "Freedom STEP",
    # Pollock
    "Pollock (P1)", "Pollock (Q1)",
    # Individual
    "Bari", "P4", "Tresa",
]

# Commands that trigger the picker / clear the current model.
_RESET_COMMANDS = {"/reset", "/model", "/change", "change lift", "🔄 change lift"}


def _model_picker_activity(
    prompt: str = "Which lift do you need help with?",
) -> Activity:
    """
    Build a model-picker reply as an Adaptive Card with family-grouped buttons.

    Teams caps the top-level `actions` array at 6 visible buttons (rest go
    into a "..." overflow menu). Putting Action.Submit buttons inside
    `ActionSet` elements in the card body bypasses that cap, so all 18
    models stay visible and can be organised by family.
    """

    def btn(title: str, model_value: Optional[str] = None) -> dict:
        return {
            "type": "Action.Submit",
            "title": title,
            "data": {
                "card_action": "pick_model",
                "model": model_value if model_value is not None else title,
            },
        }

    def family_section(title: str, models: list) -> list:
        """Return [header TextBlock, ActionSet] for a family group."""
        return [
            {
                "type": "TextBlock",
                "text": title,
                "weight": "Bolder",
                "size": "Small",
                "color": "Accent",
                "spacing": "Medium",
            },
            {
                "type": "ActionSet",
                "actions": [btn(m) for m in models],
            },
        ]

    body = [
        {
            "type": "TextBlock",
            "text": prompt,
            "weight": "Bolder",
            "size": "Medium",
            "wrap": True,
        },
        {
            "type": "TextBlock",
            "text": "Tap the model below to scope my answer.",
            "isSubtle": True,
            "size": "Small",
            "wrap": True,
            "spacing": "Small",
        },
    ]

    # Elfo family has 7 models — split into two ActionSets to stay under any
    # per-set rendering width limits.
    body += family_section("Elfo family", ["Elfo", "Elfo 2", "E3", "Elfo Cabin"])
    body += family_section(
        "Elfo (cont.)",
        ["Elfo Electronic", "Elfo Hydraulic controller", "Elfo Traction"],
    )
    body += family_section("Supermec family", ["Supermec", "Supermec 2", "Supermec 3"])
    body += family_section("Freedom family", ["Freedom", "Freedom MAXI", "Freedom STEP"])
    body += family_section("Pollock", ["Pollock (P1)", "Pollock (Q1)"])
    body += family_section("Individual models", ["Bari", "P4", "Tresa"])

    # Escape hatch
    body += [
        {
            "type": "TextBlock",
            "text": "Not sure?",
            "weight": "Bolder",
            "size": "Small",
            "color": "Accent",
            "spacing": "Medium",
        },
        {
            "type": "ActionSet",
            "actions": [btn("Other / not listed", "__other__")],
        },
    ]

    card = {
        "type": "AdaptiveCard",
        "version": "1.3",
        "body": body,
        "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
    }
    return Activity(
        type=ActivityTypes.message,
        attachments=[Attachment(
            content_type="application/vnd.microsoft.card.adaptive",
            content=card,
        )],
    )


def _reply_with_change_chip(text: str, attachments=None) -> Activity:
    """Wrap a text response with a '🔄 Change lift' chip for easy switching."""
    reply = Activity(
        type=ActivityTypes.message,
        text=text,
        attachments=attachments or [],
        suggested_actions=SuggestedActions(
            actions=[
                CardAction(
                    type=ActionTypes.im_back,
                    title="🔄 Change lift",
                    value="/reset",
                )
            ]
        ),
    )
    return reply


def _help_card() -> str:
    """Return a plain markdown help message."""
    return (
        "**LiftMind — Lift Tech Assistant** 🔧\n\n"
        "Ask me anything about lift installation, faults, specs, or procedures.\n\n"
        "**How it works:**\n"
        "1. Ask any question\n"
        "2. I'll ask you which lift — tap a chip\n"
        "3. I'll answer based on that model's manuals\n"
        "4. Use **🔄 Change lift** any time to switch models\n\n"
        "**Commands:**\n"
        "- `/model` or `/reset` — switch lift / re-show the picker\n"
        "- `/help` — show this message\n"
        "- `/add me` — authorise yourself (your @liftshop.com.au email)\n"
        "- `/add [email]` — add another Lift Shop user\n"
        "- `/remove [email]` — remove a user (admin only)\n"
        "- `/list users` — show authorised users\n"
    )


# ── Message handler ────────────────────────────────────────────────────────────

async def on_message_activity(ctx: TurnContext):
    """Handle an incoming message from Teams."""
    activity = ctx.activity
    user_id   = activity.from_property.id if activity.from_property else "unknown"
    user_name = activity.from_property.name if activity.from_property else "User"

    # ── Card submit actions (feedback / model selection) ──────────────────────
    if activity.value and isinstance(activity.value, dict):
        await _handle_card_action(ctx, user_id, user_name, activity.value)
        return

    # ── Extract text (strip @mention in channels) ─────────────────────────────
    text = (activity.text or "").strip()
    # Remove bot @mention prefix if present (Teams channels prepend "<at>BotName</at>")
    if activity.entities:
        for entity in activity.entities:
            # SDK 4.17+: entities are Entity objects, not dicts. Normalise to dict.
            ent = entity.serialize() if hasattr(entity, "serialize") else entity
            if isinstance(ent, dict) and ent.get("type") == "mention":
                mentioned = ent.get("mentioned") or {}
                mention_text = mentioned.get("name", "") if isinstance(mentioned, dict) else ""
                if mention_text:
                    text = text.replace(f"<at>{mention_text}</at>", "").strip()

    if not text:
        return

    text_lower = text.lower()

    # ── Admin/whitelist commands (no auth needed for /add me) ─────────────────
    if text_lower.startswith("/add me"):
        email = await _get_user_email(ctx)
        if not email:
            await ctx.send_activity("Could not determine your email. Please try again.")
            return
        result = _add_authorised_user(email, "self")
        await ctx.send_activity(result)
        return

    if text_lower.startswith("/add "):
        # Only authorised users can add others
        email = await _get_user_email(ctx)
        if not email or not _is_user_authorised(email):
            await ctx.send_activity(
                "⛔ You need to be authorised to add other users. "
                "Type `/add me` to request access with your @liftshop.com.au account."
            )
            return
        target_email = text[5:].strip()
        result = _add_authorised_user(target_email, email)
        await ctx.send_activity(result)
        return

    if text_lower.startswith("/remove "):
        email = await _get_user_email(ctx)
        if email.lower() != _SEED_EMAIL.lower():
            await ctx.send_activity("⛔ Only the admin can remove users.")
            return
        target_email = text[8:].strip()
        result = _remove_authorised_user(target_email)
        await ctx.send_activity(result)
        return

    if text_lower in ("/list users", "/list", "/users"):
        email = await _get_user_email(ctx)
        if email.lower() != _SEED_EMAIL.lower():
            await ctx.send_activity("⛔ Only the admin can list users.")
            return
        result = _list_authorised_users()
        await ctx.send_activity(result)
        return

    # ── Authorisation check ───────────────────────────────────────────────────
    email = await _get_user_email(ctx)
    if not email or not _is_user_authorised(email):
        await ctx.send_activity(
            "⛔ **Access denied.**\n\n"
            "This bot is for Lift Shop staff only. "
            "Type `/add me` with your @liftshop.com.au account to request access.\n\n"
            f"(Your email: `{email or 'unknown'}`)"
        )
        return

    # ── /help ─────────────────────────────────────────────────────────────────
    if text_lower in ("/help", "help", "?"):
        await ctx.send_activity(_help_card())
        return

    from liftmind.user_state import (
        get_user_model_fresh,
        set_user_model,
        clear_user_model,
        set_pending_question,
        pop_pending_question,
    )

    # ── Reset / change-lift commands → clear model, show picker ───────────────
    if text_lower in _RESET_COMMANDS:
        await asyncio.to_thread(clear_user_model, user_id)
        await ctx.send_activity(_model_picker_activity(
            "Which lift do you need help with?"
        ))
        return

    # ── Tapped a model chip (exact match to a known model) ────────────────────
    matched_model = next(
        (m for m in LIFT_MODELS if m.lower() == text_lower),
        None,
    )
    if matched_model:
        await asyncio.to_thread(set_user_model, user_id, matched_model)
        # If there was a question waiting, answer it now using this model.
        pending = await asyncio.to_thread(pop_pending_question, user_id)
        if pending:
            await _answer_with_brain(ctx, user_id, pending, matched_model)
        else:
            await ctx.send_activity(_reply_with_change_chip(
                f"Got it — **{matched_model}**. How can I help?"
            ))
        return

    # ── "Other / not listed" chip ─────────────────────────────────────────────
    if text_lower == "/other":
        await ctx.send_activity(_reply_with_change_chip(
            "No problem — type your question and I'll search across all manuals. "
            "If I need to narrow down, I'll ask."
        ))
        # Clear any model so the brain searches broadly
        await asyncio.to_thread(clear_user_model, user_id)
        return

    # ── Check whether user already has a fresh model selection ───────────────
    current_model = await asyncio.to_thread(get_user_model_fresh, user_id, 30)

    if not current_model:
        # Stash their question, show the picker. They'll get an answer right
        # after they tap a model.
        await asyncio.to_thread(set_pending_question, user_id, text)
        await ctx.send_activity(_model_picker_activity(
            "Which lift do you need help with?"
        ))
        return

    # ── Normal path: user has a model selected, answer their question ────────
    await _answer_with_brain(ctx, user_id, text, current_model)


async def _answer_with_brain(
    ctx: TurnContext,
    user_id: str,
    question: str,
    model: str,
):
    """Run the LiftMind brain for a question and reply with Change-lift chip."""
    await ctx.send_activity(Activity(type="typing"))

    try:
        result = await asyncio.to_thread(_call_brain, user_id, question)
    except Exception as exc:
        logger.error("Brain call failed for user=%s: %s", user_id, exc)
        await ctx.send_activity(_reply_with_change_chip(
            "❌ Something went wrong processing your question. Please try again."
        ))
        return

    response_text = result.get("response", "")
    sources       = result.get("sources", [])
    candidate_id  = result.get("feedback_candidate_id")

    # Prepend model context as a small header (keeps it clear which model
    # the answer is scoped to).
    response_text = f"_(answering for **{model}**)_\n\n{response_text}"

    if sources:
        source_list = "\n".join(f"- {s}" for s in sources[:5])
        response_text += f"\n\n📎 **Sources:**\n{source_list}"

    attachments = [_feedback_card(candidate_id)] if candidate_id else None
    await ctx.send_activity(_reply_with_change_chip(response_text, attachments))


def _call_brain(user_id: str, text: str) -> dict:
    """
    Synchronous wrapper around brain.process_query().
    Called via asyncio.to_thread() to avoid blocking the event loop.
    """
    from liftmind import brain
    return brain.process_query(user_id=user_id, query=text)


async def _handle_card_action(
    ctx: TurnContext,
    user_id: str,
    user_name: str,
    value: dict,
):
    """Handle Adaptive Card submit actions."""

    # ── Feedback card ─────────────────────────────────────────────────────────
    if "feedback_type" in value:
        feedback_type  = value.get("feedback_type", "skip")
        candidate_id   = value.get("candidate_id")

        if candidate_id:
            try:
                from liftmind.feedback import record_feedback_response, generate_followup_message
                fb_id = await asyncio.to_thread(
                    record_feedback_response, candidate_id, feedback_type, user_id
                )
                msg, _ = generate_followup_message(feedback_type, fb_id)
                await ctx.send_activity(msg or "Thanks for the feedback! 👍")
            except Exception as exc:
                logger.warning("Feedback record failed: %s", exc)
                await ctx.send_activity("Thanks for the feedback!")
        else:
            await ctx.send_activity("Thanks for the feedback! 👍")
        return

    # ── Model selection card ──────────────────────────────────────────────────
    if "card_action" in value:
        action = value.get("card_action")
        from liftmind.user_state import (
            set_user_model,
            clear_user_model,
            pop_pending_question,
        )

        # New Adaptive Card picker: one button per model
        if action == "pick_model":
            chosen = (value.get("model") or "").strip()
            if chosen == "__other__":
                await asyncio.to_thread(clear_user_model, user_id)
                await ctx.send_activity(_reply_with_change_chip(
                    "No problem — type your question and I'll search across all "
                    "manuals. If I need to narrow down, I'll ask."
                ))
                return
            if not chosen:
                await ctx.send_activity("No model selected — type `/reset` to try again.")
                return
            await asyncio.to_thread(set_user_model, user_id, chosen)
            pending = await asyncio.to_thread(pop_pending_question, user_id)
            if pending:
                await _answer_with_brain(ctx, user_id, pending, chosen)
            else:
                await ctx.send_activity(_reply_with_change_chip(
                    f"Got it — **{chosen}**. How can I help?"
                ))
            return

        # Legacy dropdown card (kept for backward-compat)
        if action == "clear_model":
            await asyncio.to_thread(clear_user_model, user_id)
            await ctx.send_activity("Model cleared — I'll search across all manuals. 🔍")

        elif action == "set_model":
            chosen = value.get("lift_model_choice", "").strip()
            if chosen:
                await asyncio.to_thread(set_user_model, user_id, chosen)
                await ctx.send_activity(
                    f"Model set to **{chosen}**. "
                    "Your questions will now be focused on that model's documentation. 🎯"
                )
            else:
                await ctx.send_activity("No model selected — use `/model` to try again.")
        return


# ── aiohttp routes ─────────────────────────────────────────────────────────────

class _LiftMindBot(Bot):
    """Minimal Bot implementation wrapping the turn logic."""

    async def on_turn(self, ctx: TurnContext):
        if ctx.activity.type == ActivityTypes.message:
            await on_message_activity(ctx)
        elif ctx.activity.type == ActivityTypes.conversation_update:
            if ctx.activity.members_added:
                for member in ctx.activity.members_added:
                    if member.id != ctx.activity.recipient.id:
                        await ctx.send_activity(
                            "👋 **Welcome to LiftMind!**\n\n"
                            "I'm a lift technician assistant for Lift Shop staff. "
                            "Ask me about fault codes, specs, wiring, or procedures "
                            "for any of our lift models.\n\n"
                            "Type `/help` to see what I can do, or just ask your question."
                        )


_bot = _LiftMindBot()


async def messages(request: web.Request) -> web.Response:
    """Main Bot Framework webhook endpoint (CloudAdapter)."""
    try:
        response = await adapter.process(request, _bot)
        return response or web.Response(status=200)
    except Exception as exc:
        logger.exception("CloudAdapter.process failed: %s", exc)
        return web.Response(status=500, text=f"adapter error: {exc}")


async def health(request: web.Request) -> web.Response:
    """Health check endpoint for Azure App Service."""
    return web.json_response({"status": "ok", "service": "liftmind-teams-bot"})


# ── App startup ────────────────────────────────────────────────────────────────

async def init_app() -> web.Application:
    """Initialise the aiohttp application."""
    # Ensure whitelist table exists before accepting traffic
    try:
        _ensure_whitelist_table()
    except Exception as exc:
        logger.warning("Whitelist table init failed (will retry on first request): %s", exc)

    # user_state schema is initialised lazily on first DB op (see user_state._ensure_schema)
    # so a slow/unreachable Postgres never blocks app startup.

    app = web.Application()
    app.router.add_post("/api/messages", messages)
    app.router.add_get("/health", health)
    app.router.add_get("/", health)
    return app


def main():
    app_coro = init_app()
    web.run_app(app_coro, port=PORT)


if __name__ == "__main__":
    main()

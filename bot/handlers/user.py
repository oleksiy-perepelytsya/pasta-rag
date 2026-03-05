import logging
import re
from datetime import datetime

from telegram import Update
from telegram.ext import ContextTypes


def _md_to_html(text: str) -> str:
    """Convert common LLM markdown output to Telegram-safe HTML."""
    # Escape existing HTML special chars first
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    # Code blocks (``` ... ```) — must come before inline code
    text = re.sub(r"```(?:\w+)?\n?(.*?)```", r"<pre>\1</pre>", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    # Bold: **text** or __text__
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    # Italic: *text* or _text_
    text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
    text = re.sub(r"(?<!\w)_(.+?)_(?!\w)", r"<i>\1</i>", text)
    # Headers → bold
    text = re.sub(r"^#{1,6}\s+(.+)$", r"<b>\1</b>", text, flags=re.MULTILINE)
    return text

from config import settings
from models.schemas import Message, User
from services.llm import LLMClient
from services.session_manager import SessionManager
from storage.mongo import MongoStore

logger = logging.getLogger(__name__)

UPLOAD_STATE = "AWAITING_UPLOAD"
SETPROMPT_STATE = "AWAITING_SETPROMPT"
SETMODEL_STATE = "AWAITING_SETMODEL"
SEARCH_STATE = "AWAITING_SEARCH"


class UserHandlers:
    def __init__(
        self,
        mongo: MongoStore,
        llm: LLMClient,
        session_mgr: SessionManager,
    ):
        self.mongo = mongo
        self.llm = llm
        self.session_mgr = session_mgr

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        tg_user = update.effective_user
        db_user = await self._register_user(tg_user)
        admin_note = " (you are an admin)" if db_user.is_admin else ""
        await update.message.reply_text(
            f"Hello, {tg_user.first_name}!{admin_note}\n\nJust send me a message to start chatting."
        )

    async def handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        # Delegate to admin handler states if applicable (group=0 handlers also run)
        state = context.user_data.get("state")
        if state == UPLOAD_STATE:
            # Upload text path is handled by admin handler (group=0); skip here
            return

        tg_user = update.effective_user
        text = update.message.text

        db_user = await self._register_user(tg_user)
        session = await self.session_mgr.get_or_create_session(tg_user.id)

        # Store user message
        user_msg = Message(
            session_id=session.session_id,
            telegram_id=tg_user.id,
            role="user",
            content=text,
        )
        await self.mongo.save_message(user_msg)
        await self.session_mgr.touch_session(session)

        # Build context (system prompt + RAG + session history)
        system_prompt, messages = await self.session_mgr.build_context(
            tg_user.id, text, session
        )
        messages.append({"role": "user", "content": text})

        # Determine model and token limit
        model = await self.mongo.get_config("llm_model") or settings.default_llm_model
        max_tokens_cfg = await self.mongo.get_config("max_output_tokens")
        max_output_tokens = int(max_tokens_cfg) if max_tokens_cfg else 800

        # Call LLM
        await update.message.chat.send_action("typing")
        try:
            reply = await self.llm.call_model(model, messages, system_prompt, max_output_tokens)
        except Exception as e:
            logger.error(f"LLM call failed: {e}", exc_info=True)
            await update.message.reply_text(
                "Sorry, I encountered an error generating a response. Please try again."
            )
            return

        # Store assistant message
        asst_msg = Message(
            session_id=session.session_id,
            telegram_id=tg_user.id,
            role="assistant",
            content=reply,
        )
        await self.mongo.save_message(asst_msg)
        await self.session_mgr.touch_session(session)

        # Convert markdown to Telegram HTML then send
        html = _md_to_html(reply)
        chunks = [html[i:i + 4096] for i in range(0, len(html), 4096)]
        for chunk in chunks:
            try:
                await update.message.reply_text(chunk, parse_mode="HTML")
            except Exception:
                await update.message.reply_text(chunk)

    async def _register_user(self, tg_user) -> User:
        count = await self.mongo.count_users()
        is_first = (count == 0)
        user = User(
            telegram_id=tg_user.id,
            username=tg_user.username,
            first_name=tg_user.first_name or "User",
            is_admin=is_first,
            last_active=datetime.utcnow(),
        )
        return await self.mongo.upsert_user(user)

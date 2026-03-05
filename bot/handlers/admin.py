import logging

from telegram import Update
from telegram.ext import ContextTypes

from bot.keyboards import Keyboards
from models.schemas import Document
from services.doc_loader import (
    chunk_text,
    extract_gdrive_file_id,
    extract_text_from_bytes,
    fetch_gdrive,
    fetch_url,
)
from services.session_manager import SessionManager
from storage.chroma import ChromaStore
from storage.mongo import MongoStore

logger = logging.getLogger(__name__)

UPLOAD_STATE = "AWAITING_UPLOAD"
SETPROMPT_STATE = "AWAITING_SETPROMPT"
SETMODEL_STATE = "AWAITING_SETMODEL"
SEARCH_STATE = "AWAITING_SEARCH"


class AdminHandlers:
    def __init__(
        self,
        mongo: MongoStore,
        chroma: ChromaStore,
        session_mgr: SessionManager,
    ):
        self.mongo = mongo
        self.chroma = chroma
        self.session_mgr = session_mgr

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _require_admin(self, update: Update) -> bool:
        user = await self.mongo.get_user(update.effective_user.id)
        if not user or not user.is_admin:
            await update.effective_message.reply_text("⛔ Admin only.")
            return False
        return True

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    async def cmd_admin(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        await update.message.reply_text("Admin panel:", reply_markup=Keyboards.admin_panel())

    async def cmd_setprompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        if context.args:
            prompt = " ".join(context.args)
            await self.mongo.set_config("system_prompt", prompt, update.effective_user.id)
            await update.message.reply_text("✅ System prompt updated.")
        else:
            context.user_data["state"] = SETPROMPT_STATE
            await update.message.reply_text(
                "Send the new system prompt as your next message.",
                reply_markup=Keyboards.cancel_input(),
            )

    async def cmd_setmodel(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        if context.args:
            model = context.args[0]
            await self.mongo.set_config("llm_model", model, update.effective_user.id)
            await update.message.reply_text(f"✅ Model set to: `{model}`", parse_mode="Markdown")
        else:
            context.user_data["state"] = SETMODEL_STATE
            current = await self.mongo.get_config("llm_model") or "not set"
            await update.message.reply_text(
                f"Current model: `{current}`\n\nSend the new model name "
                f"(e.g. `gemini-2.0-flash`, `claude-3-7-sonnet-20250219`, `gpt-4o`).",
                parse_mode="Markdown",
                reply_markup=Keyboards.cancel_input(),
            )

    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        await self._send_stats(update.message)

    async def _send_stats(self, message) -> None:
        user_count = await self.mongo.count_users()
        msg_count = await self.mongo.count_messages()
        session_count = await self.mongo.count_sessions()
        vectorized_count = await self.mongo.count_vectorized_sessions()
        doc_count = await self.mongo.count_documents()
        kb_count = await self.chroma.count_kb_embeddings()

        text = (
            "📊 *Statistics*\n\n"
            f"👤 Users: `{user_count}`\n"
            f"💬 Messages: `{msg_count}`\n"
            f"🗂 Sessions: `{session_count}` (vectorized: `{vectorized_count}`)\n"
            f"📄 Documents: `{doc_count}`\n"
            f"🧠 KB embeddings: `{kb_count}`"
        )
        await message.reply_text(text, parse_mode="Markdown")

    async def cmd_search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        if context.args:
            query = " ".join(context.args)
            await self._do_search(update.message, query)
        else:
            context.user_data["state"] = SEARCH_STATE
            await update.message.reply_text(
                "Send your search query.",
                reply_markup=Keyboards.cancel_input(),
            )

    async def _do_search(self, message, query: str) -> None:
        mongo_results = await self.mongo.search_messages(query, limit=5)
        kb_results = await self.chroma.query_knowledge_base(query, n_results=5)

        lines = [f"🔍 *Search results for:* `{query}`\n"]

        if mongo_results:
            lines.append("*MongoDB messages:*")
            for i, msg in enumerate(mongo_results, 1):
                snippet = msg.content[:200].replace("\n", " ")
                lines.append(f"{i}. [{msg.role}] {snippet}")
        else:
            lines.append("*MongoDB messages:* none found")

        lines.append("")

        if kb_results:
            lines.append("*Knowledge base:*")
            for i, hit in enumerate(kb_results, 1):
                meta = hit.get("metadata", {})
                fname = meta.get("filename", "?")
                snippet = hit.get("content", "")[:200].replace("\n", " ")
                score = hit.get("score", 0)
                lines.append(f"{i}. [{fname}] (score: {score:.2f}) {snippet}")
        else:
            lines.append("*Knowledge base:* none found")

        reply = "\n".join(lines)
        # Telegram message limit
        if len(reply) > 4000:
            reply = reply[:4000] + "\n…"
        await message.reply_text(reply, parse_mode="Markdown")

    async def cmd_promote(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        if not context.args:
            await update.message.reply_text("Usage: /promote <user_id or @username>")
            return

        target = context.args[0].lstrip("@")
        # Try as integer telegram_id
        try:
            target_id = int(target)
        except ValueError:
            user = await self.mongo.get_user_by_username(target)
            target_id = user.telegram_id if user else None

        if target_id is None:
            await update.message.reply_text("❌ User not found. They must have chatted with the bot first.")
            return

        ok = await self.mongo.promote_user(target_id)
        if ok:
            await update.message.reply_text(f"✅ User `{target_id}` is now an admin.", parse_mode="Markdown")
        else:
            await update.message.reply_text("❌ User not found in database.")

    async def cmd_settokens(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        if context.args:
            try:
                tokens = int(context.args[0])
            except ValueError:
                await update.message.reply_text("Usage: /settokens <number> (e.g. /settokens 1200)")
                return
            await self.mongo.set_config("max_output_tokens", str(tokens), update.effective_user.id)
            await update.message.reply_text(f"✅ Max output tokens set to `{tokens}`", parse_mode="Markdown")
        else:
            current = await self.mongo.get_config("max_output_tokens") or "800"
            await update.message.reply_text(
                f"Current max output tokens: `{current}`\n\nUsage: /settokens <number>",
                parse_mode="Markdown",
            )

    async def cmd_upload(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not await self._require_admin(update):
            return
        context.user_data["state"] = UPLOAD_STATE
        await update.message.reply_text(
            "📂 *Upload mode*\n\n"
            "Send a file (PDF, TXT, MD, DOCX) or paste a URL / Google Drive link.",
            parse_mode="Markdown",
            reply_markup=Keyboards.cancel_upload(),
        )

    # ------------------------------------------------------------------
    # Upload: file handler (group=0)
    # ------------------------------------------------------------------

    async def handle_upload_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        # Admins can drop files at any time without needing /upload first
        user = await self.mongo.get_user(update.effective_user.id)
        if not user or not user.is_admin:
            return  # silently ignore files from non-admins


        doc = update.message.document
        await update.message.reply_text(f"⏳ Processing `{doc.file_name}`…", parse_mode="Markdown")

        tg_file = await context.bot.get_file(doc.file_id)
        data = bytes(await tg_file.download_as_bytearray())

        await self._process_and_store(
            update, context,
            data=data,
            filename=doc.file_name,
            source_type="file",
            source=doc.file_id,
        )

    # ------------------------------------------------------------------
    # Upload: text/URL handler (group=0)
    # ------------------------------------------------------------------

    async def handle_upload_text(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        state = context.user_data.get("state")
        text = update.message.text.strip()

        if state == SETPROMPT_STATE:
            if not await self._require_admin(update):
                return
            await self.mongo.set_config("system_prompt", text, update.effective_user.id)
            context.user_data["state"] = None
            await update.message.reply_text("✅ System prompt updated.")
            return

        if state == SETMODEL_STATE:
            if not await self._require_admin(update):
                return
            await self.mongo.set_config("llm_model", text, update.effective_user.id)
            context.user_data["state"] = None
            await update.message.reply_text(f"✅ Model set to: `{text}`", parse_mode="Markdown")
            return

        if state == SEARCH_STATE:
            if not await self._require_admin(update):
                return
            context.user_data["state"] = None
            await self._do_search(update.message, text)
            return

        if state == UPLOAD_STATE:
            if not await self._require_admin(update):
                return
            gdrive_id = extract_gdrive_file_id(text)
            if gdrive_id:
                await update.message.reply_text("⏳ Downloading from Google Drive…")
                try:
                    data, filename = await fetch_gdrive(gdrive_id)
                except Exception as e:
                    await update.message.reply_text(f"❌ GDrive download failed: {e}")
                    return
                await self._process_and_store(
                    update, context,
                    data=data, filename=filename,
                    source_type="gdrive", source=text,
                )
            elif text.startswith(("http://", "https://")):
                await update.message.reply_text("⏳ Fetching URL…")
                try:
                    content = await fetch_url(text)
                except Exception as e:
                    await update.message.reply_text(f"❌ URL fetch failed: {e}")
                    return
                await self._store_text_chunks(
                    update, context,
                    text=content, filename=text,
                    source_type="url", source=text,
                )
            else:
                await update.message.reply_text(
                    "Please send a file, a direct URL, or a Google Drive link."
                )

    # ------------------------------------------------------------------
    # Shared processing helpers
    # ------------------------------------------------------------------

    async def _process_and_store(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        data: bytes,
        filename: str,
        source_type: str,
        source: str,
    ) -> None:
        try:
            text = extract_text_from_bytes(data, filename)
        except Exception as e:
            await update.message.reply_text(f"❌ Text extraction failed: {e}")
            return
        await self._store_text_chunks(
            update, context,
            text=text, filename=filename,
            source_type=source_type, source=source,
        )

    async def _store_text_chunks(
        self,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
        *,
        text: str,
        filename: str,
        source_type: str,
        source: str,
    ) -> None:
        if not text.strip():
            await update.message.reply_text("❌ No text could be extracted from this source.")
            return

        chunks = chunk_text(text)
        doc = Document(
            filename=filename,
            source_type=source_type,
            source=source,
            uploaded_by=update.effective_user.id,
            chunk_count=len(chunks),
        )
        doc = await self.mongo.save_document(doc)

        await update.message.reply_text(f"⏳ Embedding {len(chunks)} chunks…")
        stored = await self.chroma.add_chunks(doc.doc_id, filename, source_type, chunks)
        await self.mongo.update_document_vectorized(doc.doc_id, stored)

        context.user_data["state"] = None
        await update.message.reply_text(
            f"✅ Stored *{filename}*\n`{stored}` chunks added to knowledge base.",
            parse_mode="Markdown",
        )

    # ------------------------------------------------------------------
    # Callback query handler
    # ------------------------------------------------------------------

    async def handle_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        query = update.callback_query
        await query.answer()
        data = query.data

        if data == "upload:cancel":
            context.user_data["state"] = None
            await query.edit_message_text("Upload cancelled.")

        elif data == "admin:cancel_input":
            context.user_data["state"] = None
            await query.edit_message_text("Cancelled.")

        elif data == "admin:stats":
            await query.edit_message_text("Fetching stats…")
            await self._send_stats(query.message)

        elif data == "admin:setprompt":
            if not await self._require_admin(update):
                return
            context.user_data["state"] = SETPROMPT_STATE
            await query.edit_message_text(
                "Send the new system prompt as your next message.",
            )

        elif data == "admin:setmodel":
            if not await self._require_admin(update):
                return
            context.user_data["state"] = SETMODEL_STATE
            current = await self.mongo.get_config("llm_model") or "not set"
            await query.edit_message_text(
                f"Current model: {current}\n\nSend the new model name "
                f"(e.g. gemini-2.0-flash, claude-3-7-sonnet-20250219, gpt-4o).",
            )

        elif data == "admin:upload":
            if not await self._require_admin(update):
                return
            context.user_data["state"] = UPLOAD_STATE
            await query.edit_message_text(
                "📂 Upload mode\n\nSend a file (PDF, TXT, MD, DOCX) or paste a URL / Google Drive link.",
            )

        elif data == "admin:search":
            if not await self._require_admin(update):
                return
            context.user_data["state"] = SEARCH_STATE
            await query.edit_message_text("Send your search query.")

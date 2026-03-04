import asyncio
import logging

from telegram import Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    MessageHandler,
    filters,
)

from bot.handlers.admin import AdminHandlers
from bot.handlers.user import UserHandlers
from config import settings
from services.llm import LLMClient
from services.session_manager import SessionManager, vectorize_stale_sessions_loop
from storage.chroma import ChromaStore
from storage.mongo import MongoStore

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _error_handler(update: object, context) -> None:
    logger.error("Unhandled exception", exc_info=context.error)
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "An internal error occurred. Please try again."
            )
        except Exception:
            pass


async def main() -> None:
    # ------------------------------------------------------------------
    # 1. Init storage
    # ------------------------------------------------------------------
    mongo = MongoStore(settings.mongodb_uri, settings.mongodb_database)
    await mongo.init_db()

    chroma = ChromaStore(settings.chroma_host, settings.chroma_port)
    await chroma.initialize()

    # ------------------------------------------------------------------
    # 2. Init services
    # ------------------------------------------------------------------
    llm = LLMClient()
    session_mgr = SessionManager(mongo, chroma)

    # ------------------------------------------------------------------
    # 3. Init handlers
    # ------------------------------------------------------------------
    user_h = UserHandlers(mongo, llm, session_mgr)
    admin_h = AdminHandlers(mongo, chroma, session_mgr)

    # ------------------------------------------------------------------
    # 4. Build Telegram application
    # ------------------------------------------------------------------
    app = Application.builder().token(settings.telegram_bot_token).build()

    # User commands
    app.add_handler(CommandHandler("start", user_h.cmd_start))

    # Admin commands
    app.add_handler(CommandHandler("admin", admin_h.cmd_admin))
    app.add_handler(CommandHandler("setprompt", admin_h.cmd_setprompt))
    app.add_handler(CommandHandler("setmodel", admin_h.cmd_setmodel))
    app.add_handler(CommandHandler("stats", admin_h.cmd_stats))
    app.add_handler(CommandHandler("search", admin_h.cmd_search))
    app.add_handler(CommandHandler("upload", admin_h.cmd_upload))
    app.add_handler(CommandHandler("promote", admin_h.cmd_promote))

    # group=0: upload/state handlers — run first for every message
    app.add_handler(
        MessageHandler(filters.Document.ALL, admin_h.handle_upload_document),
        group=0,
    )
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, admin_h.handle_upload_text),
        group=0,
    )

    # group=1: regular chat handler
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, user_h.handle_message),
        group=1,
    )

    # Callback queries (inline keyboards)
    app.add_handler(CallbackQueryHandler(admin_h.handle_callback))

    # Error handler
    app.add_error_handler(_error_handler)

    # ------------------------------------------------------------------
    # 5. Start polling
    # ------------------------------------------------------------------
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("pasta-rag bot is running")

    # ------------------------------------------------------------------
    # 6. Background vectorization task
    # ------------------------------------------------------------------
    bg_task = asyncio.create_task(vectorize_stale_sessions_loop(mongo, chroma))

    # ------------------------------------------------------------------
    # 7. Run until interrupted
    # ------------------------------------------------------------------
    stop_event = asyncio.Event()
    try:
        await stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        logger.info("Shutting down…")
        bg_task.cancel()
        try:
            await bg_task
        except asyncio.CancelledError:
            pass
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())

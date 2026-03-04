from telegram import InlineKeyboardButton, InlineKeyboardMarkup


class Keyboards:
    @staticmethod
    def admin_panel() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [
                InlineKeyboardButton("📊 Stats", callback_data="admin:stats"),
                InlineKeyboardButton("✏️ Set Prompt", callback_data="admin:setprompt"),
            ],
            [
                InlineKeyboardButton("🤖 Set Model", callback_data="admin:setmodel"),
                InlineKeyboardButton("📂 Upload Doc", callback_data="admin:upload"),
            ],
            [
                InlineKeyboardButton("🔍 Search", callback_data="admin:search"),
            ],
        ])

    @staticmethod
    def cancel_upload() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel Upload", callback_data="upload:cancel")],
        ])

    @staticmethod
    def cancel_input() -> InlineKeyboardMarkup:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton("❌ Cancel", callback_data="admin:cancel_input")],
        ])

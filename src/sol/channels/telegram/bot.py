import httpx
import structlog
from aiogram import Bot, Dispatcher, types
from aiogram.enums import ChatAction

from sol.config import settings

log = structlog.get_logger()


async def handle_message(message: types.Message, bot: Bot) -> None:
    """Forward a Telegram message to the gateway and reply with the response."""
    if not message.text:
        return

    gateway_url = f"http://{settings.gateway.host}:{settings.gateway.port}/v1/messages"

    await bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            response = await client.post(
                gateway_url,
                json={
                    "channel": "telegram",
                    "user_id": str(message.from_user.id),
                    "text": message.text,
                },
            )
        except httpx.HTTPError:
            log.error("telegram.gateway_error", error="Failed to reach gateway")
            await message.reply("Sorry, I couldn't reach the server. Is Sol running?")
            return

    if response.status_code == 200:
        data = response.json()
        response_text = data.get("response_text", "")
        if response_text:
            await message.reply(response_text)
        else:
            await message.reply("[No response from Sol]")
    else:
        log.error("telegram.gateway_error", status=response.status_code)
        await message.reply("Sorry, something went wrong.")


async def start_bot() -> None:
    """Start the Telegram bot with long-polling."""
    bot = Bot(token=settings.channels.telegram.bot_token)
    dp = Dispatcher()
    dp.message.register(handle_message)

    log.info("telegram.starting")
    await dp.start_polling(bot)

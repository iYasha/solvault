from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from sol.channels.telegram.bot import handle_message


def _make_telegram_message(text: str = "Hello Sol", user_id: int = 12345, chat_id: int = 67890) -> MagicMock:
    """Create a mock aiogram Message."""
    message = AsyncMock()
    message.text = text
    message.from_user = MagicMock()
    message.from_user.id = user_id
    message.chat = MagicMock()
    message.chat.id = chat_id
    message.reply = AsyncMock()
    return message


class TestTelegramMessageHandler:
    """Telegram handler should forward messages to the gateway and reply with the response."""

    @patch("sol.channels.telegram.bot.httpx.AsyncClient")
    async def test_posts_to_gateway_and_replies(self, mock_client_cls):
        """Given a text message, when handled, then POST to gateway and reply with response_text."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response_text": "Hi there!", "session_id": "s1", "message_id": "m1"}

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        message = _make_telegram_message("Hello Sol")
        bot = AsyncMock()

        await handle_message(message, bot)

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["text"] == "Hello Sol", "Should forward message text"
        assert call_kwargs[1]["json"]["channel"] == "telegram", "Should set channel to telegram"
        message.reply.assert_called_once_with("Hi there!"), "Should reply with response_text"

    @patch("sol.channels.telegram.bot.httpx.AsyncClient")
    async def test_handles_gateway_error(self, mock_client_cls):
        """Given gateway returning 500, when handled, then user receives error message."""
        mock_response = MagicMock()
        mock_response.status_code = 500

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        message = _make_telegram_message()
        bot = AsyncMock()

        await handle_message(message, bot)

        message.reply.assert_called_once_with("Sorry, something went wrong.")

    async def test_ignores_non_text_messages(self):
        """Given a message without text, when handled, then no gateway call is made."""
        message = AsyncMock()
        message.text = None
        message.reply = AsyncMock()
        bot = AsyncMock()

        await handle_message(message, bot)

        message.reply.assert_not_called(), "Should not reply to non-text messages"

    @patch("sol.channels.telegram.bot.httpx.AsyncClient")
    async def test_handles_connection_error(self, mock_client_cls):
        """Given gateway unreachable, when handled, then user receives connection error message."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        message = _make_telegram_message()
        bot = AsyncMock()

        await handle_message(message, bot)

        message.reply.assert_called_once_with("Sorry, I couldn't reach the server. Is Sol running?")

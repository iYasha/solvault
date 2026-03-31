import tiktoken

from sol.session.models import ChatMessage


def count_tokens(text: str, model: str = "gpt-4") -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def apply_token_window(
    messages: list[ChatMessage],
    max_tokens: int,
    model: str = "gpt-4",
) -> list[ChatMessage]:
    """Return the most recent messages that fit within the token budget."""
    total = 0
    windowed: list[ChatMessage] = []

    for i, msg in enumerate(reversed(messages)):
        tokens = msg.token_count if msg.token_count > 0 else count_tokens(msg.content, model)
        if total + tokens > max_tokens and i > 0:
            break
        total += tokens
        windowed.append(msg)

    windowed.reverse()
    return windowed

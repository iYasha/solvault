from langchain_openai import ChatOpenAI

from sol.config import settings


def create_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for the local LM Studio endpoint."""
    return ChatOpenAI(
        model=settings.llm.model,
        base_url=settings.llm.endpoint,
        api_key=settings.llm.api_key or "not-needed",
        max_tokens=settings.llm.response_token_budget,
        temperature=0.7,
    )

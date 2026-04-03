from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from sol.config import settings


def get_llm() -> ChatOpenAI:
    """Create a ChatOpenAI instance configured for the local LM Studio endpoint."""
    return ChatOpenAI(
        model=settings.llm.model,
        base_url=settings.llm.endpoint,
        api_key=settings.llm.api_key or "not-needed",
        max_tokens=settings.llm.response_token_budget,
        temperature=0.7,
    )


def get_embeddings() -> OpenAIEmbeddings:
    """Create an OpenAIEmbeddings instance configured for the local embedding endpoint."""
    return OpenAIEmbeddings(
        model=settings.memory.embedding.model,
        base_url=settings.memory.embedding.endpoint,
        api_key=settings.memory.embedding.api_key or "not-needed",
        dimensions=settings.memory.embedding.dimensions,
        check_embedding_ctx_length=False,
    )


embeddings = get_embeddings()

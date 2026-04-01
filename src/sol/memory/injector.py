from sol.memory.retriever import RetrievalResult
from sol.session.token_window import count_tokens


class MemoryInjector:
    """Formats retrieved memory chunks for injection into the system prompt."""

    def build_memory_context(self, results: list[RetrievalResult], max_tokens: int = 5000) -> str:
        """Format retrieval results into a string for the system prompt.

        Returns an empty string if no results.
        """
        if not results:
            return ""

        lines = ["", "## Relevant Memories", ""]
        total_tokens = count_tokens("\n".join(lines))

        for result in results:
            entry = f"[{result.memory_type}]\n{result.content}"
            entry_tokens = count_tokens(entry)

            if total_tokens + entry_tokens > max_tokens:
                break

            lines.append(entry)
            lines.append("")
            total_tokens += entry_tokens

        # Only the header, no actual content
        if len(lines) <= 3:
            return ""

        return "\n".join(lines)

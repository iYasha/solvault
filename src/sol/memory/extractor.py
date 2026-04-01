import json
from datetime import UTC, datetime

import structlog
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from sol.memory.models import Memory
from sol.memory.schemas import MemoryFact
from sol.memory.store import MemoryStore, _float_list_to_blob

log = structlog.get_logger()

EXTRACTION_SYSTEM_PROMPT = """\
You are a memory extraction agent for Sol, a privacy-first personal AI assistant.

Your job is to analyze a conversation exchange and extract facts worth remembering \
about the user. These facts will be stored as markdown files and used to personalize \
future conversations.

## What to extract
- User preferences, habits, and working style
- Personal details: name, role, company, location, expertise
- Project context: what the user is working on, deadlines, goals
- Corrections and feedback: things the user corrected or confirmed
- References to external systems: tools, URLs, services they use

## What NOT to extract
- Ephemeral task details (one-off questions, debugging sessions)
- Information already in the conversation history (no need to duplicate)
- Code snippets or technical details that belong in code, not memory
- Anything the user explicitly asks not to remember

## Deduplication
You will be given a manifest of existing memories. Before creating a new fact, \
check if it already exists or is a minor variation. If it does:
- If the new information updates/refines the existing fact, output an update
- If it's truly new, output a new fact
- If it's already captured, skip it

## Output format
Respond with a JSON array of facts to save. Each fact is an object:
```json
[
  {
    "content": "The fact text to save",
    "type": "facts",
    "confidence": "confirmed",
    "tags": ["tag1", "tag2"],
    "expires_at": null
  }
]
```

- `type`: one of "user", "work", "facts", "conversations", "events"
- `confidence`: "confirmed" (user stated directly), "inferred" (deduced from context), \
"uncertain" (might be wrong)
- `tags`: relevant categories
- `expires_at`: ISO datetime if the fact is time-bound (e.g. "on vacation until Friday"), \
null if permanent. Use the current date provided below to compute absolute dates.

If there is nothing worth extracting, respond with an empty array: []
"""

DEDUP_PROMPT = """\
You are deciding whether a new memory is about the SAME specific topic as an \
existing memory.

Original conversation:
User: {user_message}
Assistant: {assistant_response}

Existing memory:
{existing}

New memory:
{new}

Rules:
- Only merge if both memories are about the SAME specific topic.
- Do NOT merge unrelated facts together. A person's name and their location are \
DIFFERENT topics — respond with "new".
- If the new memory updates or corrects the existing one on the same topic, merge them.
- If the existing memory already captures this fact adequately, skip it.

Respond with exactly one of:
- "replace: <merged content>" — same topic, new info updates/refines it. Write the \
final merged single-topic content after "replace: ".
- "skip" — existing memory already captures this fact.
- "new" — different topic, should be stored separately.

Respond with just "replace: ...", "skip", or "new". Nothing else.\
"""

EXTRACTION_USER_PROMPT = """\
Current date: {current_date}

## Existing memories
{manifest}

## Conversation exchange

**User:** {user_message}

**Assistant:** {assistant_response}

Analyze this exchange and extract any facts worth remembering.\
"""


class MemoryExtractor:
    """Extracts memorable facts from conversation turns using an LLM."""

    def __init__(self, llm: ChatOpenAI, store: MemoryStore) -> None:
        self.llm = llm
        self.store = store

    async def extract(self, user_message: str, assistant_response: str) -> list[str]:
        """Analyze a conversation exchange, extract facts, and save them.

        Returns list of memory IDs for newly created memories.
        """
        manifest = await self.store.build_manifest()
        messages = self._build_prompt(user_message, assistant_response, manifest)

        try:
            response = await self.llm.ainvoke(messages)
            content = str(response.content).strip()
        except Exception:
            log.warning("memory.extraction_failed", exc_info=True)
            return []

        facts = self._parse_response(content)
        return await self._save_facts(facts, user_message, assistant_response)

    def _build_prompt(self, user_message: str, assistant_response: str, manifest: str) -> list[BaseMessage]:
        """Build the extraction prompt with conversation context and manifest."""
        return [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(
                content=EXTRACTION_USER_PROMPT.format(
                    current_date=datetime.now(UTC).strftime("%Y-%m-%d"),
                    manifest=manifest or "(no existing memories)",
                    user_message=user_message,
                    assistant_response=assistant_response,
                ),
            ),
        ]

    def _parse_response(self, content: str) -> list[MemoryFact]:
        """Parse LLM JSON response into MemoryFact objects."""
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            raw_facts = json.loads(content)
        except json.JSONDecodeError:
            log.warning("memory.extraction_parse_failed", content=content[:200])
            return []

        if not isinstance(raw_facts, list):
            return []

        facts = []
        for raw in raw_facts:
            if not isinstance(raw, dict) or not raw.get("content"):
                continue
            try:
                facts.append(MemoryFact(**raw))
            except Exception:
                log.warning("memory.fact_validation_failed", raw=raw)
        return facts

    async def _save_facts(self, facts: list[MemoryFact], user_message: str, assistant_response: str) -> list[str]:
        """Save extracted facts, deduplicating against existing memories."""
        created_ids: list[str] = []
        for fact in facts:
            if not fact.content.strip():
                continue

            memory_id = await self._dedup_and_save(fact, user_message, assistant_response)
            if memory_id:
                created_ids.append(memory_id)
        return created_ids

    async def _dedup_and_save(self, fact: MemoryFact, user_message: str, assistant_response: str) -> str | None:
        """Check for similar memory, ask LLM to merge/skip, or save new."""
        # Embed the new fact
        embedding_vec = await self.store.embeddings.aembed_documents([fact.content])
        embedding_blob = _float_list_to_blob(embedding_vec[0])

        # Find similar existing memory
        similar = await self.store.find_similar(fact.content, embedding_blob)
        if similar is None:
            memory = await self.store.save(fact)
            log.info("memory.created", memory_id=memory.id, type=fact.type, content=fact.content)
            return memory.id

        # Ask LLM whether to replace, skip, or save as new
        decision = await self._ask_merge_decision(similar, fact, user_message, assistant_response)

        if decision.startswith("replace:"):
            merged_content = decision[len("replace:") :].strip()
            if not merged_content:
                merged_content = fact.content
            memory = await self.store.update(similar.id, merged_content, fact)
            log.info("memory.merged", memory_id=memory.id, content=merged_content)
            return memory.id

        if decision == "new":
            memory = await self.store.save(fact)
            log.info("memory.created", memory_id=memory.id, type=fact.type, content=fact.content)
            return memory.id

        log.info("memory.skipped_duplicate", existing_id=similar.id, new_content=fact.content[:100])
        return None

    async def _ask_merge_decision(
        self,
        existing: Memory,
        new_fact: MemoryFact,
        user_message: str,
        assistant_response: str,
    ) -> str:
        """Ask the LLM whether to replace, skip, or save as new."""
        try:
            response = await self.llm.ainvoke(
                [
                    HumanMessage(
                        content=DEDUP_PROMPT.format(
                            existing=existing.content,
                            new=new_fact.content,
                            user_message=user_message,
                            assistant_response=assistant_response,
                        ),
                    ),
                ],
            )
            return str(response.content).strip().lower()
        except Exception:
            log.warning("memory.dedup_llm_failed", exc_info=True)
            return "skip"

import json
import re
import struct

import sqlalchemy as sa
import structlog
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from sol.config import MemoryConfig
from sol.memory.models import Memory
from sol.memory.schemas import MemoryFact
from sol.session.token_window import count_tokens

log = structlog.get_logger()


def _float_list_to_blob(vec: list[float]) -> bytes:
    """Pack a float list into a little-endian float32 blob for sqlite-vec."""
    return struct.pack(f"<{len(vec)}f", *vec)


class MemoryStore:
    """CRUD for memories stored in SQLite with FTS5 + sqlite-vec indexes."""

    def __init__(self, db: AsyncSession, embeddings: OpenAIEmbeddings, config: MemoryConfig) -> None:
        self.db = db
        self.embeddings = embeddings
        self.config = config

    async def save(self, fact: MemoryFact) -> Memory:
        """Save a memory: embed, insert into memories table + FTS5 + sqlite-vec."""
        embedding_vec = await self.embeddings.aembed_documents([fact.content])
        embedding_blob = _float_list_to_blob(embedding_vec[0])

        memory = Memory(
            type=fact.type,
            content=fact.content,
            confidence=fact.confidence,
            source="conversation",
            tags_json=json.dumps(fact.tags),
            embedding=embedding_blob,
            token_count=count_tokens(fact.content),
            expires_at=fact.expires_at,
        )
        self.db.add(memory)
        await self.db.flush()

        await self.db.execute(
            sa.text("INSERT INTO memories_fts(content, memory_id) VALUES (:content, :memory_id)"),
            {"content": fact.content, "memory_id": memory.id},
        )

        await self.db.execute(
            sa.text("INSERT INTO memories_vec(memory_id, embedding) VALUES (:memory_id, :embedding)"),
            {"memory_id": memory.id, "embedding": embedding_blob},
        )

        log.info("memory.saved", memory_id=memory.id, type=fact.type)
        return memory

    async def find_similar(
        self,
        content: str,
        embedding_blob: bytes,
        vector_threshold: float = 0.5,
        max_content_length: int = 500,
    ) -> Memory | None:
        """Find the most similar existing memory via vector search or keyword fallback.

        Returns None if no match above threshold or candidate is too long.
        """
        memory = await self._find_by_vector(embedding_blob, vector_threshold)
        if memory:
            log.info("memory.dedup_match", method="vector", memory_id=memory.id, existing=memory.content[:100])
        else:
            memory = await self._find_by_keyword(content)
            if memory:
                log.info("memory.dedup_match", method="keyword", memory_id=memory.id, existing=memory.content[:100])
            else:
                log.info("memory.dedup_no_match", content=content[:100])

        if memory is None:
            return None

        # Don't merge into memories that are already too long
        if len(memory.content) > max_content_length:
            log.info("memory.skip_merge_too_long", memory_id=memory.id, length=len(memory.content))
            return None

        return memory

    async def _find_by_vector(self, embedding_blob: bytes, threshold: float) -> Memory | None:
        """Find closest memory by vector similarity."""
        result = await self.db.execute(
            sa.text("""
                SELECT memory_id, distance
                FROM memories_vec
                WHERE embedding MATCH :query
                ORDER BY distance
                LIMIT 1
            """),
            {"query": embedding_blob},
        )
        row = result.first()
        if row is None:
            return None

        similarity = 1.0 - (row[1] / 2.0)
        log.info("memory.vector_similarity", memory_id=row[0], similarity=round(similarity, 3))
        if similarity < threshold:
            return None

        mem_result = await self.db.execute(sa.select(Memory).where(Memory.id == row[0]))
        return mem_result.scalar_one_or_none()

    async def _find_by_keyword(self, content: str) -> Memory | None:
        """Find closest memory by FTS5 keyword match."""
        safe_query = re.sub(r"[^\w\s]", "", content).strip()
        result = await self.db.execute(
            sa.text("""
                SELECT memory_id, rank
                FROM memories_fts
                WHERE memories_fts MATCH :query
                ORDER BY rank
                LIMIT 1
            """),
            {"query": safe_query},
        )
        row = result.first()
        if row is None:
            return None

        log.info("memory.keyword_match", memory_id=row[0], rank=round(row[1], 3))
        mem_result = await self.db.execute(sa.select(Memory).where(Memory.id == row[0]))
        return mem_result.scalar_one_or_none()

    async def update(self, memory_id: str, new_content: str, new_fact: MemoryFact) -> Memory:
        """Update an existing memory's content, re-embed, and update indexes."""
        embedding_vec = await self.embeddings.aembed_documents([new_content])
        embedding_blob = _float_list_to_blob(embedding_vec[0])

        await self.db.execute(
            sa.update(Memory)
            .where(Memory.id == memory_id)
            .values(
                content=new_content,
                type=new_fact.type,
                confidence=new_fact.confidence,
                tags_json=json.dumps(new_fact.tags),
                embedding=embedding_blob,
                token_count=count_tokens(new_content),
            ),
        )

        # Update FTS5
        await self.db.execute(
            sa.text("DELETE FROM memories_fts WHERE memory_id = :mid"),
            {"mid": memory_id},
        )
        await self.db.execute(
            sa.text("INSERT INTO memories_fts(content, memory_id) VALUES (:content, :mid)"),
            {"content": new_content, "mid": memory_id},
        )

        # Update sqlite-vec
        await self.db.execute(
            sa.text("DELETE FROM memories_vec WHERE memory_id = :mid"),
            {"mid": memory_id},
        )
        await self.db.execute(
            sa.text("INSERT INTO memories_vec(memory_id, embedding) VALUES (:mid, :embedding)"),
            {"mid": memory_id, "embedding": embedding_blob},
        )

        await self.db.flush()
        log.info("memory.updated", memory_id=memory_id)

        result = await self.db.execute(sa.select(Memory).where(Memory.id == memory_id))
        return result.scalar_one()

    async def delete(self, memory_id: str) -> None:
        """Delete a memory and its index entries."""
        await self.db.execute(
            sa.text("DELETE FROM memories_fts WHERE memory_id = :memory_id"),
            {"memory_id": memory_id},
        )
        await self.db.execute(
            sa.text("DELETE FROM memories_vec WHERE memory_id = :memory_id"),
            {"memory_id": memory_id},
        )
        await self.db.execute(sa.delete(Memory).where(Memory.id == memory_id))
        await self.db.flush()

    async def list_all(self) -> list[Memory]:
        """List all memories ordered by creation time."""
        result = await self.db.execute(sa.select(Memory).order_by(Memory.created_at.asc()))
        return list(result.scalars().all())

    async def build_manifest(self) -> str:
        """Build a text manifest of all memories for LLM dedup context."""
        memories = await self.list_all()
        lines = []
        for mem in memories:
            preview = mem.content[:150].replace("\n", " ").strip()
            lines.append(f"- [{mem.type}] ({mem.confidence}) {preview}")
        return "\n".join(lines)

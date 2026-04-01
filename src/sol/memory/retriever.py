import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import sqlalchemy as sa
import structlog
from langchain_openai import OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from sol.config import MemoryConfig
from sol.memory.models import Memory
from sol.memory.store import _float_list_to_blob

log = structlog.get_logger()


@dataclass
class RetrievalResult:
    """A single retrieved memory with its score."""

    content: str
    memory_type: str
    score: float


class MemoryRetriever:
    """Hybrid search over memories using sqlite-vec + FTS5 with RRF fusion and temporal decay."""

    def __init__(self, db: AsyncSession, embeddings: OpenAIEmbeddings, config: MemoryConfig) -> None:
        self.db = db
        self.embeddings = embeddings
        self.config = config

    async def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """Run hybrid search and return top-K results ranked by RRF with temporal decay."""
        top_k = top_k or self.config.search_top_k
        candidate_k = top_k * 4

        query_embedding = await self.embeddings.aembed_query(query)

        vector_results = await self._vector_search(query_embedding, candidate_k)
        keyword_results = await self._keyword_search(query, candidate_k)

        if not vector_results and not keyword_results:
            return []

        return await self._rrf_merge(vector_results, keyword_results, top_k)

    async def _vector_search(self, query_embedding: list[float], k: int) -> list[tuple[str, float]]:
        """KNN search via sqlite-vec. Returns list of (memory_id, distance)."""
        blob = _float_list_to_blob(query_embedding)
        result = await self.db.execute(
            sa.text("""
                SELECT memory_id, distance
                FROM memories_vec
                WHERE embedding MATCH :query
                ORDER BY distance
                LIMIT :k
            """),
            {"query": blob, "k": k},
        )
        return [(row[0], row[1]) for row in result.all()]

    async def _keyword_search(self, query: str, k: int) -> list[tuple[str, float]]:
        """BM25 search via FTS5. Returns list of (memory_id, rank_score)."""
        safe_query = re.sub(r"[^\w\s]", "", query).strip()
        if not safe_query:
            return []
        result = await self.db.execute(
            sa.text("""
                SELECT memory_id, rank
                FROM memories_fts
                WHERE memories_fts MATCH :query
                ORDER BY rank
                LIMIT :k
            """),
            {"query": safe_query, "k": k},
        )
        return [(row[0], row[1]) for row in result.all()]

    async def _rrf_merge(
        self,
        vector_results: list[tuple[str, float]],
        keyword_results: list[tuple[str, float]],
        top_k: int,
        rrf_k: int = 60,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion to merge vector and keyword results."""
        scores: dict[str, float] = {}

        vector_weight = self.config.vector_weight
        text_weight = self.config.text_weight

        for rank, (memory_id, _distance) in enumerate(vector_results):
            scores[memory_id] = scores.get(memory_id, 0.0) + vector_weight / (rrf_k + rank + 1)

        for rank, (memory_id, _rank_score) in enumerate(keyword_results):
            scores[memory_id] = scores.get(memory_id, 0.0) + text_weight / (rrf_k + rank + 1)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return await self._resolve_memories(ranked)

    async def _resolve_memories(self, ranked: list[tuple[str, float]]) -> list[RetrievalResult]:
        """Look up memories, filter expired, apply temporal decay to scores."""
        if not ranked:
            return []

        memory_ids = [mid for mid, _ in ranked]
        score_map = dict(ranked)
        now = datetime.now(UTC)

        result = await self.db.execute(
            sa.select(Memory).where(Memory.id.in_(memory_ids)),
        )
        rows = {mem.id: mem for mem in result.scalars().all()}

        results = []
        for memory_id in memory_ids:
            if memory_id not in rows:
                continue
            mem = rows[memory_id]

            # Skip expired memories
            expires = (
                mem.expires_at.replace(tzinfo=UTC)
                if mem.expires_at and mem.expires_at.tzinfo is None
                else mem.expires_at
            )
            if expires and expires < now:
                continue

            score = score_map[memory_id] * self._decay_factor(mem, now)
            results.append(RetrievalResult(content=mem.content, memory_type=mem.type, score=score))

        # Re-sort after decay adjustment
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    def _decay_factor(self, mem: Memory, now: datetime) -> float:
        """Compute temporal decay multiplier. Permanent types always return 1.0."""
        if mem.type not in self.config.decaying_types:
            return 1.0

        created = mem.created_at.replace(tzinfo=UTC) if mem.created_at.tzinfo is None else mem.created_at
        age_days = (now - created).total_seconds() / 86400
        half_life = self.config.decay_half_life_days
        return math.pow(0.5, age_days / half_life)

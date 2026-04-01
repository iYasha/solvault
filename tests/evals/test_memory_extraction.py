import pytest

from sol.config import settings
from sol.memory.schemas import MemoryFact
from sol.memory.store import MemoryStore
from sol.session.models import Role
from tests.evals.harness import EvalScenario, judge, run_turn


@pytest.mark.eval
class TestMemoryExtractionPersonalFacts:
    """Memory extractor should identify and persist personal facts from conversations."""

    async def test_extracts_user_name_and_role(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given user states name and job, when turn runs, then a memory with that info is extracted."""
        scenario = EvalScenario(
            user_message="Hi! My name is Alex and I'm a machine learning engineer at Stripe.",
            criteria=[
                "At least one memory is extracted from this conversation",
                "An extracted memory contains the user's name (Alex)",
                "An extracted memory mentions the user's role or company",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)
        assert result.extracted_memories, "At least one memory should be extracted"

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"

    async def test_extracts_user_preference(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given user states a preference, when turn runs, then a memory about that preference is extracted."""
        scenario = EvalScenario(
            user_message="I prefer using vim over any other editor. Also I always write my code in Python.",
            criteria=[
                "At least one memory is extracted",
                "An extracted memory mentions vim or editor preference",
                "An extracted memory mentions Python",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)
        assert result.extracted_memories, "At least one memory should be extracted"

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"

    async def test_skips_ephemeral_question(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given user asks a generic factual question, when turn runs, then no personal memory is extracted."""
        scenario = EvalScenario(
            user_message="What does HTTP status code 404 mean?",
            criteria=[
                "Sol provides a correct answer about HTTP 404",
                "No personal facts about the user are extracted (this is a generic knowledge question)",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"


@pytest.mark.eval
class TestMemoryDedup:
    """Memory dedup should merge corrections about the same topic instead of creating duplicates."""

    async def test_corrects_user_name(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given a memory with old name, when user corrects it, then old name is replaced not duplicated."""
        scenario = EvalScenario(
            user_message="Actually my name is Ivan",
            history=[
                (Role.USER, "Hi! My name is Petro."),
                (Role.ASSISTANT, "Nice to meet you, Petro! How can I help you today?"),
            ],
            seed_memories=[
                MemoryFact(content="User's name is Petro", type="user", confidence="confirmed"),
            ],
            criteria=[
                "Sol acknowledges the name correction from Petro to Ivan",
                "Only one memory about the user's name should exist (the corrected one with Ivan)",
                "The old name 'Petro' should not remain as a separate memory record in the database",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)

        # Query all DB memories to pass to the judge
        store = MemoryStore(db=eval_db, embeddings=eval_embeddings, config=settings.memory)
        all_memories = await store.list_all()
        db_records = "\n".join(f"- [{m.type}] {m.content}" for m in all_memories) or "(none)"

        verdict = await judge(
            result,
            eval_judge_llm,
            extra_context={"All Memory Records in Database": db_records},
        )
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}\nDB memories: {db_records}"

import pytest

from sol.memory.schemas import MemoryFact
from sol.session.models import Role
from tests.evals.harness import EvalScenario, judge, run_turn


@pytest.mark.eval
class TestAgentResponseWithMemory:
    """Agent should use injected memory context in responses."""

    async def test_recalls_user_name_from_memory(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given a seeded name memory, when user asks for their name, then Sol responds with it."""
        scenario = EvalScenario(
            user_message="Hey Sol, do you remember my name?",
            seed_memories=[
                MemoryFact(content="User's name is Jordan", type="user", confidence="confirmed"),
            ],
            criteria=[
                "Sol's response includes the name 'Jordan'",
                "Response is friendly and conversational",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"

    async def test_recalls_user_details_from_memory(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given seeded memories about user, when user asks about themselves, then Sol uses the context."""
        scenario = EvalScenario(
            user_message="What do you know about my work?",
            seed_memories=[
                MemoryFact(
                    content="User works as a backend engineer at a fintech startup",
                    type="user",
                    confidence="confirmed",
                ),
                MemoryFact(
                    content="User is building a payments API with FastAPI",
                    type="work",
                    confidence="confirmed",
                ),
            ],
            criteria=[
                "Sol mentions the user's role (backend engineer) or company (fintech startup)",
                "Sol references the payments API or FastAPI project",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"


@pytest.mark.eval
class TestAgentResponseWithHistory:
    """Agent should use conversation history for coherent multi-turn responses."""

    async def test_references_prior_context(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given prior history about a project, when follow-up asked, then Sol references it."""
        scenario = EvalScenario(
            user_message="What framework did I mention I was using?",
            history=[
                (Role.USER, "I'm building a new web app with FastAPI and I need help with authentication."),
                (Role.ASSISTANT, "FastAPI has great auth support! You can use OAuth2 with JWT tokens."),
            ],
            criteria=[
                "Sol correctly identifies FastAPI from the conversation history",
                "Response is direct and answers the question",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"

    async def test_continues_conversation_topic(self, eval_db, eval_agent, eval_embeddings, eval_judge_llm):
        """Given history about debugging, when user asks follow-up, then Sol stays on topic."""
        scenario = EvalScenario(
            user_message="Any other ideas what could cause this?",
            history=[
                (Role.USER, "My SQLAlchemy queries are running really slow on large tables."),
                (
                    Role.ASSISTANT,
                    "Slow queries often come from missing indexes."
                    " Check if you have indexes on columns used in WHERE clauses and JOINs.",
                ),
            ],
            criteria=[
                "Sol provides additional debugging suggestions related to slow SQLAlchemy queries",
                "Response builds on the previous suggestion about indexes rather than starting fresh",
            ],
        )

        result = await run_turn(scenario, eval_agent, eval_db, eval_embeddings)

        verdict = await judge(result, eval_judge_llm)
        assert verdict.passed, f"Judge: {verdict.reason}\nAgent response: {result.response}"

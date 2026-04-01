from dataclasses import dataclass, field

from fastapi import BackgroundTasks
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sqlalchemy.ext.asyncio import AsyncSession

from sol.config import settings
from sol.core.agent import Agent
from sol.gateway.api.v1.messages import message_router, send_message
from sol.gateway.schemas import IncomingMessageRequest
from sol.memory.schemas import MemoryFact
from sol.memory.store import MemoryStore
from sol.session.models import ChatMessage, Role, Session

EVAL_CHANNEL = "cli"
EVAL_USER_ID = "eval-user"


@dataclass
class EvalScenario:
    """Definition of a single evaluation test case."""

    user_message: str
    criteria: list[str]
    history: list[tuple[str, str]] = field(default_factory=list)
    seed_memories: list[MemoryFact] = field(default_factory=list)


@dataclass
class EvalResult:
    """Outcome of running one eval scenario through the agent."""

    scenario: EvalScenario
    response: str
    extracted_memories: list[str]


@dataclass
class JudgeVerdict:
    """LLM judge's pass/fail evaluation of an EvalResult."""

    passed: bool
    reason: str


JUDGE_PROMPT = """\
You are an impartial evaluator for an AI assistant named Sol.

## Conversation History
{history}

## User Message
{user_message}

## Sol's Response
{response}

## Memories Extracted From This Conversation Turn
{extracted_memories}
{extra_context}
## Success Criteria
{criteria}

Evaluate whether Sol's behavior satisfies ALL of the success criteria listed above.

Respond with exactly two lines:
Line 1: PASS or FAIL
Line 2: Brief explanation covering all criteria.\
"""


async def run_turn(
    scenario: EvalScenario,
    agent: Agent,
    db: AsyncSession,
    embeddings: OpenAIEmbeddings,
) -> EvalResult:
    """Execute a full agent turn through the real message handler.

    Seeds memories, pre-loads history into the DB, then calls the actual
    send_message endpoint handler — which handles retrieval, injection,
    agent call, response saving, and background extraction.
    """
    store = MemoryStore(db=db, embeddings=embeddings, config=settings.memory)

    # Seed pre-existing memories
    for fact in scenario.seed_memories:
        await store.save(fact)

    # Pre-load conversation history into the database using the canonical user ID
    # (MessageRouter prepends the channel, e.g. "cli:eval-user")
    canonical_user = message_router.resolve_canonical_user(EVAL_CHANNEL, EVAL_USER_ID)
    session = Session(channel=EVAL_CHANNEL, user_id=canonical_user)
    db.add(session)
    await db.flush()
    for role, content in scenario.history:
        msg = ChatMessage(session_id=session.id, role=role, content=content)
        db.add(msg)
    await db.commit()

    # Snapshot memory state before the turn
    pre_extraction = {m.id: m.content for m in await store.list_all()}

    # Call the real endpoint handler directly
    background_tasks = BackgroundTasks()
    response = await send_message(
        body=IncomingMessageRequest(channel=EVAL_CHANNEL, user_id=EVAL_USER_ID, text=scenario.user_message),
        background_tasks=background_tasks,
        db=db,
        agent=agent,
        embeddings=embeddings,
    )

    # Run background extraction tasks (patched to use test DB via conftest)
    await background_tasks()

    # Collect newly extracted or modified memories
    all_memories = await store.list_all()
    extracted = [m.content for m in all_memories if m.id not in pre_extraction or m.content != pre_extraction[m.id]]

    return EvalResult(
        scenario=scenario,
        response=response.response_text,
        extracted_memories=extracted,
    )


async def judge(
    result: EvalResult,
    judge_llm: ChatOpenAI,
    extra_context: dict[str, str] | None = None,
) -> JudgeVerdict:
    """Ask the judge LLM to evaluate the turn result against success criteria.

    extra_context: optional dict where each key becomes a section header
    and each value becomes the section body in the judge prompt.
    """
    history_text = (
        "\n".join(f"{role.upper()}: {content}" for role, content in result.scenario.history) or "(no prior history)"
    )

    extracted_text = "\n".join(f"- {m}" for m in result.extracted_memories) or "(none)"
    criteria_text = "\n".join(f"- {c}" for c in result.scenario.criteria)

    extra_sections = ""
    if extra_context:
        extra_sections = "\n".join(f"\n## {header}\n{body}" for header, body in extra_context.items())

    prompt = JUDGE_PROMPT.format(
        history=history_text,
        user_message=result.scenario.user_message,
        response=result.response,
        extracted_memories=extracted_text,
        extra_context=extra_sections,
        criteria=criteria_text,
    )

    raw = await judge_llm.ainvoke([HumanMessage(content=prompt)])
    verdict_text = str(raw.content).strip()

    # Scan lines for the first PASS/FAIL verdict (tolerates LLM preamble)
    passed = False
    reason = verdict_text
    for i, line in enumerate(verdict_text.split("\n")):
        stripped = line.strip().upper()
        if stripped.startswith("PASS") or stripped.startswith("FAIL"):
            passed = stripped.startswith("PASS")
            remaining = verdict_text.split("\n")[i + 1 :]
            reason = "\n".join(remaining).strip() if remaining else verdict_text
            break

    return JudgeVerdict(passed=passed, reason=reason)

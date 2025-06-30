import pytest

from livekit.agents import AgentSession, llm
from livekit.plugins import openai
from agent import Assistant


def _llm() -> llm.LLM:
    return openai.LLM(model="gpt-4o-mini", temperature=0.45)

@pytest.mark.asyncio
async def test_greeting() -> None:

    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())
        result = await session.run(user_input="Hi there how are you?")
        await result.expect.message(role="assistant").judge(
            llm, intent="should offer a friendly greeting to the user"
        )
        result.expect.no_more_events()
        
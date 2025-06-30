import pytest

from livekit.agents import AgentSession, llm
from livekit.plugins import openai
from agent import Assistant


def _llm() -> llm.LLM:
    return openai.LLM(model="gpt-4o-mini", temperature=0.45)

@pytest.mark.asyncio
async def test_offers_assistance() -> None:
    async with (
        _llm() as llm,
        AgentSession(llm=llm) as session,
    ):
        await session.start(Assistant())
        result = await session.run(user_input="Hello")
        await result.expect.message(role="assistant").judge(
            llm, intent="Offers a friendly introduction and offer of assistance."
        )
        result.expect.no_more_events()
        
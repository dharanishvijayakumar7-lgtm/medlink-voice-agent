"""Run synthetic self-care conversations in six languages; no DB or phone calls."""

import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit.agents import AgentSession, llm, utils

from config import SUPPORTED_LANGUAGES, settings
from llm_factory import build_llm
from session_state import MedLinkUserData
from workflows.conversation import ConversationAgent

CASES = {
    "fever": "I am 32. Since yesterday evening I have fever, mild headache and body ache. No rash, stiff neck, breathing trouble or vomiting. I drink normally.",
    "diarrhoea": "My six-year-old daughter has had five loose stools since this morning. No blood, vomiting or fever. She is drinking and passing urine normally.",
    "reflux": "I am 40. For a week I have chest burning after spicy meals, never during walking. No vomiting, vomiting blood, black stools, breathlessness or trouble swallowing.",
    "headache": "I am 35. I have a moderate headache for two days, starting gradually. No fever, stiff neck, injury, weakness, vision trouble or vomiting.",
    "emergency": "I have crushing chest pain right now, sweating and trouble breathing. It started 20 minutes ago. I am 55.",
}

OPENINGS = {
    "fever": "I have a fever.",
    "diarrhoea": "My daughter has loose stools.",
    "reflux": "I have burning in my chest after meals.",
    "headache": "I have a headache.",
    "emergency": CASES["emergency"],
}


async def complete(model, system, text):
    ctx = llm.ChatContext()
    ctx.add_message(role="system", content=system)
    ctx.add_message(role="user", content=text)
    parts = []
    async with model.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                parts.append(chunk.delta.content)
    return "".join(parts).strip()


async def main():
    settings.enable_db = False
    settings.enable_firestore_export = False
    results = []
    output = Path("demo_audit_out/selfcare_conversations.json")
    output.parent.mkdir(exist_ok=True)
    async with utils.http_context.open():
        model = build_llm()
        for language, code in SUPPORTED_LANGUAGES.items():
            for case, facts in CASES.items():
                record = {"language": code, "case": case, "lines": []}
                try:
                    data = MedLinkUserData()
                    agent = ConversationAgent()
                    async with AgentSession(
                        llm=model, userdata=data, vad=None
                    ) as session:
                        await session.start(agent)
                        opening = await complete(
                            model,
                            f"Translate faithfully into {language} in its native script. Output only the translation.",
                            OPENINGS[case],
                        )
                        said = opening
                        for _ in range(8):
                            record["lines"].append(["caller", said])
                            result = await asyncio.wait_for(
                                session.run(user_input=said), timeout=60
                            )
                            replies = [
                                e.item.text_content
                                for e in result.events
                                if e.type == "message" and e.item.role == "assistant"
                            ]
                            record["lines"].extend(["agent", r] for r in replies)
                            if agent.policy.assessed or data.escalated:
                                break
                            said = await complete(
                                model,
                                f"You are a simulated caller speaking only {language} in its native script. Facts: {facts}. Answer the agent's last question briefly. Do not invent new symptoms; unknown details can be unknown.",
                                "\n".join(r or "" for r in replies),
                            )
                        record.update(
                            questions=len(agent.policy.questions),
                            assessed=agent.policy.assessed,
                            emergency=data.escalated,
                            kb=data.triage_entry_id,
                            actions=[d.model_dump() for d in agent.actions],
                        )
                except Exception as error:
                    record["error"] = type(error).__name__
                results.append(record)
                output.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k not in ("lines", "actions")
                        }
                    ),
                    flush=True,
                )
        await model.aclose()


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    asyncio.run(main())

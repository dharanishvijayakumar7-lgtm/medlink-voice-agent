"""Real-model text audits with fixed caller facts, not an improvising patient LLM.

uv run python scripts/audit_question_quality.py --languages en-IN
No phone calls, database writes, or Firestore exports. Saves synthetic transcripts.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit.agents import AgentSession, llm, utils

from config import SUPPORTED_LANGUAGES, settings
from llm_factory import build_llm
from session_state import MedLinkUserData
from workflows.conversation import ConversationAgent

BASE = {
    "duration": "About two years.",
    "hair_pattern": "It is gradually thinning at my front hairline. There are no separate bald patches.",
    "scalp_changes": "I have not noticed itching, flaking, redness or pain on my scalp.",
    "family_history": "My father has a similar pattern of hair thinning.",
    "recent_changes": "I do not recall an illness or major health change before it started.",
    "hair_care": "I wash gently. I do not use tight hairstyles, heat styling or chemical treatments.",
    "age": "I am 28 years old.",
    "progression": "The thinning has been gradual, not sudden.",
    "impact": "I am worried about my appearance, but can do my usual daily activities.",
}
CASES = {
    "hair_gradual": ("I have hair fall.", BASE),
    "hair_patchy": (
        "I am losing hair in separate patches.",
        {
            **BASE,
            "duration": "About three weeks.",
            "hair_pattern": "There are two separate round patches, rather than thinning at the front hairline.",
            "family_history": "I do not know of any similar hair loss in my family.",
            "progression": "I first noticed the patches suddenly three weeks ago.",
        },
    ),
    "hair_scalp": (
        "I have hair fall with an itchy scalp.",
        {
            **BASE,
            "duration": "About two months.",
            "hair_pattern": "The hair is thinning all over my scalp, not just at the hairline.",
            "scalp_changes": "The scalp is itchy and flaky. I have not noticed pain.",
            "family_history": "I do not know of any similar hair loss in my family.",
        },
    ),
    "hair_detailed": (
        "I am 28. My hair has thinned gradually at the front hairline for two years. My father has similar thinning. I have no scalp itching or flaking.",
        BASE,
    ),
    "unknown": (
        "I have persistent hiccups.",
        {
            "duration": "They have been happening for two days.",
            "progression": "They keep coming back.",
            "age": "I am 28 years old.",
            "associated": "I have not noticed any other symptoms.",
            "impact": "They make eating uncomfortable.",
        },
    ),
    "emergency": ("I have crushing chest pain and trouble breathing right now.", {}),
}


async def translate(model, text, language):
    if language == "en-IN":
        return text
    ctx = llm.ChatContext()
    ctx.add_message(
        role="system",
        content=f"Translate faithfully into {language} in its native script. Do not answer, add, or remove facts. Output only the translation.",
    )
    ctx.add_message(role="user", content=text)
    result = ""
    async with model.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                result += chunk.delta.content
    return result.strip()


async def run_case(model, language, case):
    opening, facts = CASES[case]
    row = {
        "case": case,
        "language": language,
        "lines": [],
        "facts": facts,
        "turn_seconds": [],
    }
    data = MedLinkUserData(language=language)
    agent = ConversationAgent()
    try:
        async with AgentSession(llm=model, userdata=data, vad=None) as session:
            await session.start(agent)
            said = await translate(model, opening, language)
            for _ in range(9):
                row["lines"].append(["caller", said])
                started = time.monotonic()
                result = await asyncio.wait_for(
                    session.run(user_input=said), timeout=45
                )
                row["turn_seconds"].append(round(time.monotonic() - started, 2))
                replies = [
                    e.item.text_content
                    for e in result.events
                    if e.type == "message" and e.item.role == "assistant"
                ]
                row["lines"].extend(["agent", text] for text in replies)
                if agent.policy.assessed:
                    break
                topic = agent.policy.questions[-1][0] if agent.policy.questions else ""
                # Unknown facts stay unknown. The simulated patient never invents
                # symptoms, family history, stress, or sleep problems.
                said = await translate(
                    model, facts.get(topic, "I am not sure."), language
                )
            assessment = next(
                (d for d in reversed(agent.actions) if d.action == "assess"), None
            )
            row.update(
                questions=len(agent.policy.questions),
                assessed=agent.policy.assessed,
                emergency=data.escalated,
                kb=data.triage_entry_id,
                question_topics=[t for t, _ in agent.policy.questions],
                actions=[a.model_dump() for a in agent.actions],
                complete_assessment=bool(
                    assessment
                    and all(
                        (
                            assessment.likely,
                            assessment.why,
                            assessment.do,
                            assessment.avoid,
                            assessment.seek_care,
                        )
                    )
                ),
            )
            row["passed_structure"] = (
                data.escalated and not agent.policy.questions
                if case == "emergency"
                else 4 <= row["questions"] <= 6 and row["complete_assessment"]
            )
    except Exception as error:
        row.update(error=type(error).__name__, passed_structure=False)
    return row


async def main(args):
    settings.enable_db = False
    settings.enable_firestore_export = False
    languages = (
        args.languages.split(",")
        if args.languages
        else list(SUPPORTED_LANGUAGES.values())
    )
    cases = args.cases.split(",")
    results = []
    gate = asyncio.Semaphore(2)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    async with utils.http_context.open():
        model = build_llm()
        try:

            async def run(language, case):
                async with gate:
                    row = await run_case(model, language, case)
                    results.append(row)
                    output.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        json.dumps(
                            {
                                k: row.get(k)
                                for k in (
                                    "case",
                                    "language",
                                    "questions",
                                    "passed_structure",
                                    "question_topics",
                                    "error",
                                )
                            }
                        ),
                        flush=True,
                    )

            await asyncio.gather(
                *(run(language, case) for language in languages for case in cases)
            )
        finally:
            await model.aclose()
    return 0 if all(r["passed_structure"] for r in results) else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--languages", default="")
    parser.add_argument("--cases", default="hair_gradual,hair_patchy,hair_scalp")
    parser.add_argument("--output", default="demo_audit_out/question_quality.json")
    raise SystemExit(asyncio.run(main(parser.parse_args())))

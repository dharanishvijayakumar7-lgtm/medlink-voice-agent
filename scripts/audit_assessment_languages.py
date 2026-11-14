"""Fixed synthetic assessment inputs; no patient records or caller translation.

uv run python scripts/audit_assessment_languages.py --label baseline
These are assessment-turn checks, not full calls or clinical accuracy scores.
"""

import argparse
import asyncio
import importlib.util
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from livekit.agents import AgentSession, llm, utils

from api_diagnostics import api_error_details
from config import SUPPORTED_LANGUAGES, settings
from llm_factory import build_llm
from session_state import MedLinkUserData
from workflows.conversation import ConversationAgent

# Fixed translations are intentionally visible for human review. No model
# translates the simulated caller, avoiding the earlier Tamil/Malayalam drift.
CASES = {
    "hair_gradual": {
        "en-IN": "I am 28. My hair has gradually thinned at the front hairline for two years. My father has similar thinning. I have no scalp itching, flaking or pain.",
        "hi-IN": "मेरी उम्र 28 साल है। दो साल से आगे की तरफ बाल धीरे-धीरे कम हो रहे हैं। मेरे पिता के बाल भी ऐसे ही कम हुए हैं। सिर में खुजली, पपड़ी या दर्द नहीं है।",
        "ta-IN": "எனக்கு வயது 28. இரண்டு ஆண்டுகளாக முன்பகுதியில் முடி மெதுவாகக் குறைகிறது. என் அப்பாவுக்கும் இதேபோல் முடி குறைந்துள்ளது. தலைத்தோலில் அரிப்போ செதில்களோ வலியோ இல்லை.",
        "te-IN": "నా వయసు 28. రెండు సంవత్సరాలుగా ముందు భాగంలో జుట్టు నెమ్మదిగా తగ్గుతోంది. మా నాన్నకు కూడా ఇలాగే జుట్టు తగ్గింది. తల చర్మంలో దురద, పొట్టు లేదా నొప్పి లేదు.",
        "kn-IN": "ನನಗೆ 28 ವರ್ಷ. ಎರಡು ವರ್ಷಗಳಿಂದ ಮುಂಭಾಗದಲ್ಲಿ ಕೂದಲು ನಿಧಾನವಾಗಿ ತೆಳುವಾಗುತ್ತಿದೆ. ನನ್ನ ತಂದೆಗೂ ಇದೇ ರೀತಿ ಕೂದಲು ತೆಳುವಾಗಿದೆ. ತಲೆಯ ಚರ್ಮದಲ್ಲಿ ತುರಿಕೆ, ಹೊಟ್ಟು ಅಥವಾ ನೋವು ಇಲ್ಲ.",
        "ml-IN": "എനിക്ക് 28 വയസ്സാണ്. രണ്ടു വർഷമായി മുൻഭാഗത്തെ മുടി ക്രമേണ കുറയുന്നു. അച്ഛനും ഇതുപോലെ മുടി കുറഞ്ഞിട്ടുണ്ട്. തലയോട്ടിയിൽ ചൊറിച്ചിലോ താരനോ വേദനയോ ഇല്ല.",
    },
    "fever": {
        "en-IN": "I am 32. I have had mild fever and body aches since yesterday. I can drink and pass urine normally. There is no breathing difficulty, stiff neck, rash or confusion. The cause is unknown.",
        "hi-IN": "मेरी उम्र 32 साल है। कल से हल्का बुखार और बदन दर्द है। पानी पी रही हूँ और पेशाब सामान्य है। साँस लेने में परेशानी, गर्दन में अकड़न, दाने या उलझन नहीं है। वजह नहीं पता।",
        "ta-IN": "எனக்கு வயது 32. நேற்று முதல் லேசான காய்ச்சலும் உடல்வலியும் உள்ளன. தண்ணீர் குடிக்க முடிகிறது, சிறுநீர் வழக்கம்போல் வருகிறது. மூச்சுத்திணறல், கழுத்து விறைப்பு, தோல் தடிப்பு அல்லது குழப்பம் இல்லை. காரணம் தெரியவில்லை.",
        "te-IN": "నా వయసు 32. నిన్నటి నుంచి తేలికపాటి జ్వరం, ఒళ్లు నొప్పులు ఉన్నాయి. నీళ్లు తాగగలుగుతున్నాను, మూత్రం మామూలుగానే వస్తోంది. ఊపిరి తీసుకోవడంలో ఇబ్బంది, మెడ బిగుసుకుపోవడం, దద్దుర్లు లేదా అయోమయం లేవు. కారణం తెలియదు.",
        "kn-IN": "ನನಗೆ 32 ವರ್ಷ. ನಿನ್ನೆಯಿಂದ ಸ್ವಲ್ಪ ಜ್ವರ ಮತ್ತು ಮೈ ನೋವು ಇದೆ. ನೀರು ಕುಡಿಯಲು ಸಾಧ್ಯವಾಗುತ್ತಿದೆ, ಮೂತ್ರ ಸಹಜವಾಗಿ ಬರುತ್ತಿದೆ. ಉಸಿರಾಟದ ತೊಂದರೆ, ಕುತ್ತಿಗೆ ಬಿಗಿತ, ದದ್ದು ಅಥವಾ ಗೊಂದಲ ಇಲ್ಲ. ಕಾರಣ ಗೊತ್ತಿಲ್ಲ.",
        "ml-IN": "എനിക്ക് 32 വയസ്സാണ്. ഇന്നലെ മുതൽ ചെറിയ പനിയും ശരീരവേദനയും ഉണ്ട്. വെള്ളം കുടിക്കാം, മൂത്രം സാധാരണപോലെ പോകുന്നു. ശ്വാസംമുട്ടലോ കഴുത്ത് പിടിക്കലോ ചൊറിപ്പോ ആശയക്കുഴപ്പമോ ഇല്ല. കാരണം അറിയില്ല.",
    },
    "headache": {
        "en-IN": "I am 35. I have had a moderate headache for two days. It started gradually after long hours looking at a screen. No injury, fever, stiff neck, weakness or vision changes. I do not know the cause.",
        "hi-IN": "मेरी उम्र 35 साल है। दो दिन से मध्यम सिरदर्द है। लंबे समय तक स्क्रीन देखने के बाद धीरे-धीरे शुरू हुआ। चोट, बुखार, गर्दन में अकड़न, कमजोरी या नज़र में बदलाव नहीं है। वजह नहीं पता।",
        "ta-IN": "எனக்கு வயது 35. இரண்டு நாட்களாக மிதமான தலைவலி உள்ளது. நீண்ட நேரம் திரையைப் பார்த்த பிறகு மெதுவாகத் தொடங்கியது. காயம், காய்ச்சல், கழுத்து விறைப்பு, பலவீனம் அல்லது பார்வை மாற்றம் இல்லை. காரணம் தெரியவில்லை.",
        "te-IN": "నా వయసు 35. రెండు రోజులుగా మోస్తరు తలనొప్పి ఉంది. ఎక్కువసేపు తెరను చూసిన తర్వాత నెమ్మదిగా మొదలైంది. గాయం, జ్వరం, మెడ బిగుసుకుపోవడం, బలహీనత లేదా చూపులో మార్పులు లేవు. కారణం తెలియదు.",
        "kn-IN": "ನನಗೆ 35 ವರ್ಷ. ಎರಡು ದಿನಗಳಿಂದ ಮಧ್ಯಮ ತಲೆನೋವು ಇದೆ. ಬಹಳ ಹೊತ್ತು ಪರದೆ ನೋಡಿದ ನಂತರ ನಿಧಾನವಾಗಿ ಶುರುವಾಯಿತು. ಗಾಯ, ಜ್ವರ, ಕುತ್ತಿಗೆ ಬಿಗಿತ, ದೌರ್ಬಲ್ಯ ಅಥವಾ ದೃಷ್ಟಿ ಬದಲಾವಣೆ ಇಲ್ಲ. ಕಾರಣ ಗೊತ್ತಿಲ್ಲ.",
        "ml-IN": "എനിക്ക് 35 വയസ്സാണ്. രണ്ടു ദിവസമായി മിതമായ തലവേദനയുണ്ട്. ഏറെനേരം സ്ക്രീൻ നോക്കിയ ശേഷം ക്രമേണ തുടങ്ങി. പരിക്കോ പനിയോ കഴുത്ത് പിടിക്കലോ ബലക്കുറവോ കാഴ്ചമാറ്റമോ ഇല്ല. കാരണം അറിയില്ല.",
    },
    "unknown": {
        "en-IN": "I am 28. I have had hiccups on and off for two days. They make eating uncomfortable. I do not know a trigger and have no other symptoms to report.",
        "hi-IN": "मेरी उम्र 28 साल है। दो दिन से रुक-रुककर हिचकियाँ आ रही हैं। खाने में परेशानी होती है। वजह नहीं पता और कोई दूसरा लक्षण नहीं है।",
        "ta-IN": "எனக்கு வயது 28. இரண்டு நாட்களாக விக்கல் விட்டுவிட்டு வருகிறது. சாப்பிடச் சிரமமாக உள்ளது. எதனால் வருகிறது என்று தெரியவில்லை, வேறு அறிகுறிகள் இல்லை.",
        "te-IN": "నా వయసు 28. రెండు రోజులుగా ఎక్కిళ్లు వస్తూ పోతూ ఉన్నాయి. తినడానికి ఇబ్బందిగా ఉంది. దేనివల్ల వస్తున్నాయో తెలియదు, వేరే లక్షణాలు లేవు.",
        "kn-IN": "ನನಗೆ 28 ವರ್ಷ. ಎರಡು ದಿನಗಳಿಂದ ಬಿಕ್ಕಳಿಕೆ ಬರುತ್ತಾ ಹೋಗುತ್ತಾ ಇದೆ. ತಿನ್ನಲು ಕಷ್ಟವಾಗುತ್ತಿದೆ. ಕಾರಣ ಗೊತ್ತಿಲ್ಲ, ಬೇರೆ ಲಕ್ಷಣಗಳಿಲ್ಲ.",
        "ml-IN": "എനിക്ക് 28 വയസ്സാണ്. രണ്ടു ദിവസമായി ഇടയ്ക്കിടെ എക്കിളുണ്ട്. ഭക്ഷണം കഴിക്കാൻ ബുദ്ധിമുട്ടുണ്ട്. എന്താണ് കാരണമെന്ന് അറിയില്ല, മറ്റു ലക്ഷണങ്ങളില്ല.",
    },
}

HAIR_CALLERS = {
    "en-IN": [
        "I have hair loss.",
        "About two years.",
        "It has gradually thinned at the front hairline, not in separate patches.",
        "I have no scalp itching, flaking or pain.",
        "My father has similar thinning.",
        "I am not sure.",
    ],
    "hi-IN": [
        "मेरे बाल झड़ रहे हैं।",
        "लगभग दो साल से।",
        "आगे की तरफ बाल धीरे-धीरे कम हो रहे हैं, अलग-अलग गोल जगहों से नहीं।",
        "सिर में खुजली, पपड़ी या दर्द नहीं है।",
        "मेरे पिता के बाल भी ऐसे ही कम हुए हैं।",
        "मुझे पक्का नहीं पता।",
    ],
    "ta-IN": [
        "எனக்கு முடி உதிர்கிறது.",
        "சுமார் இரண்டு ஆண்டுகளாக.",
        "முன்பகுதியில் முடி மெதுவாகக் குறைகிறது, தனித்தனி வழுக்கைத் திட்டுகள் இல்லை.",
        "தலைத்தோலில் அரிப்போ செதில்களோ வலியோ இல்லை.",
        "என் அப்பாவுக்கும் இதேபோல் முடி குறைந்துள்ளது.",
        "எனக்குத் தெரியவில்லை.",
    ],
    "te-IN": [
        "నాకు జుట్టు రాలుతోంది.",
        "సుమారు రెండు సంవత్సరాలుగా.",
        "ముందు భాగంలో జుట్టు నెమ్మదిగా తగ్గుతోంది, విడివిడిగా బట్టతల మచ్చలు లేవు.",
        "తల చర్మంలో దురద, పొట్టు లేదా నొప్పి లేదు.",
        "మా నాన్నకు కూడా ఇలాగే జుట్టు తగ్గింది.",
        "నాకు ఖచ్చితంగా తెలియదు.",
    ],
    "kn-IN": [
        "ನನ್ನ ಕೂದಲು ಉದುರುತ್ತಿದೆ.",
        "ಸುಮಾರು ಎರಡು ವರ್ಷಗಳಿಂದ.",
        "ಮುಂಭಾಗದಲ್ಲಿ ಕೂದಲು ನಿಧಾನವಾಗಿ ತೆಳುವಾಗುತ್ತಿದೆ, ಪ್ರತ್ಯೇಕ ಬೋಳು ತೇಪೆಗಳಿಲ್ಲ.",
        "ತಲೆಯ ಚರ್ಮದಲ್ಲಿ ತುರಿಕೆ, ಹೊಟ್ಟು ಅಥವಾ ನೋವು ಇಲ್ಲ.",
        "ನನ್ನ ತಂದೆಗೂ ಇದೇ ರೀತಿ ಕೂದಲು ತೆಳುವಾಗಿದೆ.",
        "ನನಗೆ ಖಚಿತವಾಗಿ ಗೊತ್ತಿಲ್ಲ.",
    ],
    "ml-IN": [
        "എനിക്ക് മുടി കൊഴിയുന്നു.",
        "ഏകദേശം രണ്ടു വർഷമായി.",
        "മുൻഭാഗത്തെ മുടി ക്രമേണ കുറയുന്നു, വേറെ കഷണ്ടിപ്പാടുകളില്ല.",
        "തലയോട്ടിയിൽ ചൊറിച്ചിലോ താരനോ വേദനയോ ഇല്ല.",
        "അച്ഛനും ഇതുപോലെ മുടി കുറഞ്ഞിട്ടുണ്ട്.",
        "എനിക്ക് ഉറപ്പില്ല.",
    ],
}


async def run_case(model, language, case):
    data = MedLinkUserData(language=language)
    agent = ConversationAgent()
    # Deliberately test the assessment turn in isolation. These placeholders
    # are policy state, not fabricated patient answers or claims of full triage.
    agent.policy.questions = [
        (f"audit_detail_{i}", "Recorded in audit fixture") for i in range(6)
    ]
    row = {
        "language": language,
        "case": case,
        "question_count_fixture": 6,
        "input": CASES[case][language],
        "proposals": [],
    }
    original = agent._propose

    async def capture(context):
        decision = await original(context)
        row["proposals"].append(decision.model_dump())
        return decision

    agent._propose = capture
    try:
        async with AgentSession(llm=model, userdata=data, vad=None) as session:
            await session.start(agent)
            ctx = llm.ChatContext()
            ctx.add_message(role="system", content=agent.instructions)
            ctx.add_message(role="user", content=row["input"])
            row["spoken"] = [p async for p in agent.llm_node(ctx, [], None)]
            row["kb"] = data.triage_entry_id
            row["actions"] = [d.model_dump() for d in agent.actions]
            row["assessed"] = agent.policy.assessed
    except Exception as exc:
        row["error"] = api_error_details(exc)
    return row


async def run_full_case(model, language):
    data = MedLinkUserData(language=language)
    agent = ConversationAgent()
    caller = HAIR_CALLERS[language]
    facts = dict(
        zip(
            ("duration", "hair_pattern", "scalp_changes", "family_history"),
            caller[1:5],
            strict=True,
        )
    )
    row = {
        "language": language,
        "case": "hair_gradual",
        "mode": "full_text_conversation",
        "lines": [],
    }
    try:
        async with AgentSession(llm=model, userdata=data, vad=None) as session:
            await session.start(agent)
            said = caller[0]
            for _ in range(8):
                row["lines"].append(["caller", said])
                result = await asyncio.wait_for(
                    session.run(user_input=said), timeout=60
                )
                row["lines"].extend(
                    ["agent", e.item.text_content]
                    for e in result.events
                    if e.type == "message" and e.item.role == "assistant"
                )
                if agent.policy.assessed:
                    break
                topic = agent.policy.questions[-1][0] if agent.policy.questions else ""
                said = facts.get(topic, caller[-1])
            row.update(
                assessed=agent.policy.assessed,
                kb=data.triage_entry_id,
                questions=len(agent.policy.questions),
                actions=[d.model_dump() for d in agent.actions],
            )
    except Exception as exc:
        row["error"] = api_error_details(exc)
    return row


async def main(args):
    global ConversationAgent
    if args.conversation_source:
        spec = importlib.util.spec_from_file_location(
            "workflows.audit_baseline", args.conversation_source
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        ConversationAgent = module.ConversationAgent
    settings.enable_db = False
    settings.enable_firestore_export = False
    output = Path("assessment_audit_out") / f"{args.label}.json"
    output.parent.mkdir(exist_ok=True)
    rows = []
    async with utils.http_context.open():
        model = build_llm()
        try:
            for language in args.languages:
                for case in args.cases:
                    row = (
                        await run_full_case(model, language)
                        if args.full_conversation
                        else await run_case(model, language, case)
                    )
                    rows.append(row)
                    output.write_text(
                        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    print(
                        json.dumps(
                            {
                                "language": language,
                                "case": case,
                                "assessed": row.get("assessed"),
                                "error": row.get("error"),
                            }
                        ),
                        flush=True,
                    )
        finally:
            await model.aclose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--conversation-source", type=Path)
    parser.add_argument("--full-conversation", action="store_true")
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=list(SUPPORTED_LANGUAGES.values()),
        default=list(SUPPORTED_LANGUAGES.values()),
    )
    parser.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main(parser.parse_args()))

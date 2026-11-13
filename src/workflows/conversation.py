"""Validated conversation actions shared by phone and text sessions.

The model proposes one action. Only the application renders it as speech; there
are no medicine tools or unconstrained model speech on this path.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config import SUPPORTED_LANGUAGES
from knowledge.triage_kb import get_triage_kb
from medicine.formulary import get_formulary
from safety.guardrails import scan_output
from safety.redflags import detect_redflag
from workflows.base import MedLinkAgent, _script_language, detect_language_request

logger = logging.getLogger("medlink.conversation")
LANGUAGES = Literal["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"]

# Used only if the provider repeatedly proposes an invalid action. These keep
# a malformed JSON response from ending a sparse conversation prematurely.
RECOVERY_QUESTIONS = {
    "en-IN": [
        "How long has this been happening?",
        "How severe is it right now?",
        "Are there any other symptoms?",
        "How old is the person who is unwell?",
    ],
    "hi-IN": [
        "यह तकलीफ कितने समय से है?",
        "अभी तकलीफ कितनी तेज है?",
        "क्या कोई और लक्षण हैं?",
        "जिस व्यक्ति को तकलीफ है उनकी उम्र कितनी है?",
    ],
    "ta-IN": [
        "இந்தத் தொந்தரவு எவ்வளவு காலமாக உள்ளது?",
        "இப்போது தொந்தரவு எவ்வளவு கடுமையாக உள்ளது?",
        "வேறு ஏதேனும் அறிகுறிகள் உள்ளனவா?",
        "உடல்நலம் சரியில்லாதவரின் வயது என்ன?",
    ],
    "te-IN": [
        "ఈ సమస్య ఎంతకాలంగా ఉంది?",
        "ఇప్పుడు సమస్య ఎంత తీవ్రంగా ఉంది?",
        "ఇంకా ఏమైనా లక్షణాలు ఉన్నాయా?",
        "అనారోగ్యంగా ఉన్న వ్యక్తి వయసు ఎంత?",
    ],
    "kn-IN": [
        "ಈ ತೊಂದರೆ ಎಷ್ಟು ಸಮಯದಿಂದ ಇದೆ?",
        "ಈಗ ತೊಂದರೆ ಎಷ್ಟು ತೀವ್ರವಾಗಿದೆ?",
        "ಬೇರೆ ಯಾವುದೇ ಲಕ್ಷಣಗಳಿವೆಯೇ?",
        "ಅನಾರೋಗ್ಯವಿರುವ ವ್ಯಕ್ತಿಯ ವಯಸ್ಸು ಎಷ್ಟು?",
    ],
    "ml-IN": [
        "ഈ ബുദ്ധിമുട്ട് എത്ര കാലമായി ഉണ്ട്?",
        "ഇപ്പോൾ ബുദ്ധിമുട്ട് എത്ര കഠിനമാണ്?",
        "മറ്റെന്തെങ്കിലും ലക്ഷണങ്ങൾ ഉണ്ടോ?",
        "അസുഖമുള്ള വ്യക്തിയുടെ പ്രായം എത്രയാണ്?",
    ],
}

FALLBACK = {
    "en-IN": "I cannot identify the cause safely from these details. Rest and look after yourself. Avoid anything that worsens the symptoms. Please get medical advice, and seek urgent help if symptoms are severe or worsening. I am a health assistant, not a doctor.",
    "hi-IN": "इन बातों से वजह सुरक्षित रूप से तय नहीं हो रही है। आराम करें और अपना ध्यान रखें। जिससे तकलीफ बढ़ती है उससे बचें। डॉक्टर से सलाह लें। तकलीफ गंभीर हो या बढ़े तो तुरंत मदद लें। मैं स्वास्थ्य सहायक हूँ, डॉक्टर नहीं।",
    "ta-IN": "இந்த விவரங்களால் காரணத்தை உறுதியாகக் கூற முடியவில்லை. ஓய்வெடுத்து உங்களை கவனித்துக் கொள்ளுங்கள். அறிகுறிகளை அதிகரிப்பவற்றைத் தவிருங்கள். மருத்துவரிடம் ஆலோசனை பெறுங்கள். கடுமையாக இருந்தாலோ அதிகரித்தாலோ உடனே உதவி பெறுங்கள். நான் சுகாதார உதவியாளர், மருத்துவர் அல்ல.",
    "te-IN": "ఈ వివరాలతో కారణాన్ని సురక్షితంగా చెప్పలేను. విశ్రాంతి తీసుకోండి, మీ ఆరోగ్యాన్ని చూసుకోండి. లక్షణాలు పెంచేవాటిని నివారించండి. వైద్యుని సలహా తీసుకోండి. లక్షణాలు తీవ్రంగా ఉంటే లేదా పెరిగితే వెంటనే సహాయం పొందండి. నేను ఆరోగ్య సహాయకురాలిని, వైద్యురాలిని కాదు.",
    "kn-IN": "ಈ ವಿವರಗಳಿಂದ ಕಾರಣವನ್ನು ಸುರಕ್ಷಿತವಾಗಿ ಹೇಳಲು ಸಾಧ್ಯವಿಲ್ಲ. ವಿಶ್ರಾಂತಿ ತೆಗೆದುಕೊಂಡು ನಿಮ್ಮನ್ನು ನೋಡಿಕೊಳ್ಳಿ. ತೊಂದರೆ ಹೆಚ್ಚಿಸುವುದನ್ನು ತಪ್ಪಿಸಿ. ವೈದ್ಯರ ಸಲಹೆ ಪಡೆಯಿರಿ. ಲಕ್ಷಣಗಳು ತೀವ್ರವಾಗಿದ್ದರೆ ಅಥವಾ ಹೆಚ್ಚಾದರೆ ತಕ್ಷಣ ಸಹಾಯ ಪಡೆಯಿರಿ. ನಾನು ಆರೋಗ್ಯ ಸಹಾಯಕಿ, ವೈದ್ಯೆಯಲ್ಲ.",
    "ml-IN": "ഈ വിവരങ്ങളിൽ നിന്ന് കാരണം സുരക്ഷിതമായി പറയാൻ കഴിയില്ല. വിശ്രമിക്കുകയും സ്വയം ശ്രദ്ധിക്കുകയും ചെയ്യുക. ലക്ഷണങ്ങൾ വർധിപ്പിക്കുന്നവ ഒഴിവാക്കുക. ഡോക്ടറുടെ ഉപദേശം തേടുക. ലക്ഷണങ്ങൾ ഗുരുതരമാകുകയോ കൂടുകയോ ചെയ്താൽ ഉടൻ സഹായം തേടുക. ഞാൻ ആരോഗ്യ സഹായിയാണ്, ഡോക്ടറല്ല.",
}
EMERGENCY = {
    "en-IN": "These symptoms may be an emergency. Call 112 now for emergency help or ask someone nearby to call. Do not drive yourself or wait to see if it passes. I cannot confirm the cause on a call. Is someone with you who can help?",
    "hi-IN": "ये लक्षण आपात स्थिति के हो सकते हैं। तुरंत 112 पर कॉल करें या पास के किसी व्यक्ति से कॉल करवाएँ। खुद गाड़ी न चलाएँ और ठीक होने का इंतजार न करें। फोन पर वजह पक्की नहीं बता सकती। क्या कोई आपकी मदद के लिए पास है?",
    "ta-IN": "இந்த அறிகுறிகள் அவசர நிலையாக இருக்கலாம். உடனே 112 எண்ணை அழைக்கவும் அல்லது அருகில் உள்ளவரை அழைக்கச் சொல்லவும். நீங்களே வண்டி ஓட்ட வேண்டாம், சரியாகும் என்று காத்திருக்க வேண்டாம். தொலைபேசியில் காரணத்தை உறுதி செய்ய முடியாது. உதவ யாராவது அருகில் இருக்கிறார்களா?",
    "te-IN": "ఈ లక్షణాలు అత్యవసర పరిస్థితి కావచ్చు. వెంటనే 112కు కాల్ చేయండి లేదా పక్కన ఉన్నవారిని చేయమని అడగండి. మీరే వాహనం నడపకండి, తగ్గుతుందని వేచి ఉండకండి. ఫోనులో కారణాన్ని నిర్ధారించలేను. సహాయం చేయడానికి ఎవరైనా మీతో ఉన్నారా?",
    "kn-IN": "ಈ ಲಕ್ಷಣಗಳು ತುರ್ತು ಪರಿಸ್ಥಿತಿಯಾಗಿರಬಹುದು. ಈಗಲೇ 112ಕ್ಕೆ ಕರೆ ಮಾಡಿ ಅಥವಾ ಹತ್ತಿರದವರನ್ನು ಕರೆ ಮಾಡಲು ಕೇಳಿ. ನೀವೇ ವಾಹನ ಚಲಾಯಿಸಬೇಡಿ, ಕಡಿಮೆಯಾಗುವುದೆಂದು ಕಾಯಬೇಡಿ. ಫೋನಿನಲ್ಲಿ ಕಾರಣವನ್ನು ಖಚಿತಪಡಿಸಲು ಸಾಧ್ಯವಿಲ್ಲ. ಸಹಾಯ ಮಾಡಲು ಯಾರಾದರೂ ನಿಮ್ಮ ಜೊತೆ ಇದ್ದಾರೆಯೇ?",
    "ml-IN": "ഈ ലക്ഷണങ്ങൾ അടിയന്തരാവസ്ഥയുടേതാകാം. ഉടൻ 112ൽ വിളിക്കുക അല്ലെങ്കിൽ അടുത്തുള്ള ഒരാളോട് വിളിക്കാൻ പറയുക. സ്വയം വാഹനം ഓടിക്കരുത്, മാറുമെന്ന് കരുതി കാത്തിരിക്കരുത്. ഫോണിലൂടെ കാരണം ഉറപ്പിക്കാനാവില്ല. സഹായിക്കാൻ ആരെങ്കിലും കൂടെയുണ്ടോ?",
}


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["ask", "assess", "answer", "emergency", "close"]
    language: LANGUAGES
    confirmed_symptoms: str = Field(
        default="",
        description="English translation of ONLY positively reported symptoms, never denied symptoms.",
    )
    denied_symptoms: str = Field(
        default="", description="English translation of explicitly denied symptoms."
    )
    topic: str = ""
    question: str = ""
    likely: str = ""
    why: str = ""
    do: str = ""
    avoid: str = ""
    seek_care: str = ""
    answer: str = ""
    opening_complete: bool = Field(
        default=False,
        description="True ONLY when the FIRST caller message already supplied duration, severity, and warning-sign information; otherwise false.",
    )


def medicine_free(text: str) -> bool:
    if not scan_output(text).is_safe:
        return False
    if re.search(r"\d\s*(?:mg|mcg|ml|tablets?|capsules?|मिलीग्राम|गोली)\b", text, re.I):
        return False
    # Includes generic names, brands, and the native-script names in the
    # formulary. Reject the entire proposal before any audio is emitted.
    for name in get_formulary().known_names():
        if len(name) > 2 and re.search(
            r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I
        ):
            return False
    return True


@dataclass
class ConversationPolicy:
    questions: list[tuple[str, str]] = field(default_factory=list)
    assessed: bool = False

    def accept(self, decision: Decision) -> str:
        if decision.action == "ask":
            if self.assessed or len(self.questions) >= 6:
                raise ValueError("Question budget exhausted; provide an assessment")
            if not decision.topic.strip() or not decision.question.strip():
                raise ValueError("Question needs a topic and text")
            if decision.question.count("?") + decision.question.count("\uff1f") > 1:
                raise ValueError("Ask one question only")
            if any(
                t.casefold() == decision.topic.casefold() or q == decision.question
                for t, q in self.questions
            ):
                raise ValueError("This question was already asked")
            text = decision.question
        elif decision.action == "assess":
            parts = [
                decision.likely,
                decision.why,
                decision.do,
                decision.avoid,
                decision.seek_care,
            ]
            if not all(p.strip() for p in parts):
                raise ValueError("Assessment must contain all five sections")
            text = " ".join(parts)
        elif decision.action in ("answer", "close"):
            if not self.assessed or not decision.answer.strip():
                raise ValueError("Assess before answering follow-up questions")
            text = decision.answer
        else:
            self.assessed = True
            return EMERGENCY[decision.language]
        if not medicine_free(text):
            raise ValueError("Self-care only: no medicine names or doses")
        if decision.action == "ask":
            self.questions.append((decision.topic, decision.question))
        if decision.action == "assess":
            self.assessed = True
        return text


INSTRUCTIONS = """You are MedLink, a warm health assistant, not a doctor.
Return only the requested JSON action, never unstructured speech.
Follow the caller's language: English, Hindi, Tamil, Telugu, Kannada, Malayalam.
Use ask to ask ONE useful, focused question, normally 3-6 total; fewer if the
caller already supplied the necessary details. Never repeat answered details.
Count age and clarification within that budget. Never ask a seventh question.
Use consistent English topic labels (age, duration, severity, warning_signs,
location, history) so the application can prevent repeats. No bundled questions.
Use assess once enough is known or the budget ends. Include five short spoken
sections in the caller's language: likely (a possible explanation, not certainty),
why (tie it to reported facts), do (practical self-care), avoid, seek_care
(specific warning signs and referral). Mention that you are not a doctor.
Keep this about 45-60 seconds. If unsure, say so rather than inventing a cause.
Never assert a symptom is absent unless the caller denied it.
No medicines, doses, supplements, branded products or medication suggestions,
even when asked. Explain that a clinician or pharmacist must advise on medicines.
Keep confirmed_symptoms as an English summary of ONLY positively reported
symptoms. Keep denied symptoms separate. Never put negated symptoms in confirmed.
Emergency symptoms require emergency action immediately, without questioning.
After assessment, use answer for relevant follow-ups; do not restart questioning.
Use close only when the caller clearly says they have finished. One farewell.
Do not invent arrangements, appointments, or emergency calls on their behalf.
"""


class ConversationAgent(MedLinkAgent):
    def __init__(self, **kwargs):
        super().__init__(instructions=INSTRUCTIONS, **kwargs)
        self.policy = ConversationPolicy()
        self.actions = []
        self.closed = False
        self.opening_complete = False

    async def on_enter(self):
        await self.session.say(
            "Hello, this is MedLink. Please tell me what is troubling you."
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # Deliberately no per-turn clinical mutations here: text and spoken
        # sessions share llm_node, so they execute identical decision logic.
        return None

    async def llm_node(self, chat_ctx, tools, model_settings):
        callers = [
            m.text_content
            for m in chat_ctx.items
            if getattr(m, "role", None) == "user" and m.text_content
        ]
        latest = callers[-1] if callers else ""
        if self.closed:
            return
        explicit = detect_language_request(latest)
        language = explicit or _script_language(latest) or self.data.language
        self.data.heard = callers[-40:]
        hit = detect_redflag(latest)
        # A long English denial list can exceed the legacy lexicon's negation
        # window. Defer those ambiguous matches to the structured decision,
        # rather than announcing an emergency for "no ... shortness of breath".
        ambiguous_denial = re.search(
            r"\b(?:no|not|never|without|don't|dont)\b", latest, re.I
        )
        if hit and hit.is_emergency and not ambiguous_denial:
            self.data.red_flag = hit
            self.data.urgency = "emergency"
            self.data.escalated = True
            self.data.disposition = "emergency"
            self.policy.assessed = True
            self._language(language)
            yield EMERGENCY[language]
            return

        # First extract an action; then ground its assessment using only the
        # caller's affirmative symptom summary, never the whole negated history.
        decision = None
        error = ""
        for _attempt in range(3):
            context = chat_ctx.copy()
            context.add_message(
                role="system",
                content=(
                    f"Return JSON matching this schema: {json.dumps(Decision.model_json_schema())}\n"
                    f"Already asked: {self.policy.questions}. Assessed: {self.policy.assessed}. "
                    f"Language hint: {language}. Remaining question budget: {6 - len(self.policy.questions)}. "
                    f"Correction: {error}. IMPORTANT: confirmed_symptoms and denied_symptoms MUST be in English even when the spoken fields are in another language."
                    + (
                        " The ONLY permitted action this turn is ask; pick one unanswered topic. Do not assess yet."
                        if len(self.policy.questions) < 3
                        and not self.opening_complete
                        and not self.policy.assessed
                        and callers
                        and len(callers) > 1
                        else ""
                    )
                ),
            )
            if decision and decision.action == "assess":
                entry = get_triage_kb().match(decision.confirmed_symptoms)
                self.data.triage_entry_id = entry.id if entry else None
                context.add_message(
                    role="system",
                    content=(
                        "Ground the assessment in this self-care reference only; omit all medicines, "
                        "supplements, ORS products and doses. If not matched, acknowledge uncertainty "
                        "and recommend medical assessment. Do not add unrelated symptoms or advice.\n"
                        + (
                            f"{entry.presentation}: {entry.self_care} Warning signs: {entry.refer_when}"
                            if entry
                            else "No matched reference."
                        )
                    ),
                )
            raw = ""
            async with self.session.llm.chat(
                chat_ctx=context,
                tools=[],
                extra_kwargs={"response_format": {"type": "json_object"}},
            ) as stream:
                async for chunk in stream:
                    if chunk.delta and chunk.delta.content:
                        raw += chunk.delta.content
            try:
                candidate = Decision.model_validate_json(raw)
                if not self.policy.questions:
                    self.opening_complete = candidate.opening_complete
                if (
                    candidate.action == "assess"
                    and not self.policy.assessed
                    and len(self.policy.questions) < 3
                    and not self.opening_complete
                ):
                    raise ValueError(
                        "Ask 3 useful questions before assessing a brief opening; ask the most important unanswered topic now."
                    )
                if candidate.action == "assess" and not candidate.confirmed_symptoms:
                    candidate.confirmed_symptoms = (
                        self.data.chief_complaint or callers[0]
                    )
                if (
                    explicit or _script_language(latest)
                ) and candidate.language != language:
                    raise ValueError(f"All spoken fields must be in {language}")
                if self.data.escalated and candidate.action not in (
                    "emergency",
                    "close",
                ):
                    candidate.action = "emergency"
                if candidate.action == "assess" and decision is None:
                    decision = candidate
                    error = (
                        "Now provide the assessment grounded in the supplied reference."
                    )
                    continue
                if candidate.action == "assess" and self.data.triage_entry_id is None:
                    self.policy.assessed = True
                    self._language(candidate.language)
                    yield FALLBACK[candidate.language]
                    return
                if candidate.action == "emergency":
                    self.data.urgency = "emergency"
                    self.data.escalated = True
                    self.data.disposition = "emergency"
                text = self.policy.accept(candidate)
                self.actions.append(candidate)
                if candidate.action == "close":
                    self.closed = True
                self._language(candidate.language)
                self.data.questions_asked = len(self.policy.questions)
                if candidate.confirmed_symptoms:
                    self.data.chief_complaint = candidate.confirmed_symptoms
                self.data.recommendations = []
                if candidate.action == "assess":
                    self.data.possible_causes = [candidate.likely]
                    self.data.possible_causes_reasoning = candidate.why
                    self.data.self_care_advice = candidate.do + " " + candidate.avoid
                    self.data.refer_when = candidate.seek_care
                logger.info(
                    "conversation action",
                    extra={
                        "action": candidate.action,
                        "question_count": len(self.policy.questions),
                        "language": self.data.language,
                    },
                )
                yield text
                return
            except ValueError as exc:
                error = str(exc)
                logger.warning(
                    "invalid conversation proposal",
                    extra={"validation_error_type": type(exc).__name__},
                )
        if not self.policy.assessed and len(self.policy.questions) < 3:
            for topic, question in zip(
                ("duration", "severity", "warning_signs", "age"),
                RECOVERY_QUESTIONS[language],
                strict=True,
            ):
                if topic not in {t for t, _ in self.policy.questions}:
                    recovery = Decision(
                        action="ask", language=language, topic=topic, question=question
                    )
                    text = self.policy.accept(recovery)
                    self.actions.append(recovery)
                    self.data.questions_asked = len(self.policy.questions)
                    self._language(language)
                    yield text
                    return
        self.policy.assessed = True
        self._language(language)
        yield FALLBACK[language]

    def _language(self, language):
        if language not in SUPPORTED_LANGUAGES.values():
            language = "en-IN"
        self.data.language = language
        if self.session.tts is not None:
            self.session.tts.update_options(target_language_code=language)

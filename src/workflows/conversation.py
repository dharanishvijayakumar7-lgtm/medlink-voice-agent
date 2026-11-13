"""Validated conversation actions shared by phone and text sessions.

The model proposes one action. Only the application renders it as speech; there
are no medicine tools or unconstrained model speech on this path.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from livekit.agents import APIError
from pydantic import BaseModel, ConfigDict, Field

from api_diagnostics import api_error_details
from config import SUPPORTED_LANGUAGES
from knowledge.triage_kb import get_triage_kb
from medicine.formulary import get_formulary
from safety.guardrails import scan_output
from safety.redflags import detect_redflag
from workflows.base import MedLinkAgent, _script_language, detect_language_request
from workflows.conversation_text import (
    GENERAL_ASSESSMENT,
    IDENTITY_NOTICES,
    SERVICE_ERROR,
)

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


MIN_QUESTIONS = 4
MAX_QUESTIONS = 6
MODEL_TIMEOUT = 15.0


class ReportedFact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    topic: str
    value: str = Field(
        description="English translation of the reported fact; do not infer."
    )
    status: Literal["reported", "denied", "unknown"] = "reported"
    quote: str = Field(
        description="Exact supporting quote from a caller message in its original language."
    )


class Decision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["ask", "repeat", "assess", "answer", "emergency", "close"]
    language: LANGUAGES
    confirmed_symptoms: str = Field(
        default="",
        description="English summary of ONLY positively reported symptoms. Keep denied symptoms separate.",
    )
    denied_symptoms: str = ""
    topic: str = ""
    question: str = ""
    important_gap: str = Field(
        default="",
        description="For question 5 or 6: the important unresolved detail that changes the advice.",
    )
    facts: list[ReportedFact] = Field(default_factory=list)
    stop_quote: str = Field(
        default="",
        description="Exact caller quote explicitly declining more questions or ending the call; otherwise empty. Duration is not a request to stop.",
    )
    likely: str = ""
    why: str = Field(
        default="",
        description="SPOKEN in the caller's language: one plain sentence combining the possible cause and why it fits a reported fact. This must make sense without the likely field.",
    )
    do: str = Field(
        default="",
        description="SPOKEN in the caller's language: one or two short sentences with practical next steps, without headings.",
    )
    avoid: str = ""
    seek_care: str = Field(
        default="",
        description="SPOKEN in the caller's language: a short relevant care instruction. No repeated identity disclaimer.",
    )
    answer: str = ""


def medicine_free(text: str) -> bool:
    if not scan_output(text).is_safe:
        return False
    if re.search(r"\d\s*(?:mg|mcg|ml|tablets?|capsules?|मिलीग्राम|गोली)\b", text, re.I):
        return False
    return not any(
        len(name) > 2
        and re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)
        for name in get_formulary().known_names()
    )


def _topic(value: str) -> str:
    topic = value.strip().casefold().replace(" ", "_")
    words = topic.split("_")
    # Keep the same clinical detail from being re-asked under a longer label.
    for core in ("duration", "age", "severity", "progression", "impact", "triggers"):
        if core in words:
            return core
    return {
        "hair_thinning_location": "hair_pattern",
        "hair_loss_pattern": "hair_pattern",
        "scalp_itching_flaking": "scalp_changes",
        "hair_care_practices": "hair_care",
        "hair_care_products": "hair_care",
    }.get(topic, topic)


def _quoted(quote: str, callers: list[str]) -> bool:
    normalized = " ".join(quote.split()).casefold()
    return bool(normalized) and any(
        normalized in " ".join(c.split()).casefold() for c in callers
    )


def _stop_requested(quote: str, latest: str) -> bool:
    # A model flag or an exact quote of "two years" must not bypass intake.
    return _quoted(quote, [latest]) and bool(
        re.search(
            r"\b(?:stop asking|no more questions|do not ask|don't ask|don't want to answer|do not want to answer|"
            r"just (?:give|tell)|end (?:the|this) call|goodbye|bye|that's all|that is all)\b|"
            r"(?:सवाल|प्रश्न).{0,20}(?:मत|नहीं)|बंद करो|अलविदा|"
            r"கேள்வி.{0,20}(?:வேண்டாம்|கேட்காத)|போதும்|"
            r"ప్రశ్న.{0,20}(?:వద్దు|అడగకు)|చాలు|"
            r"ಪ್ರಶ್ನೆ.{0,20}(?:ಬೇಡ|ಕೇಳಬೇಡ)|ಸಾಕು|"
            r"ചോദ്യം.{0,20}(?:വേണ്ട|ചോദിക്കരുത്)|മതി",
            quote,
            re.I,
        )
    )


@dataclass
class ConversationPolicy:
    questions: list[tuple[str, str]] = field(default_factory=list)
    assessed: bool = False

    def accept(
        self,
        decision: Decision,
        *,
        known_topics=None,
        caller_stopped=False,
        repeat_requested=False,
    ) -> str:
        decision.topic = _topic(decision.topic)
        if decision.action in ("ask", "repeat"):
            if self.assessed:
                raise ValueError(
                    "already_assessed: Answer the follow-up without restarting intake"
                )
            if not decision.topic or not decision.question.strip():
                raise ValueError("question_missing: Question needs a topic and text")
            if decision.question.count("?") + decision.question.count("\uff1f") > 1:
                raise ValueError("bundled_question: Ask one focused question only")
            if re.search(
                r"\?\s*\w|\band\s+(?:how|when|where|who|what|do you|have you|are you)\b",
                decision.question,
                re.I,
            ):
                raise ValueError("bundled_question: Ask one focused question only")
            if decision.action == "repeat":
                if (
                    not repeat_requested
                    or not self.questions
                    or decision.topic != self.questions[-1][0]
                ):
                    raise ValueError(
                        "repeat_not_requested: Only repeat the pending question on request"
                    )
            else:
                if len(self.questions) >= MAX_QUESTIONS:
                    raise ValueError(
                        "question_budget: No seventh question; provide an assessment"
                    )
                if (
                    len(self.questions) >= MIN_QUESTIONS
                    and not decision.important_gap.strip()
                ):
                    raise ValueError(
                        "important_gap: Explain the important gap or assess now"
                    )
                known = {_topic(t) for t in (known_topics or ())}
                if decision.topic in known or any(
                    _topic(t) == decision.topic
                    or q.casefold().strip() == decision.question.casefold().strip()
                    for t, q in self.questions
                ):
                    raise ValueError(
                        "repeated_question: This detail was already supplied or asked"
                    )
            text = decision.question
        elif decision.action == "assess":
            if (
                not self.assessed
                and len(self.questions) < MIN_QUESTIONS
                and not caller_stopped
            ):
                raise ValueError(
                    "minimum_questions: Ask four relevant questions before routine assessment"
                )
            parts = [
                decision.likely,
                decision.why,
                decision.do,
                decision.avoid,
                decision.seek_care,
            ]
            if not all(p.strip() for p in parts):
                raise ValueError("incomplete_assessment: Include all five sections")
            # Keep all five fields for records and safety validation, but speak
            # only the standalone explanation and next steps.
            text = " ".join((decision.why, decision.do, decision.seek_care))
            for notice in IDENTITY_NOTICES.values():
                text = text.replace(notice, "")
            text = " ".join(text.split())
        elif decision.action in ("answer", "close"):
            if not self.assessed or not decision.answer.strip():
                raise ValueError(
                    "assessment_required: Provide the assessment before closing"
                )
            if decision.action == "close" and not caller_stopped:
                raise ValueError("premature_close: The caller has not finished")
            text = decision.answer
        else:
            self.assessed = True
            return EMERGENCY[decision.language]
        if not medicine_free(" ".join(parts) if decision.action == "assess" else text):
            raise ValueError(
                "medicine_content: Self-care only; no medicine names or doses"
            )
        if decision.action == "ask":
            self.questions.append((decision.topic, decision.question))
        elif decision.action == "assess":
            self.assessed = True
        return text


INSTRUCTIONS = """You are MedLink, a warm voice assistant, not a doctor.
Return only the requested JSON action. Speak the caller's language: English,
Hindi, Tamil, Telugu, Kannada or Malayalam. Metadata and fact values are English;
fact quotes must be exact caller words in their original language.
For routine concerns ask FOUR to SIX useful, symptom-specific questions, one per
turn. Do not assess before four questions even after a detailed opening: explore
a relevant unanswered detail, not information already volunteered. After four,
assess unless a fifth or sixth question resolves an important gap that changes
advice; explain that gap in important_gap. Never ask a seventh question.
Use the supplied question topics and clinical reference to choose the next most
useful question. Avoid generic severity questions when pattern or associated
features are more informative (e.g. hair loss). Question topic IDs describe the
specific fact, not broad labels like history or warning_signs. Do not bundle
independent questions into one sentence. Use exactly one question mark.
Use everyday spoken wording. A brief acknowledgment is welcome when it helps
connect the caller's answer to the next question, but do not add one every turn,
repeat their answer mechanically, or use scripted filler. Be warm and direct.
Record facts supplied anywhere in the caller messages, including volunteered
details, negatives and unknowns; give an exact supporting quote for each.
Never invent an answer, infer a negative from silence, or repeat answered details.
Unknown means unknown, not normal. If all suggested questions have been answered,
ask a more specific relevant missing detail under a new descriptive topic.
Emergency takes priority at EVERY turn, regardless of the question count.
If the caller explicitly declines more questions, give a cautious assessment with
available information and put their exact request in stop_quote. Never treat a
long duration, a brief answer, or missing knowledge as a request to stop.
Use repeat to translate/rephrase the last pending question when the caller asks
to switch language or repeat it; this does not spend a new question.
For assess, still fill all five internal fields. Only why, do and seek_care are
spoken. In why, combine the possible cause and ONE supporting caller detail in
a single plain sentence, with appropriate uncertainty. It must stand alone:
the separate likely field is stored but not spoken. Put two useful next steps
in do and a short relevant care instruction in seek_care. Keep avoid for records;
there is no spoken don'ts list. Any essential precaution belongs in the next
steps. All spoken fields MUST be in the caller's language, never English metadata.
The whole spoken closing should be three or four short sentences, about 40-70
English words or equivalent in the caller's language. No headings, numbered
sections, medical jargon, or recital of the full symptom history. Explain an
unfamiliar term in everyday words or leave it out. Do not truncate care warnings.
Use the reference for advice, conditioned on what the caller actually reported.
Do not jump to unrelated conditions because of background history. Referral
must accompany useful self-care, not replace it. The introduction already says
you are a voice assistant, not a doctor: do not repeat that identity disclaimer
in routine replies. Keep uncertainty about a possible cause when appropriate.
For a complaint outside the reference library, acknowledge uncertainty about
the cause, explain the limitation, and give low-risk practical precautions and
a next step. Do not invent a diagnosis or unrelated disease-specific treatment.
No medicines, doses, supplements or branded remedies. If asked about them,
explain that a clinician must advise. Do not promise regrowth, cure, appointments
or emergency calls. After assessment use answer for relevant follow-ups, without
restarting questions. Use close only if the caller explicitly finishes, with an
exact stop_quote. No unsolicited farewell.
"""


class ConversationAgent(MedLinkAgent):
    def __init__(self, **kwargs):
        super().__init__(instructions=INSTRUCTIONS, **kwargs)
        self.policy = ConversationPolicy()
        self.actions: list[Decision] = []
        self.facts: dict[str, ReportedFact] = {}
        self.closed = False
        self.entry = None

    async def on_enter(self):
        await self.session.say(
            "Hi, I'm MedLink, a voice assistant, not a doctor. Tell me what's been bothering you."
        )

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # Text and voice sessions both use the same llm_node state machine.
        return None

    def _reference(self) -> str:
        if self.entry is None:
            return "No matched reference. Still ask four relevant questions. Do not invent a diagnosis."
        entry = self.entry
        questions = (
            [
                {"topic": q.topic, "question": q.text["en-IN"]}
                for q in entry.followup_questions
            ]
            if entry.followup_questions
            else entry.candidate_questions
        )
        return json.dumps(
            {
                "presentation": entry.presentation,
                "question_guidance": questions,
                "possible_explanations": entry.explanation,
                "self_care": entry.self_care,
                "avoid": entry.avoid,
                "seek_care": entry.refer_when,
                "warning_signs": entry.red_flags,
            },
            ensure_ascii=False,
        )

    async def _propose(self, context):
        raw = ""
        async with self.session.llm.chat(
            chat_ctx=context,
            tools=[],
            extra_kwargs={"response_format": {"type": "json_object"}},
        ) as stream:
            async for chunk in stream:
                if chunk.delta and chunk.delta.content:
                    raw += chunk.delta.content
        return Decision.model_validate_json(raw)

    def _remember(self, candidate, callers):
        for fact in candidate.facts:
            if not _quoted(fact.quote, callers):
                raise ValueError(
                    "unsupported_fact: Every fact needs an exact caller quote"
                )
        for fact in candidate.facts:
            fact.topic = _topic(fact.topic)
            self.facts[fact.topic] = fact
        if candidate.confirmed_symptoms:
            self.data.chief_complaint = candidate.confirmed_symptoms
        self.data.answers.update({t: f.value for t, f in self.facts.items()})

    def _commit(
        self, candidate, *, caller_stopped=False, repeat_requested=False, fallback=False
    ):
        if (
            candidate.action == "assess"
            and self.entry
            and candidate.language in self.entry.recovery_assessments
        ):
            # The model personalizes the explanation, not the curated self-care.
            # This prevents unsupported additions such as scalp massage to
            # stimulate circulation appearing in an otherwise valid assessment.
            guidance = self.entry.recovery_assessments[candidate.language]
            candidate.do = guidance.do
            candidate.avoid = guidance.avoid
            candidate.seek_care = guidance.seek_care
        text = self.policy.accept(
            candidate,
            known_topics=self.facts,
            caller_stopped=caller_stopped,
            repeat_requested=repeat_requested,
        )
        self.actions.append(candidate)
        self.closed = candidate.action == "close"
        self._language(candidate.language)
        self.data.questions_asked = len(self.policy.questions)
        self.data.recommendations = []
        if candidate.action == "assess":
            self.data.possible_causes = [candidate.likely]
            self.data.possible_causes_reasoning = candidate.why
            self.data.self_care_advice = candidate.do + " " + candidate.avoid
            self.data.refer_when = candidate.seek_care
            self.data.disposition = "self_care_guidance"
        if candidate.action == "emergency":
            self.data.urgency = "emergency"
            self.data.escalated = True
            self.data.disposition = "emergency"
        logger.info(
            "conversation action",
            extra={
                "action": candidate.action,
                "question_count": len(self.policy.questions),
                "topic": candidate.topic,
                "language": candidate.language,
                "knowledge_entry": self.entry.id if self.entry else None,
                "fallback": fallback,
            },
        )
        return text

    def _recovery_question(self, language):
        used = {_topic(t) for t, _ in self.policy.questions} | set(self.facts)
        if self.entry and self.entry.followup_questions:
            for question in self.entry.followup_questions:
                if question.topic not in used and language in question.text:
                    return Decision(
                        action="ask",
                        language=language,
                        topic=question.topic,
                        question=question.text[language],
                    )
            return None
        # Legacy references supply useful questions for normal generation.
        # English recovery can use those directly without another provider call.
        if self.entry and language == "en-IN":
            for index, question in enumerate(self.entry.candidate_questions):
                topic = f"{self.entry.id}_detail_{index}"
                if re.search(
                    r"\bhow (?:long|many (?:days|weeks|months|years))\b|\bwhen did\b",
                    question,
                    re.I,
                ):
                    topic = "duration"
                already = any(
                    question.casefold() == q.casefold()
                    for _, q in self.policy.questions
                )
                if topic not in used and not already and medicine_free(question):
                    return Decision(
                        action="ask", language=language, topic=topic, question=question
                    )
        for topic, question in zip(
            ("duration", "severity", "associated", "age"),
            RECOVERY_QUESTIONS[language],
            strict=True,
        ):
            if topic not in used:
                return Decision(
                    action="ask", language=language, topic=topic, question=question
                )
        return None

    def _recovery_assessment(self, language):
        if self.entry and language in self.entry.recovery_assessments:
            parts = self.entry.recovery_assessments[language].model_dump()
        else:
            parts = dict(GENERAL_ASSESSMENT[language])
            if self.entry:
                # Do not invent a native-language translation during an outage.
                # A known condition needs its own localized recovery guidance.
                if language != "en-IN":
                    return None
                sentences = re.split(r"(?<=[.!?])\s+", self.entry.self_care)
                positive = [
                    s
                    for s in sentences
                    if medicine_free(s)
                    and not re.search(r"\b(?:avoid|do not|never)\b", s, re.I)
                ]
                negative = [
                    s
                    for s in sentences
                    if medicine_free(s)
                    and re.search(r"\b(?:avoid|do not|never)\b", s, re.I)
                ]
                if self.entry.avoid and medicine_free(self.entry.avoid):
                    negative.append(self.entry.avoid)
                parts["why"] = (
                    f"The concern you described matches the guidance for {self.entry.presentation.lower()}, but an examination may be needed to identify its cause."
                )
                if positive:
                    parts["do"] = " ".join(positive)
                if negative:
                    parts["avoid"] = " ".join(negative)
                if medicine_free(self.entry.refer_when):
                    parts["seek_care"] = self.entry.refer_when
        return Decision(action="assess", language=language, **parts)

    async def llm_node(self, chat_ctx, tools, model_settings):
        callers = [
            m.text_content
            for m in chat_ctx.items
            if getattr(m, "role", None) == "user" and m.text_content
        ]
        if self.closed or not callers:
            return
        latest = callers[-1]
        explicit = detect_language_request(latest)
        language = explicit or _script_language(latest) or self.data.language
        self.data.heard = callers[-40:]
        repeat_requested = bool(
            explicit
            or re.search(
                r"\b(?:repeat|say (?:it |that )?again|did not hear|didn't hear)\b",
                latest,
                re.I,
            )
        )
        hit = detect_redflag(latest)
        ambiguous_denial = re.search(
            r"\b(?:no|not|never|without|don't|dont)\b", latest, re.I
        )
        if hit and hit.is_emergency and not ambiguous_denial:
            self.data.red_flag = hit
            yield self._commit(Decision(action="emergency", language=language))
            return

        # Establish the reference BEFORE the first question. Keep the primary
        # complaint stable so later family history cannot steal the KB match.
        if self.entry is None:
            self.entry = get_triage_kb().match(self.data.chief_complaint or callers[0])
        self.data.triage_entry_id = self.entry.id if self.entry else None
        error = ""
        stopped = False
        for attempt in range(2):
            context = chat_ctx.copy()
            context.add_message(
                role="system",
                content=(
                    f"Return JSON matching: {json.dumps(Decision.model_json_schema())}\n"
                    f"Questions already asked: {self.policy.questions}\n"
                    f"Known facts: {json.dumps({t: f.model_dump() for t, f in self.facts.items()}, ensure_ascii=False)}\n"
                    f"Assessed: {self.policy.assessed}. Language: {language}. "
                    f"Question count: {len(self.policy.questions)}; minimum FOUR, maximum SIX. "
                    f"Repeat/language switch requested: {repeat_requested}. Correction: {error}\n"
                    "Emergency and explicit caller requests to stop override the minimum. "
                    "Otherwise ask until four, then assess unless an important gap remains. "
                    "Use symptom-specific topics. Do not re-ask facts supplied in ANY caller message.\n"
                    f"Clinical reference (omit medicines and doses): {self._reference()}"
                ),
            )
            try:
                candidate = await asyncio.wait_for(
                    self._propose(context), timeout=MODEL_TIMEOUT
                )
                if (
                    explicit or _script_language(latest)
                ) and candidate.language != language:
                    raise ValueError("wrong_language: Use the requested language")
                # A valid model language choice also guides any offline recovery.
                language = candidate.language
                self._remember(candidate, callers)
                if (
                    self.entry is None
                    and not candidate.confirmed_symptoms
                    and candidate.action not in ("emergency", "close", "repeat")
                ):
                    raise ValueError(
                        "missing_complaint: Supply confirmed_symptoms as an English translation of the caller's primary complaint before choosing a question"
                    )
                stopped = bool(
                    candidate.stop_quote
                    and _stop_requested(candidate.stop_quote, latest)
                )
                if candidate.stop_quote and not stopped:
                    raise ValueError(
                        "invalid_stop_quote: Quote the caller's actual request to stop"
                    )
                if self.data.escalated and candidate.action != "close":
                    candidate.action = "emergency"
                if self.entry is None and candidate.confirmed_symptoms:
                    matched = get_triage_kb().match(candidate.confirmed_symptoms)
                    if matched:
                        self.entry = matched
                        self.data.triage_entry_id = matched.id
                        if candidate.action in ("ask", "assess"):
                            if attempt == 0:
                                error = "Use the newly matched reference for your next question or assessment."
                                continue
                            # Never speak an assessment generated before its
                            # relevant clinical reference was available.
                            break
                if (
                    repeat_requested
                    and self.policy.questions
                    and not self.policy.assessed
                    and candidate.action == "ask"
                ):
                    raise ValueError(
                        "repeat_requested: Translate/rephrase the pending question using repeat"
                    )
                if (
                    candidate.action == "ask"
                    and self.entry
                    and self.entry.followup_questions
                ):
                    topics = {q.topic for q in self.entry.followup_questions}
                    used = {_topic(t) for t, _ in self.policy.questions} | set(
                        self.facts
                    )
                    if topics - used and _topic(candidate.topic) not in topics:
                        raise ValueError(
                            "irrelevant_topic: Select a relevant unanswered topic ID from the supplied question guidance"
                        )
                yield self._commit(
                    candidate, caller_stopped=stopped, repeat_requested=repeat_requested
                )
                return
            except (APIError, TimeoutError) as exc:
                logger.warning(
                    "conversation provider unavailable",
                    extra={
                        **api_error_details(exc),
                        "question_count": len(self.policy.questions),
                    },
                )
                # Preserve state. A provider outage is not evidence about health.
                self._language(language)
                yield SERVICE_ERROR[language]
                return
            except ValueError as exc:
                error = str(exc)
                reason = (
                    error.split(":", 1)[0]
                    if re.match(r"^[a-z_]+:", error)
                    else "invalid_schema"
                )
                logger.warning(
                    "invalid conversation proposal",
                    extra={
                        "reason_code": reason,
                        "attempt": attempt + 1,
                        "question_count": len(self.policy.questions),
                        "knowledge_entry": self.entry.id if self.entry else None,
                    },
                )

        if self.policy.assessed:
            self._language(language)
            yield SERVICE_ERROR[language]
            return
        if repeat_requested and self.policy.questions:
            topic = self.policy.questions[-1][0]
            if self.entry:
                question = next(
                    (q for q in self.entry.followup_questions if q.topic == topic), None
                )
                if question and language in question.text:
                    yield self._commit(
                        Decision(
                            action="repeat",
                            language=language,
                            topic=topic,
                            question=question.text[language],
                        ),
                        repeat_requested=True,
                        fallback=True,
                    )
                    return
            self._language(language)
            yield SERVICE_ERROR[language]
            return
        if len(self.policy.questions) < MIN_QUESTIONS and not stopped:
            recovery = self._recovery_question(language)
            if recovery:
                yield self._commit(recovery, fallback=True)
                return
        else:
            recovery = self._recovery_assessment(language)
            if recovery:
                yield self._commit(recovery, caller_stopped=stopped, fallback=True)
                return
        self._language(language)
        yield SERVICE_ERROR[language]

    def _language(self, language):
        if language not in SUPPORTED_LANGUAGES.values():
            language = "en-IN"
        self.data.language = language
        if self.session.tts is not None:
            self.session.tts.update_options(target_language_code=language)

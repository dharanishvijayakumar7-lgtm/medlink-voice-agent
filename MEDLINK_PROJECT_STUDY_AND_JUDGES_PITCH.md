# MedLink: project study and judges' pitch

Prepared on 14 November 2026 from the current repository. This describes the implementation visible in source, not a fresh verification of a deployed service. Older README/report descriptions refer to an earlier workflow and should not be used as the sole basis for a presentation.

## 1. The project in one sentence

**MedLink is a multilingual, phone-based AI health guidance prototype that listens to a person's concern, asks focused questions, and explains practical self-care and when to seek professional help.**

Suggested tagline: **“A familiar language. An ordinary phone. A clearer next step.”**

Its intended role is an accessible first point of guidance. It cannot examine patients, confirm diagnoses, replace clinicians, or dispatch emergency services.

## 2. Why we built it

The product addresses a practical question: “I feel unwell. What should I do next?”

Our design starts from people for whom typing symptoms into an English app may be inconvenient or inaccessible. Intended users include rural residents, people using basic phones, older people, people uncomfortable reading medical information, and family members seeking guidance for someone else.

The barriers we are designing around are:

- **Language:** a caller should be able to explain a concern in a familiar language.
- **Interface:** an ordinary conversation should be enough to start; downloading and navigating an app should not be required for the caller.
- **Uncertainty:** people need understandable next steps and a clear indication of when professional care is needed.
- **Continuity:** with appropriate consent and access controls, a structured call summary could help avoid repeatedly explaining the same history.

These are our problem hypotheses and design motivations. The repository does not establish their prevalence through user interviews or a field study. Do not present invented statistics or claim that patients requested particular features unless the team has separate evidence.

Why voice? It allows a question-and-answer exchange without requiring reading and typing. Why phone access? It meets people through an interface they already use. Why AI? It can support an interactive conversation and language adaptation, subject to validation and supervision. Why application rules around AI? Fluent language alone does not establish safe or appropriate guidance.

## 3. What we have actually built

| Component | Current implementation | Presentation boundary |
|---|---|---|
| Voice session | Python worker using LiveKit Agents | Worker availability must be checked before a demo |
| Telephone entry | Saved LiveKit-managed US-number/SIP dispatch configuration | Not an Indian toll-free service; real handset verification remains outstanding in the saved runbook |
| Languages | English, Hindi, Tamil, Telugu, Kannada and Malayalam are configured | Equal quality across all six has not been established |
| Speech recognition | Sarvam streaming STT with automatic language detection | Recognition performance on rural phone audio needs measurement |
| Conversation | Sarvam LLM proposes structured JSON actions | Application validates proposals before rendering a response |
| Routine guidance | Focused questions followed by a short explanation, practical steps and care instructions | General guidance with uncertainty, not a confirmed diagnosis |
| Question policy | Normally four questions; a fifth or sixth needs an important unresolved gap; hard maximum six | Emergencies and explicit requests to stop bypass the minimum |
| Emergency handling | Active red-flag check plus model-proposed emergency action; fixed localized response | Instructions to seek help, not an actual emergency call or transfer |
| Knowledge retrieval | Curated YAML reference library, multilingual aliases and BM25 fallback | 23 presentations in the current file; not comprehensive coverage of medicine |
| Audio handling | Krisp voice isolation, semantic turn detection, echo filtering and output loudness levelling | Their presence does not prove noise robustness or low latency |
| Call records | Optional PostgreSQL history and Firestore summary export | Feature flags, consent and integration completion affect actual behavior |
| Tests and audits | Regression tests and saved synthetic conversations | Software checks do not establish clinical effectiveness |

### Current patient journey

1. A caller enters through a configured phone or web audio session.
2. MedLink introduces itself as a voice assistant, not a doctor.
3. The caller describes the concern and can request a supported language.
4. MedLink retrieves a relevant reference where available.
5. It asks one focused question at a time. The policy rejects recognized repeated topics and limits routine questioning.
6. A routine assessment has five internal fields: possible cause, explanation, what to do, what to avoid, and when to seek care.
7. The current spoken closing combines the explanation, practical next steps and care instructions. It does not recite five headings or a separate avoidance list.
8. Relevant follow-up questions can continue without restarting intake.
9. When the call ends, optional persistence/export runs with the application's storage checks.

Emergency handling can interrupt routine questioning at any turn.

## 4. Current architecture

```text
Caller phone or web audio
          |
LiveKit media transport / SIP dispatch
          |
Krisp input voice isolation
          |
Sarvam STT: saaras:v3-realtime
          |
ConversationAgent
  - language and conversation state
  - emergency red-flag check
  - curated reference retrieval
          |
Sarvam LLM: sarvam-105b-conversations
  proposes a JSON action
          |
Application validation and response rendering
  - action schema and question policy
  - quoted evidence for recorded facts
  - medicine-content checks
  - recovery or service-error responses
          |
Sarvam TTS: bulbul:v3, ritu voice
          |
Output audio levelling -> caller

At call end: optional PostgreSQL history / Firestore summary export
```

LiveKit supplies the real-time session infrastructure. Sarvam supplies the three AI stages. Our application supplies the health guidance workflow, reference retrieval, validation, conversation policy and persistence integration.

The active `agent.py` starts `ConversationAgent`. Older `IntakeAgent`, `TriageAgent`, `RecommendAgent` and `EscalateAgent` modules remain in the repository, but that four-agent handoff workflow is not the current entrypoint.

The legacy severity-scoring functions also remain. Do not claim that every current routine conversation is routed by that legacy numeric score; the active structured conversation path does not call that routing flow.

## 5. Exactly where Sarvam is used

Sarvam is central to the product: **listening, conversation reasoning, and speaking all use Sarvam.**

| Stage | Code | Configured default | Role |
|---|---|---|---|
| Speech to text | `src/speech/providers.py`, `build_stt()` | `sarvam.STTRealtime`, `saaras:v3-realtime`, balanced stream, `language="auto"` | Turns incoming speech into text while retaining the spoken language |
| Conversation reasoning | `src/llm_factory.py`, `build_llm()` | `sarvam.LLM`, `sarvam-105b-conversations`, temperature 0.3 | Interprets context and proposes the next structured conversation action |
| Text to speech | `src/speech/providers.py`, `build_tts()` | `sarvam.TTS`, `bulbul:v3`, speaker `ritu` | Turns accepted application text into streamed voice output |
| Session assembly | `src/agent.py`, `medlink_session()` | `AgentSession(llm=build_llm(), stt=build_stt(), tts=build_tts())` | Connects all three Sarvam components to the call |
| Language switching | `src/workflows/conversation.py`, `_language()` | `tts.update_options(target_language_code=language)` | Retargets spoken output for the selected supported language |
| Configuration | `src/config.py` | Shared `SARVAM_API_KEY`; configurable model settings | Keeps provider settings together; realtime STT model is pinned by the plugin class |

The LLM proposes actions such as `ask`, `repeat`, `assess`, `answer`, `emergency` and `close`. It is asked for JSON, parsed through a Pydantic schema, then checked by `ConversationPolicy`. Its raw JSON is not read aloud.

The STT default input is 8 kHz linear16 for telephony; TTS output is configured at 16 kHz linear16. LiveKit handles the media path. These are implementation details for technical questions, not the opening pitch.

### A judge-ready Sarvam answer

> “We use Sarvam throughout the voice AI pipeline. Saaras transcribes the caller's speech, the Sarvam conversation model proposes the next question or assessment, and Bulbul speaks the accepted response. Our Python application validates the model's structured actions and applies our question policy and content checks. LiveKit connects that pipeline to the call.”

### Why we selected Sarvam

The repository records that the team selected Sarvam after trying other providers, with Indic language quality and response speed as motivations. A consistent provider and one API key also simplify integration.

Say: “Sarvam fits our Indian-language voice use case, and our implementation uses it across all three stages.” Do not claim it is the fastest, cheapest or most accurate provider without a reproducible comparison. Configured language support and measured language quality are separate claims.

## 6. What happened to the medicine dataset?

The project contains an earlier OTC recommendation implementation: a formulary, retrieval, patient-context filters, a prescription denylist, and legacy recommendation agents.

**The active conversation flow now prohibits medicine names, doses, supplements and branded remedies.** Its `medicine_free()` check combines output scanning, dose-pattern checks and known formulary names. The formulary is therefore still relevant as a source of names to block, even though the active flow does not recommend its medicines.

The configured runtime formulary is `data/formulary.json`. The open editor file, `indian_otc_medicines_curated.json`, is not the configured runtime data source. Its visible contents also contain a JSON syntax error and conflicting dose wording. This study does not validate or recommend any entry in it.

For judges, say: “We explored curated OTC guidance earlier. The current prototype focuses on medicine-free self-care and referral guidance.” Do not demo medication prescribing as a current feature.

## 7. How this could help people

| Intended beneficiary | Potential benefit | Evidence still needed |
|---|---|---|
| Basic-phone user | Can use voice access without installing an app or using mobile data on their handset | Local phone access, affordability and usability tests |
| Person more comfortable in a regional language | Can explain a concern and hear guidance in a familiar language | Native-speaker review and comprehension checks |
| Person uncertain about a symptom | Receives practical next steps and advice about seeking care | Clinician assessment of guidance and referral appropriateness |
| Family caregiver | Can seek guidance through a conversational interface | Tests covering shared phones and who the patient is |
| Clinician or community program | Could receive a structured summary with permission | A completed handoff workflow and staff usability evaluation |

The caller-facing phone route does not require an app or mobile data. **The backend still requires internet connectivity and cloud services.** Phone connectivity and charges still apply.

Reduced delay, fewer unnecessary journeys, better understanding and lower staff workload are impact hypotheses. They are not outcomes demonstrated by this repository. Clinical outcome claims would need an appropriately designed study.

## 8. Safety and privacy: what we can honestly say

### Concrete controls

- An opening notice identifies MedLink as a voice assistant, not a doctor.
- The model proposes constrained actions rather than unrestricted speech.
- The application checks question count, assessment completeness and recognized repetition.
- Recorded model-extracted facts require an exact supporting caller quote. This checks textual support, not the truth or clinical interpretation of the fact.
- Active emergency red-flag detection runs before the model in `ConversationAgent.llm_node()`.
- Medicine names and dose-like content are checked before accepted responses are rendered.
- Where localized curated recovery guidance exists, the application replaces generated self-care, avoidance and referral fields with that guidance.
- Provider errors produce a service-error response while preserving state; an outage is not treated as medical evidence.
- PostgreSQL storage includes consent gates, phone lookup hashing, encrypted phone fields, retention and erasure functionality.

### Material limits

The reference content has not been established as clinician-reviewed. Schema validation does not certify that an explanation is medically appropriate. Finite medicine-name lists and dose patterns cannot guarantee every unsafe phrase is caught.

The emergency shortcut has an important qualification: a broad English negation check can defer a red-flag hit to the model when words such as “no” or “not” occur in the latest utterance. Therefore do not promise that every emergency always bypasses the model or that emergency sensitivity is established.

Clinical storage consent checks exist, but the active `ConversationAgent` does not expose the legacy spoken consent tool. A complete consent capture journey must be verified before presenting persistent patient records as a finished caller experience.

PostgreSQL phone protection does not mean every store anonymizes the caller. Firestore uses readable phone-based identifiers and currently permits all authenticated clients to read patient documents. Production access controls need tightening. Some PostgreSQL demographic fields are also plain text.

Automatic doctor connection, ambulance dispatch, SMS delivery and appointment booking are not implemented. Emergency speech tells the caller to seek help; it does not place a call on their behalf.

## 9. Evidence and validation

The saved 13 November follow-up audit reports **519 regression tests passed**. Its 16 synthetic text attempts produced 13 completed conversations: 12 routine assessments and one emergency response. Three regional-language attempts failed with API errors. The completed routine conversations each asked four follow-up questions; the emergency case gave urgent instructions without routine questions.

That audit also records meaningful failures: some regional-language assessments contained English explanations, two intended Tamil trials drifted into Malayalam through the caller-translation harness, some questions repeated supplied facts, and an unsupported complaint received generic questions.

An earlier implementation-verification report records 490 tests and different question behavior. These reports represent different snapshots. The current source sets a four-to-six routine question policy and speaks a shorter closing; do not combine all historical results into a claim about today's exact implementation.

For this study, a fresh `uv run pytest -q` was attempted but could not start because the execution environment could not initialize the default uv cache. A second offline attempt with a workspace cache could not query the Python interpreter due to access denial. **No fresh passing-test claim is made here.** No live API conversation or telephone call was placed for this study.

Before the event, verify the current worker, handset playback, one routine concern, one supported language switch, an emergency scenario and a refusal to provide medicine advice. Keep a prerecorded demonstration as a connectivity backup and label it clearly.

## 10. Three-minute pitch script

> Imagine someone in a village notices a health problem. They may have a basic phone, may be uncomfortable describing symptoms in English, and may be unsure whether they need professional care. Their first question is simple: “What should I do next?”
>
> We built MedLink to make that first conversation easier.
>
> MedLink is a multilingual, phone-based AI health guidance prototype. A caller describes what is troubling them. The assistant asks focused questions, explains a possible cause with uncertainty, and gives practical self-care and guidance on when to seek medical help. Our implementation supports English, Hindi, Tamil, Telugu, Kannada and Malayalam.
>
> For the caller, the interface is an ordinary conversation. There is no app to install for the phone route. The backend connects the call to our voice pipeline through LiveKit.
>
> Sarvam powers all three AI stages. Saaras converts the caller's speech into text, the Sarvam conversation model proposes the next conversation action, and Bulbul speaks the accepted reply in the selected language.
>
> Our contribution is the application around that pipeline. We retrieve curated symptom guidance, track the conversation, validate structured model responses and control what reaches the caller. Routine conversations ask one focused question at a time, normally four, with a maximum of six. Emergency handling can interrupt that routine flow.
>
> We have deliberately limited the current prototype to medicine-free guidance. It does not prescribe or confirm a diagnosis. The application checks for medicine content, and emergency responses tell callers to seek immediate help rather than promising that we have dispatched it.
>
> We have software tests and synthetic conversation audits, and those audits have also exposed issues we still need to solve, including language consistency and provider failures. Our next step is clinician and native-speaker review, followed by a supervised pilot and measured telephone testing.
>
> The value we want to deliver is a clearer first step for people who find existing digital interfaces difficult to use. With clinical partners and accessible local phone infrastructure, we want to test whether this familiar-language conversation can help people understand what to do next.
>
> MedLink: a familiar language, an ordinary phone, a clearer next step.

Rehearse aloud and adjust to the event's time limit. Put the demo between the product explanation and the architecture explanation if the format permits.

## 11. Thirty-second version

> “MedLink is a phone-based AI health guidance prototype for people who find English apps or typed interfaces difficult to use. It listens in a supported language, asks focused questions, and gives practical self-care and advice on seeking professional help. Sarvam powers speech recognition, conversation reasoning and speech generation; our application validates the conversation and response content. We have built the prototype, and our next step is clinical review and supervised real-world validation.”

## 12. How to present it to judges

Use one ordinary person's situation to introduce the need. Label that person as an illustrative scenario unless it comes from a real, consented interview. Demonstrate the interaction early, then explain why the implementation matters.

Make three points memorable:

1. **Access:** ordinary phone interaction and familiar-language conversation.
2. **Implementation:** Sarvam at all three AI stages, connected through LiveKit and our application policy.
3. **Responsibility:** a bounded prototype with explicit clinical and field-validation work ahead.

Use “designed to,” “implemented,” “observed in these simulations” and “we plan to measure” precisely. Do not turn intended benefits into measured outcomes.

### Suggested six-slide outline

| Slide | Message | Evidence or visual |
|---|---|---|
| 1 | The uncertainty before seeking care | One illustrative caller story |
| 2 | A conversation through an ordinary phone | Three steps: call, describe, understand next step |
| 3 | Show the prototype | Short routine demonstration |
| 4 | Where Sarvam fits | STT -> validated conversation -> TTS diagram |
| 5 | What works and what remains | Tests, audit findings, clinical and audio limits |
| 6 | Next milestone and ask | Clinical partner, native-language reviewers, supervised pilot |

Avoid spending presentation time listing every library. Explain a technology through the user benefit or control it enables.

## 13. Demo plan

Use a scenario already rehearsed with the current build. Gradual hair loss has a dedicated reference with localized question guidance and is a useful routine example. Describe it as a simulated caller, not a patient consultation.

| Step | Demonstration | Point to narrate |
|---|---|---|
| 1 | Start a call or web audio session | “The caller begins with speech.” |
| 2 | Give a brief routine concern | “The assistant gathers missing details one at a time.” |
| 3 | Answer the focused questions | “The application limits the question flow.” |
| 4 | Hear the short assessment | “It explains uncertainty and practical next steps.” |
| 5 | Ask a relevant follow-up | “The conversation can continue after the assessment.” |
| 6 | Show a separate rehearsed language switch | “The selected language changes the speech output.” |

If time permits, demonstrate a separate clear emergency input and show that the system responds with urgent instructions. Explain that it does not actually call emergency services. Do not interrupt an emergency demonstration with further routine questions.

The configured demonstration number is US-based. Check dialing permissions and charges before using an Indian handset. Do not describe this as a domestic toll-free service. If using a web session because handset testing is incomplete, label the access channel accurately.

## 14. Likely judge questions and strong answers

**What makes this more than connecting three APIs?**

“The APIs provide transcription, language generation and speech. We built the conversation policy, structured-action validation, symptom-reference retrieval, evidence checks for extracted facts, medicine-content restrictions, emergency responses and optional call-record integration. Those components determine how the service behaves for this use case.”

**Why would someone use this instead of a general chatbot?**

“Our intended advantage is the phone interface, supported-language voice interaction and focused health guidance workflow. We still need comparative usability testing to establish which users benefit and how much.”

**Can it diagnose illness?**

“It can suggest a possible explanation with uncertainty. It cannot confirm a diagnosis or replace an examination. We position it as guidance about next steps.”

**How do you prevent hallucinations?**

“We reduce exposure through reference retrieval, structured responses, application validation, quotation checks and content restrictions. These controls have limits; they do not prove that every explanation is correct. Clinician review and scenario evaluation remain necessary.”

**Does it recommend medicines?**

“The active version does not. We retain an earlier formulary and filtering implementation, but the current conversation path blocks medicine names and doses and focuses on self-care and seeking care.”

**Does it work without internet?**

“The caller can use a normal phone call without mobile data or an app. Our backend needs internet connectivity to LiveKit and Sarvam.”

**Have you tested all six languages?**

“All six are configured, and we have synthetic tests. The latest audit found mixed-language responses, translation-harness drift and failed attempts. We need reviewed native-language scripts and telephone testing before claiming consistent performance.”

**What if Sarvam is unavailable?**

“The application preserves conversation state and reports a service problem. There is no current cross-provider failover. Availability and safe outage handling are part of our next evaluation.”

**Will it connect me to a doctor or send an ambulance?**

“That automation is not implemented. The current service gives instructions about seeking help. Real referral and emergency-service integrations would require separate partnerships and implementation.”

**How is patient information protected?**

“We implemented consent checks and PostgreSQL phone hashing and encryption, with retention and deletion functions. The full active consent journey and Firestore access controls still need work before real patient deployment.”

**How much does each call cost?**

“We have not established a validated per-call cost. We need to measure telephony and media minutes, STT and TTS usage, LLM usage, hosting and storage, then compare that with conversation completion and quality.”

**What is the business model?**

“A possible model is institutional sponsorship through clinics, NGOs or community-health programs, keeping caller access affordable. That is a proposed model, not validated demand or existing revenue.”

**What do you need next?**

“A clinical review partner, native-language reviewers, accessible local telephony and a supervised pilot. We want to measure understanding, appropriate care guidance, language quality, reliability and cost before expanding.”

## 15. Pilot and evaluation plan

Start with clinician-authored cases and reviewed translations. Check emergency behavior, routine guidance, unsupported concerns, medicine requests, shared-phone situations, explicit refusal to answer, language switching and provider outages. Review the complete spoken output, not only JSON structure.

Then conduct supervised usability sessions with consenting participants. Avoid treating an unreviewed prototype as a standalone clinical service.

| Metric | What it establishes |
|---|---|
| Clinician-rated guidance and referral appropriateness | Whether the content and next steps are suitable for the scenario |
| Emergency sensitivity and false-alert rate | How often urgent cases are recognized and how often routine cases are escalated |
| Native-speaker rating of every spoken field | Language correctness and comprehensibility |
| User recall of next steps | Whether people understand the guidance |
| Completion and provider-error rates | Operational reliability |
| Repeated questions and follow-up count | Conversation burden |
| End-of-speech to audible-response latency | Actual interaction responsiveness |
| Cost per completed, reviewed conversation | Operational feasibility |
| Consent capture and access-control verification | Whether records are handled as intended |

Set acceptance criteria with clinical partners before evaluating. Do not invent performance percentages for the pitch.

## 16. Repository evidence map

- `src/agent.py`: active entrypoint, LiveKit/Sarvam wiring, Krisp selection and call-end saves.
- `src/config.py`: supported languages, provider defaults, data paths and feature flags.
- `src/speech/providers.py`: Sarvam streaming STT and TTS construction.
- `src/llm_factory.py`: Sarvam LLM construction.
- `src/workflows/conversation.py`: active actions, question policy, validation, red-flag shortcut, recovery and language switching.
- `src/knowledge/triage_kb.py` and `data/triage_kb.yaml`: reference matching and guidance content.
- `src/safety/redflags.py`, `src/safety/guardrails.py`: emergency matching and output checks.
- `src/db/` and `src/firestore_export.py`: optional records, consent checks and export.
- `firestore.rules`: current client access boundaries and documented production gaps.
- `sip/README.md`: configured telephone route and outstanding verification.
- `demo_audit_out/demo_followup_report.md`: 519-test historical result, conversation outcomes and language/API failures.
- `demo_audit_out/implementation_verification.md`: earlier snapshot; useful history, not the current behavior specification.

This study added documentation only. It did not change the application, medicine data, credentials or deployment.

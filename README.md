# MedLink — a voice health helpline for rural India

MedLink is a **voice AI agent you reach on an ordinary phone call**. No
smartphone, no app, no data connection. It listens to what is wrong, asks a few
targeted questions, works out how serious it is, gives safe over-the-counter
guidance when that is appropriate — and pushes hard toward a real doctor when it
is not.

It speaks **English, Hindi, Tamil, Telugu, Kannada and Malayalam**, including
the mixed way people actually talk (*"Enakku fever irukku, yesterday la irundhu
romba tired aa irukken"*).

**It is not a doctor and never pretends to be one.** It is a safe first point of
contact for people whose alternative is a long trip to a clinic, or nothing.

---

## The core design decision

An LLM is never allowed to choose a medicine, judge urgency, or decide an
emergency. Those run in deterministic code; the model handles conversation.

```
caller audio
     │
     ├─▶ red-flag detector ──── emergency phrase? ──▶ Escalate agent
     │   (deterministic, 13 categories, 6 languages)   (has NO medicine tool)
     │   runs BEFORE the LLM; suppresses its reply
     │
     ▼
Intake ──▶ Triage ──▶ Recommend        ← LiveKit agent handoffs,
                 └──▶ Escalate            one small prompt each
     │           │
     │           ├─ triage KB: which questions matter for THIS symptom,
     │           │             and which drug classes are ever appropriate
     │           │
     │           └─ severity scored in code, never by the model
     │
     ▼
medicine filter: OTC-only · India-available · age · pregnancy ·
                 contraindications · interactions · duration limit ·
                 no duplicate active ingredients
     │           (spoken text built ONLY from structured formulary fields)
     ▼
output guardrail — blocks any prescription drug the model volunteers
     │
     ▼
caller hears it
```

Four independent layers have to fail before a caller hears something unsafe.

---

## What is built

| Area | Status |
|---|---|
| Deterministic emergency detection (13 categories, 6 languages + romanised) | ✅ |
| Intake → Triage → Recommend → Escalate workflow with handoffs | ✅ |
| Triage knowledge base — 22 presentations, WHO IMCI / ICMR STG / NICE CKS | ✅ |
| OTC medicine safety pipeline + curated formulary | ✅ |
| Output guardrail (prescription-drug denylist) + prompt-injection guard | ✅ |
| PostgreSQL call history, returning-caller recall, consent gating, erasure | ✅ |
| Speech: Sarvam AI (Saaras STT + Bulbul TTS), both streaming — **active** | ✅ |
| Bhashini speech wrapper (better Indic quality) behind a one-setting switch | ✅ *(awaiting API key)* |
| Telephony (SIP inbound), doctor escalation automation, SMS | ⏳ next |

**228 tests**, `ruff` clean.

## Everything runs on free infrastructure

This was a hard constraint, and `MEDLINK_FREE_TIER_ONLY=true` (the default)
enforces it in code — the provider factory refuses to construct anything that
bills.

| Need | Choice | Cost |
|---|---|---|
| Speech (STT/TTS) | **Sarvam AI** — `saaras:v3-realtime` STT and `bulbul:v3` TTS, both native WebSocket streaming. Built for Indian languages, so all six MedLink languages are first-class. **Bhashini** (Government of India ULCA/Dhruva) stays wired as the free fallback. | prepaid credits |
| Voice-activity detection | **Silero**, on-device | free |
| Reasoning | **Fallback chain: Gemini → Groq → Cerebras.** `gemini-3.1-flash-lite` primary; falls through on a rate limit, API error, or slow first token. Kept off Sarvam so the chattiest stage does not eat the speech rate limit. | free tiers |
| Transport, turn detection, SIP | **LiveKit Cloud** free tier | free |
| Retrieval | lexical BM25 + curated multilingual aliases — no embedding model, no vector DB | free |
| Database | self-hosted PostgreSQL | free |

## Privacy

- **A raw phone number is never stored.** Lookup uses an HMAC hash; a reversible
  encrypted copy is written *only* after consent.
- Operational records (timings, triage outcome) are always kept — they are what
  makes the service auditable. The caller's **words, answers and complaint** are
  stored only with consent, and a previous call is recalled only if they
  consented at the time.
- Right to erasure by phone number; retention purge for transcripts.
- Every consent decision and data access lands in an append-only audit log.

---

## Run it

```bash
uv sync
cp .env.example .env.local     # then fill in the keys
uv run python src/agent.py console
```

**Speech runs on Sarvam; reasoning runs on its own provider chain.** One
`SARVAM_API_KEY` from [dashboard.sarvam.ai](https://dashboard.sarvam.ai) serves
STT and TTS. The LLM is deliberately elsewhere — it makes a call per caller turn,
and pinning it to the same vendor as the audio path doubled the load on one rate
limit. `llm.FallbackAdapter` moves down the chain mid-call, with no restart, when
a provider rate-limits, errors, or takes longer than
`MEDLINK_LLM_ATTEMPT_TIMEOUT` (5s) to produce a first token.

- `SARVAM_API_KEY` — required, speech only (STT + TTS).
- `GEMINI_API_KEY` / `GROQ_API_KEY` / `CEREBRAS_API_KEY` — the LLM chain, tried in that order. **At least one is required**; a provider with no key is dropped from the chain at startup, so one is enough to run. All three have a free tier.

Model IDs are named constants at the top of [`src/config.py`](src/config.py); override any of them with the `MEDLINK_*` variables in `.env.example`.
- `BHASHINI_API_KEY` / `BHASHINI_USER_ID` / `BHASHINI_PIPELINE_ID` — from [bhashini.gov.in](https://bhashini.gov.in); then set `MEDLINK_SPEECH_PROVIDER=bhashini` for stronger Tamil/Telugu/Kannada/Malayalam.

Optional call history:

```bash
docker compose up -d
export DATABASE_URL=postgresql://medlink:medlink@localhost:5432/medlink
export MEDLINK_ENABLE_DB=true
uv run alembic upgrade head
```

### Tests

```bash
uv run pytest                                   # 211 unit/integration tests, no network
lk agent simulate --scenarios scenarios.yaml    # 10 full conversation simulations
```

The pytest suite needs no API keys, no Docker and no network — the database
tests run on SQLite and the Bhashini tests use a mocked transport.

---

## Layout

```
data/
  redflags.yaml               emergency phrases, 6 languages — edit without touching code
  triage_kb.yaml              22 presentations: questions, severity, allowed drug classes
  formulary.json              curated OTC medicines, fully structured
  prescription_denylist.yaml  what must never be spoken
src/
  agent.py                    entrypoint — session wiring only
  workflows/                  intake · triage · recommend · escalate + pure routing logic
  safety/                     redflags · guardrails
  medicine/                   formulary retrieval + the hard safety filter
  knowledge/                  triage KB retrieval
  speech/                     Bhashini wrapper + provider factory
  db/                         models · repository · PII crypto
```

Built on [LiveKit Agents](https://github.com/livekit/agents). See
[AGENTS.md](AGENTS.md) for development conventions.

## About the commit history

This repository was created from the MIT-licensed
[`agent-starter-python`](https://github.com/livekit-examples/agent-starter-python)
template, and its full upstream history was kept for provenance. As a result the
GitHub contributor list and the older commits (anything before
`Add MedLink safety core`) belong to the LiveKit template, not to MedLink. All
MedLink work is authored by the repository owner.

## Disclaimer

MedLink provides general health information, not medical advice or diagnosis.
The clinical content is drawn from published guidelines but **has not been
reviewed by a licensed clinician** and is not fit for real patient use in its
current state. Do not deploy it to real callers without clinical sign-off and
regulatory review.

## License

MIT — see [LICENSE](LICENSE).

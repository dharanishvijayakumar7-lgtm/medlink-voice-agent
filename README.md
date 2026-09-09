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
| Bhashini speech (free) behind a provider factory | ✅ *(awaiting API key for a live smoke test)* |
| Telephony (SIP inbound), doctor escalation automation, SMS | ⏳ next |

**211 tests**, `ruff` clean.

## Everything runs on free infrastructure

This was a hard constraint, and `MEDLINK_FREE_TIER_ONLY=true` (the default)
enforces it in code — the provider factory refuses to construct anything that
bills.

| Need | Choice | Cost |
|---|---|---|
| Speech (STT/TTS) | **Bhashini** — Government of India ULCA/Dhruva | free |
| Voice-activity detection | **Silero**, on-device | free |
| Reasoning | **Gemini free tier** via Google AI Studio (*not* billed Google Cloud — `vertexai=False` is enforced) | free |
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

The agent starts and works with **no keys at all** — it falls back to LiveKit
Inference and logs a warning. Add keys to improve it:

- `GOOGLE_API_KEY` — free Gemini key from [aistudio.google.com/apikey](https://aistudio.google.com/apikey) (no card)
- `BHASHINI_API_KEY` / `BHASHINI_USER_ID` / `BHASHINI_PIPELINE_ID` — from [bhashini.gov.in](https://bhashini.gov.in)

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

## Disclaimer

MedLink provides general health information, not medical advice or diagnosis.
The clinical content is drawn from published guidelines but **has not been
reviewed by a licensed clinician** and is not fit for real patient use in its
current state. Do not deploy it to real callers without clinical sign-off and
regulatory review.

## License

MIT — see [LICENSE](LICENSE).

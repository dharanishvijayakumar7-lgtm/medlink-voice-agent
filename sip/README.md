# MedLink inbound telephony

## Current setup

- Dial **+1 484 398 9008** (`+14843989008`), a LiveKit-managed US local number.
- Project: `medlink-sih` (`p_3kagp74ekiz`).
- Number ID: `PN_PPN_QLpxiVr4dEm7`.
- Dispatch rule: `SDR_j4XMGsxuN5ga` (MedLink inbound dispatch).
- Agent name: `medlink-agent`.

The LiveKit API confirmed the number and inbound service are active and assigned
to this rule on November 12, 2026. Real phone audio verification remains pending.

```text
Caller -> LiveKit-managed US number -> MedLink dispatch rule
       -> individual medlink-call room -> medlink-agent -> Sarvam STT/LLM/TTS
```

Managed numbers need no third-party inbound SIP trunk, carrier password, Python
changes, or Docker changes. Do not add the managed number to `inbound-trunk.json`.

## Dispatch configuration

`dispatch-rule.json` records the current routing in CLI format: camelCase fields
inside `dispatch_rule`. The dashboard/API displays equivalent snake_case fields.
Its `trunkIds` contains the legacy VoiceLink trunk and the managed number ID,
matching the saved live rule. The dashboard adds the number ID when it is selected
under Phone numbers; the API confirmed this association.

`roomConfig.agents[0].agentName` must match `LIVEKIT_AGENT_NAME`, whose default is
`medlink-agent`. Explicit agent dispatch is required. The empty deployment value
in the live JSON corresponds to the default production selection in the dashboard.

For this existing project, edit **Telephony -> Dispatch rules -> MedLink inbound
dispatch** instead of creating an overlapping rule. Keep the new number selected
under Phone numbers. The expired VoiceLink trunk can remain selected; it is not
needed for managed calls.

For another project, replace the project-specific IDs before creating a rule:

```powershell
lk sip dispatch create sip/dispatch-rule.json
```

Assign the managed number to that rule using **Telephony -> Phone numbers ->
number menu -> Assign dispatch rule**. Never reuse IDs from another project.

## Run the agent

From the `medlink-agent` directory:

```powershell
uv run src/agent.py start
```

Keep the terminal open for local hosting. `.env.local` must contain the existing
project's `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`, and
`SARVAM_API_KEY`. Do not print or commit credentials. The Dockerfile already runs
this command. If the intended deployed worker is already online, do not start a
second local copy: multiple workers with the same name may receive calls.

The CLI is optional for dashboard setup. To install it on Windows if needed:
`winget install LiveKit.LiveKitCLI`.

## Demo verification

### Current conversation policy

Phone calls now start `ConversationAgent`, a shared structured-action flow for
English, Hindi, Tamil, Telugu, Kannada, and Malayalam. It normally asks 3–6
focused questions across the whole conversation, with a hard ceiling of six.
Detailed opening descriptions can need fewer; emergencies bypass questioning.
The final response explains a possible cause, why it fits, self-care, things to
avoid, and when to get medical help. It offers no medicine recommendations or
doses. Caller questions afterward do not start another diagnostic questionnaire.

Language follows the caller automatically after the initial English greeting.
The existing Sarvam models and protected-reply audio policy are unchanged.
The legacy four-agent/medicine workflow and its settings remain in the repository
for compatibility, but are not the active phone entrypoint. `MEDLINK_DEMO_MODE`
does not shorten this new flow or force an early farewell.

Run `uv run python scripts/audit_selfcare.py` for synthetic conversations covering
four routine complaints and an emergency in every language. Results are saved
in `demo_audit_out/selfcare_conversations.json`; no phone calls are placed and
database/Firestore writes are disabled for the audit. These simulations are not
a substitute for native-speaker review or testing audio on the actual phone.

### Audio setup

Set `MEDLINK_ALLOW_BARGE_IN=false` in `.env.local`. The agent finishes each reply
before accepting the next answer. Wait until it stops speaking: speech and noise
received during a reply are discarded and will not be answered later. This is
independent of `MEDLINK_DEMO_MODE`; that setting does not shorten the active flow.

Restart the existing local worker after applying these changes (stop it with
Ctrl+C, then run `uv run src/agent.py start`). Do not start a second worker.
Startup logs should show `allow_barge_in: false`; the input filter log should
show `krisp_telephony` for a phone caller. The first call after a restart must
also be checked, to catch startup-related stalls.

The agent uses LiveKit Krisp VIVA voice isolation, with its telephony variant
selected for SIP participants. It uses the existing LiveKit Cloud credentials;
no additional Sarvam, Bhashini, or Krisp API key is needed. Voice isolation can
incur LiveKit usage charges. See the [LiveKit audio filter documentation](https://docs.livekit.io/transport/media/noise-cancellation/).

Check six calls: quiet, environmental noise, and nearby speech, each on
speakerphone and earpiece. During each, introduce noise while the agent speaks,
then answer after it finishes. Confirm complete audible replies, correct answers
afterward, successful language switching and handoffs, and normal hangup.
Inspect worker logs for `agent speech committed` with `interrupted: false`,
`speech pipeline error`, false interruptions, and event-loop blocking warnings.
Provider synthesis completion alone does not prove the phone played all audio.

Set `MEDLINK_ALLOW_BARGE_IN=true` and restart to restore adaptive caller
interruptions. Keep other endpointing and audio-level settings unchanged until
real calls justify further tuning.

### Call checklist

1. Check remaining minutes in LiveKit's usage/billing dashboard. Published Build
   allowances are one US local number and 50 inbound minutes; this does not
   establish the project's current remaining allowance.
2. Confirm the worker registers as `medlink-agent`.
3. Dial `+14843989008` from the actual demo phone. Indian mobiles may need ISD
   activation and incur international charges. This is a US local number.
4. Complete three calls, checking greeting, two-way audio, requested language
   switching, and normal hangup. Caller ID comes from `sip.phoneNumber`;
   `MEDLINK_DEV_CALLER_PHONE` is only a development fallback.
5. Inspect **LiveKit -> Telephony -> Calls** and worker logs. No call record:
   check dialling/ISD and number status. Room without agent: check worker
   registration and exact agent name. Agent joins but fails to reply: check
   Sarvam API/credit errors and audio logs.
6. If summaries are enabled, verify export under the real caller's number.
   Sarvam API/hackathon credits and LiveKit allowances are separate.

## Legacy VoiceLink configuration

The old number `+919429397308` expired. Do not advertise it as active.
`inbound-trunk.json` is retained only for the legacy VoiceLink trunk
`ST_PmuSNZGEDAJ6`; do not apply it for the managed US number.

Historically VoiceLink used TCP 5060, PCMU/PCMA, and
`3kagp74ekiz.india.sip.livekit.cloud`, with digest credentials stored only on the
live trunk. Past failures included missing DID routing, a missing `.india.` host,
and an empty client wallet despite reseller funds. These settings do not apply
to the new managed number.

## References

- [LiveKit Phone Numbers](https://docs.livekit.io/telephony/start/phone-numbers/)
- [Telephony testing](https://docs.livekit.io/telephony/testing/)
- [LiveKit pricing](https://livekit.com/pricing)

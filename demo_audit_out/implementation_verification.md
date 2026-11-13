# Demo implementation verification — 13 November 2026

The active phone entrypoint uses `ConversationAgent`. Routine conversations have
a shared maximum of six follow-up question turns, normally three to six. A detailed
opening can require fewer. Assessment responses require five fields: possible
cause, explanation, self-care, avoidance, and when to seek care. The active flow
does not use the legacy medicine recommendation tools. English, Hindi, Tamil,
Telugu, Kannada, and Malayalam are supported.

## Verified results

- Final regression suite: **490 passed**.
- Six saved fever simulations, one per language: **three question turns each**,
  followed by a validated five-field assessment.
- Additional English diarrhoea, reflux, and headache simulations: **three
  question turns each**, followed by a validated five-field assessment.
- English emergency simulation: **zero routine questions** before emergency
  instructions. The response included calling 112 and seeking nearby help.
- Average across the nine routine simulations: **3.0 question turns**.
- Reflux selected the acidity reference, rather than the unrelated vomiting
  reference observed in the earlier audit.
- All ten simulations completed without a recorded exception. Invalid model
  proposals occurred in some turns and were corrected before speech was emitted.

The six-language records are in `six_language_fever_smoke.json`. The four
additional English records are in `selfcare_conversations.json`. These are
synthetic text conversations using the configured Sarvam model, not real calls.
Question turns are not a clinical measure of diagnostic accuracy. Simulated
callers can introduce extra details, and the results have not received clinician
or native-speaker review. Some responses still contain generic advice; passing
schema validation does not establish that every sentence is appropriate.

## Demo startup

Stop the existing worker with Ctrl+C, then run from `medlink-agent`:

```powershell
uv run src/agent.py start
```

The local configuration has `MEDLINK_ALLOW_BARGE_IN=false`. Replies finish before
new caller speech is accepted; callers should wait for the agent to finish.
Krisp voice isolation remains enabled. Existing Sarvam and LiveKit credentials
are sufficient; no Bhashini integration or replacement model was added.

Worker restart and end-to-end phone audio verification remain outstanding.
These text tests do not verify cellular audio, background-noise performance,
network latency, or full playback on the demo handset. Place a short call after
restarting, checking a complete assessment and language switching.

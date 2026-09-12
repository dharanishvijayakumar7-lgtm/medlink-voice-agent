# SIP / telephony config

Applied with the LiveKit CLI (`lk`). Checked in so the telephony setup is
reproducible rather than living only in a dashboard.

## Project facts

- LiveKit project: `medlink-sih` (`p_3kagp74ekiz`)
- **SIP subdomain: `3kagp74ekiz`** — the Project ID minus `p_`. NOT the URL
  subdomain `medlink-sih-1n96eczu`. Getting this wrong is the most common
  setup failure.
- India region-pinned SIP endpoint (what the provider must point at):

      3kagp74ekiz.india.sip.livekit.cloud;transport=tcp

  Region pinning is mandatory for Indian numbers - calls fail to connect on the
  global endpoint, with no obvious error.

## dispatch-rule.json — APPLIED

    lk sip dispatch create sip/dispatch-rule.json

Created `SDR_j4XMGsxuN5ga`. Routes each caller into their own
`medlink-call_*` room and dispatches the agent named `medlink-agent`.

`agentName` must match `settings.agent_name` in src/config.py. The agent sets an
explicit `agent_name`, which disables LiveKit's automatic dispatch - without this
rule a call connects and the caller hears silence.

## inbound-trunk.json — APPLIED

    lk sip inbound create sip/inbound-trunk.json

Created `ST_qLDzQ27bzhWf`.

Provider is **VoiceLink** (Elision Technolab LLP), DOT-licensed VNO, chosen over
Plivo because it accepts **individual KYC with a PAN card** rather than requiring
a registered business entity, and it documents a LiveKit inbound SIP integration
directly.

- DID: `+919429397308` (mobile series)
- `allowed_addresses`: `160.30.71.108`, the resolved A record of
  `sip.voicelink.co.in` (VoiceLink's SIP signalling host). Not optional — without
  it the trunk accepts SIP from anywhere, inviting toll fraud on a per-minute
  number. Re-resolve and update if VoiceLink ever changes that host.

VoiceLink side, for reference:

| Setting | Value |
|---|---|
| SIP Server Host | `3kagp74ekiz.india.sip.livekit.cloud` |
| SIP Server Port | `5060` |
| Transport | TCP (UDP if TCP will not establish) |
| Their signalling | `sip.voicelink.co.in:3300` |

The DID must also be routed to the trunk on VoiceLink's side (DID Call Routing) —
creating the trunk alone does not route calls to it.

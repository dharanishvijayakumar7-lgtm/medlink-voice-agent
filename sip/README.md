# SIP / telephony config

Applied with the LiveKit CLI (`lk`). Checked in so the telephony setup is
reproducible rather than living only in a dashboard.

**Status: working end to end** — a real call to `+91 94293 97308` reaches the
agent.

```
Indian mobile -> VoiceLink DID -> VoiceLink SIP trunk
             -> LiveKit inbound trunk -> dispatch rule -> medlink-agent
```

## Project facts

- LiveKit project: `medlink-sih` (`p_3kagp74ekiz`)
- **SIP subdomain: `3kagp74ekiz`** — the Project ID minus `p_`. NOT the URL
  subdomain `medlink-sih-1n96eczu`.
- LiveKit SIP listens on TCP **5060** (and 5061 for TLS). Nothing listens on
  3300 — that is VoiceLink's own port.

## dispatch-rule.json — applied as `SDR_j4XMGsxuN5ga`

    lk sip dispatch create sip/dispatch-rule.json

Routes each caller into their own `medlink-call_<caller>_<random>` room and
dispatches the agent named `medlink-agent`. `agentName` must match
`settings.agent_name` in src/config.py: the agent sets an explicit `agent_name`,
which disables LiveKit's automatic dispatch, so without this rule a call
connects to silence.

Note: `roomPrefix` does not hide the caller's number — it still appears in the
room name, and room names are logged.

## inbound-trunk.json — applied as `ST_PmuSNZGEDAJ6`

    lk sip inbound create sip/inbound-trunk.json
    lk sip inbound update --id <trunk id> \
        --auth-user <voicelink trunk username> --auth-pass <voicelink password>

- Both `+919429397308` and `919429397308` are listed: VoiceLink shows the DID
  without the `+`, and the trunk only matches numbers it lists.
- **Credentials are set on the live trunk but deliberately not in this file.**
  They must match the Username / Registration Password on the VoiceLink trunk.
  Pass them as flags; never commit them.
- No `allowed_addresses`: that field has to be enabled for the project by
  LiveKit support, and a trunk with empty `numbers` plus an un-enabled
  `allowed_addresses` matches nothing. Explicit numbers plus digest auth is what
  keeps the trunk from accepting arbitrary SIP.

## VoiceLink side

Provider: VoiceLink (Elision Technolab LLP), a DOT-licensed VNO. Chosen because
it accepts individual KYC with a PAN card and documents a LiveKit integration.

**SIP Trunk (Voice Services → SIP Trunk Management), BOT Provider = Custom:**

| Field | Value |
|---|---|
| SIP Server Host | `3kagp74ekiz.india.sip.livekit.cloud` |
| SIP Server Port | `5060` |
| Transport | TCP |
| Audio Format | PCMU + PCMA |

The `.india.` hostname *is* region pinning for inbound SIP — there is no
dashboard toggle. Indian numbers need it.

**DID Call Routing:** Inbound → SIP trunk → the LiveKit trunk. Creating the
trunk alone routes nothing.

## Troubleshooting — what actually broke, in order

1. **No DID call routing** on VoiceLink. Calls went nowhere.
2. **Host missing `.india.`.**
3. **Insufficient wallet balance — the real blocker.** The account is a
   *Reseller*; the DID and trunk belong to a *client* sub-account, and calls are
   billed to the client wallet, which was empty even though the reseller wallet
   had funds. Fix: enable **Bypass Wallet** on the client (Client Management),
   or Allocate Wallet from reseller to client.

To tell which side is failing, check both dashboards before changing config:

- **LiveKit Cloud → Telephony → Calls.** No call listed means the INVITE never
  arrived; the problem is upstream at VoiceLink.
- **VoiceLink → Reports → Call Logs Report.** The *Disconnect Reason* field
  names the cause (it is what showed the wallet problem).

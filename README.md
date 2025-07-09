<a href="https://livekit.io/">
  <img src="./.github/assets/livekit-mark.png" alt="LiveKit logo" width="100" height="100">
</a>

# Voice AI Assistant with LiveKit Agents

<p>
  <a href="https://cloud.livekit.io/projects/p_/sandbox"><strong>Deploy a sandbox app</strong></a>
  •
  <a href="https://docs.livekit.io/agents/">LiveKit Agents Docs</a>
  •
  <a href="https://livekit.io/cloud">LiveKit Cloud</a>
  •
  <a href="https://blog.livekit.io/">Blog</a>
</p>

A simple voice AI assistant built with [LiveKit Agents for Python](https://github.com/livekit/agents).

## Dev Setup

Clone the repository and install dependencies to a virtual environment:

```console
cd agent-starter-python
uv sync
```

Set up the environment by copying `.env.example` to `.env` and filling in the required values:

- `LIVEKIT_URL`
- `LIVEKIT_API_KEY`
- `LIVEKIT_API_SECRET`
- `OPENAI_API_KEY`
- `DEEPGRAM_API_KEY`

You can also do this automatically using the LiveKit CLI:

```bash
lk app env -w .env
```

Run the agent:

```console
uv run python src/agent.py dev
```

This agent requires a frontend application to communicate with. Use a [starter app](https://docs.livekit.io/agents/start/frontend/#starter-apps), our hosted [Sandbox](https://cloud.livekit.io/projects/p_/sandbox) frontends, or the [LiveKit Agents Playground](https://agents-playground.livekit.io/).


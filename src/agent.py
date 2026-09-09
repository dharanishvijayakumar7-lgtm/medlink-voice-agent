"""MedLink voice agent - entrypoint.

Wires the LiveKit session and starts the workflow. All conversation logic lives
in `workflows/`; all safety logic in `safety/` and `medicine/`. Keep this file
thin - the Dockerfile runs it directly (`uv run src/agent.py start`).
"""

import logging

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)
from livekit.plugins import ai_coustics

from config import settings
from db import repository as history
from llm_factory import build_llm
from session_state import MedLinkUserData
from speech import build_stt, build_tts
from workflows.intake import IntakeAgent

logger = logging.getLogger("agent")

server = AgentServer()


def _caller_phone(ctx: JobContext) -> str | None:
    """Best-effort caller number for SIP calls (used for returning-caller lookup)."""
    try:
        for participant in ctx.room.remote_participants.values():
            number = (participant.attributes or {}).get("sip.phoneNumber")
            if number:
                return number
    except Exception:  # pragma: no cover - never break a call over this
        logger.debug("could not read caller phone", exc_info=True)
    return None


@server.rtc_session(agent_name=settings.agent_name)
async def medlink_session(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    phone = _caller_phone(ctx)
    userdata = MedLinkUserData(
        caller_phone=phone,
        channel="pstn" if phone else "web",
    )
    ctx.log_context_fields["call_id"] = userdata.call_id

    # Look up a returning caller and open the call record before we greet, so
    # the intake agent can acknowledge prior history. Never fatal.
    await history.start_call(userdata)

    session = AgentSession[MedLinkUserData](
        userdata=userdata,
        llm=build_llm(),
        stt=build_stt(),
        tts=build_tts(),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            interruption={"mode": "adaptive"},
            preemptive_generation={"enabled": True},
        ),
    )

    async def _log_outcome():
        await history.finish_call(userdata)
        logger.info(
            "call ended",
            extra={
                "call_id": userdata.call_id,
                "disposition": userdata.disposition,
                "urgency": userdata.urgency,
                "escalated": userdata.escalated,
                "questions_asked": userdata.questions_asked,
                "summary": userdata.clinical_summary(),
            },
        )

    ctx.add_shutdown_callback(_log_outcome)

    await session.start(
        agent=IntakeAgent(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=ai_coustics.audio_enhancement(
                    model=ai_coustics.EnhancerModel.QUAIL_VF_S
                ),
            ),
        ),
    )

    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)

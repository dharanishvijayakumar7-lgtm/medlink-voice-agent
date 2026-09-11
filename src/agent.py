"""MedLink voice agent - entrypoint.

Wires the LiveKit session and starts the workflow. All conversation logic lives
in `workflows/`; all safety logic in `safety/` and `medicine/`. Keep this file
thin - the Dockerfile runs it directly (`uv run src/agent.py start`).
"""

import contextlib
import logging
import sys

from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    room_io,
)

# Import the provider plugins on the main thread: LiveKit refuses to register a
# plugin from the job worker thread, so the lazy `from livekit.plugins import
# ...` inside llm_factory / speech.providers must find it already registered.
# `sarvam` serves all three stages (STT, LLM, TTS) off SARVAM_API_KEY.
from livekit.plugins import ai_coustics, sarvam  # noqa: F401

from config import SUPPORTED_LANGUAGES, settings
from db import repository as history
from llm_factory import build_llm
from session_state import MedLinkUserData
from speech import build_stt, build_tts
from workflows.intake import IntakeAgent

logger = logging.getLogger("agent")

server = AgentServer()


def _caller_phone(ctx: JobContext) -> str | None:
    """Best-effort caller number for SIP calls (used for returning-caller lookup).

    Falls back to ``MEDLINK_DEV_CALLER_PHONE`` so console sessions, which carry
    no caller ID, still create a patient record and can exercise the
    returning-caller path. A real SIP number always wins.
    """
    try:
        for participant in ctx.room.remote_participants.values():
            number = (participant.attributes or {}).get("sip.phoneNumber")
            # Must be a real string. In console mode the room is a MagicMock, so
            # this lookup returns a truthy mock that blew up downstream in
            # normalise_phone() and killed every database write for the call.
            if isinstance(number, str) and number.strip():
                return number
    except Exception:  # pragma: no cover - never break a call over this
        logger.debug("could not read caller phone", exc_info=True)

    if settings.dev_caller_phone:
        logger.warning(
            "no caller ID; using MEDLINK_DEV_CALLER_PHONE - development only"
        )
        return settings.dev_caller_phone
    return None


@server.rtc_session(agent_name=settings.agent_name)
async def medlink_session(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    phone = _caller_phone(ctx)
    # A dev fallback number is not a real phone call, so don't log it as one.
    is_sip_call = bool(phone) and phone != settings.dev_caller_phone
    userdata = MedLinkUserData(
        caller_phone=phone,
        channel="pstn" if is_sip_call else "web",
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
            # "stt" = trust Sarvam's server-side VAD endpointing, which already
            # decides where an utterance ends and only emits a final there.
            # This replaced inference.TurnDetector(), a LiveKit-hosted end-of-turn
            # model that cost a network round trip on every single turn and then
            # usually agreed with Sarvam anyway.
            turn_detection="stt",
            interruption={"mode": "adaptive"},
            # Dead air after the caller stops talking. See config.py - the
            # LiveKit default ceiling of 3.0s was the largest single component
            # of per-turn latency.
            endpointing={
                "min_delay": settings.endpointing_min_delay,
                "max_delay": settings.endpointing_max_delay,
            },
            # Preemptive generation starts a speculative LLM call before the
            # caller's turn is confirmed and discards it if they keep talking.
            # ON: it is the largest remaining latency win, because the reply is
            # already streaming by the time the turn commits. The cost is extra
            # Sarvam LLM calls against the same credits - set
            # MEDLINK_PREEMPTIVE_GENERATION=false if the spend shows.
            preemptive_generation={"enabled": settings.preemptive_generation},
        ),
    )

    @session.on("user_input_transcribed")
    def _follow_caller_language(ev) -> None:
        """Point the TTS at whatever language the caller actually spoke.

        Sarvam's STT identifies the language per utterance (`language="auto"`),
        but the TTS was constructed with a fixed `target_language_code`, so every
        reply was synthesised as if it were English no matter what the LLM wrote.
        A Tamil answer read with an English voice is what made the agent sound
        like it only spoke Hindi and English.

        Only final transcripts carry a settled language - the interim ones
        default to en-IN and would otherwise flip the voice mid-sentence.
        """
        if not getattr(ev, "is_final", False):
            return
        code = getattr(ev, "language", None)
        if not code or code not in SUPPORTED_LANGUAGES.values():
            return  # unknown, or outside the six MedLink supports
        if code == userdata.language:
            return

        userdata.language = code
        try:
            session.tts.update_options(target_language_code=code)
            logger.info("caller language detected, TTS switched to %s", code)
        except Exception:
            logger.exception("could not switch TTS to %s", code)

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
    # Windows terminals often default to a legacy codepage (e.g. cp1252) that
    # cannot encode the emoji the LiveKit CLI prints on startup, which crashes
    # `console` mode with a UnicodeEncodeError. Force UTF-8 on the std streams.
    for _stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    cli.run_app(server)

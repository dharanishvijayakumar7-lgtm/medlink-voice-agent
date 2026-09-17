"""MedLink voice agent - entrypoint.

Wires the LiveKit session and starts the workflow. All conversation logic lives
in `workflows/`; all safety logic in `safety/` and `medicine/`. Keep this file
thin - the Dockerfile runs it directly (`uv run src/agent.py start`).
"""

import asyncio
import contextlib
import logging
import sys

from livekit import rtc
from livekit.agents import (
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    room_io,
)

# Import the provider plugins on the main thread: LiveKit refuses to register a
# plugin from the job worker thread, so the lazy `from livekit.plugins import
# ...` inside llm_factory / speech.providers must find it already registered.
# `sarvam` serves all three stages (STT, LLM, TTS) off SARVAM_API_KEY.
from livekit.plugins import ai_coustics, sarvam  # noqa: F401

import firestore_export
from config import settings
from db import repository as history
from llm_factory import build_llm
from session_state import MedLinkUserData
from speech import build_stt, build_tts
from workflows.intake import IntakeAgent

logger = logging.getLogger("agent")

server = AgentServer()


# How long to wait for the caller to appear in the room after connecting. A SIP
# caller is normally already there when the agent is dispatched, so this only
# bounds the pathological case.
CALLER_WAIT_TIMEOUT = 5.0


async def _caller_phone(ctx: JobContext) -> str | None:
    """Best-effort caller number for SIP calls (used for returning-caller lookup).

    Must run after ``ctx.connect()``: before that the room has no remote
    participants, so the SIP caller's ``sip.phoneNumber`` attribute was never
    found and every real phone call was filed under the dev fallback number.

    Falls back to ``MEDLINK_DEV_CALLER_PHONE`` so console sessions, which carry
    no caller ID, still create a patient record and can exercise the
    returning-caller path. A real SIP number always wins.
    """
    if ctx.room.name != "console":  # console mode has no participants to wait for
        try:
            # Default kinds cover SIP and ordinary participants, so a web tester
            # returns immediately instead of waiting out the timeout.
            participant = await asyncio.wait_for(
                ctx.wait_for_participant(), timeout=CALLER_WAIT_TIMEOUT
            )
            attributes = participant.attributes or {}
            if participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP:
                number = attributes.get("sip.phoneNumber")
                # Must be a real string - a mocked room returns a truthy mock
                # that would blow up downstream in normalise_phone().
                if isinstance(number, str) and number.strip():
                    logger.info("caller identified from SIP", extra={"room": ctx.room.name})
                    return number
            # Loud on purpose: a silent fallback here is what filed real phone
            # calls under the dev number with no trace in the logs.
            logger.warning(
                "joined participant has no usable caller number",
                extra={
                    "room": ctx.room.name,
                    "participant_kind": participant.kind,
                    "attribute_keys": sorted(attributes),
                },
            )
        except asyncio.TimeoutError:
            logger.warning(
                "no participant joined within %ss", CALLER_WAIT_TIMEOUT,
                extra={"room": ctx.room.name},
            )
        except Exception:  # never break a call over caller ID
            logger.exception("could not read caller phone", extra={"room": ctx.room.name})

    if settings.dev_caller_phone:
        logger.warning(
            "no caller ID; using MEDLINK_DEV_CALLER_PHONE - development only"
        )
        return settings.dev_caller_phone
    return None


def _interruption_options() -> dict:
    """How the caller may cut in while the agent is talking."""
    if not settings.allow_barge_in:
        # The agent always finishes its sentence. Whatever is heard meanwhile -
        # its own echo, bystanders, the caller - is dropped, so nothing can
        # chop its speech. The escape hatch for a speakerphone in a crowded room.
        return {"enabled": False}
    return {
        "mode": "adaptive",
        # Phone lines carry echo and noise. On a real call, 1- and 3-character
        # "utterances" at ~0.46 confidence kept interrupting the agent and
        # restarting its reply. A real barge-in is at least two words and most
        # of a second of speech.
        "min_words": 2,
        "min_duration": 0.8,
        # A pause that turns out not to be the caller resumes after this. The
        # default 2.0 s left a hole in the middle of a word on speakerphone.
        "false_interruption_timeout": settings.false_interruption_timeout,
    }


@server.rtc_session(agent_name=settings.agent_name)
async def medlink_session(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    # Connect before anything else: the SIP caller (and their phone number) is
    # only visible as a remote participant once the agent is in the room.
    await ctx.connect()
    phone = await _caller_phone(ctx)
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
            # Semantic end-of-turn model: decides whether the caller has actually
            # FINISHED, not just paused. It was swapped for turn_detection="stt"
            # to save a round trip, and on a real phone call that made the agent
            # cut in on a natural ~1s pause mid-sentence and answer half a thought.
            # Being talked over is worse than a slightly slower reply.
            turn_detection=inference.TurnDetector(),
            interruption=_interruption_options(),
            # Wait after the caller stops talking. The turn detector moves this
            # toward min_delay when it is confident they finished, and toward
            # max_delay when they sound mid-thought.
            endpointing={
                "min_delay": settings.endpointing_min_delay,
                "max_delay": settings.endpointing_max_delay,
            },
            # Preemptive generation starts a reply before the caller's turn is
            # confirmed. Off by default: combined with eager endpointing it made
            # the agent start answering while the caller was still talking.
            preemptive_generation={"enabled": settings.preemptive_generation},
        ),
    )

    @session.on("user_input_transcribed")
    def _log_stt_final(ev) -> None:
        """Diagnostic only: shows whether the STT is producing words at all.

        This used to also switch the TTS to whatever language was detected. That
        made the agent flip language on a single mixed-in word, so the voice now
        only changes when the caller actually asks (workflows.base).
        """
        if not getattr(ev, "is_final", False):
            return
        transcript = getattr(ev, "transcript", "") or ""
        logger.info(
            "stt final",
            extra={
                "call_id": userdata.call_id,
                "language": getattr(ev, "language", None),
                "chars": len(transcript.strip()),
            },
        )

    # The agent's own voice comes back into the mic on speakerphone, and a call
    # from a room full of people carries other voices too. Either can cut the
    # agent off mid-sentence. LiveKit resumes the speech after
    # `false_interruption_timeout` (2.0s by default), so this shows up as a gap
    # rather than as mangled words - these two lines are here so the logs say
    # whether it is happening at all, instead of it being guessed at again.
    @session.on("agent_state_changed")
    def _track_agent_speech(ev) -> None:
        """Tell the echo guard when the agent's voice is going down the line.

        Only while it is speaking - and for a moment after - can a transcript
        be its own voice coming back through a speakerphone.
        """
        if ev.new_state == "speaking":
            userdata.agent_speech.started()
        elif ev.old_state == "speaking":
            userdata.agent_speech.stopped()

    @session.on("agent_false_interruption")
    def _log_false_interruption(ev) -> None:
        logger.warning(
            "agent was cut off by something that was not speech",
            extra={"call_id": userdata.call_id, "resumed": ev.resumed},
        )

    @session.on("overlapping_speech")
    def _log_overlapping_speech(ev) -> None:
        logger.info(
            "caller and agent spoke at once",
            extra={
                "call_id": userdata.call_id,
                # False means it was judged a backchannel, not a real barge-in.
                "interruption": ev.is_interruption,
                "seconds": ev.total_duration,
            },
        )

    async def _log_outcome():
        await history.finish_call(userdata)
        # App-facing copy of the call summary. Independent of Postgres, bounded
        # by its own timeout, and never raises - see firestore_export.
        await firestore_export.export_call(userdata)
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

    # Save the call exactly once, as early as possible.
    # The job's shutdown callback alone was not enough: on hangup LiveKit closes
    # the *session* at once, but the *job* lingers until the empty room times out.
    # A worker stopped in that window (Ctrl+C, redeploy, crash) lost the whole
    # call - a real phone call ended with no Postgres summary and nothing in
    # Firestore. finish_call inserts rows, so it must never run twice.
    outcome: asyncio.Task | None = None

    def _save_outcome() -> asyncio.Task:
        nonlocal outcome
        if outcome is None:
            outcome = asyncio.create_task(_log_outcome())
        return outcome

    @session.on("close")
    def _on_session_close(_ev) -> None:
        _save_outcome()  # caller hung up (or the session errored): save now

    async def _on_job_shutdown() -> None:
        # Fallback if the session never emitted close; otherwise waits for the
        # save already in flight so the worker doesn't exit mid-write.
        await _save_outcome()

    ctx.add_shutdown_callback(_on_job_shutdown)

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


if __name__ == "__main__":
    # Windows terminals often default to a legacy codepage (e.g. cp1252) that
    # cannot encode the emoji the LiveKit CLI prints on startup, which crashes
    # `console` mode with a UnicodeEncodeError. Force UTF-8 on the std streams.
    for _stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(AttributeError, ValueError):
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    cli.run_app(server)

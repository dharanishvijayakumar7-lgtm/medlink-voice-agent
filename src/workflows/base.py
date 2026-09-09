"""Shared base for every MedLink workflow agent.

Its one critical job is the **red-flag safety hook**: `on_user_turn_completed`
runs on every transcribed user turn *before* the LLM sees it. If a deterministic
emergency phrase matches, the LLM's reply is suppressed entirely
(`StopResponse`) and the session is force-handed to the escalation agent. The
model never gets the chance to talk a caller out of an emergency.
"""

from __future__ import annotations

import logging

from livekit.agents import Agent, ChatContext, ChatMessage, StopResponse

from config import settings
from safety.redflags import detect_redflag
from session_state import MedLinkUserData

logger = logging.getLogger("medlink.workflow")

# Spoken while the caller is still on the line, in every agent's voice.
SHARED_STYLE = """\
# How you speak
- You are talking to someone on a phone call, often in rural India, who may not
  have much schooling. Use short, plain sentences. No medical jargon.
- Speak warmly and unhurriedly. Never rush or lecture.
- Ask ONE question at a time, then stop and listen.
- Reply in whatever language the caller uses, including mixed Hindi/Tamil/English.
- Say medicine names slowly and clearly.
- Never claim to be a doctor and never state a confident diagnosis.
"""


class MedLinkAgent(Agent):
    """Base class carrying the safety hook and shared voice guidance."""

    @property
    def data(self) -> MedLinkUserData:
        return self.session.userdata

    async def on_user_turn_completed(
        self, turn_ctx: ChatContext, new_message: ChatMessage
    ) -> None:
        text = new_message.text_content or ""

        # --- deterministic emergency guard (runs before the LLM) ---
        hit = detect_redflag(text)
        if hit is not None and hit.is_emergency and not self.data.emergency_handled:
            logger.warning(
                "red flag detected",
                extra={
                    "call_id": self.data.call_id,
                    "category": hit.category_id,
                    "matched": hit.matched_term,
                },
            )
            self.data.red_flag = hit
            self.data.emergency_handled = True

            # Import here to avoid a circular import at module load.
            from workflows.escalate import EscalateAgent

            self.session.update_agent(EscalateAgent(chat_ctx=self.chat_ctx))
            # Suppress this agent's reply; EscalateAgent.on_enter speaks instead.
            raise StopResponse()

        if hit is not None and self.data.red_flag is None:
            # Non-emergency (urgent) flag: record it, let the conversation continue.
            self.data.red_flag = hit

        await self.on_turn(turn_ctx, new_message)

    async def on_turn(self, turn_ctx: ChatContext, new_message: ChatMessage) -> None:
        """Subclass hook for per-turn work (e.g. KB retrieval). Default: nothing."""
        return None

    def _context_block(self) -> str:
        """Case context appended to an agent's instructions at handoff time."""
        lines = [f"# What you already know\n{self.data.clinical_summary()}"]
        if self.data.is_returning_caller and self.data.previous_summary:
            lines.append(
                f"\n# This caller has spoken to us before\n{self.data.previous_summary}\n"
                "Acknowledge it briefly and ask if that issue is better."
            )
        lines.append(
            f"\n# Disclaimer you must convey before ending\n{settings.disclaimer}"
        )
        return "\n".join(lines)

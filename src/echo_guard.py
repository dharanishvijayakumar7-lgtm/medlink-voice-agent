"""Stop the agent from hearing its own voice.

On speakerphone the agent's voice leaves the loudspeaker, goes into the phone's
microphone, and some of it survives the phone's echo cancellation and comes back
up the line. Sarvam's STT transcribes it cleanly - it is clear speech - and that
transcript is what LiveKit checks before pausing the agent for a caller who is
talking over it. The agent then paused itself mid-word, waited, resumed, heard
its own voice again, and paused again. On the earpiece there is almost no path
from the speaker back to the microphone, which is why that mode was fine.

The session runs without a local VAD, so every interim transcript goes through
LiveKit's interruption check (`_interrupt_by_audio_activity`), and the only gate
there is `min_words`. A transcript that never reaches it cannot pause anything.
`MedLinkAgent.stt_node` drops the ones this module identifies as echo.

What makes something echo rather than a caller: it repeats the agent's words
*in the order the agent said them*. Callers reuse the agent's words all the
time when they answer ("do you have a headache?" - "yes, I have a headache"), so
word overlap alone would throw real answers away. Order is the tell.

Two strengths of check:

* **While the agent is talking**, a caller is very unlikely to be saying the
  same words in the same order at the same moment, so two words in sequence is
  enough. It has to be this sensitive: LiveKit pauses on a two-word interim.
* **Just after it stops**, the caller is probably answering, and answers echo
  the question. Only a long verbatim run counts there - the tail end of the
  agent's own sentence arriving late through the STT.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field

# Same word pattern as the formulary and triage KB: `\w` alone splits Indic
# words at their vowel signs, and the danda full stops (U+0964/5) are left out
# of the Indic range so they do not stick to the last word of a sentence.
_WORD_RE = re.compile(r"[^\w\u0900-\u0963\u0966-\u0D7F]+")

# How much of what the agent said recently is kept to compare against. A long
# reply is a few sentences; this covers it without matching half the call.
RECENT_WORDS = 80
# Echo arrives after the audio that caused it, and the STT adds its own delay on
# top, so the guard stays on for a moment after the agent stops talking.
ECHO_TAIL_SECONDS = 2.5

# While the agent is talking.
SPEAKING_MIN_RUN = 2
SPEAKING_MIN_SHARE = 0.6
# Just after it stops, when the caller is most likely answering.
TAIL_MIN_RUN = 4
TAIL_MIN_SHARE = 0.8


def words(text: str) -> list[str]:
    return [w for w in _WORD_RE.split(text.casefold()) if w]


def longest_shared_run(heard: list[str], said: list[str]) -> int:
    """Length of the longest word sequence that appears, in order, in both."""
    if not heard or not said:
        return 0
    best = 0
    previous = [0] * (len(said) + 1)
    for h in heard:
        current = [0] * (len(said) + 1)
        for j, s in enumerate(said, 1):
            if h == s:
                current[j] = previous[j - 1] + 1
                best = max(best, current[j])
        previous = current
    return best


def is_echo(
    transcript: str,
    recent_agent_words,
    *,
    min_run: int = SPEAKING_MIN_RUN,
    min_share: float = SPEAKING_MIN_SHARE,
) -> bool:
    """True when a transcript is the agent's own recent words, in order."""
    heard = words(transcript)
    said = list(recent_agent_words)
    if len(heard) < min_run or not said:
        return False
    if longest_shared_run(heard, said) < min_run:
        return False
    vocabulary = _vocabulary(said)
    share = sum(1 for w in heard if w in vocabulary) / len(heard)
    return share >= min_share


def _vocabulary(said: list[str]) -> set[str]:
    """The agent's words, plus runs of them written closed up.

    The agent spells an acronym out so it is pronounced letter by letter
    ("ओ आर एस"), and the STT writes it back as one word ("ओआरएस"). Without the
    joined forms, that one word counted against a real echo, and a Hindi echo at
    -26 dB slipped past the guard.
    """
    vocabulary = set(said)
    for size in (2, 3):
        vocabulary.update("".join(said[i : i + size]) for i in range(len(said) - size + 1))
    return vocabulary


@dataclass
class AgentSpeech:
    """What the agent has said recently, and whether it is still talking.

    Lives on the session's userdata, so it carries across agent handoffs.
    """

    recent: deque[str] = field(default_factory=lambda: deque(maxlen=RECENT_WORDS))
    speaking: bool = False
    stopped_at: float = 0.0
    # Text arrives from the LLM in pieces that can split a word ("parac" +
    # "etamol"), so the unfinished tail waits here until the word is complete.
    _pending: str = ""

    def feed(self, chunk: str) -> None:
        text = self._pending + chunk
        if text and not text[-1].isspace():
            cut = max(text.rfind(c) for c in " \n\t")
            self._pending = text[cut + 1 :]
            text = text[: cut + 1]
        else:
            self._pending = ""
        self.recent.extend(words(text))

    def flush(self) -> None:
        if self._pending:
            self.recent.extend(words(self._pending))
            self._pending = ""

    def started(self) -> None:
        self.speaking = True

    def stopped(self, now: float | None = None) -> None:
        self.speaking = False
        self.stopped_at = time.monotonic() if now is None else now

    def is_echo(self, transcript: str, now: float | None = None) -> bool:
        if not self.recent:
            return False
        if self.speaking:
            return is_echo(transcript, self.recent)
        now = time.monotonic() if now is None else now
        if now - self.stopped_at >= ECHO_TAIL_SECONDS:
            return False
        return is_echo(
            transcript,
            self.recent,
            min_run=TAIL_MIN_RUN,
            min_share=TAIL_MIN_SHARE,
        )

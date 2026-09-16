"""The agent's voice must be louder without being mangled.

The first version of the leveller recomputed one gain per 50 ms frame and applied
it as a constant, which stepped the level 4-6 dB at every frame boundary and made
words crackle on a real call. These tests pin the properties that prevent that:
the gain is continuous, it moves slowly, and silence does not reset it.
"""

import numpy as np
import pytest

from audio_gain import _CEILING, _INT16_PEAK, Leveller

SAMPLE_RATE = 16000
# What the Sarvam plugin actually emits (tts.py: frame_size_ms=50).
FRAME = SAMPLE_RATE // 20


def tone(amplitude: float, samples: int = FRAME, freq: float = 180.0) -> np.ndarray:
    t = np.arange(samples) / SAMPLE_RATE
    return (np.sin(2 * np.pi * freq * t) * amplitude * _INT16_PEAK).astype(np.int16)


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean((samples.astype(np.float64) / _INT16_PEAK) ** 2)))


def leveller(**kwargs) -> Leveller:
    defaults = {"target_rms": 0.12, "max_gain": 6.0, "makeup": 2.0,
                "sample_rate": SAMPLE_RATE}
    return Leveller(**{**defaults, **kwargs})


# --------------------------------------------------------------- loudness ---


def test_quiet_speech_is_lifted():
    quiet = tone(0.03)
    assert rms(leveller().process(quiet)) > rms(quiet) * 2


def test_the_first_frame_is_already_loud():
    """A slow envelope starting at unity would leave the opening word quiet."""
    lev = leveller(makeup=2.0)
    first = lev.process(tone(0.03))
    # Straight in at the makeup gain, not creeping up to it over a second.
    assert rms(first) > rms(tone(0.03)) * 1.8


def test_a_near_silent_frame_is_not_amplified_into_hiss():
    """The cap exists so room tone between words is not dragged up to speech level."""
    out = leveller(max_gain=6.0).process(tone(0.0005))
    assert rms(out) < 0.01


def test_nothing_clips():
    loud = tone(0.9)
    out = leveller(makeup=6.0).process(loud)
    assert np.max(np.abs(out)) < _CEILING * _INT16_PEAK + 1


# ------------------------------------------------------------- continuity ---
# The actual bug: a gain step between frames is a click, and twenty of them a
# second is the crackle the caller heard.


def test_the_gain_does_not_step_between_frames():
    """The applied gain must be continuous across a frame boundary.

    This is the bug that mangled the voice: the old code held one gain for a
    whole frame, so the level jumped 4-6 dB at every seam - twenty clicks a
    second, inside words.

    A constant input isolates it. The signal never changes, so every change in
    the output is the leveller's gain and nothing else.
    """
    lev = leveller()
    frames = [np.full(FRAME, int(0.05 * _INT16_PEAK), dtype=np.int16) for _ in range(8)]

    out = np.concatenate([lev.process(f) for f in frames]).astype(np.float64)
    steps = np.abs(np.diff(out))

    # The gain still moves - it just has to get there gradually. Every step,
    # seams included, must be a rounding-level nudge rather than a jump.
    assert steps.max() <= 2.0, f"gain jumps by {steps.max():.0f} counts somewhere"
    # And it really did move, so the test is not passing on a frozen gain.
    assert abs(out[-1] - out[0]) > 10


def test_the_gain_moves_slowly_enough_to_ignore_syllables():
    """Vowel-to-consonant dips must not move the gain; only chunk drift should.

    The old coefficients gave a ~140 ms release, which tracked syllables and
    swung the gain 7.6 dB inside a single sentence.
    """
    lev = leveller()
    gains = []
    # A sentence: loud vowels with consonant dips, no change in overall level.
    for amplitude in [0.05, 0.05, 0.008, 0.004, 0.05, 0.05, 0.04, 0.006, 0.05] * 2:
        lev.process(tone(amplitude))
        gains.append(lev.gain)
    swing_db = 20 * np.log10(max(gains) / min(gains))
    assert swing_db < 3.0


def test_silence_does_not_reset_the_gain():
    """A consonant can measure as silence; resetting there dropped the level mid-word."""
    lev = leveller()
    for _ in range(10):
        lev.process(tone(0.03))
    settled = lev.gain

    for _ in range(5):
        lev.process(np.zeros(FRAME, dtype=np.int16))
    assert lev.gain == pytest.approx(settled)


def test_silence_stays_quiet():
    out = leveller().process(np.zeros(FRAME, dtype=np.int16))
    assert not np.any(out)


# ------------------------------------------------------------ chunk drift ---


def test_a_quieter_chunk_is_brought_back_toward_the_louder_one():
    """The reason the envelope exists: Sarvam returns each chunk at its own level."""
    lev = leveller()
    # Both levels must sit above the speech gate, or the quieter chunk is never
    # measured - that is the point of the gate.
    loud = [lev.process(tone(0.12)) for _ in range(60)][-1]
    quiet = [lev.process(tone(0.04)) for _ in range(60)][-1]

    # Input differed by ~9.5 dB; the output gap must be meaningfully smaller.
    gap_in = 20 * np.log10(0.12 / 0.04)
    gap_out = 20 * np.log10(rms(loud) / rms(quiet))
    assert gap_out < gap_in - 3


# ------------------------------------------------------------------ edges ---


def test_an_empty_frame_is_passed_through():
    assert leveller().process(np.array([], dtype=np.int16)).size == 0


def test_output_stays_int16():
    assert leveller().process(tone(0.05)).dtype == np.int16


def test_pauses_do_not_drag_the_gain_up():
    """Quiet frames are amplified but must not be measured.

    Measuring every frame is what broke the first attempt at this: a 10 ms frame
    landing in a pause asks for a huge gain, and with a slow envelope those
    frames won hands down - the gain settled at 3.5x instead of 1.5x and pushed
    the output into the ceiling.
    """
    speech_only = leveller()
    for _ in range(60):
        speech_only.process(tone(0.10))

    with_pauses = leveller()
    for i in range(120):
        with_pauses.process(tone(0.10) if i % 2 else tone(0.003))

    assert with_pauses.gain == pytest.approx(speech_only.gain, rel=0.1)

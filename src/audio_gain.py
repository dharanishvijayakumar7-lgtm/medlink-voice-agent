"""Lift the agent's spoken audio before it reaches the caller.

Sarvam cannot do this for us: the plugin only sends `loudness` for `bulbul:v2`
(see `livekit/plugins/sarvam/tts.py`) and we run `bulbul:v3`, so the volume knob
is dropped. The level also drifts between the chunks Sarvam synthesises a reply
in, each returned at its own level.

The first version of this module corrected that drift with a fast automatic gain
control, and it mangled the speech: it recomputed one gain per 50 ms frame and
applied it as a constant, so the level stepped 4-6 dB at every frame boundary -
twenty clicks a second, inside words - and its ~140 ms release time-constant made
it track syllables, ducking vowels and pumping up consonants.

This version cannot do that:

* a fixed makeup gain does the actual lifting, so the first word of a reply is
  already loud rather than waiting for an envelope to catch up
* the gain is **ramped across the samples of each frame**, so it is continuous
  over the whole reply and there is no step to hear
* the envelope moves with a ~1.2 s time-constant, slow enough to ignore syllables
  and correct only the drift between Sarvam's chunks
* silence holds the gain where it is instead of resetting it
* the ceiling is a smooth `tanh` rather than a per-frame rescale, which would
  itself be a per-frame gain change
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_INT16_PEAK = 32767.0
# Only frames at least this loud (fraction of full scale, about -34 dBFS) are
# used to estimate the speech level. Measuring every frame does not work: a
# 10 ms frame landing on a pause or the tail of a consonant asks for a huge
# gain, and with a slow envelope those frames dominate and drag the gain up -
# measured, it settled at 3.5x instead of the intended 1.5x and drove the
# output into the ceiling. Quiet frames are still amplified, just not measured.
_SPEECH_GATE = 0.02
# How long the envelope takes to follow a change in level. Slow on purpose: at
# anything near syllable speed it ducks vowels and lifts consonants, which is
# what made the first version crackle. It only has to track the drift between
# the chunks Sarvam synthesises a reply in, and that moves over seconds.
# Derived per frame from the frame's own duration, so it does not silently
# change when the plugin's frame size or the sample rate does - the frames
# actually arriving are 10 ms, not the 50 ms the plugin's config suggests.
_ENVELOPE_TAU = 1.2
# Headroom under full scale. `tanh` approaches this asymptotically, so samples
# get close to it but never reach it and never wrap.
_CEILING = 0.95


@dataclass
class Leveller:
    """Loudness for one spoken reply, applied without discontinuities.

    One instance per reply: the gain carries across frames, which is what keeps
    the level steady from the first word to the last.

    Args:
        target_rms: level speech is corrected toward, as a fraction of full
            scale. Keep it well under the ceiling: Sarvam's output has an ~18 dB
            peak-to-RMS ratio, so a high target drives the peaks into the
            ceiling and the softening below turns into audible distortion.
        max_gain: hard cap, so a near-silent chunk is not amplified into hiss.
        makeup: the gain speech starts at, before the envelope has measured
            anything. Worth keeping near the level the envelope settles at, so
            the opening of a reply matches the rest of it.
        sample_rate: of the frames passed to `process`, so the envelope's speed
            is in seconds rather than in frames.
    """

    target_rms: float
    max_gain: float
    makeup: float = 2.0
    sample_rate: int = 16000
    gain: float = field(default=0.0)
    level: float = field(default=0.0)

    def __post_init__(self) -> None:
        # Start at the makeup gain rather than at unity: with an envelope this
        # slow, starting neutral would leave the opening second of every reply
        # quiet, which is the complaint this module exists to fix.
        if self.gain <= 0.0:
            self.gain = min(self.makeup, self.max_gain)
        if self.level <= 0.0:
            self.level = self.target_rms / self.gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        """Level one frame of int16 samples and return int16 samples."""
        if samples.size == 0:
            return samples

        audio = samples.astype(np.float32) / _INT16_PEAK
        rms = float(np.sqrt(np.mean(np.square(audio))))

        previous = self.gain
        if rms >= _SPEECH_GATE:
            # Follow how loud the speech itself is, one exponential step sized
            # for the time this frame covers. Anything quieter is a pause or a
            # consonant and is left out of the estimate - but not out of the
            # gain, which keeps running so the level never steps mid-word.
            step = 1.0 - np.exp(-(samples.size / self.sample_rate) / _ENVELOPE_TAU)
            self.level += (rms - self.level) * float(step)
            self.gain = min(
                max(self.target_rms / max(self.level, 1e-6), self.makeup * 0.5),
                self.max_gain,
            )

        # The ramp is the whole point - one gain per frame would step at the
        # seam, and a step in level is a click.
        ramp = np.linspace(previous, self.gain, samples.size, dtype=np.float32)
        # Smooth ceiling: linear well below it, compressing as it approaches,
        # so loud passages soften instead of clipping.
        out = np.tanh(audio * ramp / _CEILING) * _CEILING
        return (out * _INT16_PEAK).astype(np.int16)

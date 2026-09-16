"""Listen to the agent's voice without making a phone call.

Every round of voice fixes so far has cost a real call and a guess. This runs a
sentence through the same streaming path production uses - the Sarvam WebSocket,
the same model, speaker, codec and sample rate - and writes three WAV files so
the problem can be located by ear:

    raw.wav       exactly what Sarvam sent back, untouched
    levelled.wav  after audio_gain.Leveller, which is what the caller hears
    phone.wav     resampled to 8 kHz, what the phone line actually carries

If raw.wav already sounds wrong, the fault is the codec or sample rate - set
MEDLINK_TTS_CODEC=mp3 and MEDLINK_TTS_SAMPLE_RATE=8000 in .env.local to go back
to the old settings, no code change needed. If only levelled.wav sounds wrong,
it is the gain stage. If only phone.wav does, it is the narrowband line and no
amount of processing on our side will fix it.

Usage:
    uv run python scripts/tts_probe.py
    uv run python scripts/tts_probe.py --text "Your own sentence here"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import wave
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import utils

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_gain import Leveller
from config import settings
from speech.providers import build_tts

# Long enough to hear the level settle and to catch a chunk boundary mid-reply,
# which is where the voice used to drift.
DEFAULT_TEXT = (
    "Hello, this is MedLink. I am sorry to hear you have been unwell. "
    "Can you tell me how long you have had this problem? "
    "Take paracetamol five hundred milligrams if you have a fever, "
    "drink plenty of fluids, and rest as much as you can today."
)
PHONE_RATE = 8000
OUT_DIR = Path(__file__).resolve().parent.parent / "tts_probe_out"


def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(samples.astype(np.int16).tobytes())
    seconds = samples.size / sample_rate
    peak = np.max(np.abs(samples)) / 32767 if samples.size else 0.0
    rms = np.sqrt(np.mean((samples.astype(np.float64) / 32767) ** 2)) if samples.size else 0.0
    print(
        f"  {path.name:<14} {seconds:5.1f}s @ {sample_rate} Hz   "
        f"peak {peak:5.1%}   RMS {rms:6.3f} ({20 * np.log10(max(rms, 1e-9)):5.1f} dBFS)"
    )


async def synthesize(text: str) -> tuple[list[rtc.AudioFrame], int]:
    """Push text through the streaming WebSocket path, exactly as a call does."""
    frames: list[rtc.AudioFrame] = []
    # The plugins expect the HTTP session a worker would normally provide.
    async with utils.http_context.open():
        tts = build_tts()
        stream = tts.stream()

        async def collect() -> None:
            async for event in stream:
                frames.append(event.frame)

        collector = asyncio.create_task(collect())
        stream.push_text(text)
        stream.end_input()
        await collector
        await stream.aclose()
        rate = frames[0].sample_rate if frames else tts.sample_rate
        await tts.aclose()
    return frames, rate


def to_samples(frames: list[rtc.AudioFrame]) -> np.ndarray:
    if not frames:
        return np.array([], dtype=np.int16)
    return np.concatenate(
        [np.frombuffer(f.data, dtype=np.int16) for f in frames]
    )


def levelled(frames: list[rtc.AudioFrame], rate: int) -> np.ndarray:
    """The production gain stage, one Leveller per reply - as workflows.base does."""
    leveller = Leveller(
        target_rms=settings.tts_target_rms,
        max_gain=settings.tts_gain_max,
        makeup=settings.tts_makeup_gain,
        sample_rate=rate,
    )
    return np.concatenate(
        [leveller.process(np.frombuffer(f.data, dtype=np.int16)) for f in frames]
    )


def to_phone(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate == PHONE_RATE:
        return samples
    resampler = rtc.AudioResampler(input_rate=rate, output_rate=PHONE_RATE)
    frame = rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=rate,
        num_channels=1,
        samples_per_channel=samples.size,
    )
    out = [*resampler.push(frame), *resampler.flush()]
    return to_samples(out)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default=DEFAULT_TEXT, help="sentence to speak")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print(
        f"\nSarvam {settings.sarvam_tts_model} / {settings.sarvam_tts_speaker}, "
        f"{settings.tts_codec} @ {settings.tts_sample_rate} Hz\n"
        f"makeup {settings.tts_makeup_gain}x, target RMS {settings.tts_target_rms}, "
        f"cap {settings.tts_gain_max}x\n"
    )

    frames, rate = await synthesize(args.text)
    if not frames:
        print("No audio came back from Sarvam - check SARVAM_API_KEY and credits.")
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    raw = to_samples(frames)
    loud = levelled(frames, rate)

    print(f"{len(frames)} frames, {frames[0].samples_per_channel} samples each:")
    write_wav(args.out / "raw.wav", raw, rate)
    write_wav(args.out / "levelled.wav", loud, rate)
    write_wav(args.out / "phone.wav", to_phone(loud, rate), PHONE_RATE)
    print(f"\nListen to them in {args.out}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

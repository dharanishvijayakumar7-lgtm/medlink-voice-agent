"""Measure the agent's voice, instead of guessing at it.

Every round of voice trouble so far has cost a real phone call and a hypothesis.
This runs the production speech path offline and puts numbers on it.

What it measures, per sentence and per language:

* **Round-trip intelligibility.** The sentence is synthesised by Sarvam, pushed
  through the 8 kHz telephony path, and fed back into Sarvam's own STT. The
  transcript is compared against what we asked it to say. This is the objective
  answer to "are the words clear and complete" - if the recogniser cannot get
  the words back off the phone line, neither can a caller in a noisy room.
  Drug names and doses are scored separately: "paracetamol" or "500" coming
  back wrong is a safety problem, a dropped "the" is not.
* **Truncation.** Energy in the last 100 ms, which catches audio that stops
  mid-word.
* **Gaps.** The longest silence inside one reply. A pause at a comma is normal
  speech - only a long hole is how a chunk seam or a dropped line shows up.
* **Level and continuity.** RMS, peak, and the frame-to-frame gain movement that
  caught the crackle bug - a level that steps between frames is heard as a click.

Usage:
    uv run python scripts/audio_audit.py                  # English
    uv run python scripts/audio_audit.py --all-languages  # all six
    uv run python scripts/audio_audit.py --keep-wavs      # also write the audio
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import difflib
import logging
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from livekit import rtc
from livekit.agents import APIStatusError, utils
from livekit.agents import stt as stt_api

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from audio_gain import Leveller
from config import settings
from speech.providers import build_stt, build_tts

PHONE_RATE = 8000
OUT_DIR = Path(__file__).resolve().parent.parent / "audio_audit_out"
# A silence longer than this inside one reply is a hole rather than phrasing.
# 300 ms flagged every comma: Bulbul pauses noticeably at punctuation in the
# Indic voices, which is natural speech, not a fault. Only a hole this long is
# the kind of break a caller would hear as the line dropping.
GAP_MS = 600
SILENCE_RMS = 0.002
# Terms that must survive the line intact. A wrong dose is a safety problem.
CRITICAL = re.compile(r"\b(paracetamol|ors|zinc|\d+)\b", re.IGNORECASE)
# Difference above this means the line is not carrying the words. Measured on
# characters, not words: word-level scoring punished the recogniser for writing
# an acronym closed up ("ஓ ஆர் எஸ்" heard as "ஓஆர்எஸ்"), which scored 43% while
# being a perfect transcription. On characters the same pair scores 0%.
DIFF_LIMIT = 0.20
# A frame-to-frame level change above this is audible as a click.
STEP_LIMIT_DB = 1.0

# Real lines from the agent's own vocabulary: a dose, a number, a referral
# warning, the disclaimer, and a long sentence that crosses chunk boundaries.
SENTENCES: dict[str, list[str]] = {
    "en-IN": [
        "Take one paracetamol tablet, five hundred milligrams, every six hours.",
        "Give ORS after every loose motion, and zinc for fourteen days.",
        "Please see a doctor if the fever lasts more than three days.",
        "I am a health assistant, not a doctor, so this is general guidance only.",
        "From what you have told me this sounds most like a viral fever, which "
        "usually settles in three to four days with rest and plenty of fluids, "
        "but please do not cover yourself with heavy blankets.",
    ],
    "hi-IN": [
        "पैरासिटामोल की एक गोली, पाँच सौ मिलीग्राम, हर छह घंटे में लें।",
        "हर दस्त के बाद ओ आर एस दें, और चौदह दिन तक जिंक दें।",
        "अगर बुखार तीन दिन से ज्यादा रहे तो डॉक्टर को दिखाइए।",
    ],
    "ta-IN": [
        "பாராசிட்டமால் ஒரு மாத்திரை, ஐநூறு மில்லிகிராம், ஆறு மணி நேரத்திற்கு ஒருமுறை.",
        "ஒவ்வொரு பேதிக்குப் பிறகும் ஓ ஆர் எஸ் கொடுங்கள்.",
        "காய்ச்சல் மூன்று நாட்களுக்கு மேல் இருந்தால் மருத்துவரைப் பாருங்கள்.",
    ],
    "te-IN": [
        "పారాసిటమాల్ ఒక మాత్ర, ఐదు వందల మిల్లీగ్రాములు, ప్రతి ఆరు గంటలకు.",
        "ప్రతి విరేచనం తర్వాత ఓ ఆర్ ఎస్ ఇవ్వండి.",
        "జ్వరం మూడు రోజులకు మించి ఉంటే వైద్యుడిని కలవండి.",
    ],
    "kn-IN": [
        "ಪ್ಯಾರಾಸಿಟಮಾಲ್ ಒಂದು ಮಾತ್ರೆ, ಐನೂರು ಮಿಲ್ಲಿಗ್ರಾಂ, ಪ್ರತಿ ಆರು ಗಂಟೆಗೊಮ್ಮೆ.",
        "ಪ್ರತಿ ಭೇದಿಯ ನಂತರ ಓ ಆರ್ ಎಸ್ ಕೊಡಿ.",
        "ಜ್ವರ ಮೂರು ದಿನಕ್ಕಿಂತ ಹೆಚ್ಚು ಇದ್ದರೆ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.",
    ],
    "ml-IN": [
        "പാരസെറ്റമോൾ ഒരു ഗുളിക, അഞ്ഞൂറ് മില്ലിഗ്രാം, ഓരോ ആറ് മണിക്കൂറിലും.",
        "ഓരോ വയറിളക്കത്തിനു ശേഷവും ഓ ആർ എസ് കൊടുക്കുക.",
        "പനി മൂന്ന് ദിവസത്തിൽ കൂടുതൽ നീണ്ടാൽ ഡോക്ടറെ കാണുക.",
    ],
}


@dataclass
class Result:
    language: str
    text: str
    heard: str = ""
    seconds: float = 0.0
    rms_dbfs: float = 0.0
    peak: float = 0.0
    wer: float = 0.0
    char_diff: float = 0.0
    longest_gap_ms: int = 0
    critical_lost: list[str] = field(default_factory=list)
    gaps: int = 0
    tail_energy: float = 0.0
    worst_gain_step_db: float = 0.0

    @property
    def clean(self) -> bool:
        return not (
            self.char_diff > DIFF_LIMIT
            or self.critical_lost
            or self.longest_gap_ms > GAP_MS
            or self.worst_gain_step_db > STEP_LIMIT_DB
        )


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[^\w\u0900-\u0963\u0966-\u0D7F]+", text.casefold()) if w]


def char_difference(reference: str, hypothesis: str) -> float:
    """How much of the sentence came back altered, ignoring word spacing.

    Robust to the recogniser spelling an acronym closed up, which is a writing
    choice rather than a mishearing.
    """
    flat_ref, flat_hyp = "".join(words(reference)), "".join(words(hypothesis))
    if not flat_ref:
        return 0.0
    return 1 - difflib.SequenceMatcher(None, flat_ref, flat_hyp).ratio()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Levenshtein distance over words, as a fraction of the reference length."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        return 0.0
    previous = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        current = [i]
        for j, h in enumerate(hyp, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (r != h))
            )
        previous = current
    return previous[-1] / len(ref)


async def speak(tts, text: str) -> tuple[list[rtc.AudioFrame], int]:
    stream = tts.stream()
    frames: list[rtc.AudioFrame] = []

    async def collect() -> None:
        async for event in stream:
            frames.append(event.frame)

    task = asyncio.create_task(collect())
    stream.push_text(text)
    stream.end_input()
    await task
    await stream.aclose()
    return frames, (frames[0].sample_rate if frames else tts.sample_rate)


def _frame(samples: np.ndarray) -> rtc.AudioFrame:
    return rtc.AudioFrame(
        data=samples.tobytes(),
        sample_rate=PHONE_RATE,
        num_channels=1,
        samples_per_channel=samples.size,
    )


async def transcribe(stt, samples: np.ndarray) -> str:
    """Push 8 kHz audio back into the recogniser the agent itself uses."""
    stream = stt.stream()
    heard: list[str] = []

    async def collect() -> None:
        async for event in stream:
            final = event.type == stt_api.SpeechEventType.FINAL_TRANSCRIPT
            if final and event.alternatives:
                heard.append(event.alternatives[0].text)

    task = asyncio.create_task(collect())
    chunk = PHONE_RATE // 50  # 20 ms, as the room would deliver it
    for i in range(0, samples.size, chunk):
        stream.push_frame(_frame(samples[i : i + chunk]))
    # Trailing silence so the server's VAD closes the utterance.
    stream.push_frame(_frame(np.zeros(PHONE_RATE, dtype=np.int16)))
    stream.end_input()
    with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=30)
    task.cancel()
    await stream.aclose()
    return " ".join(heard)


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
    if not out:
        return samples
    return np.concatenate([np.frombuffer(f.data, dtype=np.int16) for f in out])


def analyse(result: Result, levelled: np.ndarray, rate: int, gains: list[float]) -> None:
    audio = levelled.astype(np.float64) / 32767
    result.seconds = audio.size / rate
    result.peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    speech = audio[np.abs(audio) > SILENCE_RMS]
    rms = float(np.sqrt(np.mean(speech**2))) if speech.size else 0.0
    result.rms_dbfs = 20 * np.log10(max(rms, 1e-9))

    # Truncation: is there still energy running at the very end?
    tail = audio[-int(0.1 * rate) :]
    result.tail_energy = float(np.sqrt(np.mean(tail**2))) if tail.size else 0.0

    # Gaps: runs of quiet inside the reply, ignoring lead-in and run-out.
    window = max(rate // 100, 1)  # 10 ms
    envelope = np.array(
        [
            np.sqrt(np.mean(audio[i : i + window] ** 2))
            for i in range(0, max(audio.size - window, 0), window)
        ]
    )
    loud = np.where(envelope > SILENCE_RMS)[0]
    if loud.size:
        run = longest = 0
        for quiet in envelope[loud[0] : loud[-1] + 1] <= SILENCE_RMS:
            run = run + 1 if quiet else 0
            longest = max(longest, run)
        result.longest_gap_ms = longest * 10
        result.gaps = int(result.longest_gap_ms > GAP_MS)

    if len(gains) > 1:
        g = np.maximum(np.array(gains), 1e-9)
        result.worst_gain_step_db = float(
            np.abs(20 * np.log10(g[1:] / g[:-1])).max()
        )


async def audit_language(language: str, keep: bool, out_dir: Path) -> list[Result]:
    results: list[Result] = []
    tts = build_tts()
    tts.update_options(target_language_code=language)
    stt = build_stt()
    try:
        for text in SENTENCES[language]:
            result = Result(language=language, text=text)
            frames, rate = await speak(tts, text)
            if not frames:
                print(f"  !! no audio came back for: {text[:50]}")
                results.append(result)
                continue

            leveller = Leveller(
                target_rms=settings.tts_target_rms,
                max_gain=settings.tts_gain_max,
                makeup=settings.tts_makeup_gain,
                sample_rate=rate,
            )
            blocks, gains = [], []
            for f in frames:
                blocks.append(leveller.process(np.frombuffer(f.data, dtype=np.int16)))
                gains.append(leveller.gain)
            analyse(result, np.concatenate(blocks), rate, gains)

            phone = to_phone(np.concatenate(blocks), rate)
            result.heard = await transcribe(stt, phone)
            result.wer = word_error_rate(text, result.heard)
            result.char_diff = char_difference(text, result.heard)
            expected = {m.casefold() for m in CRITICAL.findall(text)}
            result.critical_lost = sorted(expected - set(words(result.heard)))

            if keep:
                out_dir.mkdir(parents=True, exist_ok=True)
                with wave.open(str(out_dir / f"{language}_{len(results)}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(PHONE_RATE)
                    w.writeframes(phone.tobytes())
            results.append(result)
    finally:
        await tts.aclose()
        await stt.aclose()
    return results


def report(results: list[Result]) -> int:
    print(
        f"\n{'lang':<7}{'sec':>5}{'dBFS':>7}{'peak':>6}{'diff':>6}{'WER':>6}"
        f"{'gap ms':>8}{'tail':>7}{'step':>7}  text"
    )
    flagged = 0
    for r in results:
        mark = "" if r.clean else "  <-- CHECK"
        flagged += not r.clean
        print(
            f"{r.language:<7}{r.seconds:5.1f}{r.rms_dbfs:7.1f}{r.peak:6.2f}"
            f"{r.char_diff:6.0%}{r.wer:6.0%}{r.longest_gap_ms:8d}{r.tail_energy:7.3f}"
            f"{r.worst_gain_step_db:7.2f}  {r.text[:34]}{mark}"
        )
        if r.critical_lost:
            print(f"{'':>47}LOST FROM THE LINE: {r.critical_lost}")
        if r.char_diff > DIFF_LIMIT:
            print(f"{'':>47}heard: {r.heard[:66]!r}")
    print(
        "\ndiff = how much of the sentence came back altered when the agent's own"
        "\n       voice was read off the 8 kHz line, scored on characters. This is"
        f"\n       the intelligibility number; over {DIFF_LIMIT:.0%} is a real problem."
        "\nWER  = the same thing scored per word, for information only. It punishes"
        "\n       the recogniser for writing an acronym closed up, so it reads high."
        f"\ngap  = longest silence inside the reply. A pause at a comma is normal;"
        f"\n       over {GAP_MS} ms is a hole."
        "\ntail = energy in the final 100 ms; 0.000 can mean the audio was cut off."
        f"\nstep = worst frame-to-frame level jump; over {STEP_LIMIT_DB} dB is an audible click."
    )
    return flagged


# --------------------------------------------------------- speakerphone echo ---
# How strongly the agent's own voice comes back up the line when the caller's
# phone is on speaker. It depends entirely on the handset's echo cancellation,
# which we cannot see, so a range is tried: -10 dB is a poor phone in a small
# room, -26 dB a good one.
ECHO_LEVELS_DB = (-10, -18, -26)
ECHO_DELAY_MS = 120
# A phone loudspeaker and microphone pass roughly the telephony band.
ECHO_BAND_HZ = (300, 3400)


def speakerphone_echo(audio: np.ndarray, rate: int, level_db: float) -> np.ndarray:
    """The agent's voice as it would come back through a speakerphone."""
    signal = audio.astype(np.float64)
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(signal.size, 1 / rate)
    spectrum[(freqs < ECHO_BAND_HZ[0]) | (freqs > ECHO_BAND_HZ[1])] = 0
    banded = np.fft.irfft(spectrum, n=signal.size)
    delay = int(rate * ECHO_DELAY_MS / 1000)
    echo = np.concatenate([np.zeros(delay), banded]) * 10 ** (level_db / 20)
    rng = np.random.default_rng(0)
    echo += rng.normal(0, 30, echo.size)  # a little room and line noise
    return np.clip(echo, -32767, 32767).astype(np.int16)


async def audit_echo(language: str) -> int:
    """Show whether echo would pause the agent, and whether the guard stops it."""
    from echo_guard import AgentSpeech

    tts, stt = build_tts(), build_stt()
    tts.update_options(target_language_code=language)
    problems = 0
    print(
        f"\n{'level':>7}  {'words':>5}  {'would pause':>11}  {'guard drops':>11}  heard"
    )
    try:
        for text in SENTENCES[language]:
            frames, rate = await speak(tts, text)
            if not frames:
                continue
            leveller = Leveller(
                target_rms=settings.tts_target_rms,
                max_gain=settings.tts_gain_max,
                makeup=settings.tts_makeup_gain,
                sample_rate=rate,
            )
            played = np.concatenate(
                [leveller.process(np.frombuffer(f.data, dtype=np.int16)) for f in frames]
            )
            speech = AgentSpeech()
            speech.feed(text + " ")
            speech.started()

            print(f"  {text[:60]}")
            for level in ECHO_LEVELS_DB:
                echo = to_phone(speakerphone_echo(played, rate, level), rate)
                try:
                    heard = await transcribe(stt, echo)
                except APIStatusError as err:
                    # Each echo level is a fresh STT session, and a full run is
                    # dozens of them. Sarvam rate-limits that; stop cleanly.
                    print(f"\n  stopped: Sarvam refused the request ({err.status_code}).")
                    return problems
                n = len(words(heard))
                # LiveKit's gate in this session: min_words=2 on any transcript.
                would_pause = n >= 2
                dropped = speech.is_echo(heard)
                leak = would_pause and not dropped
                problems += leak
                print(
                    f"{level:>5} dB  {n:>5}  {'yes' if would_pause else 'no':>11}  "
                    f"{'yes' if dropped else 'no':>11}  {heard[:44]!r}"
                    + ("  <-- WOULD STILL PAUSE" if leak else "")
                )
    finally:
        await tts.aclose()
        await stt.aclose()
    return problems


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all-languages", action="store_true")
    parser.add_argument("--language", default="en-IN", choices=sorted(SENTENCES))
    parser.add_argument("--keep-wavs", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--echo",
        action="store_true",
        help="simulate a speakerphone: does the agent's own voice pause it?",
    )
    args = parser.parse_args()

    if args.echo:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")
        languages = sorted(SENTENCES) if args.all_languages else [args.language]
        async with utils.http_context.open():
            leaks = 0
            for language in languages:
                print(f"\n--- {language}: speakerphone echo")
                leaks += await audit_echo(language)
        print(
            "\nwould pause  = the echo transcribes to 2+ words, which is all LiveKit"
            "\n               needs in this session to pause the agent mid-word."
            "\nguard drops  = the echo guard recognises it as the agent's own voice."
            f"\n\n{leaks} echo transcript(s) would still pause the agent."
        )
        return 0

    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    languages = sorted(SENTENCES) if args.all_languages else [args.language]
    print(
        f"Sarvam {settings.sarvam_tts_model}/{settings.sarvam_tts_speaker} -> "
        f"{settings.tts_codec} @ {settings.tts_sample_rate} Hz -> {PHONE_RATE} Hz "
        f"-> {settings.sarvam_stt_model}\n"
        f"makeup {settings.tts_makeup_gain}x  target RMS {settings.tts_target_rms}"
        f"  cap {settings.tts_gain_max}x"
    )

    results: list[Result] = []
    async with utils.http_context.open():
        for language in languages:
            print(f"\n--- {language} ({len(SENTENCES[language])} sentences)")
            results += await audit_language(language, args.keep_wavs, args.out)

    flagged = report(results)
    print(f"\n{len(results) - flagged}/{len(results)} sentences clean.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

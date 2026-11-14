"""Synthesize actual assessment closings, save 8 kHz audio and STT round trips.

This is an intelligibility proxy, not a native-speaker or handset evaluation.
"""

import argparse
import asyncio
import json
import logging
import sys
import wave
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from audio_audit import Result, analyse, char_difference, speak, to_phone, transcribe
from livekit.agents import utils

from audio_gain import Leveller
from config import settings
from speech.providers import build_stt, build_tts
from speech.wording import normalize_for_speech


async def main(args):
    source = json.loads(args.input.read_text(encoding="utf-8"))
    output = Path("assessment_audit_out") / args.label
    output.mkdir(exist_ok=True, parents=True)
    rows = []
    async with utils.http_context.open():
        for row in source:
            language = row["language"]
            if (
                row["case"] != "hair_gradual"
                or language not in args.languages
                or not row.get("spoken")
            ):
                continue
            for speaker in args.speakers or [
                settings.tts_speaker_for_language(language)
            ]:
                tts = build_tts()
                stt = build_stt()
                result = Result(language=language, text=" ".join(row["spoken"]))
                record = {
                    "speaker": speaker,
                    "language": language,
                    "input": result.text,
                }
                try:
                    tts.update_options(target_language_code=language, speaker=speaker)
                    if args.tuned:
                        tts.update_options(
                            pace=0.95,
                            temperature=0.4,
                            min_buffer_size=100,
                            max_chunk_length=500,
                        )
                    frames, rate = await speak(tts, normalize_for_speech(result.text))
                    if not frames:
                        raise RuntimeError("No audio returned")
                    leveller = Leveller(
                        target_rms=settings.tts_target_rms,
                        max_gain=settings.tts_gain_max,
                        makeup=settings.tts_makeup_gain,
                        sample_rate=rate,
                    )
                    blocks, gains = [], []
                    for frame in frames:
                        blocks.append(
                            leveller.process(np.frombuffer(frame.data, dtype=np.int16))
                        )
                        gains.append(leveller.gain)
                    samples = np.concatenate(blocks)
                    analyse(result, samples, rate, gains)
                    phone = to_phone(samples, rate)
                    path = output / f"{language}_{speaker}.wav"
                    with wave.open(str(path), "wb") as wav:
                        wav.setnchannels(1)
                        wav.setsampwidth(2)
                        wav.setframerate(8000)
                        wav.writeframes(phone.tobytes())
                    result.heard = await transcribe(stt, phone)
                    result.char_diff = char_difference(result.text, result.heard)
                    record.update(asdict(result), wav=str(path))
                except Exception as exc:
                    record["error"] = type(exc).__name__
                finally:
                    await tts.aclose()
                    await stt.aclose()
                rows.append(record)
                (output / "results.json").write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in record.items()
                            if k
                            in (
                                "speaker",
                                "language",
                                "seconds",
                                "char_diff",
                                "longest_gap_ms",
                                "tail_energy",
                                "error",
                            )
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("assessment_audit_out/improved.json")
    )
    parser.add_argument("--label", default="audio")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "ml-IN"],
    )
    parser.add_argument("--speakers", nargs="+")
    parser.add_argument("--tuned", action="store_true")
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(main(parser.parse_args()))

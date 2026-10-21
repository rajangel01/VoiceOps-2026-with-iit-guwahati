
import os
import json
import argparse
from pathlib import Path

from dotenv import load_dotenv
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

from src.rag_agent import respond_with_sources

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "voice_output"


def process_audio(audio_path: str, history=None):
    audio_path = Path(audio_path).resolve()

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if not os.getenv("GNANI_API_KEY"):
        raise RuntimeError(
            "GNANI_API_KEY missing. Check the .env file."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()

    # 1. Convert customer audio to text.
    stt_result = stt.transcribe(
        str(audio_path),
        language_code="en-IN"
    )

    transcript = stt_result.get("transcript", "").strip()

    if not transcript:
        raise RuntimeError(
            "Speech recognition returned an empty transcript."
        )

    print(f"\nCustomer transcript: {transcript}")

    # 2. Send recognized text to the existing RAG chatbot.
    answer, sources = respond_with_sources(transcript, history)

    print(f"\nAssistant: {answer}")

    # 3. Generate spoken response.
    audio_output = OUTPUT_DIR / f"{audio_path.stem}_reply.wav"

    tts.synthesize(
        answer,
        voice="Kaveri",
        model="timbre-v2.5",
        language="en-IN",
        output_file=str(audio_output)
    )

    # 4. Save transcript, answer, and retrieved evidence.
    result = {
        "input_audio": str(audio_path),
        "transcript": transcript,
        "answer": answer,
        "retrieved_sources": [
            {
                "source": item.get("source"),
                "section_id": item.get("section_id"),
                "score": item.get("score")
            }
            for item in sources
        ],
        "output_audio": str(audio_output)
    }

    json_output = OUTPUT_DIR / f"{audio_path.stem}_result.json"
    json_output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    text_output = OUTPUT_DIR / f"{audio_path.stem}_reply.txt"
    text_output.write_text(answer, encoding="utf-8")

    print(f"\nReply audio: {audio_output}")
    print(f"Reply text:  {text_output}")
    print(f"Result JSON: {json_output}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="ZentroMart Gnani voice support agent"
    )
    parser.add_argument(
        "audio",
        help="Path to a customer audio file"
    )
    args = parser.parse_args()

    process_audio(args.audio)


if __name__ == "__main__":
    main()

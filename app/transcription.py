# Audio transcription service using OpenAI Whisper API

import os
import tempfile

from openai import OpenAI


def transcribe_audio_chunk(client: OpenAI, audio_bytes: bytes) -> str:
    """Write audio bytes to a temp .webm file, send to Whisper API, and return transcribed text.

    The temp file is always deleted in the finally block, even if the API call fails.
    """
    tmp_path = None
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
        tmp_path = tmp.name
        tmp.write(audio_bytes)
        tmp.close()

        with open(tmp_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
            )

        return transcription.text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

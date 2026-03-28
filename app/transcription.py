# Audio transcription service using OpenAI Whisper API

import logging
import os
import re
import tempfile

from openai import OpenAI

logger = logging.getLogger(__name__)


# --- Whisper Hallucination Filter ---

# Layer 1: Exact matches (after stripping punctuation/emoji, lowercased)
_HALLUCINATION_EXACT = {
    "you", "you you", "you you you", "you you you you",
    "thank you", "thank you for watching", "thanks for watching",
    "thank you so much for watching", "thanks for watching this video",
    "subtitles by", "subtitles by amaraorg", "amaraorg",
    "please subscribe", "like and subscribe",
    "share this video with your friends on social media",
    "share this video with your friends",
    "bye", "bye bye", "bye bye bye",
    "ahh", "oh", "hmm", "uh", "um",
}

# Layer 2: Regex for repetitive phrases
_HALLUCINATION_PATTERNS = [
    r"^(you\s*[.,!?]?\s*){2,}$",
    r"^(thank you[.,!?]?\s*){2,}$",
    r"^(thanks for watching[.,!?]?\s*){2,}$",
    r"^(please subscribe[.,!?]?\s*){2,}$",
    r"^(bye[.,!?]?\s*){2,}$",
    r"^(i know[.,!?]?\s*){2,}$",
    r"^(okay[.,!?]?\s*){2,}$",
]
_HALLUCINATION_RE = [re.compile(p, re.IGNORECASE) for p in _HALLUCINATION_PATTERNS]

# Layer 3: Keyword detection — if text contains any of these, it's hallucination
_HALLUCINATION_KEYWORDS = [
    # English
    "subscribe", "share this video", "thanks for watching",
    "thank you for watching", "thank you so much for watching",
    "subtitles by", "amara.org", "follow me on", "check out my",
    # Vietnamese — cover spelling variations (ký/kí, kênh, video)
    "đăng ký", "đăng kí", "đăng ký kênh", "đăng kí kênh",
    "nhấn nút đăng ký", "nhấn nút đăng kí",
    "cảm ơn bạn đã xem", "cảm ơn đã xem", "cảm ơn các bạn đã xem",
    "chia sẻ video", "nhấn like", "phụ đề bởi",
    "không bỏ lỡ những video", "những video hấp dẫn",
    "hãy subscribe", "hãy đăng ký", "hãy đăng kí",
    # Japanese
    "チャンネル登録", "ご視聴ありがとう",
    # Russian
    "Субтитры", "подписывайтесь", "Спасибо за просмотр",
    # Chinese
    "订阅", "感谢观看",
    # Korean
    "구독", "시청해주셔서",
    # General
    "social media", "my channel", "my website",
]
_HALLUCINATION_KEYWORDS_LOWER = [kw.lower() for kw in _HALLUCINATION_KEYWORDS]


def _is_hallucination(text: str) -> bool:
    """Detect common Whisper hallucination patterns."""
    stripped = text.strip()
    if not stripped:
        return True

    # Normalize: strip punctuation for exact matching
    cleaned = re.sub(r"[.,!?;:'\"\-…。\s]", "", stripped).lower()

    # If nothing left after removing punctuation and whitespace, it's noise
    if not cleaned:
        return True

    # Rebuild with spaces for phrase matching
    phrase = re.sub(r"[.,!?;:'\"\-…。]", "", stripped).strip().lower()
    phrase = re.sub(r"\s+", " ", phrase)

    if phrase in _HALLUCINATION_EXACT:
        return True

    for pattern in _HALLUCINATION_RE:
        if pattern.match(stripped):
            return True

    # Layer 3: keyword detection
    text_lower = stripped.lower()
    for kw in _HALLUCINATION_KEYWORDS_LOWER:
        if kw in text_lower:
            return True

    return False


def transcribe_audio_chunk(client: OpenAI, audio_bytes: bytes) -> str:
    """Write audio bytes to a temp .webm file, send to Whisper API, and return transcribed text.

    The temp file is always deleted in the finally block, even if the API call fails.
    Returns empty string if the result looks like a hallucination.
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

        text = transcription.text
        logger.info("Whisper raw output: %r (len=%d bytes=%d)", text, len(text), len(audio_bytes))
        if _is_hallucination(text):
            logger.info("Filtered as hallucination: %r", text)
            return ""
        return text
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

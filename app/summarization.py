# Transcript summarization service using OpenAI Chat API

from openai import OpenAI

SYSTEM_PROMPT = (
    "You are a transcript summarizer. Your ONLY job is to summarize what was actually said. "
    "Rules: "
    "1. ONLY include information that was explicitly spoken in the transcript. "
    "2. Do NOT add, infer, or fabricate any information not present in the transcript. "
    "3. Do NOT mention topics that were not discussed. "
    "4. Use bullet points organized by topic. Be thorough — capture all key points and action items. "
    "5. Respond in the same language as the transcript (Vietnamese for Vietnamese, English for English). "
    "6. IGNORE and EXCLUDE these AI transcription artifacts (they are NOT real speech): "
    "subscribe, like and subscribe, share this video, thanks for watching, "
    "đăng ký kênh, cảm ơn đã xem, không bỏ lỡ những video, "
    "subtitles by, amara.org, repeated nonsense words, "
    "and any text in a language that doesn't match the main conversation."
)


def summarize_transcript(client: OpenAI, transcript: str) -> str:
    """Send transcript to Chat API and return the summary string.

    Raises on any API error so the caller can handle it.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
    )
    return response.choices[0].message.content

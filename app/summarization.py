# Transcript summarization service using OpenAI Chat API

from openai import OpenAI

SYSTEM_PROMPT = (
    "Summarize the following meeting/video transcript concisely, "
    "highlighting the main points and any action items."
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

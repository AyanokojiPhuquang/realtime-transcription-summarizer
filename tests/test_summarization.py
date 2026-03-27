"""Unit tests for the summarization service."""

from unittest.mock import MagicMock, patch

import pytest

from app.summarization import SYSTEM_PROMPT, summarize_transcript


class TestSummarizeTranscript:
    """Tests for summarize_transcript function."""

    def test_returns_summary_from_chat_api(self):
        """Verify that summarize_transcript returns the Chat API response content."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "This is a summary."
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        result = summarize_transcript(mock_client, "Some transcript text")

        assert result == "This is a summary."

    def test_sends_correct_system_prompt(self):
        """Verify the correct system prompt is sent to the Chat API."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Summary"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        summarize_transcript(mock_client, "transcript")

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}

    def test_sends_transcript_as_user_message(self):
        """Verify the transcript is sent as the user message."""
        mock_client = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Summary"
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[mock_choice]
        )

        transcript = "Meeting notes: discussed project timeline."
        summarize_transcript(mock_client, transcript)

        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[1] == {"role": "user", "content": transcript}

    def test_raises_on_api_error(self):
        """Verify that API errors propagate to the caller."""
        from openai import APIError

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = APIError(
            message="Service unavailable",
            request=MagicMock(),
            body=None,
        )

        with pytest.raises(APIError):
            summarize_transcript(mock_client, "Some transcript")

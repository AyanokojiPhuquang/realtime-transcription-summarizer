# Unit tests for transcribe_audio_chunk

import os
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import pytest

from app.transcription import transcribe_audio_chunk


class TestTranscribeAudioChunk:
    """Unit tests for the transcribe_audio_chunk function."""

    def test_returns_transcribed_text(self):
        """Verify that transcribed text is returned on success."""
        client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "Hello world"
        client.audio.transcriptions.create.return_value = mock_result

        result = transcribe_audio_chunk(client, b"fake audio bytes")

        assert result == "Hello world"

    def test_calls_whisper_api_with_whisper_model(self):
        """Verify the Whisper API is called with model='whisper-1'."""
        client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "test"
        client.audio.transcriptions.create.return_value = mock_result

        transcribe_audio_chunk(client, b"fake audio bytes")

        call_kwargs = client.audio.transcriptions.create.call_args
        assert call_kwargs.kwargs["model"] == "whisper-1"

    def test_temp_file_cleaned_up_on_success(self):
        """Verify temp file is deleted after successful transcription."""
        client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "test"
        client.audio.transcriptions.create.return_value = mock_result

        # Track temp files created
        created_files = []
        original_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(**kwargs):
            tmp = original_ntf(**kwargs)
            created_files.append(tmp.name)
            return tmp

        with patch("app.transcription.tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            transcribe_audio_chunk(client, b"fake audio bytes")

        assert len(created_files) == 1
        assert not os.path.exists(created_files[0])

    def test_temp_file_cleaned_up_on_api_error(self):
        """Verify temp file is deleted even when Whisper API raises an error."""
        client = MagicMock()
        client.audio.transcriptions.create.side_effect = Exception("API error")

        created_files = []
        original_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(**kwargs):
            tmp = original_ntf(**kwargs)
            created_files.append(tmp.name)
            return tmp

        with patch("app.transcription.tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            with pytest.raises(Exception, match="API error"):
                transcribe_audio_chunk(client, b"fake audio bytes")

        assert len(created_files) == 1
        assert not os.path.exists(created_files[0])

    def test_temp_file_has_webm_suffix(self):
        """Verify the temp file is created with .webm suffix."""
        client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "test"
        client.audio.transcriptions.create.return_value = mock_result

        created_files = []
        original_ntf = tempfile.NamedTemporaryFile

        def tracking_ntf(**kwargs):
            tmp = original_ntf(**kwargs)
            created_files.append(tmp.name)
            return tmp

        with patch("app.transcription.tempfile.NamedTemporaryFile", side_effect=tracking_ntf):
            transcribe_audio_chunk(client, b"fake audio bytes")

        assert created_files[0].endswith(".webm")

    def test_audio_bytes_written_to_temp_file(self):
        """Verify the audio bytes are written to the temp file before API call."""
        audio_data = b"test audio content 12345"
        file_contents = []

        client = MagicMock()
        mock_result = MagicMock()
        mock_result.text = "test"

        def capture_file_content(**kwargs):
            # Read the file that was passed to the API
            f = kwargs.get("file")
            if f:
                file_contents.append(f.read())
            return mock_result

        client.audio.transcriptions.create.side_effect = capture_file_content

        transcribe_audio_chunk(client, audio_data)

        assert len(file_contents) == 1
        assert file_contents[0] == audio_data

"""Unit tests for FastAPI app setup in app/main.py."""

import json
from unittest.mock import patch, MagicMock

import openai as openai_module
import pytest
from fastapi.testclient import TestClient

from app.main import app, openai_client, OPENAI_API_KEY, summarize


@pytest.fixture
def client():
    return TestClient(app)


class TestAppSetup:
    def test_openai_client_type(self):
        """Verify the OpenAI client is either an OpenAI instance or None (if key missing)."""
        from openai import OpenAI
        assert openai_client is None or isinstance(openai_client, OpenAI)

    def test_app_is_fastapi_instance(self):
        """Verify app is a FastAPI instance."""
        from fastapi import FastAPI
        assert isinstance(app, FastAPI)


class TestServeIndex:
    def test_get_root_returns_html(self, client):
        """GET / should return the index.html file."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_get_root_contains_expected_content(self, client):
        """GET / should serve the actual index.html content."""
        response = client.get("/")
        assert "Realtime Transcription Summarizer" in response.text


class TestStaticFiles:
    def test_static_app_js_served(self, client):
        """Static files should be accessible under /static/."""
        response = client.get("/static/app.js")
        assert response.status_code == 200

    def test_static_index_html_served(self, client):
        """index.html should also be accessible under /static/."""
        response = client.get("/static/index.html")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_static_nonexistent_returns_404(self, client):
        """Requesting a nonexistent static file should return 404."""
        response = client.get("/static/nonexistent.xyz")
        assert response.status_code == 404


class TestWebSocketAudio:
    """Tests for the WebSocket /ws/audio endpoint."""

    def test_websocket_transcription_success(self, client):
        """Send a binary audio chunk, mock Whisper, verify transcription message returned."""
        mock_result = MagicMock()
        mock_result.text = "Hello world"

        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.transcribe_audio_chunk", return_value="Hello world") as mock_transcribe:
            mock_client.__bool__ = lambda self: True  # truthy so it passes the None check
            with client.websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"\x00" * 100)
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "transcription"
                assert msg["text"] == "Hello world"
                mock_transcribe.assert_called_once()

    def test_websocket_rate_limit_error(self, client):
        """On RateLimitError, send error message and continue processing."""
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.headers = {}

        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.transcribe_audio_chunk", side_effect=openai_module.RateLimitError(
                 message="rate limited", response=rate_limit_resp, body=None
             )):
            mock_client.__bool__ = lambda self: True
            with client.websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"\x00" * 100)
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "error"
                assert "Rate limit exceeded" in msg["message"]

    def test_websocket_api_error(self, client):
        """On APIError, send error message and continue processing."""
        api_err_resp = MagicMock()
        api_err_resp.status_code = 500
        api_err_resp.headers = {}

        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.transcribe_audio_chunk", side_effect=openai_module.APIError(
                 message="server error", request=MagicMock(), body=None
             )):
            mock_client.__bool__ = lambda self: True
            with client.websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"\x00" * 100)
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "error"
                assert "Transcription API error" in msg["message"]

    def test_websocket_generic_error(self, client):
        """On unexpected error, send error message and continue processing."""
        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.transcribe_audio_chunk", side_effect=RuntimeError("something broke")):
            mock_client.__bool__ = lambda self: True
            with client.websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"\x00" * 100)
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "error"
                assert "something broke" in msg["message"]

    def test_websocket_no_api_key_sends_error_and_closes(self, client):
        """If openai_client is None, send error and close the connection."""
        with patch("app.main.openai_client", None):
            with client.websocket_connect("/ws/audio") as ws:
                data = ws.receive_text()
                msg = json.loads(data)
                assert msg["type"] == "error"
                assert "API key" in msg["message"]

    def test_websocket_multiple_chunks_buffer_accumulates(self, client):
        """Multiple binary chunks should each produce a transcription response."""
        call_count = 0

        def mock_transcribe(client_arg, data):
            nonlocal call_count
            call_count += 1
            return f"chunk{call_count}"

        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.transcribe_audio_chunk", side_effect=mock_transcribe):
            mock_client.__bool__ = lambda self: True
            with client.websocket_connect("/ws/audio") as ws:
                ws.send_bytes(b"\x00" * 100)
                msg1 = json.loads(ws.receive_text())
                assert msg1["text"] == "chunk1"

                ws.send_bytes(b"\x00" * 100)
                msg2 = json.loads(ws.receive_text())
                assert msg2["text"] == "chunk2"

    def test_websocket_continues_after_error(self, client):
        """After an error on one chunk, the next chunk should still be processed."""
        calls = [0]

        def mock_transcribe(client_arg, data):
            calls[0] += 1
            if calls[0] == 1:
                raise RuntimeError("first chunk fails")
            return "recovered"

        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.transcribe_audio_chunk", side_effect=mock_transcribe):
            mock_client.__bool__ = lambda self: True
            with client.websocket_connect("/ws/audio") as ws:
                # First chunk: error
                ws.send_bytes(b"\x00" * 100)
                msg1 = json.loads(ws.receive_text())
                assert msg1["type"] == "error"

                # Second chunk: success
                ws.send_bytes(b"\x00" * 100)
                msg2 = json.loads(ws.receive_text())
                assert msg2["type"] == "transcription"
                assert msg2["text"] == "recovered"


class TestSummarizeEndpoint:
    """Tests for the POST /summarize endpoint."""

    def test_summarize_success(self, client):
        """POST /summarize with valid text should return a summary."""
        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.summarize_transcript", return_value="This is a summary.") as mock_summarize:
            mock_client.__bool__ = lambda self: True
            response = client.post("/summarize", json={"text": "Some transcript text."})
            assert response.status_code == 200
            data = response.json()
            assert data["summary"] == "This is a summary."
            mock_summarize.assert_called_once_with(mock_client, "Some transcript text.")

    def test_summarize_empty_text_returns_422(self, client):
        """POST /summarize with empty text should return 422 (Pydantic min_length=1 validation)."""
        response = client.post("/summarize", json={"text": ""})
        assert response.status_code == 422

    def test_summarize_missing_text_returns_422(self, client):
        """POST /summarize with missing text field should return 422."""
        response = client.post("/summarize", json={})
        assert response.status_code == 422

    def test_summarize_no_api_key_returns_502(self, client):
        """POST /summarize when openai_client is None should return 502."""
        with patch("app.main.openai_client", None):
            response = client.post("/summarize", json={"text": "Some text"})
            assert response.status_code == 502
            assert "API key" in response.json()["detail"]

    def test_summarize_rate_limit_returns_502(self, client):
        """POST /summarize on RateLimitError should return 502 with rate limit message."""
        rate_limit_resp = MagicMock()
        rate_limit_resp.status_code = 429
        rate_limit_resp.headers = {}

        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.summarize_transcript", side_effect=openai_module.RateLimitError(
                 message="rate limited", response=rate_limit_resp, body=None
             )):
            mock_client.__bool__ = lambda self: True
            response = client.post("/summarize", json={"text": "Some text"})
            assert response.status_code == 502
            assert "rate limited" in response.json()["detail"].lower()

    def test_summarize_api_error_returns_502(self, client):
        """POST /summarize on APIError should return 502 with descriptive message."""
        with patch("app.main.openai_client") as mock_client, \
             patch("app.main.summarize_transcript", side_effect=openai_module.APIError(
                 message="server error", request=MagicMock(), body=None
             )):
            mock_client.__bool__ = lambda self: True
            response = client.post("/summarize", json={"text": "Some text"})
            assert response.status_code == 502
            assert "Summarization failed" in response.json()["detail"]

"""Unit tests for Pydantic models in app/models.py."""

import pytest
from pydantic import ValidationError

from app.models import (
    ErrorMessage,
    ErrorResponse,
    SummarizeRequest,
    SummarizeResponse,
    TranscriptionMessage,
)


class TestSummarizeRequest:
    def test_valid_request(self):
        req = SummarizeRequest(text="Hello world")
        assert req.text == "Hello world"

    def test_empty_text_rejected(self):
        with pytest.raises(ValidationError):
            SummarizeRequest(text="")

    def test_missing_text_rejected(self):
        with pytest.raises(ValidationError):
            SummarizeRequest()


class TestSummarizeResponse:
    def test_valid_response(self):
        resp = SummarizeResponse(summary="A brief summary.")
        assert resp.summary == "A brief summary."


class TestErrorResponse:
    def test_valid_error(self):
        err = ErrorResponse(detail="Something went wrong")
        assert err.detail == "Something went wrong"


class TestTranscriptionMessage:
    def test_default_type(self):
        msg = TranscriptionMessage(text="hello")
        assert msg.type == "transcription"
        assert msg.text == "hello"

    def test_json_serialization(self):
        msg = TranscriptionMessage(text="snippet")
        data = msg.model_dump()
        assert data == {"type": "transcription", "text": "snippet"}


class TestErrorMessage:
    def test_default_type(self):
        msg = ErrorMessage(message="error occurred")
        assert msg.type == "error"
        assert msg.message == "error occurred"

    def test_json_serialization(self):
        msg = ErrorMessage(message="rate limit")
        data = msg.model_dump()
        assert data == {"type": "error", "message": "rate limit"}

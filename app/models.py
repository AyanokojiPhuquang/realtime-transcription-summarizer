# Pydantic models for request/response schemas and WebSocket messages

from typing import Literal

from pydantic import BaseModel, Field


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Full transcription text to summarize")


class SummarizeResponse(BaseModel):
    summary: str = Field(..., description="Generated summary of the transcript")


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Error description")


class TranscriptionMessage(BaseModel):
    type: Literal["transcription"] = "transcription"
    text: str


class ErrorMessage(BaseModel):
    type: Literal["error"] = "error"
    message: str

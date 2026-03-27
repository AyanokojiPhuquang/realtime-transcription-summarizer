# Realtime Transcription Summarizer

A local web app that captures browser tab audio via the Screen Capture API, streams it over WebSocket to a FastAPI backend for real-time transcription using OpenAI Whisper, and generates a summary via the Chat API when the session ends.

## Setup

1. Install dependencies:

```bash
uv sync
```

2. Configure your OpenAI API key:

```bash
cp .env.template .env
```

Edit `.env` and set your `OPENAI_API_KEY`.

## Run the Server

```bash
uv run uvicorn app.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

## Run Tests

```bash
uv run pytest
```

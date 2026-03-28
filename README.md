# Realtime Transcription Summarizer

A web app that captures browser tab or microphone audio, streams it over WebSocket to a FastAPI backend for real-time transcription using OpenAI Whisper, and generates a summary via the Chat API when the session ends.

## Run with Docker (recommended)

1. Create `.env` file with your API key:

```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

2. Build and start:

```bash
docker compose up --build -d
```

3. Open [http://localhost:3000](http://localhost:3000)

To stop:

```bash
docker compose down
```

## Run locally (development)

1. Install dependencies:

```bash
uv sync
```

2. Configure your OpenAI API key:

```bash
cp .env.template .env
```

Edit `.env` and set your `OPENAI_API_KEY`.

3. Start the server:

```bash
uv run uvicorn app.main:app --reload
```

Then open [http://localhost:8000](http://localhost:8000)

## Run Tests

```bash
uv run pytest
```

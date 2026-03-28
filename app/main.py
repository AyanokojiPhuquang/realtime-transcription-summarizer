# FastAPI application entry point
import asyncio
import os
import logging
from pathlib import Path

import openai as openai_module
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI

from app.models import SummarizeRequest, SummarizeResponse, TranscriptionMessage, ErrorMessage
from app.summarization import summarize_transcript
from app.transcription import transcribe_audio_chunk

logger = logging.getLogger(__name__)

# Configure logging to show INFO level
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Validate OPENAI_API_KEY at module load time
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is not set.")

# Initialize OpenAI client (None if key is missing; endpoints will fail gracefully)
openai_client: OpenAI | None = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Create FastAPI app
app = FastAPI()

# Path to static directory
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def serve_index():
    """Serve the index.html file at the root URL."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    """Accept WebSocket connections for audio streaming and transcription."""
    await websocket.accept()

    # If no OpenAI client, send error and close
    if openai_client is None:
        error_msg = ErrorMessage(message="OpenAI API key is not configured. Cannot transcribe audio.")
        await websocket.send_text(error_msg.model_dump_json())
        await websocket.close()
        return

    transcription_buffer = ""
    pending_tasks: set[asyncio.Task] = set()
    send_lock = asyncio.Lock()

    async def process_chunk(data: bytes, seq: int):
        """Transcribe a chunk and send the result back, preserving order."""
        nonlocal transcription_buffer
        try:
            logger.info("Processing chunk #%d, size=%d bytes", seq, len(data))
            text = await asyncio.to_thread(transcribe_audio_chunk, openai_client, data)
            if not text.strip():
                logger.info("Chunk #%d: empty text, skipping", seq)
                return  # Skip empty/hallucinated results
            logger.info("Chunk #%d: sending text=%r", seq, text[:100])
            transcription_buffer += text
            msg = TranscriptionMessage(text=text)
            async with send_lock:
                await websocket.send_text(msg.model_dump_json())
        except openai_module.RateLimitError:
            error_msg = ErrorMessage(message="Rate limit exceeded. Please wait before sending more audio.")
            async with send_lock:
                await websocket.send_text(error_msg.model_dump_json())
        except openai_module.APIError as e:
            error_msg = ErrorMessage(message=f"Transcription API error: {e}")
            async with send_lock:
                await websocket.send_text(error_msg.model_dump_json())
        except Exception as e:
            error_msg = ErrorMessage(message=f"Transcription error: {e}")
            async with send_lock:
                await websocket.send_text(error_msg.model_dump_json())

    seq = 0
    try:
        while True:
            data = await websocket.receive_bytes()
            seq += 1
            task = asyncio.create_task(process_chunk(data, seq))
            pending_tasks.add(task)
            task.add_done_callback(pending_tasks.discard)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected. Cleaning up.")
    finally:
        # Cancel any in-flight transcription tasks
        for task in pending_tasks:
            task.cancel()
        transcription_buffer = ""


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Accept transcription text and return a Chat API summary."""
    if openai_client is None:
        raise HTTPException(status_code=502, detail="OpenAI API key is not configured.")

    try:
        result = await asyncio.to_thread(summarize_transcript, openai_client, request.text)
        return SummarizeResponse(summary=result)
    except openai_module.RateLimitError:
        raise HTTPException(status_code=502, detail="Summarization rate limited. Please try again.")
    except openai_module.APIError as e:
        raise HTTPException(status_code=502, detail=f"Summarization failed: {e}")

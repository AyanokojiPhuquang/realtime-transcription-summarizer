// Frontend logic for Realtime Transcription Summarizer

// --- State ---
let mediaStream = null;
let mediaRecorder = null;
let websocket = null;
let fullTranscript = "";
let isCapturing = false;
let recordingInterval = null;
let audioStream = null;
let sentenceBuffer = ""; // Buffer to accumulate text until a sentence-ending punctuation is found

// --- DOM Elements ---
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const videoPreview = document.getElementById("videoPreview");
const transcriptionArea = document.getElementById("transcriptionArea");
const summaryArea = document.getElementById("summaryArea");
const errorArea = document.getElementById("errorArea");

// --- Event Listeners ---
startBtn.addEventListener("click", async () => {
  try {
    await startCapture();
  } catch (err) {
    if (err.name === "NotAllowedError") {
      displayError("Screen share permission was denied");
    } else {
      displayError("Failed to start capture: " + err.message);
    }
    resetUI();
  }
});

stopBtn.addEventListener("click", () => {
  stopCapture();
});

// --- Core Capture Logic ---

async function startCapture() {
  clearError();
  fullTranscript = "";
  sentenceBuffer = "";
  transcriptionArea.textContent = "";
  summaryArea.textContent = "Summary will appear here after stopping capture.";

  mediaStream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: true,
  });

  videoPreview.srcObject = mediaStream;

  const videoTrack = mediaStream.getVideoTracks()[0];
  if (videoTrack) {
    videoTrack.addEventListener("ended", onStreamEnded);
  }

  const audioTrack = mediaStream.getAudioTracks()[0];
  if (!audioTrack) {
    throw new Error("No audio track available. Please share a tab with audio.");
  }

  audioStream = new MediaStream([audioTrack]);

  // Open WebSocket
  const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
  websocket = new WebSocket(`${wsProtocol}//${location.host}/ws/audio`);

  isCapturing = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;

  websocket.onmessage = onWebSocketMessage;

  websocket.onerror = () => {
    displayError("WebSocket connection error");
    stopCapture();
  };

  websocket.onclose = (event) => {
    if (isCapturing) {
      if (event.code !== 1000) {
        displayError("WebSocket connection closed unexpectedly (code: " + event.code + ")");
      }
      stopCapture();
    }
  };

  websocket.onopen = () => {
    startRecordingCycle();
  };
}

/**
 * Instead of using timeslice (which produces non-standalone chunks),
 * we stop and restart the MediaRecorder every 4 seconds.
 * Each stop produces a complete, valid WebM file that Whisper can process.
 */
function startRecordingCycle() {
  startNewRecorder();
  recordingInterval = setInterval(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop(); // triggers ondataavailable with a complete WebM blob, then onstop starts a new one
    }
  }, 2000);
}

function startNewRecorder() {
  if (!audioStream || !isCapturing) return;

  const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
    ? "audio/webm;codecs=opus"
    : "audio/webm";

  mediaRecorder = new MediaRecorder(audioStream, { mimeType });

  const chunks = [];

  mediaRecorder.ondataavailable = (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  };

  mediaRecorder.onstop = () => {
    if (chunks.length > 0) {
      const blob = new Blob(chunks, { type: mimeType });
      // Only send if blob is large enough to contain actual audio (skip tiny header-only blobs)
      if (blob.size > 1000 && websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(blob);
      }
    }
    // Start a new recording cycle if still capturing
    if (isCapturing) {
      startNewRecorder();
    }
  };

  mediaRecorder.onerror = (event) => {
    displayError("Recording error: " + (event.error?.message || "Unknown error"));
    stopCapture();
  };

  mediaRecorder.start();
}

function stopCapture() {
  if (!isCapturing) return;
  isCapturing = false;

  if (recordingInterval) {
    clearInterval(recordingInterval);
    recordingInterval = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  audioStream = null;

  videoPreview.srcObject = null;

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.close();
  }
  websocket = null;
  mediaRecorder = null;

  startBtn.disabled = false;
  stopBtn.disabled = true;

  // Flush any remaining text in the sentence buffer
  if (sentenceBuffer.trim()) {
    transcriptionArea.textContent = sentenceBuffer.trim();
    sentenceBuffer = "";
  }

  if (fullTranscript.trim()) {
    requestSummary(fullTranscript);
  }
}

function onStreamEnded() {
  stopCapture();
}

// --- Message Handling and Display ---

function onWebSocketMessage(event) {
  try {
    const msg = JSON.parse(event.data);
    if (msg.type === "transcription" && msg.text) {
      fullTranscript += msg.text + " ";
      sentenceBuffer += msg.text + " ";

      // Split on sentence-ending punctuation: . ! ? … and their combinations
      const sentences = sentenceBuffer.split(/(?<=[.!?…。]+)\s+/);

      if (sentences.length > 1) {
        // We have at least one complete sentence — display the last complete one
        const completeSentences = sentences.slice(0, -1);
        for (const sentence of completeSentences) {
          transcriptionArea.textContent = sentence.trim();
        }
        // Keep the remaining incomplete part in the buffer
        sentenceBuffer = sentences[sentences.length - 1];
      } else {
        // No complete sentence yet — show what we have so far
        transcriptionArea.textContent = sentenceBuffer.trim();
      }
    } else if (msg.type === "error" && msg.message) {
      displayError(msg.message);
    }
  } catch {
    // Non-JSON message, ignore
  }
}

async function requestSummary(transcript) {
  summaryArea.textContent = "Generating summary...";
  try {
    const response = await fetch("/summarize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: transcript }),
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => null);
      const detail = errData?.detail || `Server error (${response.status})`;
      displayError("Summarization failed: " + detail);
      summaryArea.textContent = "Summarization failed. See error above.";
      return;
    }

    const data = await response.json();
    summaryArea.textContent = data.summary || "No summary returned.";
  } catch (err) {
    displayError("Failed to get summary: " + err.message);
    summaryArea.textContent = "Summarization failed. See error above.";
  }
}

// --- Error Handling ---

function displayError(message) {
  errorArea.textContent = message;
  errorArea.classList.remove("hidden");
}

function clearError() {
  errorArea.textContent = "";
  errorArea.classList.add("hidden");
}

function resetUI() {
  isCapturing = false;

  if (recordingInterval) {
    clearInterval(recordingInterval);
    recordingInterval = null;
  }

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    try { mediaRecorder.stop(); } catch {}
  }

  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  audioStream = null;
  videoPreview.srcObject = null;

  if (websocket) {
    try { websocket.close(); } catch {}
    websocket = null;
  }

  mediaRecorder = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

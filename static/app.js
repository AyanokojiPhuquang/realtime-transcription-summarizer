// Frontend logic for Realtime Transcription Summarizer

// --- State ---
let mediaStream = null;
let mediaRecorder = null;
let websocket = null;
let fullTranscript = "";
let isCapturing = false;
let recordingInterval = null;
let audioStream = null;
let sentenceBuffer = "";

// --- DOM Elements ---
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const videoPreview = document.getElementById("videoPreview");
const videoContainer = document.getElementById("videoContainer");
const micPlaceholder = document.getElementById("micPlaceholder");
const transcriptionArea = document.getElementById("transcriptionArea");
const summaryArea = document.getElementById("summaryArea");
const errorArea = document.getElementById("errorArea");

// --- Event Listeners ---
startBtn.addEventListener("click", async () => {
  try {
    await startCapture();
  } catch (err) {
    if (err.name === "NotAllowedError") {
      const source = getSelectedSource();
      displayError(source === "microphone"
        ? "Microphone permission was denied"
        : "Screen share permission was denied");
    } else {
      displayError("Failed to start capture: " + err.message);
    }
    resetUI();
  }
});

stopBtn.addEventListener("click", () => {
  stopCapture();
});

// --- Helper ---

function getSelectedSource() {
  return document.querySelector('input[name="audioSource"]:checked').value;
}

// --- Core Capture Logic ---

async function startCapture() {
  clearError();
  fullTranscript = "";
  sentenceBuffer = "";
  transcriptionArea.textContent = "";
  summaryArea.textContent = "Summary will appear here after stopping capture.";

  const source = getSelectedSource();

  if (source === "microphone") {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: false, audio: true });
    videoContainer.classList.add("hidden");
    micPlaceholder.classList.remove("hidden");
    micPlaceholder.classList.add("flex");
    videoPreview.srcObject = null;
  } else {
    mediaStream = await navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
    videoContainer.classList.remove("hidden");
    micPlaceholder.classList.add("hidden");
    micPlaceholder.classList.remove("flex");
    videoPreview.srcObject = mediaStream;
    const videoTrack = mediaStream.getVideoTracks()[0];
    if (videoTrack) {
      videoTrack.addEventListener("ended", onStreamEnded);
    }
  }

  const audioTrack = mediaStream.getAudioTracks()[0];
  if (!audioTrack) {
    throw new Error("No audio track available.");
  }

  audioStream = new MediaStream([audioTrack]);

  const wsProtocol = location.protocol === "https:" ? "wss:" : "ws:";
  websocket = new WebSocket(`${wsProtocol}//${location.host}/ws/audio`);

  isCapturing = true;
  startBtn.disabled = true;
  stopBtn.disabled = false;
  document.querySelectorAll('input[name="audioSource"]').forEach(r => r.disabled = true);

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

function startRecordingCycle() {
  startNewRecorder();
  // Mic uses 5s chunks for better Whisper accuracy, browser tab uses 2s
  const interval = getSelectedSource() === "microphone" ? 5000 : 2000;
  recordingInterval = setInterval(() => {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }, interval);
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
      if (blob.size > 1000 && websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(blob);
      }
    }
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
  videoContainer.classList.remove("hidden");
  micPlaceholder.classList.add("hidden");
  micPlaceholder.classList.remove("flex");

  if (websocket && websocket.readyState === WebSocket.OPEN) {
    websocket.close();
  }
  websocket = null;
  mediaRecorder = null;

  startBtn.disabled = false;
  stopBtn.disabled = true;
  document.querySelectorAll('input[name="audioSource"]').forEach(r => r.disabled = false);

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

      const sentences = sentenceBuffer.split(/(?<=[.!?…。]+)\s+/);

      if (sentences.length > 1) {
        const completeSentences = sentences.slice(0, -1);
        for (const sentence of completeSentences) {
          transcriptionArea.textContent = sentence.trim();
        }
        sentenceBuffer = sentences[sentences.length - 1];
      } else {
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
  videoContainer.classList.remove("hidden");
  micPlaceholder.classList.add("hidden");
  micPlaceholder.classList.remove("flex");

  if (websocket) {
    try { websocket.close(); } catch {}
    websocket = null;
  }

  mediaRecorder = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
}

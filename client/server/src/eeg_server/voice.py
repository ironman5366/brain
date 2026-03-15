"""Voice controller — local STT (faster-whisper) and TTS (kokoro) for agent↔user communication."""

import asyncio
import io
import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Generator

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Lazy-loaded models (heavy imports)
_whisper_model = None
_tts_pipeline = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model (base)...")
        _whisper_model = WhisperModel("base", compute_type="int8")
        logger.info("Whisper model loaded")
    return _whisper_model


def _get_tts():
    global _tts_pipeline
    if _tts_pipeline is None:
        from kokoro import KPipeline
        logger.info("Loading Kokoro TTS pipeline...")
        _tts_pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
        logger.info("Kokoro TTS loaded")
    return _tts_pipeline


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.webm") -> str:
    """Transcribe audio bytes with faster-whisper. Returns text."""
    whisper = _get_whisper()
    # Write to temp file (faster-whisper needs a file path)
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        segments, _info = whisper.transcribe(f.name, language="en")
        text = " ".join(seg.text.strip() for seg in segments).strip()
    return text


def synthesize_stream(text: str) -> Generator[bytes, None, None]:
    """Generate speech with kokoro, yielding WAV-format audio chunks.

    Uses kokoro's generator to yield audio sentence-by-sentence.
    Each chunk is a complete WAV file that can be concatenated for streaming.
    """
    tts = _get_tts()
    sample_rate = 24000  # kokoro default

    # For streaming, we write a single WAV with all chunks concatenated.
    # We use a raw PCM approach: write the WAV header first with unknown size,
    # then stream PCM chunks. The browser handles incomplete WAV fine.
    header_written = False
    for _gs, _ps, audio in tts(text, voice="af_heart", split_pattern=r"[.!?;]+"):
        if audio is None:
            continue
        audio_np = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        if not header_written:
            # Write WAV header with max size (browser handles truncation)
            buf = io.BytesIO()
            # Write a proper WAV header, then we'll stream PCM data
            sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
            yield buf.getvalue()
            header_written = True
        else:
            # Subsequent chunks: just raw PCM16 bytes (append to the WAV stream)
            buf = io.BytesIO()
            sf.write(buf, audio_np, sample_rate, format="RAW", subtype="PCM_16")
            yield buf.getvalue()


def synthesize_full(text: str) -> bytes:
    """Generate speech with kokoro, returning a complete WAV file."""
    tts = _get_tts()
    sample_rate = 24000
    all_audio = []
    for _gs, _ps, audio in tts(text, voice="af_heart"):
        if audio is None:
            continue
        audio_np = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
        all_audio.append(audio_np)
    if not all_audio:
        return b""
    combined = np.concatenate(all_audio)
    buf = io.BytesIO()
    sf.write(buf, combined, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@dataclass
class VoiceRequest:
    """A pending voice question waiting for the user's verbal response."""
    request_id: str
    question: str
    response: str | None = None
    error: str | None = None
    _done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class VoiceController:
    """Manages voice communication between Claude and the user."""

    _pending: VoiceRequest | None = None
    _subscribers: set[asyncio.Queue] = field(default_factory=set)
    _inbox: list[dict] = field(default_factory=list)

    async def ask(self, context: str, question: str) -> dict:
        """Push question to browser (TTS + record), block until user responds."""
        if self._pending is not None and not self._pending._done.is_set():
            self._pending.error = "Cancelled by new request"
            self._pending._done.set()

        request_id = uuid.uuid4().hex[:12]
        request = VoiceRequest(request_id=request_id, question=question)
        self._pending = request

        await self._push_event("voice_ask", {
            "request_id": request_id,
            "question": question,
        })
        logger.info("Voice ask %s: %s", request_id, question[:80])

        try:
            await asyncio.wait_for(request._done.wait(), timeout=120)
        except asyncio.TimeoutError:
            logger.warning("Voice ask %s timed out", request_id)
            return {
                "request_id": request_id,
                "error": "Voice request timed out after 120s",
                "connected_clients": len(self._subscribers),
            }

        if request.error:
            return {"request_id": request_id, "error": request.error}

        return {"request_id": request_id, "response": request.response}

    def submit_response(self, request_id: str, response: str) -> dict:
        """Called when browser transcribes the user's spoken response to a voice_ask."""
        if self._pending is None or self._pending.request_id != request_id:
            return {"ok": False, "error": "No matching pending request"}
        self._pending.response = response
        self._pending._done.set()
        logger.info("Voice response for %s: %s", request_id, response[:80])
        return {"ok": True}

    async def notify(self, text: str) -> dict:
        """Push a status text to the browser top bar."""
        await self._push_event("status_update", {"text": text})
        return {"ok": True}

    def add_to_inbox(self, message: str) -> dict:
        """Add a user-initiated voice message to the inbox."""
        entry = {
            "id": uuid.uuid4().hex[:12],
            "message": message,
            "timestamp": time.time(),
        }
        self._inbox.append(entry)
        logger.info("Voice inbox +1 (%d total): %s", len(self._inbox), message[:80])
        return {"ok": True, "id": entry["id"]}

    def get_inbox(self) -> list[dict]:
        """Return and clear all pending user-initiated messages."""
        messages = list(self._inbox)
        self._inbox.clear()
        return messages

    def status(self) -> dict:
        pending = None
        if self._pending and not self._pending._done.is_set():
            pending = {
                "request_id": self._pending.request_id,
                "question": self._pending.question,
            }
        return {
            "active": pending is not None,
            "pending": pending,
            "inbox_count": len(self._inbox),
            "connected_clients": len(self._subscribers),
        }

    async def events(self):
        """Async generator yielding SSE events for one browser client."""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.add(queue)
        try:
            yield self._format_sse("connected", {
                "active": self._pending is not None
                and not self._pending._done.is_set(),
            })
            while True:
                event = await queue.get()
                yield event
        finally:
            self._subscribers.discard(queue)

    async def _push_event(self, event_type: str, data: dict) -> None:
        formatted = self._format_sse(event_type, data)
        for queue in self._subscribers:
            await queue.put(formatted)

    @staticmethod
    def _format_sse(event_type: str, data: dict) -> str:
        payload = json.dumps({"type": event_type, **data})
        return f"data: {payload}\n\n"

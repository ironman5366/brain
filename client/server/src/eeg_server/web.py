import asyncio
import json
import logging
import threading
import time
from pathlib import Path

import msgpack
from fastapi import FastAPI, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pylsl import StreamInlet, resolve_byprop

from .bandpower import compute_band_powers
from .spectrum import compute_spectrum
from .ball import BallController
from .board import EEGStream
from .config import BoardMode, ServerConfig
from .cyton import CytonHeadset
from .impedance import ImpedanceChecker, SyntheticImpedanceChecker
from .bci import BCIController
from .calibration import CalibrationController
from .cyton import CYTON_WIRE_COLORS, CYTON_PIN_LABELS
from .signal_quality import analyze_signal_quality
from .session import SessionManager
from .control import ControlSignalComputer
from .voice import VoiceController

logger = logging.getLogger(__name__)


class WebSocketBridge:
    """Bridges an LSL stream to WebSocket clients via binary msgpack frames."""

    def __init__(self, config: ServerConfig, eeg_stream: EEGStream):
        self.config = config
        self.eeg_stream = eeg_stream
        self._inlet: StreamInlet | None = None
        self._clients: set[WebSocket] = set()
        self._broadcast_task: asyncio.Task | None = None

    async def connect_lsl(self) -> None:
        """Resolve and connect to the LSL stream."""
        loop = asyncio.get_event_loop()
        logger.info("Resolving LSL stream '%s'...", self.config.lsl_stream_name)
        streams = await loop.run_in_executor(
            None,
            lambda: resolve_byprop(
                "name", self.config.lsl_stream_name, timeout=10.0
            ),
        )
        if not streams:
            raise RuntimeError(
                f"Could not find LSL stream '{self.config.lsl_stream_name}'"
            )
        self._inlet = StreamInlet(streams[0], max_buflen=360)
        logger.info("Connected to LSL stream")

    async def start(self) -> None:
        """Start the broadcast loop."""
        await self.connect_lsl()
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())

    async def stop(self) -> None:
        """Stop broadcasting."""
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        if self._inlet:
            self._inlet.close_stream()

    async def add_client(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients.add(ws)
        logger.info("WebSocket client connected (%d total)", len(self._clients))

        # Send initial metadata frame
        meta = msgpack.packb(
            {
                "type": "meta",
                "sr": self.eeg_stream.sampling_rate,
                "ch": self.eeg_stream.channel_names,
            }
        )
        try:
            await ws.send_bytes(meta)
        except Exception:
            self._clients.discard(ws)

    def remove_client(self, ws: WebSocket) -> None:
        self._clients.discard(ws)
        logger.info("WebSocket client disconnected (%d total)", len(self._clients))

    async def _broadcast_loop(self) -> None:
        """Pull chunks from LSL and broadcast to all WebSocket clients."""
        loop = asyncio.get_event_loop()
        buffer_sec = self.config.ws_buffer_ms / 1000.0

        while True:
            try:
                samples, timestamps = await loop.run_in_executor(
                    None,
                    lambda: self._inlet.pull_chunk(
                        timeout=buffer_sec, max_samples=512
                    ),
                )

                if len(timestamps) > 0:
                    frame = msgpack.packb(
                        {
                            "type": "data",
                            "t": timestamps,
                            "d": samples,
                        }
                    )

                    disconnected = set()
                    for ws in self._clients.copy():
                        try:
                            await ws.send_bytes(frame)
                        except Exception:
                            disconnected.add(ws)
                    self._clients -= disconnected
                else:
                    await asyncio.sleep(0.01)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("Broadcast error: %s", e)
                await asyncio.sleep(0.1)


class NavigateRequest(BaseModel):
    view: str


class StartSessionRequest(BaseModel):
    protocol_id: str
    protocol_version: str = "1.0.0"


class MarkerBatch(BaseModel):
    session_id: str
    markers: list[dict]


class ResponseRequest(BaseModel):
    session_id: str
    block_id: str
    trial_index: int
    response_key: str
    reaction_time_ms: float
    timestamp: float
    correct: bool | None = None


class StopSessionRequest(BaseModel):
    session_id: str


class FlashRequest(BaseModel):
    sequences: int = 5


class FlashDoneRequest(BaseModel):
    marker_count: int = 0


class ProposeRequest(BaseModel):
    letter: str
    message: str = ""


class FeedbackRequest(BaseModel):
    accepted: bool


class MessageRequest(BaseModel):
    text: str


class PlaySoundRequest(BaseModel):
    frequency: int = 440
    duration_ms: int = 200
    novel: bool = False


class TargetRequest(BaseModel):
    x: float | None = None
    y: float | None = None


class CalibrateRequest(BaseModel):
    session_id: str


def create_app(config: ServerConfig, eeg_stream: EEGStream) -> FastAPI:
    app = FastAPI(title="EEG Server")
    bridge = WebSocketBridge(config, eeg_stream)
    impedance_lock = threading.Lock()
    sessions_dir = Path(__file__).resolve().parent.parent.parent.parent / "sessions"
    session_mgr = SessionManager(sessions_dir, eeg_stream)

    bci = BCIController()
    bci.load_classifier(sessions_dir)
    calibration = CalibrationController()
    control = ControlSignalComputer()
    ball = BallController()
    voice = VoiceController()

    # --- Navigation (agent-driven UI routing) ---
    _nav_subscribers: set[asyncio.Queue] = set()
    _nav_current_view = "dashboard"

    VALID_VIEWS = {"dashboard", "eeg", "impedance", "bandpower", "fft", "experiment", "bci", "calibration", "ball", "asymmetry"}

    async def _nav_broadcast(view: str) -> None:
        msg = f"data: {json.dumps({'type': 'navigate', 'view': view})}\n\n"
        for q in list(_nav_subscribers):
            q.put_nowait(msg)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        await bridge.start()
        async def ball_loop():
            delay = 1.0 / ball.tick_hz
            while True:
                try:
                    await ball.tick(eeg_stream, control)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error("Ball loop error: %s", e)
                await asyncio.sleep(delay)

        app.state.ball_task = asyncio.create_task(ball_loop())

    @app.on_event("shutdown")
    async def shutdown():
        ball_task = getattr(app.state, "ball_task", None)
        if ball_task is not None:
            ball_task.cancel()
            try:
                await ball_task
            except asyncio.CancelledError:
                pass
        await bridge.stop()

    @app.get("/api/status")
    async def get_status():
        return {
            "board_connected": eeg_stream.is_running,
            "board_mode": eeg_stream.config.board_mode.value,
            "sampling_rate": eeg_stream.sampling_rate,
            "channels": eeg_stream.channel_names,
            "num_channels": len(eeg_stream.eeg_channels),
            "num_ws_clients": len(bridge._clients),
            "error": eeg_stream.error,
        }

    @app.get("/api/bandpower")
    async def get_band_powers(window_sec: float = 2.0):
        """Compute current EEG band powers using Welch's method."""
        if not eeg_stream.is_running:
            return {"error": "Board not streaming"}

        num_samples = int(eeg_stream.sampling_rate * window_sec)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_band_powers(
                eeg_stream.get_recent_data(num_samples),
                eeg_stream.eeg_channels,
                eeg_stream.sampling_rate,
            ),
        )
        return result

    @app.get("/api/control")
    async def get_control_signals(window_sec: float = 1.0):
        """Compute control signals (alpha asymmetry + concentration) for ball control."""
        if not eeg_stream.is_running:
            return {"error": "Board not streaming"}

        ball_status_payload = ball.status()
        if (
            ball_status_payload["state"] == "running"
            and ball_status_payload["control"] is not None
        ):
            return ball_status_payload["control"]

        num_samples = int(eeg_stream.sampling_rate * window_sec)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: control.update(
                eeg_stream.get_recent_data(num_samples),
                eeg_stream.eeg_channels,
                eeg_stream.channel_names,
                eeg_stream.sampling_rate,
            ),
        )
        return result

    @app.get("/api/spectrum")
    async def get_spectrum(window_sec: float = 2.0, nfft: int = 512):
        """Compute frequency spectrum (PSD) using Welch's method."""
        if not eeg_stream.is_running:
            return {"error": "Board not streaming"}

        num_samples = max(int(eeg_stream.sampling_rate * window_sec), nfft)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: compute_spectrum(
                eeg_stream.get_recent_data(num_samples),
                eeg_stream.eeg_channels,
                eeg_stream.sampling_rate,
                nfft,
            ),
        )
        return result

    @app.post("/api/impedance/start")
    async def start_impedance_check():
        """Run impedance check on all channels. Returns SSE stream with per-channel results."""
        if not impedance_lock.acquire(blocking=False):
            return StreamingResponse(
                iter(
                    [
                        f"data: {json.dumps({'type': 'error', 'message': 'Impedance check already in progress'})}\n\n"
                    ]
                ),
                media_type="text/event-stream",
                status_code=409,
            )

        # Get thresholds from headset (or defaults for synthetic)
        if eeg_stream.headset is not None:
            thresholds = eeg_stream.headset.get_impedance_thresholds()
        else:
            # Synthetic mode — use Cyton defaults for display
            thresholds = CytonHeadset().get_impedance_thresholds()

        async def generate():
            loop = asyncio.get_event_loop()
            try:
                yield f"data: {json.dumps({'type': 'start', 'channels': eeg_stream.channel_names, 'thresholds': thresholds})}\n\n"

                if config.board_mode == BoardMode.SYNTHETIC:
                    checker = SyntheticImpedanceChecker(len(eeg_stream.eeg_channels))
                else:
                    # Pause normal EEG acquisition
                    await loop.run_in_executor(None, eeg_stream.pause_stream)
                    yield f"data: {json.dumps({'type': 'status', 'message': 'Stream paused'})}\n\n"

                    # Restart board stream for impedance measurement
                    await loop.run_in_executor(
                        None, eeg_stream.board.start_stream
                    )

                    checker = ImpedanceChecker(
                        eeg_stream.board,
                        eeg_stream.board_id,
                        eeg_stream.eeg_channels,
                        eeg_stream.headset,
                    )

                # Measure each channel
                results: dict[str, float] = {}
                for ch_idx in range(len(eeg_stream.eeg_channels)):
                    z = await loop.run_in_executor(
                        None, checker._measure_channel, ch_idx
                    )
                    name = eeg_stream.channel_names[ch_idx]
                    results[name] = z
                    yield f"data: {json.dumps({'type': 'channel', 'index': ch_idx, 'name': name, 'impedance': z})}\n\n"

                yield f"data: {json.dumps({'type': 'done', 'results': results})}\n\n"

            except Exception as e:
                logger.error("Impedance check error: %s", e)
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                # Use synchronous calls in finally — await doesn't work
                # reliably in an async generator's finally block when the
                # client disconnects (Python async generator limitation).
                if config.board_mode != BoardMode.SYNTHETIC:
                    try:
                        eeg_stream.board.stop_stream()
                    except Exception:
                        pass
                    try:
                        eeg_stream.resume_stream()
                    except Exception as e:
                        logger.error("Failed to resume stream: %s", e)
                impedance_lock.release()

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.websocket("/ws/eeg")
    async def websocket_eeg(websocket: WebSocket):
        await bridge.add_client(websocket)
        try:
            # Keep connection alive — wait for client disconnect
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            bridge.remove_client(websocket)

    # --- Navigation Endpoints ---

    @app.get("/api/nav/events")
    async def nav_events():
        """SSE stream for agent-driven UI navigation."""
        async def generate():
            nonlocal _nav_current_view
            queue: asyncio.Queue = asyncio.Queue()
            _nav_subscribers.add(queue)
            try:
                yield f"data: {json.dumps({'type': 'state', 'view': _nav_current_view})}\n\n"
                while True:
                    yield await queue.get()
            finally:
                _nav_subscribers.discard(queue)

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.post("/api/nav/goto")
    async def nav_goto(req: NavigateRequest):
        """Navigate the browser UI to a specific view."""
        nonlocal _nav_current_view
        if req.view not in VALID_VIEWS:
            raise HTTPException(400, f"Unknown view: {req.view}. Valid: {sorted(VALID_VIEWS)}")
        _nav_current_view = req.view
        await _nav_broadcast(req.view)
        return {"ok": True, "view": req.view}

    # --- Asymmetry Check Endpoints ---

    _asymmetry_instruction = ""

    @app.post("/api/asymmetry/instruction")
    async def set_asymmetry_instruction(req: MessageRequest):
        nonlocal _asymmetry_instruction
        _asymmetry_instruction = req.text
        return {"ok": True}

    @app.get("/api/asymmetry/instruction")
    async def get_asymmetry_instruction():
        return {"text": _asymmetry_instruction}

    # --- Session / Experiment Endpoints ---

    @app.post("/api/session/start")
    async def start_session(req: StartSessionRequest):
        """Start a recording session. Returns session_id and server timestamp."""
        if not eeg_stream.is_running:
            raise HTTPException(400, "Board not streaming")
        try:
            session = session_mgr.start_session(req.protocol_id, req.protocol_version)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        return {
            "session_id": session.session_id,
            "started_at": session.started_at,
        }

    @app.post("/api/session/marker")
    async def add_markers(req: MarkerBatch):
        """Add a batch of event markers to the active session."""
        active = session_mgr.active_session
        if active is None or active.session_id != req.session_id:
            raise HTTPException(404, "No active session with that ID")
        active.add_markers(req.markers)
        return {"ok": True, "count": len(req.markers)}

    class AutoMarkerRequest(BaseModel):
        code: str
        metadata: dict = {}

    @app.post("/api/session/marker/auto")
    async def add_marker_auto(req: AutoMarkerRequest):
        """Add a single marker to the active session (no session_id needed).

        Designed for MCP tool callers that don't track session IDs.
        """
        active = session_mgr.active_session
        if active is None:
            raise HTTPException(404, "No active recording session")
        marker = {
            "code": req.code,
            "timestamp": 0,
            "server_timestamp": time.time(),
            "metadata": req.metadata,
        }
        active.add_markers([marker])
        return {"ok": True, "session_id": active.session_id}

    @app.post("/api/session/response")
    async def add_response(req: ResponseRequest):
        """Add a user response to the active session."""
        active = session_mgr.active_session
        if active is None or active.session_id != req.session_id:
            raise HTTPException(404, "No active session with that ID")
        active.add_response(req.model_dump())
        return {"ok": True}

    @app.post("/api/session/stop")
    async def stop_session(req: StopSessionRequest):
        """Stop the active session, save to disk, return summary."""
        active = session_mgr.active_session
        if active is None or active.session_id != req.session_id:
            raise HTTPException(404, "No active session with that ID")
        loop = asyncio.get_event_loop()
        session = await loop.run_in_executor(None, session_mgr.stop_session)
        return {
            "session_id": session.session_id,
            "duration_sec": round(session.markers[-1].server_timestamp - session.started_at, 1) if session.markers else 0,
            "total_markers": len(session.markers),
            "total_responses": len(session.responses),
        }

    @app.get("/api/sessions")
    async def list_sessions():
        """List all saved experiment sessions."""
        return session_mgr.list_sessions()

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get full session metadata."""
        try:
            return session_mgr.get_session(session_id)
        except FileNotFoundError:
            raise HTTPException(404, f"Session not found: {session_id}")

    @app.get("/api/sessions/{session_id}/report")
    async def get_report(session_id: str):
        """Get a session's markdown report."""
        content = session_mgr.get_report(session_id)
        if content is None:
            raise HTTPException(404, "No report for this session")
        return {"content": content}

    @app.put("/api/sessions/{session_id}/report")
    async def save_report(session_id: str, body: dict):
        """Save a markdown report for a session."""
        try:
            session_mgr.save_report(session_id, body["content"])
        except FileNotFoundError:
            raise HTTPException(404, f"Session not found: {session_id}")
        return {"ok": True}

    # --- BCI Speller Endpoints ---

    @app.get("/api/bci/events")
    async def bci_events():
        """SSE stream for the BCI UI."""
        return StreamingResponse(bci.events(), media_type="text/event-stream")

    @app.post("/api/bci/start")
    async def bci_start():
        """Start a BCI speller session."""
        nonlocal _nav_current_view
        if not eeg_stream.is_running:
            raise HTTPException(400, "Board not streaming")
        try:
            result = await bci.start(session_mgr)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        _nav_current_view = "bci"
        await _nav_broadcast("bci")
        return result

    @app.post("/api/bci/stop")
    async def bci_stop():
        """Stop the BCI session and save data."""
        try:
            return await bci.stop(session_mgr)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/bci/flash")
    async def bci_flash(req: FlashRequest):
        """Run N flash sequences. Blocks until the UI finishes flashing."""
        try:
            return await bci.flash(req.sequences)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/bci/flash-done")
    async def bci_flash_done(req: FlashDoneRequest):
        """Called by UI when flashing is complete."""
        return bci.flash_complete(req.marker_count)

    @app.get("/api/bci/epochs")
    async def bci_epochs(
        channels: str = "C3,C4,P7,P8",
        window_start_ms: int = 250,
        window_end_ms: int = 500,
        filter_low: float = 0.5,
        filter_high: float = 30.0,
        artifact_threshold_uv: float = 150.0,
    ):
        """Get P300 row/column scores from the current session."""
        loop = asyncio.get_event_loop()
        try:
            ch_list = [c.strip() for c in channels.split(",")]
            return await loop.run_in_executor(
                None,
                lambda: bci.get_epochs(
                    session_mgr,
                    eeg_stream,
                    channels=ch_list,
                    window_start_ms=window_start_ms,
                    window_end_ms=window_end_ms,
                    filter_low=filter_low,
                    filter_high=filter_high,
                    artifact_threshold_uv=artifact_threshold_uv,
                ),
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/bci/snapshot")
    async def bci_snapshot():
        """Dump raw EEG + markers to a .npz file for custom analysis."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None, lambda: bci.snapshot(session_mgr, eeg_stream)
            )
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/bci/status")
    async def bci_status():
        """Get current BCI state."""
        return bci.status(session_mgr)

    @app.post("/api/bci/propose")
    async def bci_propose(req: ProposeRequest):
        """Propose a letter. Blocks until user accepts or rejects."""
        try:
            return await bci.propose(req.letter, req.message)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/bci/feedback")
    async def bci_feedback(req: FeedbackRequest):
        """UI submits accept/reject for a proposed letter."""
        return bci.submit_feedback(req.accepted)

    @app.post("/api/bci/message")
    async def bci_message(req: MessageRequest):
        """Show a message in the BCI UI."""
        return await bci.send_message(req.text)

    @app.post("/api/bci/play-sound")
    async def bci_play_sound(req: PlaySoundRequest):
        """Play a sound in the user's browser."""
        return await bci.play_sound(req.frequency, req.duration_ms, req.novel)

    # --- Ball Control Endpoints ---

    @app.get("/api/ball/events")
    async def ball_events():
        """SSE stream for the ball-control UI."""
        return StreamingResponse(ball.events(), media_type="text/event-stream")

    @app.get("/api/ball/status")
    async def ball_status():
        """Current server-owned ball state and latest control snapshot."""
        return ball.status()

    @app.post("/api/ball/start")
    async def ball_start():
        """Start an agent-driven ball-control run."""
        nonlocal _nav_current_view
        if not eeg_stream.is_running:
            raise HTTPException(400, "Board not streaming")
        try:
            result = await ball.start(session_mgr, control)
        except RuntimeError as e:
            raise HTTPException(409, str(e))
        _nav_current_view = "ball"
        await _nav_broadcast("ball")
        return result

    @app.post("/api/ball/stop")
    async def ball_stop():
        """Stop the active ball-control run and save EEG data."""
        try:
            return await ball.stop(session_mgr)
        except RuntimeError as e:
            raise HTTPException(409, str(e))

    @app.post("/api/ball/reset")
    async def ball_reset():
        """Recenter the ball and reset control normalization."""
        return await ball.reset(control)

    @app.post("/api/ball/target")
    async def ball_target(req: TargetRequest):
        """Set or clear a target position on the ball canvas."""
        return await ball.set_target(req.x, req.y)

    @app.post("/api/ball/message")
    async def ball_message(req: MessageRequest):
        """Show a message in the ball-control UI."""
        return await ball.send_message(req.text)

    @app.post("/api/bci/calibrate")
    async def bci_calibrate(req: CalibrateRequest):
        """Train P300 classifier from a completed copy-spelling session."""
        loop = asyncio.get_event_loop()
        try:
            return await loop.run_in_executor(
                None,
                lambda: bci.calibrate(session_mgr, eeg_stream, req.session_id),
            )
        except (RuntimeError, FileNotFoundError) as e:
            raise HTTPException(400, str(e))

    # --- Calibration Endpoints ---

    @app.get("/api/calibration/events")
    async def calibration_events():
        """SSE stream for the calibration UI."""
        return StreamingResponse(calibration.events(), media_type="text/event-stream")

    @app.post("/api/calibration/check-impedance")
    async def calibration_check_impedance():
        """Run impedance check, return results enriched with wire colors."""
        if not impedance_lock.acquire(blocking=False):
            raise HTTPException(409, "Impedance check already in progress")

        loop = asyncio.get_event_loop()
        try:
            if eeg_stream.headset is not None:
                thresholds = eeg_stream.headset.get_impedance_thresholds()
                wire_colors = eeg_stream.headset.wire_colors
                pin_labels = eeg_stream.headset.pin_labels
            else:
                headset = CytonHeadset()
                thresholds = headset.get_impedance_thresholds()
                wire_colors = CYTON_WIRE_COLORS
                pin_labels = CYTON_PIN_LABELS

            if config.board_mode == BoardMode.SYNTHETIC:
                checker = SyntheticImpedanceChecker(len(eeg_stream.eeg_channels))
            else:
                await loop.run_in_executor(None, eeg_stream.pause_stream)
                await loop.run_in_executor(None, eeg_stream.board.start_stream)
                checker = ImpedanceChecker(
                    eeg_stream.board,
                    eeg_stream.board_id,
                    eeg_stream.eeg_channels,
                    eeg_stream.headset,
                )

            raw_results = await loop.run_in_executor(None, checker.measure_all)

            channels = []
            for ch_idx, z in raw_results.items():
                name = eeg_stream.channel_names[ch_idx]
                if z < thresholds["good"]:
                    rating = "good"
                elif z < thresholds["ok"]:
                    rating = "ok"
                else:
                    rating = "bad"

                channels.append({
                    "index": ch_idx,
                    "name": name,
                    "wire_color": wire_colors.get(name, "unknown"),
                    "pin": pin_labels.get(name, "unknown"),
                    "impedance_ohms": z,
                    "impedance_kohms": round(z / 1000, 1),
                    "rating": rating,
                })

            result = {
                "channels": channels,
                "thresholds": thresholds,
                "all_good": all(c["rating"] == "good" for c in channels),
            }

            await calibration.push_impedance_update(result)
            return result

        except Exception as e:
            logger.error("Calibration impedance check error: %s", e)
            raise HTTPException(500, str(e))
        finally:
            if config.board_mode != BoardMode.SYNTHETIC:
                try:
                    await loop.run_in_executor(None, eeg_stream.board.stop_stream)
                except Exception:
                    pass
                try:
                    await loop.run_in_executor(None, eeg_stream.resume_stream)
                except Exception as e:
                    logger.error("Failed to resume stream: %s", e)
            impedance_lock.release()

    @app.get("/api/calibration/check-signal")
    async def calibration_check_signal(
        duration_sec: float = 3.0,
        line_freq: float = 60.0,
    ):
        """Analyze live EEG signal quality on all channels."""
        if not eeg_stream.is_running:
            return {"error": "Board not streaming"}

        num_samples = int(eeg_stream.sampling_rate * duration_sec)
        data = eeg_stream.get_recent_data(num_samples)

        if data.shape[1] < eeg_stream.sampling_rate:
            return {"error": "Insufficient data", "samples": int(data.shape[1])}

        if eeg_stream.headset is not None:
            wire_colors = eeg_stream.headset.wire_colors
            pin_labels = eeg_stream.headset.pin_labels
        else:
            wire_colors = CYTON_WIRE_COLORS
            pin_labels = CYTON_PIN_LABELS

        loop = asyncio.get_event_loop()
        channel_results = await loop.run_in_executor(
            None,
            lambda: analyze_signal_quality(
                data,
                eeg_stream.eeg_channels,
                eeg_stream.channel_names,
                eeg_stream.sampling_rate,
                line_freq,
            ),
        )

        for ch in channel_results:
            ch["wire_color"] = wire_colors.get(ch["name"], "unknown")
            ch["pin"] = pin_labels.get(ch["name"], "unknown")

        return {
            "channels": channel_results,
            "duration_sec": duration_sec,
            "all_good": all(c["rating"] == "good" for c in channel_results),
        }

    @app.post("/api/calibration/message")
    async def calibration_message(req: MessageRequest):
        """Show a message from the agent in the calibration UI."""
        return await calibration.send_message(req.text)

    @app.get("/api/calibration/status")
    async def calibration_status():
        """Get current calibration state summary."""
        return calibration._get_state()

    # --- Voice Agent (Claude ↔ user via OpenAI Realtime) ---

    class VoiceAskRequest(BaseModel):
        context: str
        question: str

    @app.get("/api/voice/events")
    async def voice_events():
        """SSE stream for pushing voice requests to the browser."""
        return StreamingResponse(
            voice.events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/api/voice/ask")
    async def voice_ask(req: VoiceAskRequest):
        """Blocking: Claude sends a question, waits for user's verbal response."""
        return await voice.ask(req.context, req.question)

    @app.get("/api/voice/speak")
    async def voice_speak(text: str):
        """Stream TTS audio for the given text. Returns WAV audio."""
        from .voice import synthesize_full
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(None, synthesize_full, text)
        return StreamingResponse(
            iter([audio_bytes]),
            media_type="audio/wav",
            headers={"Cache-Control": "no-cache"},
        )

    @app.post("/api/voice/transcribe")
    async def voice_transcribe(
        audio: UploadFile,
        request_id: str | None = Form(None),
    ):
        """Transcribe uploaded audio with faster-whisper.

        If request_id is provided, submits the transcript as a response to
        a pending voice_ask. Otherwise, adds it to the inbox.
        """
        from .voice import transcribe_audio
        audio_bytes = await audio.read()
        loop = asyncio.get_event_loop()
        text = await loop.run_in_executor(
            None, transcribe_audio, audio_bytes, audio.filename or "audio.webm"
        )

        if request_id:
            voice.submit_response(request_id, text)
        else:
            voice.add_to_inbox(text)

        return {"text": text}

    @app.get("/api/voice/inbox")
    async def voice_inbox():
        """Return and clear all pending user-initiated voice messages."""
        return voice.get_inbox()

    @app.get("/api/voice/status")
    async def voice_status():
        """Check voice controller state."""
        return voice.status()

    class VoiceNotifyRequest(BaseModel):
        text: str

    @app.post("/api/voice/notify")
    async def voice_notify(req: VoiceNotifyRequest):
        """Push a status text to the browser top bar."""
        return await voice.notify(req.text)

    return app

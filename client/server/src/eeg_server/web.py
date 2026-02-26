import asyncio
import json
import logging
import threading

import msgpack
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pylsl import StreamInlet, resolve_byprop

from .bandpower import compute_band_powers
from .spectrum import compute_spectrum
from .board import EEGStream
from .config import BoardMode, ServerConfig
from .cyton import CytonHeadset
from .impedance import ImpedanceChecker, SyntheticImpedanceChecker

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


def create_app(config: ServerConfig, eeg_stream: EEGStream) -> FastAPI:
    app = FastAPI(title="EEG Server")
    bridge = WebSocketBridge(config, eeg_stream)
    impedance_lock = threading.Lock()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def startup():
        await bridge.start()

    @app.on_event("shutdown")
    async def shutdown():
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
                # Resume normal EEG if we're on a real board
                if config.board_mode != BoardMode.SYNTHETIC:
                    try:
                        await loop.run_in_executor(
                            None, eeg_stream.board.stop_stream
                        )
                    except Exception:
                        pass
                    try:
                        await loop.run_in_executor(None, eeg_stream.resume_stream)
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

    return app

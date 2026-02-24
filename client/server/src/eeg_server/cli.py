import logging
import signal
import sys

import click
import uvicorn

from .board import EEGStream
from .config import BoardMode, ServerConfig
from .web import create_app

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--mode",
    type=click.Choice(["cyton", "synthetic"]),
    default="synthetic",
    help="Board mode: 'cyton' for real hardware, 'synthetic' for simulated data",
)
@click.option(
    "--serial-port",
    default="",
    help="Serial port for Cyton board (e.g. /dev/cu.usbserial-DM0258EH)",
)
@click.option("--port", default=8765, type=int, help="Server port")
@click.option("--host", default="0.0.0.0", help="Server host")
def main(mode: str, serial_port: str, port: int, host: str) -> None:
    """Start the EEG streaming server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = ServerConfig(
        board_mode=BoardMode(mode),
        serial_port=serial_port,
        host=host,
        port=port,
    )

    # Start board acquisition + LSL stream
    eeg_stream = EEGStream(config)

    def shutdown(signum, frame):
        logger.info("Shutting down...")
        eeg_stream.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        eeg_stream.start()
    except Exception as e:
        logger.error("Failed to start board: %s", e)
        sys.exit(1)

    # Create and run FastAPI app
    app = create_app(config, eeg_stream)

    try:
        uvicorn.run(app, host=host, port=port, log_level="info")
    finally:
        eeg_stream.stop()
